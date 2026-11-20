"""Phase-1 controls E1 and E3: render the controlled inputs and re-extract features (protocols.md section 8).

Post hoc, outside the locked protocol; never replaces a locked result. The rules are the subsection
"Phase-1 controls E1-E4" of protocols.md (E0 definitions, the E1 and E3 rows, the GPU plan). No hashed
fm_*.py file is edited: every arm loads through fm_extract.load_encoder with the spec stored in the
input lock (weights checked against their pinned SHA256), and E3 changes one step either as a PIL
step before the locked transform or by replacing one name of fm_extract inside the worker process
(fm_extract.mirage_input, or IMAGENET_MEAN/IMAGENET_STD read by the RETFound transform), recorded
by name and value in identity.json. No label is read and no score is computed here.

Two commands, run on Colab by colab_fm_control.ipynb:

    render   (CPU, the locked Python environment): decode (E0), content box (A2 rule repeated to a
             fixed point), E1-M / E1-L / E1-C rendered twice with equal pixel hashes, saved as 8-bit
             RGB PNG under <out>/<set>/<original path>.png; render.csv (receipt per image),
             thumbs16.npz, pixel_stats.csv, edge_residual.csv, 640-px external copies (E2f) and the
             40 PIL-decoded originals for the Inception decoder check.
    extract  (GPU, one process per arm and set, and per E3 variant): features.npz and
             features_external.npz in the locked cache format, keyed by the original paths in
             inventory order; control.npz; identity.json; complete.json. Scoring runs on the
             laptop (fm_posthoc.py).

    python fm_control_extract.py render --root <checkout> --lock <input lock> --external <query lock> --out <phase1>/render
    python fm_control_extract.py extract --root <checkout> --lock <input lock> --external <query lock> \
        --render <phase1>/render --arm MIRAGE --set E1-M --assets <assets/MIRAGE> --out <phase1>/out/E1-M/MIRAGE
    python fm_control_extract.py extract ... --arm RETFound --set E3 --variant R-pad --out <phase1>/out/E3/RETFound/R-pad
"""
from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from contextlib import contextmanager, nullcontext
import hashlib
from io import BytesIO
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image

import fm_core as fm
import fm_extract as extract
import fm_pad_extract as pad
import fm_prepare as prep

# The locked objects, captured before any override, so a variant can call the locked step itself.
LOCKED_MIRAGE_INPUT = extract.mirage_input
LOCKED_MEAN = tuple(extract.IMAGENET_MEAN)
LOCKED_STD = tuple(extract.IMAGENET_STD)

ARMS = ("MIRAGE", "Inception", "RETFound", "DINOv2")
E1_SETS = ("E1-M", "E1-L", "E1-C")
SETS = (*E1_SETS, "E3")
SIDE = 256  # E1-M: one common size for every arm
LOW = 128  # E1-L: the resolution floor
MIN_SIDE = 32  # a content box with a side under this keeps the full frame (flagged, counted)
THUMB = 16
NEAR_BLACK, NEAR_BLACK_SHARE = 8, 0.95  # edge run: rows/columns with >= 95% of pixels <= 8
EXTERNAL_WIDTH = 640  # E2f pass 2: the original scaled to 640 px wide
DECODER_ROWS = dict(Kermany=20, OCTID=10, OCTDL=10)  # Inception decoder check, first rows in inventory order
CONTROL = 20  # control rows for MIRAGE, RETFound (full control is E3-ref) and Inception-v3
REPEAT = 5  # inputs encoded twice at the end, recorded
PARTIAL = ".phase1_partial"  # marks an output folder this script started, so an interrupted run may clear it

# Every controlled input and every E3 variant with its exact parameters; the hash goes into identity.json.
VARIANTS = {
    "E1-M": dict(set="E1-M", arms=list(ARMS), steps=[
        "decode: PIL Image.open, .convert('RGB') (alpha dropped)",
        "content box: A2 rule (grey range <= 8; rows top then bottom, columns left then right) on the grey original, "
        "repeated inside the remaining box until it removes nothing; empty or a side < 32 px: full frame, flagged",
        "centre square of side s = min(w, h) of the box, offsets floor((w - s) / 2), floor((h - s) / 2)",
        "resize to 256 x 256, PIL BICUBIC, reducing_gap=None", "save as 8-bit RGB PNG",
        "each arm reads the PNG as its locked extraction reads a file (Inception-v3: stage3_classifier.load_image; "
        "others: PIL open, .copy(), encode)"]),
    "E1-L": dict(set="E1-L", arms=list(ARMS), steps=[
        "as E1-M up to the centre square", "resize to 128 x 128, PIL BOX, reducing_gap=None",
        "resize to 256 x 256, PIL BICUBIC, reducing_gap=None", "save as 8-bit RGB PNG", "read as E1-M"]),
    "E1-C": dict(set="E1-C", arms=list(ARMS), steps=[
        "as E1-M up to the content box", "crop to the content box only (no square, no resize)",
        "save as 8-bit RGB PNG", "read as E1-M; each arm then squashes or crops as locked"]),
    "MIRAGE/E3-ref": dict(set="E3", arm="MIRAGE", override="fm_extract.mirage_input",
                          value="locked fm_extract.mirage_input (512 x 512 bilinear in the file's mode, L, /255, min-max); "
                                "constant min-max inputs retain their pixel values"),
    "MIRAGE/M-crop": dict(set="E3", arm="MIRAGE", override="fm_extract.mirage_input",
                          value="short edge to 512 bilinear in the file's mode (long edge int(512 * long / short)), "
                                "centre crop 512 x 512 at floor((w - 512) / 2), floor((h - 512) / 2), then L, /255, min-max; "
                                "constant min-max inputs retain their pixel values"),
    "MIRAGE/M-bicubic": dict(set="E3", arm="MIRAGE", override="fm_extract.mirage_input",
                             value="512 x 512 BICUBIC in the file's mode, then L, /255, min-max; "
                                   "constant min-max inputs retain their pixel values"),
    "MIRAGE/M-nostretch": dict(set="E3", arm="MIRAGE", override="fm_extract.mirage_input",
                               value="512 x 512 bilinear in the file's mode, then L, /255; no per-image min-max"),
    "MIRAGE/M-res256": dict(set="E3", arm="MIRAGE", override="fm_extract.mirage_input",
                            value="256 x 256 bilinear in the file's mode, then the same path as MIRAGE/E3-ref"),
    "RETFound/E3-ref": dict(set="E3", arm="RETFound", prestep="identity (return the decoded image unchanged)",
                            override="fm_extract.IMAGENET_MEAN, fm_extract.IMAGENET_STD",
                            value=dict(mean=list(LOCKED_MEAN), std=list(LOCKED_STD))),
    "RETFound/R-squash": dict(set="E3", arm="RETFound", prestep=".convert('RGB'), resize to 256 x 256 BICUBIC"),
    "RETFound/R-pad": dict(set="E3", arm="RETFound", prestep=".convert('RGB'), black pad to square (fm_pad_extract.pad_to_square)"),
    "RETFound/R-bilinear": dict(set="E3", arm="RETFound",
                                prestep=".convert('RGB'), short edge to 256 BILINEAR (long edge int(256 * long / short))"),
    "RETFound/R-stretch": dict(set="E3", arm="RETFound",
                               prestep=".convert('RGB'), per-image min-max over all channels to 0-255, numpy.round, uint8 "
                                       "(a constant image is left unchanged)"),
    "RETFound/R-norm05": dict(set="E3", arm="RETFound", prestep="none",
                              override="fm_extract.IMAGENET_MEAN, fm_extract.IMAGENET_STD",
                              value=dict(mean=[.5, .5, .5], std=[.5, .5, .5])),
    "RETFound/R-res128": dict(set="E3", arm="RETFound",
                              prestep=".convert('RGB'), short edge to 128 BICUBIC (long edge int(128 * long / short))"),
}
E3_VARIANTS = {arm: tuple(name.split("/")[1] for name in VARIANTS if name.startswith(arm + "/")) for arm in ("MIRAGE", "RETFound")}


def registry_hash() -> str:
    return prep.object_hash(VARIANTS)


# ======================================================================= pure geometry ===
Box = tuple[int, int, int, int]  # left, top, right, bottom (right and bottom exclusive)


def peel(gray: np.ndarray) -> tuple[int, int, int, int]:
    """One pass of the A2 rule (fm_core.border_share): flat rows from the top, then the bottom; flat
    columns (over the full height) from the left, then the right. Returns the four counts."""
    a = np.asarray(gray)
    if a.ndim != 2 or not a.size:
        raise ValueError("expected a nonempty two-dimensional grayscale image")
    h, w = a.shape
    rows = np.ptp(a, axis=1) <= fm.PAD_RANGE
    cols = np.ptp(a, axis=0) <= fm.PAD_RANGE

    def leading(flags: np.ndarray, limit: int) -> int:
        stops = np.flatnonzero(~flags[:limit])
        return int(stops[0]) if stops.size else limit

    top = leading(rows, h)
    bottom = leading(rows[::-1], h - top)
    left = leading(cols, w)
    right = leading(cols[::-1], w - left)
    return top, bottom, left, right


def content_box(gray: np.ndarray) -> tuple[Box, Box, bool, int]:
    """(first-pass box, final box, flagged, passes). The A2 rule is applied to the grey original, then
    again inside the remaining box until it removes nothing. The first pass equals fm_core.border_share.
    If the final box is empty or has a side under 32 px, the full frame is kept and the image flagged."""
    a = np.asarray(gray)
    h, w = a.shape
    box, first, passes = (0, 0, w, h), None, 0
    while True:
        left, top, right, bottom = box
        t, b, l, r = peel(a[top:bottom, left:right])
        passes += 1
        new = (left + l, top + t, max(right - r, left + l), max(bottom - b, top + t))
        if first is None:
            first = new
        if new == box:
            break
        box = new
        if box[2] - box[0] <= 0 or box[3] - box[1] <= 0:
            break
    flagged = min(box[2] - box[0], box[3] - box[1]) < MIN_SIDE
    return first, ((0, 0, w, h) if flagged else box), flagged, passes


def box_share(box: Box, width: int, height: int) -> float:
    """Share of the frame outside a box (for the first pass this is fm_core.border_share)."""
    return 1 - max(box[2] - box[0], 0) * max(box[3] - box[1], 0) / (width * height)


def centre_square(box: Box) -> Box:
    """The centred square of side min(w, h) inside a box, offsets floored."""
    left, top, right, bottom = box
    w, h = right - left, bottom - top
    s = min(w, h)
    return left + (w - s) // 2, top + (h - s) // 2, left + (w - s) // 2 + s, top + (h - s) // 2 + s


def decode(path: Path) -> Image.Image:
    """E0 decoding: PIL open, then .convert('RGB'), which drops alpha as every locked recipe does."""
    with Image.open(path) as image:
        return image.convert("RGB")


def grey(image: Image.Image) -> np.ndarray:
    """E0 grey: .convert('RGB').convert('L'), as in fm_extract.geometry."""
    return np.asarray(as_rgb(image).convert("L"))


def as_rgb(image: Image.Image) -> Image.Image:
    return image if image.mode == "RGB" else image.convert("RGB")


def image_meta(path: Path) -> dict:
    """Mode, container format and smallest alpha value of the file as stored (255 without alpha)."""
    with Image.open(path) as image:
        return mode_and_alpha(image)


def mode_and_alpha(image: Image.Image) -> dict:
    has_alpha = "A" in image.getbands() or "transparency" in image.info
    low = image.convert("RGBA").getchannel("A").getextrema()[0] if has_alpha else 255
    return dict(mode=image.mode, format=image.format or "", has_alpha=has_alpha, min_alpha=low)


def frames(image: Image.Image) -> dict:
    """Content box, flag and centre square of a decoded RGB image."""
    first, box, flagged, passes = content_box(grey(image))
    return dict(first=first, box=box, flagged=flagged, passes=passes, square=centre_square(box))


def render_e1m(image: Image.Image, geometry: dict | None = None) -> Image.Image:
    geometry = geometry or frames(image)
    return as_rgb(image).crop(geometry["square"]).resize((SIDE, SIDE), Image.Resampling.BICUBIC, reducing_gap=None)


def render_e1l(image: Image.Image, geometry: dict | None = None) -> Image.Image:
    geometry = geometry or frames(image)
    low = as_rgb(image).crop(geometry["square"]).resize((LOW, LOW), Image.Resampling.BOX, reducing_gap=None)
    return low.resize((SIDE, SIDE), Image.Resampling.BICUBIC, reducing_gap=None)


def render_e1c(image: Image.Image, geometry: dict | None = None) -> Image.Image:
    geometry = geometry or frames(image)
    return as_rgb(image).crop(geometry["box"])


RENDER = {"E1-M": render_e1m, "E1-L": render_e1l, "E1-C": render_e1c}


def render_all(image: Image.Image) -> tuple[dict, dict[str, Image.Image]]:
    """Geometry and the three E1 renders of one decoded image."""
    geometry = frames(image)
    return geometry, {name: function(image, geometry) for name, function in RENDER.items()}


def png_bytes(image: Image.Image) -> bytes:
    if image.mode != "RGB":
        raise ValueError("rendered inputs are 8-bit RGB")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def edge_run(gray: np.ndarray) -> float:
    """Longest near-black edge run (rows or columns with >= 95% of pixels <= 8) as a share of the side."""
    a = np.asarray(gray)
    if a.ndim != 2 or not a.size:
        raise ValueError("expected a nonempty two-dimensional grayscale image")
    dark = a <= NEAR_BLACK
    rows, cols = dark.mean(axis=1) >= NEAR_BLACK_SHARE, dark.mean(axis=0) >= NEAR_BLACK_SHARE

    def run(flags: np.ndarray) -> int:
        stops = np.flatnonzero(~flags)
        return int(stops[0]) if stops.size else len(flags)

    h, w = a.shape
    return max(run(rows) / h, run(rows[::-1]) / h, run(cols) / w, run(cols[::-1]) / w)


def thumb16(image: Image.Image) -> np.ndarray:
    """Grey (E0) squashed to 16 x 16 with PIL BOX; 256 values / 255."""
    return grey_thumb16(grey(image))


def grey_thumb16(gray: np.ndarray) -> np.ndarray:
    small = Image.fromarray(gray).resize((THUMB, THUMB), Image.Resampling.BOX, reducing_gap=None)
    return np.asarray(small, dtype=np.float64).reshape(-1) / 255


STAT_NAMES = ("mean", "sd", "p01", "p50", "p99", "share_le8", "share_ge250", "entropy_bits")


def pixel_stats(gray: np.ndarray) -> list[float]:
    """The 8 S statistics of E4: mean, SD (ddof 0), 1st/50th/99th percentile (numpy linear), share <= 8,
    share >= 250, entropy in bits of the 256-bin histogram."""
    a = np.asarray(gray)
    if a.dtype != np.uint8 or a.ndim != 2 or not a.size:
        raise ValueError("expected an 8-bit grey image")
    values = a.astype(np.float64).reshape(-1)
    p = np.bincount(a.reshape(-1), minlength=256) / a.size
    p = p[p > 0]
    return [float(values.mean()), float(values.std()), *map(float, np.percentile(values, [1, 50, 99])),
            float(np.mean(a <= 8)), float(np.mean(a >= fm.NEAR_WHITE)), float(-(p * np.log2(p)).sum())]


def external_copy(image: Image.Image) -> Image.Image:
    """E2f pass 2: the original (RGB) scaled to 640 px wide, BICUBIC."""
    rgb = as_rgb(image)
    return rgb.resize((EXTERNAL_WIDTH, max(1, round(rgb.height * EXTERNAL_WIDTH / rgb.width))),
                      Image.Resampling.BICUBIC, reducing_gap=None)


# ======================================================================= E3 variants ===
def short_edge(image: Image.Image, side: int, method: int) -> Image.Image:
    """Short edge to `side`, long edge int(side * long / short) (torchvision's Resize(int) size rule)."""
    w, h = image.size
    size = (side, int(side * h / w)) if w <= h else (int(side * w / h), side)
    return image.resize(size, method)


def minmax(pixels: np.ndarray) -> np.ndarray:
    low, high = pixels.min(), pixels.max()
    if not high > low:
        return pixels
    return (pixels - low) / (high - low)


def _m_crop(image: Image.Image) -> np.ndarray:
    resized = short_edge(image, 512, Image.Resampling.BILINEAR)
    left, top = (resized.width - 512) // 2, (resized.height - 512) // 2
    cropped = resized.crop((left, top, left + 512, top + 512))
    return minmax(np.array(cropped.convert("L"), dtype=np.float32) / 255)


def _m_bicubic(image: Image.Image) -> np.ndarray:
    return minmax(np.array(image.resize((512, 512), Image.Resampling.BICUBIC).convert("L"), dtype=np.float32) / 255)


def _m_nostretch(image: Image.Image) -> np.ndarray:
    return np.array(image.resize((512, 512), Image.Resampling.BILINEAR).convert("L"), dtype=np.float32) / 255


def _m_reference(image: Image.Image) -> np.ndarray:
    try:
        return LOCKED_MIRAGE_INPUT(image)
    except ValueError as exc:
        if str(exc) != "MIRAGE min-max input needs a nonconstant image":
            raise
        return _m_nostretch(image)


def _m_res256(image: Image.Image) -> np.ndarray:
    return _m_reference(image.resize((256, 256), Image.Resampling.BILINEAR))


def mirage_input_variant(name: str) -> Callable[[Image.Image], np.ndarray]:
    """A replacement for fm_extract.mirage_input with the same signature; E3-ref is the locked function."""
    table = {"E3-ref": _m_reference, "M-crop": _m_crop, "M-bicubic": _m_bicubic,
             "M-nostretch": _m_nostretch, "M-res256": _m_res256}
    if name not in table:
        raise ValueError(f"unknown MIRAGE variant {name}")
    return table[name]


def _stretch(image: Image.Image) -> Image.Image:
    a = np.asarray(image, dtype=np.float64)
    low, high = a.min(), a.max()
    if not high > low:
        return image.copy()
    return Image.fromarray(np.round((a - low) / (high - low) * 255).astype(np.uint8), "RGB")


def _identity(image: Image.Image) -> Image.Image:
    return image


def retfound_prestep(name: str) -> Callable[[Image.Image], Image.Image] | None:
    """The PIL step placed before the locked RETFound transform (None: no step)."""
    if name == "E3-ref":
        return _identity
    if name == "R-norm05":
        return None
    table = {
        "R-squash": lambda im: im.resize((256, 256), Image.Resampling.BICUBIC),
        "R-pad": lambda im: pad.pad_to_square(im)[0],
        "R-bilinear": lambda im: short_edge(im, 256, Image.Resampling.BILINEAR),
        "R-stretch": _stretch,
        "R-res128": lambda im: short_edge(im, 128, Image.Resampling.BICUBIC),
    }
    if name not in table:
        raise ValueError(f"unknown RETFound variant {name}")
    step = table[name]
    return lambda image: step(image.convert("RGB"))


def retfound_constants(name: str) -> tuple[list[float], list[float]]:
    if name == "R-norm05":
        return [.5, .5, .5], [.5, .5, .5]
    return list(LOCKED_MEAN), list(LOCKED_STD)


def overrides(arm: str, variant: str) -> dict:
    """The names of fm_extract replaced in this worker and their values (for identity.json)."""
    if arm == "MIRAGE":
        function = mirage_input_variant(variant)
        return {"fm_extract.mirage_input": f"fm_control_extract.{function.__name__}"}
    mean, std = retfound_constants(variant)
    record = {"fm_extract.IMAGENET_MEAN": mean, "fm_extract.IMAGENET_STD": std}
    if variant == "E3-ref":
        record["prestep"] = "fm_control_extract._identity"
    return record


@contextmanager
def applied(arm: str, variant: str):
    """Replace the recorded names inside this process (load the encoder inside the block)."""
    if (arm not in E3_VARIANTS) or variant not in E3_VARIANTS[arm]:
        raise ValueError(f"E3 has no variant {variant} for {arm}")
    saved = extract.mirage_input, extract.IMAGENET_MEAN, extract.IMAGENET_STD
    try:
        if arm == "MIRAGE":
            extract.mirage_input = mirage_input_variant(variant)
        else:
            extract.IMAGENET_MEAN, extract.IMAGENET_STD = retfound_constants(variant)
        yield
    finally:
        extract.mirage_input, extract.IMAGENET_MEAN, extract.IMAGENET_STD = saved


def input_check(arm: str, variant: str, image: Image.Image) -> tuple[bool | None, bool]:
    """Record input change and whether a constant input bypassed min-max."""
    if arm == "MIRAGE":
        pixels = mirage_input_variant(variant)(image)
        changed = variant != "E3-ref" and not np.array_equal(pixels, _m_reference(image))
        return changed, bool(variant != "M-nostretch" and np.ptp(pixels) == 0)
    step = retfound_prestep(variant)
    if variant == "E3-ref" or step is None:
        return (None if variant == "R-norm05" else False), False
    rgb = image.convert("RGB")
    return pad.pixel_hash(step(image)) != pad.pixel_hash(rgb), bool(variant == "R-stretch" and np.ptp(np.asarray(rgb)) == 0)


# ======================================================================= render (CPU) ===
RENDER_HEADER = ["path", "sha256", "mode", "format", "has_alpha", "min_alpha", "width", "height",
                 "first_left", "first_top", "first_right", "first_bottom", "first_share",
                 "box_left", "box_top", "box_right", "box_bottom", "passes", "flagged",
                 "square_left", "square_top", "square_side",
                 *[f"{s}_{k}" for s in E1_SETS for k in ("width", "height", "png_sha256", "pixel_sha256")]]
EDGE_SOURCES = ("original", *E1_SETS)
EDGE_HEADER = ["path", *[f"{s}_{k}" for s in EDGE_SOURCES for k in ("border_share", "edge_run", "near_white_share")]]
STATS_HEADER = ["path", *[f"{s}_{k}" for s in ("original", "E1-M") for k in STAT_NAMES]]


def rendered_path(render: Path, name: str, path: str) -> Path:
    """Where the render of `path` for set `name` lives (the original path plus .png)."""
    return render / name / prep.safe_id(path + ".png")


def source_of(path: str) -> str:
    parts = path.split("/")
    if parts[:3] == ["data", "raw", "OCT2017"]:
        return "Kermany"
    if parts[:2] == ["data", "external"] and len(parts) > 2:
        return parts[2]
    return "rare" if parts[:3] == ["data", "raw", "rare"] else "other"


def decoder_rows(main: prep.Rows, external: prep.Rows) -> prep.Rows:
    """The first 20 Kermany, 10 OCTID and 10 OCTDL rows in inventory order (fewer if absent)."""
    chosen = []
    for source, n in DECODER_ROWS.items():
        chosen += [r for r in (main if source == "Kermany" else external) if source_of(r["path"]) == source][:n]
    return chosen


def fresh(out: Path) -> Path:
    """An output folder: kept if complete; cleared if this script left it unfinished; refused otherwise."""
    if (out / "complete.json").exists():
        return out
    if out.exists() and any(out.iterdir()):
        if not (out / PARTIAL).exists():
            raise ValueError(f"{out} holds files this script did not start; choose an empty folder")
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / PARTIAL).write_text("started\n", encoding="utf-8")
    return out


def finish(out: Path, files: list[str], written: dict[str, str] | None = None) -> None:
    if written is None:
        prep.complete(out, files)
    else:
        hashes = {**written, **{name: prep.sha256(out / name) for name in files}}
        prep.verify_child_coverage(out, hashes)
        prep.write_json(out / "complete.json", {"files": hashes})
    (out / PARTIAL).unlink()


def save_png(target: Path, image: Image.Image) -> str:
    """Write a lossless PNG and return the hash of the bytes written."""
    data = png_bytes(image)
    target.parent.mkdir(parents=True, exist_ok=True)
    prep.write_once(target, data)
    return hashlib.sha256(data).hexdigest()


def safe_hardware() -> dict:
    try:
        return extract.hardware()
    except Exception:  # no GPU driver (laptop dry run): the render is CPU work
        return dict(gpu="none (CPU)")


def select(rows: prep.Rows, only: set[str] | None) -> prep.Rows:
    return rows if only is None else [r for r in rows if r["path"] in only]


def render(root: Path, lock_folder: Path, external: Path | None, out: Path, only: set[str] | None = None) -> dict:
    """Render every inventory image (main, then external) twice; write the receipts. Reads no labels."""
    started = time.perf_counter()
    lock = prep.load_lock(lock_folder)
    main_rows = prep.read_csv(lock_folder / "extraction_inventory.csv")
    ext_rows: prep.Rows = []
    if external is not None:
        from fm_external import verify_external
        verify_external(external, lock_folder)
        ext_rows = prep.read_csv(external / "external_extraction_inventory.csv")
    main_rows, ext_rows = select(main_rows, only), select(ext_rows, only)
    out = prep.out_dir(root, out)
    if (out / "complete.json").exists():
        prep.verify_complete(out)
        print("render: already complete")
        return prep.read_json(out / "identity.json")
    fresh(out)
    files, receipt, edges, stats, thumbs = {}, [], [], [], {k: [] for k in ("raw", *E1_SETS)}
    counts = dict(flagged=0, alpha_below_255=0, modes={}, formats={}, passes={})
    rows = [*main_rows, *ext_rows]
    decoder_inventory = decoder_rows(main_rows, ext_rows)
    decoder_hashes = {row["path"]: "" for row in decoder_inventory}
    for i, row in enumerate(rows):
        path = root / prep.safe_id(row["path"])
        source = path.read_bytes()
        if hashlib.sha256(source).hexdigest() != row["sha256"]:
            raise ValueError(f"input image content differs from the lock: {row['path']}")
        with Image.open(BytesIO(source)) as image:
            meta = mode_and_alpha(image)
            original = image.convert("RGB")
        geometry, first = render_all(original)
        again = render_all(original)[1]  # repeat all geometry and resize operations on the checked decode
        raw_gray = grey(original)
        first_share = box_share(geometry["first"], original.width, original.height)
        line = [row["path"], row["sha256"], meta["mode"], meta["format"], meta["has_alpha"], meta["min_alpha"],
                original.width, original.height, *geometry["first"], first_share, *geometry["box"], geometry["passes"],
                geometry["flagged"], geometry["square"][0], geometry["square"][1], geometry["square"][2] - geometry["square"][0]]
        edge = [row["path"], first_share, edge_run(raw_gray), fm.near_white_share(raw_gray)]
        grays = {}
        for name in E1_SETS:
            image = first[name]
            digest = pad.pixel_hash(image)
            if pad.pixel_hash(again[name]) != digest:
                raise ValueError(f"{name} render of {row['path']} differs between two renders")
            target = rendered_path(out, name, row["path"])
            png_hash = save_png(target, image)
            with Image.open(target) as saved:
                if pad.pixel_hash(saved) != digest:
                    raise ValueError(f"{name} saved PNG of {row['path']} differs from the repeated render")
            files[target.relative_to(out).as_posix()] = png_hash
            line += [image.width, image.height, png_hash, digest]
            g = grays[name] = grey(image)
            edge += [fm.border_share(g), edge_run(g), fm.near_white_share(g)]
            thumbs[name].append(grey_thumb16(g))
        thumbs["raw"].append(grey_thumb16(raw_gray))
        stats.append([row["path"], *pixel_stats(raw_gray), *pixel_stats(grays["E1-M"])])
        receipt.append(line)
        edges.append(edge)
        counts["flagged"] += geometry["flagged"]
        counts["alpha_below_255"] += meta["min_alpha"] < 255
        for key, value in (("modes", meta["mode"]), ("formats", meta["format"]), ("passes", str(geometry["passes"]))):
            counts[key][value] = counts[key].get(value, 0) + 1
        if i >= len(main_rows):
            target = out / "external640" / prep.safe_id(row["path"] + ".png")
            files[target.relative_to(out).as_posix()] = save_png(target, external_copy(original))
        if row["path"] in decoder_hashes:
            target = out / "decoder" / prep.safe_id(row["path"] + ".png")
            digest = save_png(target, original)
            files[target.relative_to(out).as_posix()] = decoder_hashes[row["path"]] = digest
        if (i + 1) % 500 == 0:
            print(f"render: {i + 1}/{len(rows)}", flush=True)
    # The Inception decoder check: originals decoded by PIL (E0) and saved as PNG, encoded by the E1-M worker.
    decoder = [[row["path"], row["sha256"], decoder_hashes[row["path"]]] for row in decoder_inventory]
    prep.write_csv(out / "render.csv", RENDER_HEADER, receipt)
    prep.write_csv(out / "edge_residual.csv", EDGE_HEADER, edges)
    prep.write_csv(out / "pixel_stats.csv", STATS_HEADER, stats)
    prep.write_csv(out / "decoder.csv", ["path", "sha256", "png_sha256"], decoder)
    buffer = BytesIO()
    np.savez(buffer, paths=np.array([r["path"] for r in rows]), sha256=np.array([r["sha256"] for r in rows]),
             **{name: np.array(values, dtype=np.float64).reshape(len(rows), THUMB * THUMB) for name, values in thumbs.items()})
    prep.write_once(out / "thumbs16.npz", buffer.getvalue())
    import PIL
    identity = dict(
        step="phase1_render", label="post hoc, Phase-1 rules E0/E1 (protocols.md section 8); outside the locked protocol",
        variants={k: v for k, v in VARIANTS.items() if k in E1_SETS}, registry_sha256=registry_hash(),
        script_sha256=prep.text_hash(Path(__file__)), code=prep.code_identity(),
        code_matches_lock=prep.code_identity() == lock["code"], pillow=PIL.__version__,
        environment=extract.environment(), hardware=safe_hardware(),
        lock_sha256=prep.sha256(lock_folder / "input_lock.json"),
        external_lock_sha256=prep.sha256(external / "external_lock.json") if external else None,
        images=dict(main=len(main_rows), external=len(ext_rows), decoder=len(decoder)),
        subset=sorted(only) if only is not None else None, rendered_twice="pixel hashes equal for every image and set",
        counts=counts, render_csv_sha256=prep.sha256(out / "render.csv"),
        seconds=dict(total=time.perf_counter() - started))
    prep.write_json(out / "identity.json", identity)
    finish(out, ["render.csv", "edge_residual.csv", "pixel_stats.csv", "decoder.csv", "thumbs16.npz", "identity.json"], files)
    return identity


# ======================================================================= extract (GPU) ===
def read_rendered(render_folder: Path, lock_folder: Path, external: Path) -> dict[str, dict[str, str]]:
    """Bind the render receipts to both locks. Each PNG is checked as the worker reads it."""
    files = prep.read_json(render_folder / "complete.json")["files"]
    prep.verify_files(render_folder, {name: files[name] for name in ("render.csv", "decoder.csv", "identity.json")})
    identity = prep.read_json(render_folder / "identity.json")
    if identity["subset"] is not None or identity["lock_sha256"] != prep.sha256(lock_folder / "input_lock.json") or \
            identity["external_lock_sha256"] != prep.sha256(external / "external_lock.json") or \
            identity["registry_sha256"] != registry_hash():
        raise ValueError("render is incomplete or belongs to different locks or variants")
    return {r["path"]: r for r in prep.read_csv(render_folder / "render.csv")}


def source_file(render_folder: Path | None, root: Path, name: str, row: dict, receipt: dict | None) -> Path:
    """The file the arm reads for this row: the checked render for E1 sets, the checked original for E3."""
    if name == "E3":
        path = root / prep.safe_id(row["path"])
        if prep.sha256(path) != row["sha256"]:
            raise ValueError(f"input image content differs from the lock: {row['path']}")
        return path
    entry = (receipt or {}).get(row["path"])
    if entry is None or entry["sha256"] != row["sha256"]:
        raise ValueError(f"no render of the locked content for {row['path']}")
    path = rendered_path(render_folder, name, row["path"])
    if prep.sha256(path) != entry[f"{name}_png_sha256"]:
        raise ValueError(f"{name} render differs from its receipt: {row['path']}")
    return path


def decoder_png(render_folder: Path, row: dict) -> Path:
    """A PIL-decoded original saved as PNG by the render step, checked against decoder.csv."""
    path = render_folder / "decoder" / prep.safe_id(row["path"] + ".png")
    if prep.sha256(path) != row["png_sha256"]:
        raise ValueError(f"decoder-check PNG differs from its receipt: {row['path']}")
    return path


def encode_file(encoder: extract.Encoder, path: Path, prestep: Callable | None = None) -> np.ndarray:
    """One file exactly as the locked extraction reads it (Keras: stage3 load_image; others: PIL open,
    .copy(), encode), with an E3 PIL step before the locked transform if given."""
    if encoder.stage_path and prestep is None:
        return encoder.stage_path(path)()
    with Image.open(path) as image:
        decoded = image.copy()
    return encoder.encode([prestep(decoded) if prestep else decoded])


def encode_rows(encoder: extract.Encoder, rows: prep.Rows, locate: Callable[[dict], Path],
                prestep: Callable | None = None, log: list | None = None, check: Callable | None = None) -> np.ndarray:
    out = np.empty((len(rows), extract.DIMENSIONS[encoder.arm]), dtype=np.float32)
    for i, row in enumerate(rows):
        path = locate(row)
        values = encode_file(encoder, path, prestep)
        extract.validate_features(values, 1, extract.DIMENSIONS[encoder.arm])
        out[i] = values[0]
        if check is not None and log is not None:
            with Image.open(path) as image:
                width, height = image.size
                log.append([row["path"], width, height, width == height, *check(image.copy())])
        if (i + 1) % 500 == 0:
            print(f"{encoder.arm}: {i + 1}/{len(rows)}", flush=True)
    return out


def control_rows(arm: str, name: str, main: prep.Rows, ext: prep.Rows) -> tuple[prep.Rows, prep.Rows]:
    """E0 control: unchanged originals through the locked path. DINOv2: all images. MIRAGE and RETFound: the
    first 20 here (their full control is E3-ref). Inception-v3: the first 20. E3 workers: none (E3-ref is it)."""
    if name == "E3":
        return [], []
    if arm == "DINOv2":
        return main, ext
    return main[:CONTROL], []


def tf_resize_identity(paths: Sequence[Path]) -> dict:
    """Inception-v3: tf.image.resize of a 256 x 256 array to 256 x 256 returns it unchanged."""
    import tensorflow as tf
    arrays = [np.arange(SIDE * SIDE * 3, dtype=np.int64).reshape(SIDE, SIDE, 3) % 251]
    for path in paths:
        arrays.append(tf.io.decode_image(tf.io.read_file(str(path)), channels=3, expand_animations=False).numpy())
    failures = 0
    for array in arrays:
        if array.shape[:2] != (SIDE, SIDE):
            failures += 1
        else:
            resized = tf.image.resize(array, [SIDE, SIDE]).numpy()
            failures += not np.array_equal(resized, array.astype(np.float32))
    return dict(arrays=len(arrays), unchanged=failures == 0, failures=failures)


def out_files(arm: str, name: str) -> list[str]:
    files = ["features.npz", "features_external.npz"]
    if name != "E3":
        files.append("control.npz")
        if arm == "DINOv2":
            files.append("control_external.npz")
    else:
        files.append("variant_inputs.csv")
    if arm == "Inception" and name == "E1-M":
        files.append("decoder_check.npz")
    return files + ["identity.json"]


def run_extract(root: Path, lock_folder: Path, external: Path, render_folder: Path | None, arm: str, name: str,
                variant: str | None, assets: Path, out: Path, load: Callable = extract.load_encoder) -> dict:
    """One arm and one feature set (or one E3 variant). `load` is injectable for the laptop tests."""
    if arm not in ARMS or name not in SETS:
        raise ValueError("unknown arm or set")
    if (name == "E3") != (variant is not None):
        raise ValueError("--variant is required for E3 and only for E3")
    if name == "E3" and (arm not in E3_VARIANTS or variant not in E3_VARIANTS[arm]):
        raise ValueError(f"E3 has no variant {variant} for {arm}")
    started = time.perf_counter()
    lock = prep.load_lock(lock_folder)  # also checks that the hashed code is the locked code
    from fm_external import verify_external
    verify_external(external, lock_folder)
    locked = lock["arms"][arm]
    if locked["status"] != "available":
        raise ValueError("arm unavailable at the input lock")
    spec = locked["identity"]["spec"]
    main_rows = prep.read_csv(lock_folder / "extraction_inventory.csv")
    ext_rows = prep.read_csv(external / "external_extraction_inventory.csv")
    out = prep.out_dir(root, out)
    if (out / "complete.json").exists():
        prep.verify_complete(out)
        print(f"{arm} {name} {variant or ''}: already complete")
        return prep.read_json(out / "identity.json")
    fresh(out)
    receipt = read_rendered(render_folder, lock_folder, external) if name != "E3" else None
    locate = lambda row: source_file(render_folder, root, name, row, receipt)  # noqa: E731
    prestep = retfound_prestep(variant) if name == "E3" and arm == "RETFound" else None
    record = overrides(arm, variant) if name == "E3" else {}
    with (applied(arm, variant) if name == "E3" else nullcontext()):
        encoder = load(spec, assets)
        loaded = time.perf_counter()
        changes: list[list] = []
        check = (lambda image: input_check(arm, variant, image)) if name == "E3" else None
        identity_check = (tf_resize_identity([locate(r) for r in main_rows[:REPEAT]])
                          if arm == "Inception" and name == "E1-M" else None)
        main = encode_rows(encoder, main_rows, locate, prestep, changes, check)
        ext = encode_rows(encoder, ext_rows, locate, prestep, changes, check)
        encoded = time.perf_counter()
        first = [main[i:i + 1] for i in range(REPEAT)]
        again = [encode_file(encoder, locate(r), prestep) for r in main_rows[:REPEAT]]
        control_main, control_ext = control_rows(arm, name, main_rows, ext_rows)
        original = lambda row: source_file(None, root, "E3", row, None)  # noqa: E731  (checked originals)
        control = encode_rows(encoder, control_main, original) if control_main else None
        control_external = encode_rows(encoder, control_ext, original) if control_ext else None
        decoder, decoder_list = None, []
        if arm == "Inception" and name == "E1-M":
            decoder_list = prep.read_csv(render_folder / "decoder.csv")
            decoder = encode_rows(encoder, decoder_list, lambda row: decoder_png(render_folder, row))
        checked = time.perf_counter()
    paths = lambda rows: [r["path"] for r in rows]  # noqa: E731
    digests = lambda rows: [r["sha256"] for r in rows]  # noqa: E731
    extract.save_features(out / "features.npz", paths(main_rows), digests(main_rows), main)
    extract.save_features(out / "features_external.npz", paths(ext_rows), digests(ext_rows), ext)
    if control is not None:
        extract.save_features(out / "control.npz", paths(control_main), digests(control_main), control)
    if control_external is not None:
        extract.save_features(out / "control_external.npz", paths(control_ext), digests(control_ext), control_external)
    if decoder is not None:
        extract.save_features(out / "decoder_check.npz", paths(decoder_list), digests(decoder_list), decoder)
    if name == "E3":
        prep.write_csv(out / "variant_inputs.csv",
                       ["path", "width", "height", "square", "input_changed", "constant_minmax_input"], changes)
    current = extract.environment()
    entry = VARIANTS[name] if name != "E3" else VARIANTS[f"{arm}/{variant}"]
    identity = dict(
        step="phase1_control_extract", arm=arm, set=name, variant=variant,
        label=f"post hoc, Phase-1 rule {'E1' if name != 'E3' else 'E3'} (protocols.md section 8); outside the locked "
              "protocol; never replaces a locked result",
        transform=entry, overrides=record, registry_sha256=registry_hash(), script_sha256=prep.text_hash(Path(__file__)),
        spec=spec, spec_sha256=prep.object_hash(spec), code=prep.code_identity(),
        code_matches_lock=prep.code_identity() == locked["identity"]["code"], environment=current,
        environment_matches_lock=current == locked["identity"]["environment"], hardware=safe_hardware(),
        lock_sha256=prep.sha256(lock_folder / "input_lock.json"), external_lock_sha256=prep.sha256(external / "external_lock.json"),
        render_csv_sha256=prep.sha256(render_folder / "render.csv") if name != "E3" else None,
        render_identity_sha256=prep.sha256(render_folder / "identity.json") if name != "E3" else None,
        images=dict(main=len(main_rows), external=len(ext_rows)),
        control=dict(main=len(control_main), external=len(control_ext),
                     path="unchanged originals through the locked extraction path" if control_main else
                     "none in this worker: E3-ref is the control of E3"),
        decoder_check=len(decoder) if decoder is not None else None, tf_resize_identity=identity_check,
        inputs_changed=sum(1 for c in changes if c[4] is True) if name == "E3" else None,
        constant_minmax_inputs=sum(1 for c in changes if c[5]) if name == "E3" else None,
        repeat_check=extract.repeat_check(first, again), load_report=encoder.load_report,
        seconds=dict(loading=loaded - started, encoding=encoded - loaded, checks=checked - encoded,
                     total=time.perf_counter() - started))
    prep.write_json(out / "identity.json", identity)
    finish(out, out_files(arm, name))
    return identity


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="step", required=True)
    command = sub.add_parser("render")
    for name in ("root", "lock", "out"):
        command.add_argument("--" + name, type=Path, required=True)
    command.add_argument("--external", type=Path, help="external query lock (Colab always passes it)")
    command.add_argument("--only", type=Path, help="laptop dry run: a text file of inventory paths to render")
    command = sub.add_parser("extract")
    for name in ("root", "lock", "external", "assets", "out"):
        command.add_argument("--" + name, type=Path, required=True)
    command.add_argument("--render", type=Path, help="the render folder (E1 sets)")
    command.add_argument("--arm", choices=ARMS, required=True)
    command.add_argument("--set", dest="name", choices=SETS, required=True)
    command.add_argument("--variant", choices=sorted({v for vs in E3_VARIANTS.values() for v in vs}))
    args = parser.parse_args()
    if args.step == "render":
        only = set(args.only.read_text(encoding="utf-8").split()) if args.only else None
        identity = render(args.root, args.lock, args.external, args.out, only)
        print(f"render: {identity['images']}; counts {identity['counts']}; pillow {identity['pillow']}")
    else:
        if args.name != "E3" and args.render is None:
            parser.error("--render is required for the E1 sets")
        identity = run_extract(args.root, args.lock, args.external, args.render, args.arm, args.name, args.variant,
                               args.assets, args.out)
        print(f"{args.arm} {args.name} {args.variant or ''}: {identity['images']} images; environment matches lock: "
              f"{identity['environment_matches_lock']}; repeat check {identity['repeat_check']}")


if __name__ == "__main__":
    main()
