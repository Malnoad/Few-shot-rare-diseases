"""Re-tests RT1 and RT2: re-extract features for two suspicious Phase-1 results (protocols.md section 13).

DRAFT kit. Section 13 is a draft rule; nothing here may run on real weights until Lam approves it and the
rule is committed and pushed. Post hoc, outside the locked protocol; never replaces a locked result. No
label is read and no score is computed here (scoring: fm_posthoc.py rt, on the laptop).

No hashed fm_*.py file and no line of fm_control_extract.py is edited; both are reused by import.

RT1, RETFound read-out (token and normalisation choice). Preprocessing, weights and the trunk are exactly
the locked ones: the model is built and loaded by fm_extract.load_encoder (assets checked against their
pinned SHA256, the locked key checks, FP32, inference mode). Two names are replaced inside the worker
process and recorded by name and value in identity.json:
    fm_extract.import_source   wrapped only to keep a handle on the model the locked loader builds;
    model.forward_features     replaced, after loading, by a read-out of the same trunk pass. The token
                               sequence is taken by a forward hook on the last block while the upstream
                               forward_features runs unchanged, so every call also checks that our R-ref
                               read-out reproduces the locked output (hook_check in identity.json).
Variants: R-ref (locked read-out; the control), R-meanraw, R-clsnorm, R-meannorm, and the upstream check
on the first 20 main and first 20 external inventory rows. The two `norm` variants need the checkpoint's
norm.weight and norm.bias; without them they are written as "not available" and never substituted.

RT2, Inception-v3 decoder only. I-pil: every image decoded as E0 decodes it (PIL, RGB) and saved as PNG
with no crop and no resize, then read by the locked loader (stage3_classifier.load_image). I-ref: the
unchanged originals through the same locked path (the control).

    python fm_retest_extract.py render  --root <checkout> --lock <input lock> --external <query lock> \
        --out <retest>/render --png <retest>/png
    python fm_retest_extract.py extract --root <checkout> --lock <input lock> --external <query lock> \
        --variant I-pil --render <retest>/render --png <retest>/png --assets <assets/Inception> --out <retest>/out/RT2/I-pil
    python fm_retest_extract.py extract ... --variant R-clsnorm --assets <assets/RETFound> --out <retest>/out/RT1/R-clsnorm

Laptop pre-check for RT2 (CPU TensorFlow; no model, no label, no score). Lam decides whether it runs before approval:

    python -I fm_retest_extract.py pixels --root D:/Few-shot --inventory <lock>/extraction_inventory.csv \
        --out <outside the repository>/pixels.csv
"""
from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
import hashlib
from io import BytesIO
from pathlib import Path
import time

import numpy as np
from PIL import Image

import fm_control_extract as control
import fm_extract as extract
import fm_pad_extract as pad
import fm_prepare as prep

RT1_ARM, RT2_ARM = "RETFound", "Inception"
RT1_REF, RT2_REF = "R-ref", "I-ref"
UPSTREAM = "upstream-check"
NEEDS_NORM = ("R-clsnorm", "R-meannorm")  # read-outs through the checkpoint's trained encoder norm
UPSTREAM_ROWS = control.CONTROL  # 20: first rows of the main inventory, and of the external inventory
PROBE = ("upstream_global_pool_true", "R-ref", "upstream_global_pool_false", "R-clsnorm")
LABEL = ("post hoc, re-test rule {retest} (protocols.md section 13), written after the Phase-1 results; outside the "
         "locked protocol; never replaces a locked result")
_TOKENS = ("tokens: output of the last transformer block, [batch, 1 + 196, 1024], class token first, read by a "
           "forward hook while the upstream forward_features runs unchanged")
_NORM = "norm: LayerNorm(1024, eps=1e-6) holding the checkpoint's own norm.weight and norm.bias"
_LOCKED_INPUT = "preprocessing, weights and trunk exactly as locked (fm_extract.load_encoder)"

# Every variant and set with its exact definition; the hash goes into identity.json.
VARIANTS = {
    "RT1/R-ref": dict(retest="RT1", arm=RT1_ARM, input=_LOCKED_INPUT, tokens=_TOKENS, override="model.forward_features",
                      value="fc_norm(mean of the 196 patch tokens); fc_norm is the locked LayerNorm at its default "
                            "values (ones, zeros, eps=1e-6), as upstream global_pool=True computes it"),
    "RT1/R-meanraw": dict(retest="RT1", arm=RT1_ARM, input=_LOCKED_INPUT, tokens=_TOKENS,
                          override="model.forward_features", value="mean of the 196 patch tokens; no normalisation layer"),
    "RT1/R-clsnorm": dict(retest="RT1", arm=RT1_ARM, input=_LOCKED_INPUT, tokens=_TOKENS, norm=_NORM,
                          override="model.forward_features",
                          value="class token of norm(all 197 tokens), as upstream global_pool=False computes it"),
    "RT1/R-meannorm": dict(retest="RT1", arm=RT1_ARM, input=_LOCKED_INPUT, tokens=_TOKENS, norm=_NORM,
                           override="model.forward_features",
                           value="norm(all 197 tokens), then the mean of the 196 patch tokens"),
    "RT1/upstream-check": dict(retest="RT1", arm=RT1_ARM, input=_LOCKED_INPUT, images="first 20 rows of the main "
                               "inventory and first 20 rows of the external inventory",
                               value="models_vit.VisionTransformer.forward_features of the pinned upstream source used "
                                     "directly, built with global_pool=True and with global_pool=False (checkpoint "
                                     "loaded with strict=False, head keys dropped), beside R-ref and R-clsnorm"),
    "RT2/I-pil": dict(retest="RT2", arm=RT2_ARM, steps=[
        "decode: PIL Image.open, .convert('RGB') (alpha dropped), as E0", "no crop, no resize",
        "save as 8-bit RGB PNG", "read by the locked loader (stage3_classifier.load_image), locked weights"]),
    "RT2/I-ref": dict(retest="RT2", arm=RT2_ARM, steps=[
        "the unchanged original file read by the locked loader (stage3_classifier.load_image), locked weights"]),
}
RT1_VARIANTS = tuple(name.split("/")[1] for name in VARIANTS if name.startswith("RT1/"))
RT2_VARIANTS = tuple(name.split("/")[1] for name in VARIANTS if name.startswith("RT2/"))


def registry_hash() -> str:
    return prep.object_hash(VARIANTS)


def retest_of(variant: str) -> str:
    if variant in RT1_VARIANTS:
        return "RT1"
    if variant in RT2_VARIANTS:
        return "RT2"
    raise ValueError(f"unknown re-test variant {variant}")


# ============================================================ RT1 read-outs (framework-free) ===
# Each read-out takes the token sequence [batch, 1 + patches, dim] (class token first) and returns
# [batch, 1, dim], the shape the locked loader checks before it drops the middle axis. They use only
# indexing and .mean(dim=, keepdim=), so the laptop tests run them on a NumPy stand-in.
def patch_mean(tokens):
    """Mean of the patch tokens (the class token at index 0 is left out), kept as [batch, 1, dim]."""
    return tokens[:, 1:, :].mean(dim=1, keepdim=True)


def _r_ref(tokens, fc_norm, norm):
    return fc_norm(patch_mean(tokens))


def _r_meanraw(tokens, fc_norm, norm):
    return patch_mean(tokens)


def _r_clsnorm(tokens, fc_norm, norm):
    return norm(tokens)[:, :1, :]


def _r_meannorm(tokens, fc_norm, norm):
    return patch_mean(norm(tokens))


READOUTS = {"R-ref": _r_ref, "R-meanraw": _r_meanraw, "R-clsnorm": _r_clsnorm, "R-meannorm": _r_meannorm}


def to_numpy(values) -> np.ndarray:
    """A framework tensor (or the NumPy stand-in) as a float32 array on the host."""
    return np.asarray(values.cpu().numpy() if hasattr(values, "cpu") else values, dtype=np.float32)


def _hooked(model) -> tuple[Callable, dict]:
    """The model's own forward_features and a holder the last block's output is written to on every call."""
    seen: dict = {}
    model.blocks[-1].register_forward_hook(lambda module, inputs, output: seen.__setitem__("tokens", output))
    return model.forward_features, seen


def attach_readout(model, variant: str, norm=None) -> dict:
    """Replace model.forward_features by a variant read-out of the same trunk pass.

    The upstream forward_features still runs unchanged on every call; its output is compared with our R-ref
    read-out of the hooked tokens, and the largest difference is kept in the returned record. A `norm`
    variant without the trained norm is an error here: the caller reports it as not available."""
    if variant not in READOUTS:
        raise ValueError(f"RT1 has no read-out {variant}")
    if variant in NEEDS_NORM and norm is None:
        raise ValueError(f"{variant} needs the checkpoint's trained norm; it is not available and is never substituted")
    locked_forward, seen = _hooked(model)
    record = dict(calls=0, largest_difference=0.0, largest_value=0.0)

    def forward_features(x):
        seen.clear()
        locked = locked_forward(x)
        tokens = seen["tokens"]
        reference = READOUTS[RT1_REF](tokens, model.fc_norm, norm)
        record.update(calls=record["calls"] + 1,
                      largest_difference=max(record["largest_difference"], float(abs(locked - reference).max())),
                      largest_value=max(record["largest_value"], float(abs(locked).max())))
        return reference if variant == RT1_REF else READOUTS[variant](tokens, model.fc_norm, norm)

    model.forward_features = forward_features
    return record


def attach_probe(model, norm=None, upstream_false: Callable | None = None) -> dict[str, list[np.ndarray]]:
    """Upstream check: on every call keep the upstream class's own output (global_pool=True: this model's
    unchanged forward_features; global_pool=False: `upstream_false`) beside our R-ref and R-clsnorm read-outs
    of the same input. The locked output is returned, so the locked loader's shape check still applies."""
    if (norm is None) != (upstream_false is None):
        raise ValueError("the global_pool=False pair needs both the trained norm and the upstream model")
    locked_forward, seen = _hooked(model)
    kept: dict[str, list[np.ndarray]] = {name: [] for name in PROBE}

    def forward_features(x):
        seen.clear()
        locked = locked_forward(x)
        tokens = seen["tokens"]
        kept["upstream_global_pool_true"].append(to_numpy(locked[:, 0, :]))
        kept["R-ref"].append(to_numpy(READOUTS[RT1_REF](tokens, model.fc_norm, norm)[:, 0, :]))
        if upstream_false is not None:
            kept["upstream_global_pool_false"].append(to_numpy(upstream_false(x)))
            kept["R-clsnorm"].append(to_numpy(READOUTS["R-clsnorm"](tokens, model.fc_norm, norm)[:, 0, :]))
        return locked

    model.forward_features = forward_features
    return kept


def hook_check(record: dict) -> dict:
    """The per-worker read-out check: largest |locked output - our R-ref read-out| over largest |locked output|."""
    ratio = record["largest_difference"] / record["largest_value"] if record["largest_value"] else None
    return dict(calls=record["calls"], max_difference_ratio=ratio,
                compares="upstream forward_features (unchanged) with R-ref computed from the hooked tokens")


# ================================================================== RT1 loading (Colab) ===
@dataclass
class Capture:
    """What the locked loader built, kept by the wrapped import_source."""
    build: Callable | None = None  # the pinned source's own RETFound_mae
    model: object | None = None  # the model the locked loader built, loaded and moved to the GPU by it


def capturing_import_source(capture: Capture) -> Callable[[Path], object]:
    """A replacement for fm_extract.import_source that returns the pinned source unchanged except that
    RETFound_mae also remembers the one model it builds."""
    locked_import = extract.import_source

    def import_source(path: Path):
        source = locked_import(path)
        capture.build = source.RETFound_mae

        def RETFound_mae(**kwargs):
            if capture.model is not None:
                raise ValueError("the locked loader built more than one RETFound model")
            capture.model = capture.build(**kwargs)
            return capture.model

        source.RETFound_mae = RETFound_mae
        return source

    return import_source


def capture_locked(spec: dict, assets: Path, load: Callable = extract.load_encoder) -> tuple[extract.Encoder, Capture]:
    """Load RETFound through the locked loader and keep a handle on its model."""
    capture = Capture()
    saved = extract.import_source
    extract.import_source = capturing_import_source(capture)
    try:
        encoder = load(spec, assets)
    finally:
        extract.import_source = saved
    if capture.model is None:
        raise ValueError("the locked loader did not build RETFound through fm_extract.import_source")
    return encoder, capture


def _checkpoint_state(assets: Path) -> dict:
    """The checkpoint's state mapping; the locked loader has just checked the file against its pinned SHA256."""
    return extract.load_retfound_checkpoint(assets / extract.REQUIRED_ASSETS[RT1_ARM][0])["model"]


# PyTorch is not installed on the laptop: trained_norm and upstream_model run on Colab only and are covered
# there by the R-ref control, the hook check and the upstream check, not by the laptop tests.
def trained_norm(assets: Path) -> tuple[object | None, dict]:
    """The checkpoint's trained encoder norm as a frozen FP32 LayerNorm on the GPU, with a report.
    None when the checkpoint holds no complete norm.weight / norm.bias pair (the variants are then not available)."""
    import torch
    state = _checkpoint_state(assets)
    found = [key for key in ("norm.weight", "norm.bias") if key in state]
    if len(found) != 2:
        return None, dict(available=False, keys_found=found, reason="the checkpoint holds no norm.weight and norm.bias")
    dimension = extract.DIMENSIONS[RT1_ARM]
    weight, bias = state["norm.weight"].float(), state["norm.bias"].float()
    if weight.shape != (dimension,) or bias.shape != (dimension,):
        raise ValueError("the checkpoint's norm does not have the encoder width")
    norm = torch.nn.LayerNorm(dimension, eps=1e-6)  # eps as models_vit.RETFound_mae's norm_layer
    norm.load_state_dict({"weight": weight, "bias": bias}, strict=True)
    norm = norm.float().cuda().eval().requires_grad_(False)
    return norm, dict(available=True, eps=1e-6, at_default_values=bool((weight == 1).all() and (bias == 0).all()),
                      **{f"{name}_sha256": hashlib.sha256(tensor.cpu().numpy().tobytes()).hexdigest()
                         for name, tensor in (("weight", weight), ("bias", bias))})


def upstream_model(assets: Path, capture: Capture) -> tuple[Callable, dict]:
    """The pinned upstream class built with global_pool=False and used directly (its own norm, its own
    forward_features): the checkpoint loaded with strict=False after dropping the head keys, as the locked
    loader does for global_pool=True. Only the head may be missing and only decoder keys may be unexpected."""
    state = {key: value for key, value in _checkpoint_state(assets).items() if key not in ("head.weight", "head.bias")}
    model = capture.build(global_pool=False)
    info = model.load_state_dict(state, strict=False)
    missing, unexpected = list(info.missing_keys), list(info.unexpected_keys)
    if set(missing) != {"head.weight", "head.bias"} or any(
            not (key.startswith("decoder_") or key == "mask_token") for key in unexpected):
        raise ValueError(f"upstream global_pool=False loading mismatch: missing={missing}, unexpected={unexpected}")
    model.float().cuda().eval().requires_grad_(False)
    return model.forward_features, dict(missing_keys=missing, unexpected_keys=unexpected)


def load_readout(spec: dict, assets: Path, variant: str, load: Callable = extract.load_encoder,
                 norm_of: Callable = trained_norm, upstream_of: Callable = upstream_model) -> tuple[extract.Encoder, dict]:
    """The locked RETFound encoder with one read-out attached. Returns the encoder and a report whose
    `available` is False when the variant needs a trained norm the checkpoint does not hold; `record` is
    updated while the encoder runs (hook check, or the upstream check's kept outputs)."""
    encoder, capture = capture_locked(spec, assets, load)
    norm, norm_report = norm_of(assets) if variant in (*NEEDS_NORM, UPSTREAM) else (None, None)
    report = dict(variant=variant, norm=norm_report, available=True,
                  overrides={"fm_extract.import_source": "fm_retest_extract.capturing_import_source (the pinned source "
                                                         "unchanged; RETFound_mae also keeps the model it builds)"})
    if variant == UPSTREAM:
        upstream_false, loading = upstream_of(assets, capture) if norm is not None else (None, None)
        report.update(record=attach_probe(capture.model, norm, upstream_false), upstream_global_pool_false=loading
                      or "not available: the checkpoint holds no norm.weight and norm.bias")
        report["overrides"]["model.forward_features"] = "fm_retest_extract.attach_probe (returns the locked output)"
    elif variant in NEEDS_NORM and norm is None:
        report.update(available=False, reason=norm_report["reason"])
    else:
        report["record"] = attach_readout(capture.model, variant, norm)
        report["overrides"]["model.forward_features"] = f"fm_retest_extract.{READOUTS[variant].__name__}"
    return encoder, report


# ======================================================================== shared pieces ===
def checked_original(root: Path, row: dict) -> Path:
    """The unchanged original of an inventory row, checked against its locked SHA256."""
    return control.source_file(None, root, "E3", row, None)


def locked_inputs(lock_folder: Path, external: Path, arm: str) -> tuple[dict, dict, prep.Rows, prep.Rows]:
    """The input lock (which also checks that the hashed code is the locked code), the arm's locked entry and
    both inventories."""
    lock = prep.load_lock(lock_folder)
    from fm_external import verify_external
    verify_external(external, lock_folder)
    locked = lock["arms"][arm]
    if locked["status"] != "available":
        raise ValueError("arm unavailable at the input lock")
    return lock, locked, prep.read_csv(lock_folder / "extraction_inventory.csv"), \
        prep.read_csv(external / "external_extraction_inventory.csv")


def base_identity(retest: str, arm: str, variant: str, locked: dict, lock_folder: Path, external: Path) -> dict:
    current = extract.environment()
    spec = locked["identity"]["spec"]
    return dict(
        step="retest_extract", retest=retest, arm=arm, set=retest, variant=variant, label=LABEL.format(retest=retest),
        transform=VARIANTS[f"{retest}/{variant}"], registry_sha256=registry_hash(),
        script_sha256=prep.text_hash(Path(__file__)), control_script_sha256=prep.text_hash(Path(control.__file__)),
        spec=spec, spec_sha256=prep.object_hash(spec), code=prep.code_identity(),
        code_matches_lock=prep.code_identity() == locked["identity"]["code"], environment=current,
        environment_matches_lock=current == locked["identity"]["environment"], hardware=control.safe_hardware(),
        lock_sha256=prep.sha256(lock_folder / "input_lock.json"),
        external_lock_sha256=prep.sha256(external / "external_lock.json"))


def repeated(encoder: extract.Encoder, rows: prep.Rows, locate: Callable[[dict], Path], first: np.ndarray) -> dict:
    """The first inputs (5 at most) encoded a second time, recorded as in the Phase-1 workers."""
    rows = rows[:control.REPEAT]
    again = [control.encode_file(encoder, locate(row)) for row in rows]
    return extract.repeat_check([first[i:i + 1] for i in range(len(rows))], again)


def _paths(rows: prep.Rows) -> list[str]:
    return [row["path"] for row in rows]


def _digests(rows: prep.Rows) -> list[str]:
    return [row["sha256"] for row in rows]


# ============================================================================ RT1 worker ===
def run_rt1(root: Path, lock_folder: Path, external: Path, variant: str, assets: Path, out: Path,
            load: Callable = load_readout) -> dict:
    """One RETFound read-out on all 5,927 images, or the upstream check on 40. `load` is injectable for the laptop tests."""
    if variant not in RT1_VARIANTS:
        raise ValueError(f"RT1 has no variant {variant}")
    started = time.perf_counter()
    _, locked, main_rows, ext_rows = locked_inputs(lock_folder, external, RT1_ARM)
    out = prep.out_dir(root, out)
    if (out / "complete.json").exists():
        prep.verify_complete(out)
        print(f"RT1 {variant}: already complete")
        return prep.read_json(out / "identity.json")
    control.fresh(out)
    encoder, readout = load(locked["identity"]["spec"], assets, variant)
    loaded = time.perf_counter()
    locate = lambda row: checked_original(root, row)  # noqa: E731
    files, extra = [], {}
    if variant == UPSTREAM:
        rows = [*main_rows[:UPSTREAM_ROWS], *ext_rows[:UPSTREAM_ROWS]]
        control.encode_rows(encoder, rows, locate)
        kept = {name: np.concatenate(values) for name, values in readout["record"].items() if values}
        for name, values in kept.items():
            if len(values) != len(rows):
                raise ValueError(f"upstream check kept {len(values)} outputs of {name} for {len(rows)} images")
            extract.save_features(out / f"{name}.npz", _paths(rows), _digests(rows), values)
            files.append(f"{name}.npz")
        pairs = {"global_pool_true_vs_R-ref": ("upstream_global_pool_true", "R-ref"),
                 "global_pool_false_vs_R-clsnorm": ("upstream_global_pool_false", "R-clsnorm")}
        extra = dict(images=dict(main=min(UPSTREAM_ROWS, len(main_rows)), external=min(UPSTREAM_ROWS, len(ext_rows))),
                     upstream_check={name: extract.repeat_check([kept[a]], [kept[b]]) if a in kept else "not available"
                                     for name, (a, b) in pairs.items()},
                     upstream_global_pool_false=readout["upstream_global_pool_false"])
    elif readout["available"]:
        main = control.encode_rows(encoder, main_rows, locate)
        ext = control.encode_rows(encoder, ext_rows, locate)
        extra = dict(images=dict(main=len(main_rows), external=len(ext_rows)),
                     repeat_check=repeated(encoder, main_rows, locate, main), hook_check=hook_check(readout["record"]))
        extract.save_features(out / "features.npz", _paths(main_rows), _digests(main_rows), main)
        extract.save_features(out / "features_external.npz", _paths(ext_rows), _digests(ext_rows), ext)
        files = ["features.npz", "features_external.npz"]
    else:
        extra = dict(reason=readout["reason"], note="not available; never substituted")
    encoded = time.perf_counter()
    identity = dict(**base_identity("RT1", RT1_ARM, variant, locked, lock_folder, external),
                    available=readout["available"], overrides=readout["overrides"], norm=readout["norm"],
                    control="R-ref is the control of RT1: all images against the locked features, on the laptop",
                    load_report=encoder.load_report, **extra,
                    seconds=dict(loading=loaded - started, encoding=encoded - loaded, total=time.perf_counter() - started))
    prep.write_json(out / "identity.json", identity)
    control.finish(out, [*files, "identity.json"])
    return identity


# ======================================================================= RT2 render (CPU) ===
RENDER_HEADER = ["path", "sha256", "mode", "format", "has_alpha", "min_alpha", "width", "height", "png_sha256",
                 "pixel_sha256"]
PNG_NOTE = ("the PNGs are not archived: each is the PIL RGB decode of a locked original saved losslessly, and its "
            "SHA256 and pixel hash are in render.csv")


def png_path(png: Path, path: str) -> Path:
    """Where the I-pil PNG of an original lives (the original path plus .png)."""
    return png / "I-pil" / prep.safe_id(path + ".png")


def render_pil(root: Path, lock_folder: Path, external: Path, out: Path, png: Path, only: set[str] | None = None) -> dict:
    """RT2 I-pil inputs: every inventory image (main, then external) decoded by PIL to RGB and saved as PNG.

    Receipts (render.csv, identity.json) go to `out`; the PNGs go to `png`. Run again on a complete `out`
    (for example after its archive was restored), it rebuilds missing PNGs and stops if one differs from its receipt."""
    started = time.perf_counter()
    lock, _, main_rows, ext_rows = locked_inputs(lock_folder, external, RT2_ARM)
    main_rows, ext_rows = control.select(main_rows, only), control.select(ext_rows, only)
    rows = [*main_rows, *ext_rows]
    out, png = prep.out_dir(root, out), prep.out_dir(root, png)
    receipts = None
    if (out / "complete.json").exists():
        prep.verify_complete(out)
        receipts = {r["path"]: r for r in prep.read_csv(out / "render.csv")}
        if list(receipts) != _paths(rows):
            raise ValueError("the render receipts belong to other inventories")
    else:
        control.fresh(out)
    lines, counts = [], dict(alpha_below_255=0, modes={}, formats={})
    for i, row in enumerate(rows):
        source = (root / prep.safe_id(row["path"])).read_bytes()
        if hashlib.sha256(source).hexdigest() != row["sha256"]:
            raise ValueError(f"input image content differs from the lock: {row['path']}")
        with Image.open(BytesIO(source)) as image:
            meta = control.mode_and_alpha(image)
            decoded = image.convert("RGB")  # E0 decoding
        data = control.png_bytes(decoded)
        digest, pixels = hashlib.sha256(data).hexdigest(), pad.pixel_hash(decoded)
        if receipts is not None and (receipts[row["path"]]["png_sha256"], receipts[row["path"]]["pixel_sha256"]) != (digest, pixels):
            raise ValueError(f"I-pil PNG of {row['path']} differs from the render of record")
        target = png_path(png, row["path"])
        target.parent.mkdir(parents=True, exist_ok=True)
        prep.write_once(target, data)
        with Image.open(target) as saved:
            if pad.pixel_hash(saved) != pixels:
                raise ValueError(f"saved PNG of {row['path']} differs from its PIL decode")
        lines.append([row["path"], row["sha256"], meta["mode"], meta["format"], meta["has_alpha"], meta["min_alpha"],
                      decoded.width, decoded.height, digest, pixels])
        counts["alpha_below_255"] += meta["min_alpha"] < 255
        for key, value in (("modes", meta["mode"]), ("formats", meta["format"])):
            counts[key][value] = counts[key].get(value, 0) + 1
        if (i + 1) % 500 == 0:
            print(f"render: {i + 1}/{len(rows)}", flush=True)
    if receipts is not None:
        print("render: receipts already complete; every PNG matches them")
        return prep.read_json(out / "identity.json")
    prep.write_csv(out / "render.csv", RENDER_HEADER, lines)
    import PIL
    identity = dict(
        step="retest_render", retest="RT2", label=LABEL.format(retest="RT2"), transform=VARIANTS["RT2/I-pil"],
        registry_sha256=registry_hash(), script_sha256=prep.text_hash(Path(__file__)), code=prep.code_identity(),
        code_matches_lock=prep.code_identity() == lock["code"], pillow=PIL.__version__, environment=extract.environment(),
        hardware=control.safe_hardware(), lock_sha256=prep.sha256(lock_folder / "input_lock.json"),
        external_lock_sha256=prep.sha256(external / "external_lock.json"),
        images=dict(main=len(main_rows), external=len(ext_rows)), subset=sorted(only) if only is not None else None,
        counts=counts, png=PNG_NOTE, render_csv_sha256=prep.sha256(out / "render.csv"),
        seconds=dict(total=time.perf_counter() - started))
    prep.write_json(out / "identity.json", identity)
    control.finish(out, ["render.csv", "identity.json"])
    return identity


def read_receipts(render_folder: Path, lock_folder: Path, external: Path) -> dict[str, dict[str, str]]:
    """Bind the I-pil receipts to both locks and to this registry. Each PNG is checked as the worker reads it."""
    prep.verify_complete(render_folder)
    identity = prep.read_json(render_folder / "identity.json")
    if identity["subset"] is not None or identity["lock_sha256"] != prep.sha256(lock_folder / "input_lock.json") or \
            identity["external_lock_sha256"] != prep.sha256(external / "external_lock.json") or \
            identity["registry_sha256"] != registry_hash():
        raise ValueError("I-pil render is incomplete or belongs to different locks or variants")
    return {r["path"]: r for r in prep.read_csv(render_folder / "render.csv")}


def checked_png(png: Path, row: dict, receipts: dict[str, dict[str, str]]) -> Path:
    """The I-pil PNG of an inventory row, checked against the receipt of the locked content."""
    entry = receipts.get(row["path"])
    if entry is None or entry["sha256"] != row["sha256"]:
        raise ValueError(f"no I-pil PNG of the locked content for {row['path']}")
    path = png_path(png, row["path"])
    if prep.sha256(path) != entry["png_sha256"]:
        raise ValueError(f"I-pil PNG differs from its receipt: {row['path']}")
    return path


# ============================================================================ RT2 worker ===
PIXEL_HEADER = ["path", "source", "format", "width", "height", "same_shape", "identical", "max_abs_difference",
                "mean_abs_difference"]


def pixel_difference(first: np.ndarray, second: np.ndarray) -> tuple[bool, bool, int | None, float | None]:
    """(same shape, identical, largest and mean absolute difference) of two decoded 8-bit images."""
    if first.shape != second.shape:
        return False, False, None, None
    gap = np.abs(first.astype(np.int16) - second.astype(np.int16))
    return True, not gap.any(), int(gap.max()), float(gap.mean())


def tf_pixels(original: bytes, png: bytes) -> tuple[bool, bool, int | None, float | None]:
    """RT2 pixel check: the bytes of both files decoded by the locked loader's own decoder call
    (stage3_classifier.load_image: tf.io.decode_image, channels=3), so it shows what PIL decoding changed."""
    import tensorflow as tf
    first, second = (tf.io.decode_image(data, channels=3, expand_animations=False).numpy() for data in (original, png))
    return pixel_difference(first, second)


def add_pixels(lines: list[list], summary: dict, path: str, container: str, width, height, result: tuple) -> None:
    """One decoder_pixels.csv row, and the count per source."""
    source = control.source_of(path)
    lines.append([path, source, container, width, height, *result])
    group = summary.setdefault(source, dict(images=0, identical=0))
    group["images"] += 1
    group["identical"] += result[1]


def decoder_pixels(root: Path, inventory: Path, out: Path, kermany: bool = False, pixels: Callable = tf_pixels) -> dict:
    """Laptop pre-check for RT2, with no model, no label and no score: for the rare images of an extraction
    inventory (and the Kermany ones with `kermany`), whether PIL decoding changes the pixels TensorFlow gives
    the locked loader. The PNG is made in memory exactly as render_pil makes it. `out` must lie outside
    every project checkout: it is a design input, not a result."""
    wanted = ("rare", "Kermany") if kermany else ("rare",)
    rows = [row for row in prep.read_csv(inventory) if control.source_of(row["path"]) in wanted]
    prep.experiment_out(out.parent)
    lines, summary = [], {}
    for row in rows:
        source = (root / prep.safe_id(row["path"])).read_bytes()
        if hashlib.sha256(source).hexdigest() != row["sha256"]:
            raise ValueError(f"input image content differs from the inventory: {row['path']}")
        with Image.open(BytesIO(source)) as image:
            container, decoded = image.format or "", image.convert("RGB")
        add_pixels(lines, summary, row["path"], container, decoded.width, decoded.height,
                   pixels(source, control.png_bytes(decoded)))
    out.parent.mkdir(parents=True, exist_ok=True)
    prep.write_csv(out, PIXEL_HEADER, lines)
    return summary


def run_rt2(root: Path, lock_folder: Path, external: Path, render_folder: Path | None, png: Path | None, variant: str,
            assets: Path, out: Path, main_only: bool = False, load: Callable = extract.load_encoder,
            pixels: Callable = tf_pixels) -> dict:
    """I-pil or I-ref through the locked Inception-v3 loader: main images, then the external ones unless
    `main_only`. `load` and `pixels` are injectable for the laptop tests."""
    if variant not in RT2_VARIANTS:
        raise ValueError(f"RT2 has no set {variant}")
    if (variant == "I-pil") != (render_folder is not None and png is not None):
        raise ValueError("--render and --png are required for I-pil and only for I-pil")
    started = time.perf_counter()
    _, locked, main_rows, ext_rows = locked_inputs(lock_folder, external, RT2_ARM)
    if main_only:
        ext_rows = []
    out = prep.out_dir(root, out)
    if (out / "complete.json").exists():
        prep.verify_complete(out)
        print(f"RT2 {variant}: already complete")
        return prep.read_json(out / "identity.json")
    control.fresh(out)
    receipts = read_receipts(render_folder, lock_folder, external) if variant == "I-pil" else None
    original = lambda row: checked_original(root, row)  # noqa: E731
    locate = original if receipts is None else (lambda row: checked_png(png, row, receipts))
    encoder = load(locked["identity"]["spec"], assets)
    loaded = time.perf_counter()
    main = control.encode_rows(encoder, main_rows, locate)
    ext = control.encode_rows(encoder, ext_rows, locate) if ext_rows else None
    repeat = repeated(encoder, main_rows, locate, main)
    encoded = time.perf_counter()
    files, summary = ["features.npz"], None
    extract.save_features(out / "features.npz", _paths(main_rows), _digests(main_rows), main)
    if ext is not None:
        extract.save_features(out / "features_external.npz", _paths(ext_rows), _digests(ext_rows), ext)
        files.append("features_external.npz")
    if receipts is not None:
        lines, summary = [], {}
        for row in (*main_rows, *ext_rows):
            entry = receipts[row["path"]]
            add_pixels(lines, summary, row["path"], entry["format"], entry["width"], entry["height"],
                       pixels(original(row).read_bytes(), locate(row).read_bytes()))
        prep.write_csv(out / "decoder_pixels.csv", PIXEL_HEADER, lines)
        files.append("decoder_pixels.csv")
    identity = dict(
        **base_identity("RT2", RT2_ARM, variant, locked, lock_folder, external), available=True,
        overrides={} if receipts is None else {
            "input file": "fm_retest_extract.render_pil: the original decoded by PIL to RGB and saved as PNG, then read by "
                          "the locked stage3_classifier.load_image"},
        main_only=main_only, images=dict(main=len(main_rows), external=len(ext_rows)),
        render_csv_sha256=prep.sha256(render_folder / "render.csv") if receipts is not None else None,
        render_identity_sha256=prep.sha256(render_folder / "identity.json") if receipts is not None else None,
        control="I-ref is the control of RT2: its images against the locked features, on the laptop",
        decoder_pixels=summary, repeat_check=repeat, load_report=encoder.load_report,
        seconds=dict(loading=loaded - started, encoding=encoded - loaded, total=time.perf_counter() - started))
    prep.write_json(out / "identity.json", identity)
    control.finish(out, [*files, "identity.json"])
    return identity


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="step", required=True)
    command = sub.add_parser("render", description="RT2 (CPU): the I-pil PNGs and their receipts.")
    for name in ("root", "lock", "external", "out", "png"):
        command.add_argument("--" + name, type=Path, required=True)
    command.add_argument("--only", type=Path, help="laptop dry run: a text file of inventory paths to render")
    command = sub.add_parser("pixels", description="RT2 laptop pre-check (CPU TensorFlow; no model, no score): do the "
                             "TensorFlow and PIL decoders give the same pixels for the rare images?")
    command.add_argument("--root", type=Path, required=True, help="checkout whose data/raw holds the images (read-only)")
    command.add_argument("--inventory", type=Path, required=True, help="extraction_inventory.csv of the input lock")
    command.add_argument("--out", type=Path, required=True, help="CSV to write, outside every project checkout")
    command.add_argument("--kermany", action="store_true", help="also count the Kermany images of the inventory")
    command = sub.add_parser("extract", description="One RT1 read-out or one RT2 set (GPU, one encoder per process).")
    for name in ("root", "lock", "external", "assets", "out"):
        command.add_argument("--" + name, type=Path, required=True)
    command.add_argument("--variant", choices=(*RT2_VARIANTS, *RT1_VARIANTS), required=True)
    command.add_argument("--render", type=Path, help="I-pil: the render receipts folder")
    command.add_argument("--png", type=Path, help="I-pil: the folder holding the PNGs")
    command.add_argument("--main-only", action="store_true", help="RT2: leave out OCTID and OCTDL (its nine-way is then not scored)")
    args = parser.parse_args()
    if args.step == "render":
        only = set(args.only.read_text(encoding="utf-8").split()) if args.only else None
        identity = render_pil(args.root, args.lock, args.external, args.out, args.png, only)
        print(f"render: {identity['images']}; counts {identity['counts']}; pillow {identity['pillow']}")
    elif args.step == "pixels":
        counts = decoder_pixels(args.root, args.inventory, args.out, args.kermany)
        print(f"pixels (images, and those identical under both decoders): {counts}")
    elif retest_of(args.variant) == "RT1":
        if args.render or args.png or args.main_only:
            parser.error("--render, --png and --main-only belong to RT2")
        identity = run_rt1(args.root, args.lock, args.external, args.variant, args.assets, args.out)
        print(f"RT1 {args.variant}: available {identity['available']}; {identity.get('images')}; "
              f"hook check {identity.get('hook_check')}; upstream check {identity.get('upstream_check')}")
    else:
        identity = run_rt2(args.root, args.lock, args.external, args.render, args.png, args.variant, args.assets,
                           args.out, args.main_only)
        print(f"RT2 {args.variant}: {identity['images']}; environment matches lock: {identity['environment_matches_lock']}; "
              f"repeat check {identity['repeat_check']}; pixels {identity['decoder_pixels']}")


if __name__ == "__main__":
    main()
