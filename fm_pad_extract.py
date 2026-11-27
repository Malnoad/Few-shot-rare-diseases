"""G1: pad-to-square re-extraction of MIRAGE and Inception-v3 (protocols.md section 8; GPU, Colab only).

A labelled post hoc sensitivity check outside A2 A-11; it never replaces a locked result. Each
image is padded with black (0) to a square around its centre, then given to the arm's own
locked transform and weights, unchanged: fm_extract.load_encoder with the spec stored in the
input lock, so every weight file is checked against its pinned SHA256. No hashed fm_*.py file is
edited, and no label is read (the inventories hold paths and content hashes only).

Inputs: every image the two locked arms used, i.e. extraction_inventory.csv (146 rare, 4,000
Kermany fitting, 1,000 Kermany test) and external_extraction_inventory.csv (781 OCTID/OCTDL),
each checked against its locked SHA256 before use. Outputs in --out, one folder per arm:
features.npz and features_external.npz (the locked cache format), control.npz (the first 20
inventory images without padding, through the locked path, to compare with the locked features
on the laptop), padding.csv (per image: size, square side, offsets, padded-pixel SHA256),
identity.json (spec, code, environment, hardware, lock hashes, repeat check, costs) and
complete.json. Scoring runs on the laptop: fm_posthoc.py g1.

    python fm_pad_extract.py extract --root <checkout> --lock <restored input lock> \
        --external <external query lock> --arm MIRAGE --assets <assets/MIRAGE> --out <out/MIRAGE>
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import time

import numpy as np
from PIL import Image

import fm_extract as extract
import fm_prepare as prep

ARMS = ("MIRAGE", "Inception")  # the two arms whose locked transform squashes the image to a square
CONTROL = 20  # unpadded control images, compared with the locked features on the laptop
REPEAT = 5  # padded images encoded twice at the end (A2 A-6 style agreement, recorded, not a gate)
PADDING_HEADER = ["path", "sha256", "mode", "width", "height", "side", "left", "top", "padded_sha256"]


def pad_box(width: int, height: int) -> tuple[int, int, int]:
    """Square side and the offsets that centre the image (extra pixel, if odd, right/bottom)."""
    side = max(width, height)
    return side, (side - width) // 2, (side - height) // 2


def black(mode: str):
    """Black in the image's own mode; alpha stays opaque so padding is black, not transparent."""
    fills = {"L": 0, "RGB": (0, 0, 0), "RGBA": (0, 0, 0, 255)}
    if mode not in fills:
        raise ValueError(f"no black fill defined for image mode {mode}")
    return fills[mode]


def pad_to_square(image: Image.Image) -> tuple[Image.Image, tuple[int, int, int]]:
    """The image pasted unchanged onto a black square around its centre (no resampling)."""
    side, left, top = pad_box(*image.size)
    canvas = Image.new(image.mode, (side, side), black(image.mode))
    canvas.paste(image, (left, top))
    return canvas, (side, left, top)


def pixel_hash(image: Image.Image) -> str:
    return hashlib.sha256(f"{image.mode}:{image.width}x{image.height}:".encode() + image.tobytes()).hexdigest()


def decode(encoder: extract.Encoder, path: Path) -> Image.Image:
    """The arm's locked decoder: TensorFlow's for the Keras reference, PIL's otherwise."""
    if encoder.decode_path:
        return encoder.decode_path(path)
    with Image.open(path) as image:
        return image.copy()


def encode_padded(encoder: extract.Encoder, root: Path, rows: prep.Rows, log: list) -> np.ndarray:
    """Verify, pad and encode every inventory row, in inventory order."""
    out = np.empty((len(rows), extract.DIMENSIONS[encoder.arm]), dtype=np.float32)
    for i, row in enumerate(rows):
        path = root / prep.safe_id(row["path"])
        if prep.sha256(path) != row["sha256"]:
            raise ValueError(f"input image content differs from the lock: {row['path']}")
        image = decode(encoder, path)
        padded, (side, left, top) = pad_to_square(image)
        values = encoder.encode([padded])
        extract.validate_features(values, 1, extract.DIMENSIONS[encoder.arm])
        out[i] = values[0]
        log.append([row["path"], row["sha256"], image.mode, image.width, image.height, side, left, top, pixel_hash(padded)])
        if (i + 1) % 500 == 0:
            print(f"{encoder.arm}: {i + 1}/{len(rows)}", flush=True)
    return out


def control(encoder: extract.Encoder, root: Path, rows: prep.Rows) -> np.ndarray:
    """The first rows without padding, through exactly the locked extraction path."""
    values = []
    for row in rows:
        path = root / prep.safe_id(row["path"])
        if encoder.stage_path:
            values.append(encoder.stage_path(path)()[0])
        else:
            with Image.open(path) as image:
                values.append(encoder.encode([image.copy()])[0])
    return np.stack(values).astype(np.float32)


def run(root: Path, lock_folder: Path, external: Path, arm: str, assets: Path, out: Path,
        load: callable = extract.load_encoder) -> dict:
    """One arm. `load` is injectable so the laptop tests can pass a stand-in encoder."""
    if arm not in ARMS:
        raise ValueError("G1 covers MIRAGE and Inception only")
    started = time.perf_counter()
    lock = prep.load_lock(lock_folder)  # also checks that the hashed code is the locked code
    from fm_external import verify_external  # fm_external imports fm_extract
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
        print(f"{arm}: already complete")
        return prep.read_json(out / "identity.json")
    encoder = load(spec, assets)
    loaded = time.perf_counter()
    log: list[list] = []
    main = encode_padded(encoder, root, main_rows, log)
    ext = encode_padded(encoder, root, ext_rows, log)
    encoded = time.perf_counter()
    check = control(encoder, root, main_rows[:CONTROL])
    first = [main[i:i + 1] for i in range(REPEAT)]
    again = [encoder.encode([pad_to_square(decode(encoder, root / prep.safe_id(r["path"])))[0]]) for r in main_rows[:REPEAT]]
    extract.save_features(out / "features.npz", [r["path"] for r in main_rows], [r["sha256"] for r in main_rows], main)
    extract.save_features(out / "features_external.npz", [r["path"] for r in ext_rows], [r["sha256"] for r in ext_rows], ext)
    np.savez(out / "control.npz", paths=np.array([r["path"] for r in main_rows[:CONTROL]]), feats=check)
    prep.write_csv(out / "padding.csv", PADDING_HEADER, log)
    current = extract.environment()
    identity = dict(
        step="g1_pad_extract", arm=arm, label="post hoc sensitivity check outside A2 A-11 (protocols.md section 8, G1)",
        transform="pad with black (0) to a square around the centre, then the locked transform of this arm",
        spec=spec, spec_sha256=prep.object_hash(spec), code=prep.code_identity(),
        code_matches_lock=prep.code_identity() == locked["identity"]["code"],
        g1_script_sha256=prep.text_hash(Path(__file__)), environment=current,
        environment_matches_lock=current == locked["identity"]["environment"], hardware=extract.hardware(),
        lock_sha256=prep.sha256(lock_folder / "input_lock.json"), external_lock_sha256=prep.sha256(external / "external_lock.json"),
        images=dict(main=len(main_rows), external=len(ext_rows), control=len(check)),
        repeat_check=extract.repeat_check(first, again), load_report=encoder.load_report,
        seconds=dict(loading=loaded - started, encoding=encoded - loaded, total=time.perf_counter() - started))
    prep.write_json(out / "identity.json", identity)
    prep.complete(out, ["features.npz", "features_external.npz", "control.npz", "padding.csv", "identity.json"])
    return identity


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="step", required=True)
    command = sub.add_parser("extract")
    for name in ("root", "lock", "external", "assets", "out"):
        command.add_argument("--" + name, type=Path, required=True)
    command.add_argument("--arm", choices=ARMS, required=True)
    args = parser.parse_args()
    identity = run(args.root, args.lock, args.external, args.arm, args.assets, args.out)
    print(f"{args.arm}: {identity['images']} images; environment matches lock: {identity['environment_matches_lock']}; "
          f"repeat check {identity['repeat_check']}")


if __name__ == "__main__":
    main()
