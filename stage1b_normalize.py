"""C0 classifier-input copies. Raw, basic and accepted GAN inputs stay read-only.

python stage1b_normalize.py --arm framing
python stage1b_normalize.py --root D:/Few-shot --workspace . --arm format --sample-per-class 2
See docs/research/C0_DESIGN.md. No augmentation, GAN training or QC is rerun.
"""
import argparse
import hashlib
import json
import struct
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from stage3_classifier import RARE, fix, rel_images

VERSION = "v1"
ARMS = ("current", "format", "framing")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def arm_root(workspace, arm):
    if arm not in ARMS[1:]:
        raise ValueError("Only format/framing have normalized copies")
    return (Path(workspace).resolve() / "data/c0" / VERSION / arm).resolve()


def normalized_path(path):
    # Append rather than replace: preserves sort order and avoids foo.jpg/foo.png collisions.
    return fix(path) + ".png"


def normalize_image(image, arm):
    """Discard alpha without compositing, then apply the fixed arm recipe."""
    if arm not in ARMS[1:]:
        raise ValueError(arm)
    if image.dtype == np.uint16:
        image = (image // 257).astype(np.uint8)
    if image.dtype != np.uint8:
        raise ValueError(f"Unsupported pixel dtype: {image.dtype}")
    gray = image.copy() if image.ndim == 2 else cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY)
    if arm == "format":
        return gray
    height, width = gray.shape
    side = max(width, height)
    left, top = (side - width) // 2, (side - height) // 2
    canvas = cv2.copyMakeBorder(gray, top, side - height - top, left, side - width - left,
                                cv2.BORDER_CONSTANT, value=0)
    return cv2.resize(canvas, (256, 256), interpolation=cv2.INTER_LINEAR)


def read_image(path):
    image = cv2.imdecode(np.frombuffer(Path(path).read_bytes(), np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Cannot decode {path}")
    return image


def image_header(path):
    with Path(path).open("rb") as stream:
        header = stream.read(33)
    if header[:8] == b"\x89PNG\r\n\x1a\n":
        width, height = struct.unpack(">II", header[16:24])
        mode = {0: "L", 2: "RGB", 3: "P", 4: "LA", 6: "RGBA"}[header[25]]
        if header[24] != 8:
            mode += f"/{header[24]}bit"
        return dict(width=width, height=height, mode=mode, encoding="PNG")
    if header[:2] != b"\xff\xd8":
        raise ValueError(f"Expected PNG or JPEG: {path}")
    image = read_image(path)
    return dict(width=image.shape[1], height=image.shape[0],
                mode="L" if image.ndim == 2 else "RGB", encoding="JPEG")


def selected_inputs(root, sample_per_class=0):
    """Keep manifest order and the exact Stage 3 augmentation filename ordering."""
    root = Path(root)
    man = pd.read_csv(root / "data/manifest.csv")
    man["path"] = man.path.map(fix)
    if sample_per_class:
        # All real rare images retain L7's full 124-train /22-test geometry check.
        common = man[~man.label.isin(RARE)].groupby(["label", "split"], sort=False).head(sample_per_class)
        man = man[man.index.isin(common.index) | man.label.isin(RARE)].copy()
    records = [dict(source_path=r.path, label=r.label, split=r.split, kind="raw")
               for r in man.itertuples()]
    for cls in RARE:
        for kind, folder in [("basic", f"data/aug_basic/{cls}"), ("gan", f"data/gan/accepted/{cls}")]:
            paths = rel_images(root, folder)
            if not sample_per_class and (len(paths) < 2000 if kind == "basic" else not paths):
                raise ValueError(f"Missing {kind} pool for {cls}: restore real images first")
            if sample_per_class:
                paths = paths[:sample_per_class]
            records.extend(dict(source_path=p, label=cls, split="pool", kind=kind) for p in paths)
    return man, records


def validate_source(root, relative):
    path = (Path(root) / relative).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError(f"Input escapes source root: {relative}")
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"Missing/placeholder image: {path}; restore real data, not Part 2 placeholders")
    return path


def prepare(root, workspace, arm, sample_per_class=0):
    root, workspace = Path(root).resolve(), Path(workspace).resolve()
    dest = arm_root(workspace, arm)
    # Reject junctions/links into the original data as well as accidental overlapping roots.
    if not dest.is_relative_to(workspace / "data/c0") or dest.is_relative_to(root / "data/raw"):
        raise ValueError("Normalized output must stay in workspace/data/c0, outside raw data")
    man, records = selected_inputs(root, sample_per_class)
    recipe = dict(version=VERSION, arm=arm, sample_per_class=sample_per_class,
                  manifest_sha256=sha256(root / "data/manifest.csv"), script_sha256=sha256(__file__))
    recipe_file = dest / "recipe.json"
    if recipe_file.exists() and json.loads(recipe_file.read_text()) != recipe:
        raise ValueError(f"Recipe changed: use a fresh workspace; do not reuse {dest}")
    # Validate every input before creating any copies (especially restored placeholders).
    for record in records:
        validate_source(root, record["source_path"])
    dest.mkdir(parents=True, exist_ok=True)
    atomic_json(recipe_file, recipe)
    inventory_file = dest / "inventory.csv"
    previous = pd.read_csv(inventory_file).set_index("source_path").to_dict("index") if inventory_file.exists() else {}
    rows = []
    for i, record in enumerate(records, 1):
        relative = record["source_path"]
        src = validate_source(root, relative)
        out = dest / normalized_path(relative)
        source_hash = sha256(src)
        old = previous.get(relative)
        if old and (old["source_sha256"] != source_hash or sha256(out) != old["output_sha256"]):
            raise ValueError(f"Content changed: {relative}; use a fresh workspace/cache")
        source_header = image_header(src)
        source_width, source_height = source_header["width"], source_header["height"]
        if not old:
            out.parent.mkdir(parents=True, exist_ok=True)
            tmp = out.with_suffix(".tmp")
            ok, encoded = cv2.imencode(".png", normalize_image(read_image(src), arm),
                                       [cv2.IMWRITE_PNG_COMPRESSION, 6])
            if not ok:
                raise ValueError(f"Cannot encode {out}")
            tmp.write_bytes(encoded.tobytes())
            tmp.replace(out)
        header = image_header(out)
        expected_size = (source_width, source_height) if arm == "format" else (256, 256)
        if header != dict(width=expected_size[0], height=expected_size[1], mode="L", encoding="PNG"):
            raise ValueError(f"Normalization header mismatch: {out}")
        rows.append(dict(**record, path=normalized_path(relative), source_sha256=source_hash,
                         output_sha256=sha256(out), width=expected_size[0], height=expected_size[1]))
        if i % 5000 == 0 or i == len(records):
            print(f"{arm}: verified {i}/{len(records)} images", flush=True)
    inventory = pd.DataFrame(rows)
    # A complete inventory is the normalization completion marker.
    tmp = inventory_file.with_suffix(".tmp")
    inventory.to_csv(tmp, index=False)
    tmp.replace(inventory_file)
    man["path"] = man.path.map(normalized_path)
    man.to_csv(dest / "data/manifest.csv", index=False)
    atomic_json(dest / "complete.json", dict(recipe=recipe, inventory_sha256=sha256(inventory_file),
                                             images=len(rows)))
    print(f"Copies: {dest}; sources untouched")
    return dest


def header_check(root, arm, inventory=None):
    """L7 size-only rule; count ties so constant sizes cannot masquerade as signal."""
    root = Path(root)
    man = pd.read_csv(root / "data/manifest.csv")
    man["path"] = man.path.map(fix)
    headers = []
    paths = man.path.tolist() if inventory is None else inventory.path.tolist()
    for path in paths:
        headers.append(dict(path=path, **image_header(validate_source(root, path))))
    headers = pd.DataFrame(headers)
    if arm in ARMS[1:] and not ((headers["mode"] == "L") & (headers.encoding == "PNG")).all():
        raise ValueError("Normalized encoding/mode mismatch")
    if arm == "framing" and not ((headers.width == 256) & (headers.height == 256)).all():
        raise ValueError("Framing dimensions mismatch")
    rare = man[man.label.isin(RARE)].merge(headers, on="path", validate="one_to_one")
    train, test = rare[rare.split == "train"], rare[rare.split == "test"]
    medians = np.stack([train.loc[train.label == c, ["width", "height"]].median().to_numpy() for c in RARE])
    distances = abs(test[["width", "height"]].to_numpy()[:, None] - medians).sum(2)
    pred = np.array(RARE)[distances.argmin(1)]
    ties = (distances == distances.min(1, keepdims=True)).sum(1)
    result = dict(arm=arm, images=len(headers), header_signatures=len(headers.drop_duplicates(
        ["width", "height", "mode", "encoding"])), rare_test=len(test),
        size_correct=int((pred == test.label).sum()), size_ties=int((ties > 1).sum()),
        encoding_modes=headers.groupby(["encoding", "mode"]).size().to_dict())
    result["encoding_modes"] = {f"{a}/{b}": int(n) for (a, b), n in result["encoding_modes"].items()}
    return result, headers


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=".", help="original source checkout")
    ap.add_argument("--workspace", default=".", help="copies go below this checkout's data/c0/v1")
    ap.add_argument("--arm", required=True, choices=ARMS[1:])
    ap.add_argument("--sample-per-class", type=int, default=0,
                    help="smoke test: retain all rare raw images, sample common/augmented pools; 0=all")
    a = ap.parse_args()
    if a.sample_per_class < 0:
        ap.error("sample-per-class must be >=0")
    dest = prepare(a.root, a.workspace, a.arm, a.sample_per_class)
    inventory = pd.read_csv(dest / "inventory.csv")
    result, headers = header_check(dest, a.arm, inventory)
    headers.to_csv(dest / "headers.csv", index=False)
    atomic_json(dest / "header_check.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
