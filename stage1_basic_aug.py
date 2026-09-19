"""
Stage 1 - basic augmentation of the few-shot rare classes  (GAN-FSL-OCT reproduction)

Expands each rare class from its real TRAIN images (manifest) to 2,000 images at 256x256,
using the paper's "basic augmentation": flip, +-5% shift, +-30 deg rotation, 0-20% zoom,
+-10% brightness, Gaussian elastic deformation (sigma 10, alpha 2).
These images are CycleGAN domain B (Stage 2) and 40% of each rare class in Stage 3.

Usage (from the project root):
  python stage1_basic_aug.py            # skip classes that already have 2,000 images, verify all
  python stage1_basic_aug.py --force    # regenerate everything

Outputs:
  data/aug_basic/<class>/<class>_0000.png ... _1999.png
  data/aug_basic/_preview_<class>.jpg   8x6 sheet to eyeball (outside the class folders)
"""
import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

RARE = ["CSC", "MH", "MacTel", "Stargardt", "RP"]
REAL_TRAIN = dict(CSC=27, MH=26, MacTel=25, Stargardt=19, RP=27)  # verified in demo.ipynb
IMG = 256


def fix(p):
    return str(p).replace("\\", "/")


def make_aug(seed):
    import albumentations as A

    steps = [
        A.HorizontalFlip(p=0.5),
        A.Affine(translate_percent=(-0.05, 0.05), rotate=(-30, 30), scale=(1.0, 1.2), p=1.0),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0, p=1.0),
        A.ElasticTransform(alpha=2, sigma=10, p=0.5),
    ]
    try:
        return A.Compose(steps, seed=seed)  # albumentations >= 2.0
    except TypeError:
        return A.Compose(steps)


def preview(files, path):
    files = random.Random(0).sample(files, min(48, len(files)))
    tiles = [cv2.resize(cv2.imread(str(f)), (128, 128)) for f in files]
    tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % 8)
    cv2.imwrite(str(path), np.vstack([np.hstack(tiles[i : i + 8]) for i in range(0, len(tiles), 8)]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".")
    ap.add_argument("--n", type=int, default=2000, help="images per rare class")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    root = Path(a.root)
    df = pd.read_csv(root / "data/manifest.csv")
    random.seed(a.seed)
    np.random.seed(a.seed)
    ok = True

    for cls in RARE:
        src = [root / fix(p) for p in df[(df.label == cls) & (df.split == "train")].path]
        if len(src) != REAL_TRAIN[cls]:
            print(f"Note: {cls} has {len(src)} real train images (verified: {REAL_TRAIN[cls]})")
        out = root / "data/aug_basic" / cls
        out.mkdir(parents=True, exist_ok=True)
        existing = sorted(out.glob("*.png"))

        if len(existing) == a.n and not a.force:
            print(f"{cls:10s} {len(existing)} images already present -> kept")
        else:
            for f in existing:
                f.unlink()
            aug = make_aug(a.seed + RARE.index(cls))
            for i in range(a.n):
                img = cv2.imread(str(src[i % len(src)]))
                if img is None:
                    sys.exit(f"cannot read {src[i % len(src)]}")
                img = cv2.resize(img, (IMG, IMG))
                cv2.imwrite(str(out / f"{cls}_{i:04d}.png"), aug(image=img)["image"])
            print(f"{cls:10s} generated {a.n} images from {len(src)} real train images")

        files = sorted(out.glob("*.png"))
        bad = [f for f in files[:50] if (im := cv2.imread(str(f))) is None or im.shape != (IMG, IMG, 3)]
        if len(files) != a.n or bad:
            ok = False
            print(f"  PROBLEM {cls}: {len(files)} files, {len(bad)} unreadable/wrong-size in first 50")
        preview(files, root / "data/aug_basic" / f"_preview_{cls}.jpg")

    print("\nPreview sheets: data/aug_basic/_preview_<class>.jpg "
          "(layers should bend or shift but stay continuous)")
    print("VERIFIED" if ok else "NOT VERIFIED: see messages above")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
