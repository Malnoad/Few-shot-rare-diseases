"""
Stage 0 - build and verify data/manifest.csv  (GAN-FSL-OCT reproduction, Yoo et al. MBEC 2021)

Columns: path (relative to project root), label, pid, split (train / val / test).

Reproduces the procedure run and verified in demo.ipynb exactly:
  * Kermany OCT2017 official train/test split (independent patients); 250 test per major class.
  * Rare classes: fixed random test hold-out (seed 42) of 5/5/4/4/4 images, as in the paper.
  * 10% validation carved from train, grouped by patient (GroupShuffleSplit, seed 42),
    then rare-class val images moved back to train, so every real rare image is available
    for augmentation and the CycleGAN. Rare-class validation images come from augmented
    data in Stage 3, as in the paper.

Usage (from the project root):
  python stage0_manifest.py            # build if missing, otherwise verify only
  python stage0_manifest.py --force    # rebuild (only if you change the raw data)

Stages 1 and 2 were built from this manifest, so it is never overwritten without --force.
"""
import argparse
import random
import re
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

MAJOR = ["CNV", "DME", "DRUSEN", "NORMAL"]
RARE_TEST = dict(CSC=5, MH=5, MacTel=4, Stargardt=4, RP=4)  # order matters for the seeded sampling

# Counts verified in demo.ipynb: label -> (test, train, val)
EXPECTED = {
    "CNV": (250, 32536, 4669),
    "DME": (250, 10153, 1195),
    "DRUSEN": (250, 7733, 883),
    "NORMAL": (250, 23536, 2779),
    "CSC": (5, 27, 0),
    "MH": (5, 26, 0),
    "MacTel": (4, 25, 0),
    "RP": (4, 27, 0),
    "Stargardt": (4, 19, 0),
}


def fix(p):
    return str(p).replace("\\", "/")


def build(root: Path) -> pd.DataFrame:
    rows = []
    for split in ["train", "test"]:
        for p in sorted((root / "data/raw/OCT2017" / split).rglob("*.jpeg")):
            m = re.match(r"([A-Z]+)-(\d+)-(\d+)", p.stem)
            if not m:
                print("  skipped unexpected file name:", p)
                continue
            rows.append(dict(path=str(p.relative_to(root)), label=m[1], pid=m[2], split=split))

    random.seed(42)
    for cls, n in RARE_TEST.items():
        files = sorted((root / "data/raw/rare" / cls).glob("*"))
        if len(files) < n:
            sys.exit(f"data/raw/rare/{cls} has {len(files)} files, need at least {n}")
        test = set(random.sample(files, n))
        for f in files:
            rows.append(dict(path=str(f.relative_to(root)), label=cls, pid=f"{cls}_{f.stem}",
                             split="test" if f in test else "train"))

    df = pd.DataFrame(rows)
    tr = df[df.split == "train"]
    gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    _, val_idx = next(gss.split(tr, groups=tr.pid))
    df.loc[tr.index[val_idx], "split"] = "val"
    # keep every real rare image for Stage 1/2 (same result as the demo's fix-up cell)
    df.loc[df.label.isin(RARE_TEST) & (df.split == "val"), "split"] = "train"
    return df


def verify(df: pd.DataFrame, root: Path) -> bool:
    print(pd.crosstab(df.label, df.split, margins=True), "\n")
    ct = pd.crosstab(df.label, df.split).reindex(columns=["test", "train", "val"], fill_value=0)
    ok = True

    for lab, exp in EXPECTED.items():
        got = tuple(int(x) for x in ct.loc[lab]) if lab in ct.index else None
        if got != exp:
            ok = False
            print(f"MISMATCH {lab}: got (test, train, val) = {got}, verified = {exp}")
    extra = sorted(set(ct.index) - set(EXPECTED))
    if extra:
        ok = False
        print("Unexpected labels:", extra)

    missing = [p for p in df.path.map(fix) if not (root / p).exists()]
    if missing:
        ok = False
        print(f"{len(missing)} manifest paths do not exist, e.g. {missing[:3]}")

    # leakage checks
    maj = df[df.label.isin(MAJOR)]
    tv = maj[maj.split != "test"].groupby("pid").split.nunique()
    if (tv > 1).any():
        ok = False
        print(f"LEAK: {(tv > 1).sum()} patients appear in both train and val")
    overlap = set(maj[maj.split == "test"].pid) & set(maj[maj.split != "test"].pid)
    if overlap:  # comes from the Kermany release itself, reported not fixed
        print(f"Note: {len(overlap)} patient IDs appear in both Kermany train and test "
              f"(official split; left as published)")
    if df.path.duplicated().any():
        ok = False
        print("Duplicate paths in manifest")

    print("VERIFIED: matches demo.ipynb" if ok else "NOT VERIFIED: see messages above")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".", help="project root (default: current folder)")
    ap.add_argument("--force", action="store_true", help="rebuild even if data/manifest.csv exists")
    a = ap.parse_args()

    root = Path(a.root)
    out = root / "data/manifest.csv"
    if out.exists() and not a.force:
        print(f"{out} exists: verifying only (use --force to rebuild)\n")
        df = pd.read_csv(out, dtype={"pid": str})
    else:
        df = build(root)
        df.to_csv(out, index=False)
        print(f"wrote {out} ({len(df)} rows)\n")
    sys.exit(0 if verify(df, root) else 1)


if __name__ == "__main__":
    main()
