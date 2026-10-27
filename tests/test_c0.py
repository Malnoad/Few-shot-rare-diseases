"""Training-free checks at C0's image, split and cache boundaries."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import cv2

from c0_experiment import auc, axis_result, cache_groups, paired_contrast, recall_metrics
from stage1b_normalize import header_check, normalize_image, normalized_path, prepare, selected_inputs, sha256
from stage3_classifier import CLASSES, RARE


class C0Tests(unittest.TestCase):
    def test_pixel_recipe_preserves_scan_and_drops_alpha(self):
        rgba = np.full((2, 4, 4), 90, np.uint8)
        rgba[:, :, 3] = 0
        gray = normalize_image(rgba, "format")
        self.assertEqual((gray.dtype, gray.shape), (np.dtype("uint8"), (2, 4)))
        self.assertTrue((np.array(gray) == 90).all())
        framed = np.array(normalize_image(rgba, "framing"))
        self.assertEqual(framed.shape, (256, 256))
        self.assertEqual(int(framed[128, 128]), 90)
        self.assertTrue((framed[:32] == 0).all())
        self.assertTrue((framed[-32:] == 0).all())
        self.assertEqual(normalized_path("a/b.jpeg"), "a/b.jpeg.png")
        self.assertNotEqual(normalized_path("a/b.jpeg"), normalized_path("a/b.png"))

    def make_source(self, root):
        records = []
        for cls in CLASSES:
            for split in ["train", "test"]:
                for i in range(2):
                    suffix = ".PNG" if cls in RARE else ".jpeg"
                    relative = f"data/raw/{cls}/{split}/{i}{suffix}"
                    path = root / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    shape = (10, 12 + CLASSES.index(cls), 4) if cls in RARE else (10, 12 + CLASSES.index(cls))
                    cv2.imwrite(str(path), np.full(shape, 80, np.uint8))
                    records.append(dict(path=relative, label=cls, pid=f"{cls}-{i}", split=split))
            for folder in [f"data/aug_basic/{cls}", f"data/gan/accepted/{cls}"] if cls in RARE else []:
                path = root / folder / "a.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(path), np.full((8, 8, 3), 128, np.uint8))
        pd.DataFrame(records).to_csv(root / "data/manifest.csv", index=False)

    def test_copies_splits_pools_resume_and_content_guard(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.make_source(root)
            original = {str(p): sha256(p) for p in (root / "data/raw").rglob("*") if p.is_file()}
            dest = prepare(root, root, "framing", sample_per_class=1)
            man, records = selected_inputs(root, 1)
            copied = pd.read_csv(dest / "data/manifest.csv")
            self.assertEqual(copied.path.tolist(), man.path.map(normalized_path).tolist())
            self.assertEqual(copied.split.tolist(), man.split.tolist())
            inventory = pd.read_csv(dest / "inventory.csv")
            self.assertEqual(inventory.source_path.tolist(), [r["source_path"] for r in records])
            result, _ = header_check(dest, "framing", inventory)
            self.assertEqual(result["header_signatures"], 1)
            self.assertEqual(result["size_ties"], result["rare_test"])
            before = sha256(dest / "inventory.csv")
            prepare(root, root, "framing", sample_per_class=1)
            self.assertEqual(before, sha256(dest / "inventory.csv"))
            self.assertEqual(original, {p: sha256(p) for p in original})
            groups = cache_groups(dest, "gan", 1)
            self.assertEqual(groups["basic_CSC"], ["data/aug_basic/CSC/a.png.png"])
            self.assertEqual(groups["gan_CSC"], ["data/gan/accepted/CSC/a.png.png"])
            changed = dest / inventory.path.iloc[0]
            cv2.imwrite(str(changed), np.full((256, 256), 23, np.uint8))
            with self.assertRaisesRegex(ValueError, "Content changed"):
                prepare(root, root, "framing", sample_per_class=1)

    def test_placeholders_fail_before_copies(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.make_source(root)
            (root / "data/raw/NORMAL/train/0.jpeg").write_bytes(b"")
            with self.assertRaisesRegex(ValueError, "placeholder"):
                prepare(root, root, "format", sample_per_class=1)
            self.assertFalse((root / "data/c0").exists())

    def test_axis_and_metrics_have_expected_units(self):
        self.assertEqual(auc(np.array([1, 1]), np.array([1, 1])), 0.5)
        self.assertEqual(auc(np.array([2, 3]), np.array([0, 1])), 1)
        diagnostic, _ = axis_result(np.array([[0.0], [1.0]]), ["NORMAL", "CSC"],
                                    np.array([[0.1], [0.9]]), ["NORMAL", "CSC"])
        self.assertEqual(diagnostic["auc"], 1)
        frame = pd.DataFrame(dict(label=CLASSES, pred=CLASSES))
        metrics = recall_metrics(frame)
        self.assertEqual(metrics["rare_macro"], 1)
        self.assertEqual(metrics["common_macro"], 1)
        self.assertEqual(metrics["rare_correct"], 5)

    def test_decision_requires_complete_seeds_and_common_control(self):
        pair = pd.DataFrame(dict(seed=range(1, 21), rare_macro_left=0.8, rare_macro_right=0.6,
                                 common_macro_left=0.9, common_macro_right=0.9))
        self.assertTrue(paired_contrast(pair)["rule_pass"])
        self.assertIsNone(paired_contrast(pair.iloc[:-1])["rule_pass"])
        pair["common_macro_right"] = 0.8
        self.assertFalse(paired_contrast(pair)["rule_pass"])
        pair["common_macro_right"] = 0.9
        pair.loc[:4, "rare_macro_right"] = 0.85
        self.assertFalse(paired_contrast(pair)["rule_pass"])


if __name__ == "__main__":
    unittest.main()
