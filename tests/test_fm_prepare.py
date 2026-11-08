"""Input-preparation gates on a tiny synthetic dataset (FM_PROTOCOL.md sections 4, 6, 8)."""
import csv
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fm_core as fm
import fm_prepare as prep


class Prepare(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "root"
        self.out = Path(self.tmp.name) / "out"
        self.rng = np.random.default_rng(0)
        self.rows = []
        for label in fm.RARE:
            for i in range(12):
                self.add(label, "train", i)
            self.add(label, "test", 99)
        for label in fm.NINE[:4]:
            for i in range(3):
                self.add(label, "train", i)
            self.add(label, "test", 99)
        self.write_manifest()

    def tearDown(self):
        self.tmp.cleanup()

    def path(self, label, split, i):
        return f"data/raw/{label}/{split}_{i:02d}.png"

    def add(self, label, split, i, pixels=None):
        path = self.path(label, split, i)
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        pixels = self.rng.integers(0, 256, (32, 32, 4), dtype=np.uint8) if pixels is None else pixels
        Image.fromarray(pixels, "RGBA").save(target)
        self.rows.append([path.replace("/", "\\"), label, f"{label}_{i}", split])
        return target

    def write_manifest(self):
        with (self.root / "data/manifest.csv").open("w", newline="", encoding="utf-8") as stream:
            csv.writer(stream).writerows([["path", "label", "pid", "split"], *self.rows])

    def read(self, name):
        return prep.read_csv(self.out / name)

    def run_both(self, reviews=None):
        prep.screen(self.root, self.out)
        return prep.freeze(self.root, self.out, reviews)

    def test_clean_run_writes_full_rosters(self):
        result = self.run_both()
        self.assertEqual(set(result["groups_per_class"].values()), {12})
        self.assertEqual(result["report_k"], [1, 2, 4, 8])
        development = self.read("episodes_development.csv")
        self.assertEqual(len(development), 5 * 5 * 100)
        self.assertTrue(all(row["s08"] for row in development))
        roles = [r["role"] for r in self.read("nine_way_roster.csv")]
        self.assertEqual((roles.count("fitting"), roles.count("query")), (12 + 60, 9))

    def test_rerun_is_byte_identical(self):
        self.run_both()
        first = {p.name: p.read_bytes() for p in self.out.iterdir()}
        shutil.rmtree(self.out)
        self.run_both()
        self.assertEqual(first, {p.name: p.read_bytes() for p in self.out.iterdir()})

    def test_changed_output_is_not_overwritten(self):
        self.run_both()
        target = self.out / "groups.csv"
        target.write_bytes(target.read_bytes() + b"tampered\n")
        with self.assertRaises(ValueError):
            prep.freeze(self.root, self.out)

    def test_alpha_only_copy_shares_a_group(self):
        original = self.root / self.path("CSC", "train", 0)
        pixels = np.array(Image.open(original))
        pixels[..., 3] = 7
        self.add("CSC", "train", 50, pixels)
        self.write_manifest()
        self.assertEqual(self.run_both()["groups_per_class"]["CSC"], 12)
        self.assertEqual(self.read("groups.csv")[0]["member"], self.path("CSC", "train", 0))

    def test_fitting_copy_of_a_legacy_query_is_excluded_but_test_stays(self):
        shutil.copy(self.root / self.path("MH", "test", 99), self.root / self.path("MH", "train", 0))
        result = self.run_both()
        self.assertEqual(result["excluded_rows"], 1)
        self.assertEqual(result["groups_per_class"]["MH"], 11)
        self.assertEqual(self.read("exclusions.csv")[0]["path"], self.path("MH", "train", 0))
        self.assertEqual(result["query_rows"], 9)

    def test_flagged_pairs_need_a_decision_and_rejected_stay_apart(self):
        original = self.root / self.path("RP", "train", 0)
        pixels = np.array(Image.open(original))
        pixels[0, 0, 0] ^= 1  # new pixels, near-identical hash
        self.add("RP", "train", 51, pixels)
        self.write_manifest()
        prep.screen(self.root, self.out)
        with self.assertRaises(ValueError):
            prep.freeze(self.root, self.out)
        pair = next(p for p in self.read("screen_pairs.csv") if p["kind"] == "dhash")
        reviews = Path(self.tmp.name) / "reviews.csv"
        reviews.write_text(f"a,b,decision\n{pair['a']},{pair['b']},rejected\n", encoding="utf-8")
        self.assertEqual(prep.freeze(self.root, self.out, reviews)["groups_per_class"]["RP"], 13)

    def test_too_few_groups_stops_before_any_roster(self):
        self.rows = [r for r in self.rows if not (r[1] == "RP" and r[3] == "train" and r[0].endswith(("_09.png", "_10.png", "_11.png")))]
        self.write_manifest()
        with self.assertRaises(ValueError):
            self.run_both()
        self.assertFalse((self.out / "groups.csv").exists())

    def test_cross_disease_duplicate_stops(self):
        shutil.copy(self.root / self.path("CSC", "train", 0), self.root / self.path("MH", "train", 0))
        with self.assertRaises(ValueError):
            self.run_both()

    def test_outputs_may_not_go_under_data(self):
        with self.assertRaises(ValueError):
            prep.screen(self.root, self.root / "data/out")


if __name__ == "__main__":
    unittest.main()
