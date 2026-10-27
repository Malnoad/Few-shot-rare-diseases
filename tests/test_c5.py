"""Training-free C5 arithmetic, immutable pools, source exclusions and cache checks."""
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
import pandas as pd

from c5_topup import candidate_count, capped_pool, checked_copy, fresh_sources, inventory, plan, publish, write_json
from stage2_cyclegan import select_qc
from stage3_classifier import RARE, features, gan_cache_group, gan_images


class C5Tests(unittest.TestCase):
    def test_candidate_budgets_from_committed_qc(self):
        result = plan(Path(__file__).resolve().parents[1]).set_index("disease")
        self.assertEqual(result.candidates.to_dict(), dict(CSC=0, MH=718, MacTel=6739, Stargardt=1749, RP=12508))
        self.assertEqual(candidate_count(3000, 0, 0), 0)
        with self.assertRaises(ValueError):
            candidate_count(100, 0, 4000)

    def test_original_qc_ranking_threshold_ties_and_cap(self):
        df = pd.DataFrame(dict(file=["b", "a", "c", "d", "e"],
                               d_rare=[.1, .1, .2, .3, .4],
                               d_normal=[.11, .12, .8, .3, .2]))
        ranked, kept = select_qc(df, 2, ranking="d_rare")
        self.assertEqual(kept.file.tolist(), ["a", "b"])
        self.assertEqual(ranked["pass"].tolist(), [True, True, True, False, False])
        self.assertEqual(capped_pool(["old2", "old1"], df, target=4), ["old1", "old2", "a", "b"])
        self.assertEqual(capped_pool(["old"], df, target=10), ["old", "a", "b", "c"])
        self.assertEqual(capped_pool(["old"], df, target=1), ["old"])
        with self.assertRaises(ValueError):
            capped_pool(["a"], df)

    def test_fresh_train_normals_and_append_only_prefix(self):
        man = pd.DataFrame(dict(path=["raw/n1.jpeg", "raw/n2.jpeg", "raw/n3.jpeg", "raw/n4.jpeg", "raw/n5.jpeg", "raw/n6.jpeg"],
                                label=["NORMAL"] * 5 + ["MH"],
                                split=["train"] * 4 + ["test", "train"]))
        self.assertEqual(fresh_sources(man, {"n1"}, {"n2"}, 1), ["raw/n3.jpeg"])
        self.assertEqual(fresh_sources(man, {"n1"}, {"n2"}, 2), ["raw/n3.jpeg", "raw/n4.jpeg"])
        with self.assertRaises(ValueError):
            fresh_sources(man, {"n1"}, {"n2"}, 3)

    def test_explicit_pool_and_immutable_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old = root / "data/gan/accepted/MH/old.png"
            old.parent.mkdir(parents=True)
            old.write_bytes(b"original")
            self.assertEqual(gan_images(root, "MH"), ["data/gan/accepted/MH/old.png"])
            with self.assertRaisesRegex(ValueError, "incomplete"):
                gan_images(root, "MH", "v2")
            dest = root / "data/gan/accepted_v2/MH/old.png"
            checked_copy(old, dest)
            checked_copy(old, dest)
            expected = {c: {f"{i}.png": "hash" for i in range(2999)} | {"old.png": "hash"} for c in RARE}
            (dest.parent.parent / "complete.json").write_text(json.dumps(expected))
            with self.assertRaisesRegex(ValueError, "differs"):
                gan_images(root, "MH", "v2")
            for i in range(2999):
                (dest.parent / f"{i}.png").touch()
            self.assertEqual(len(gan_images(root, "MH", "v2")), 3000)
            dest.write_bytes(b"different")
            with self.assertRaisesRegex(ValueError, "immutable"):
                checked_copy(old, dest)
            self.assertEqual(old.read_bytes(), b"original")

    def test_pool_cache_names_and_path_list_guard_without_tensorflow(self):
        self.assertEqual(gan_cache_group("MH", "v1"), "gan_MH")
        self.assertEqual(gan_cache_group("MH", "v2"), "gan_v2_MH")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = ["data/gan/accepted/MH/a.png"]
            np.savez(root / "gan_MH.npz", paths=paths, feats=np.ones((1, 2)))
            def must_extract():
                raise RuntimeError("fresh extraction requested")
            self.assertEqual(features("gan_MH", paths, root, root, must_extract).shape, (1, 2))
            with self.assertRaisesRegex(RuntimeError, "fresh extraction"):
                features("gan_MH", ["data/gan/accepted_v2/MH/a.png"], root, root, must_extract)
            with self.assertRaisesRegex(RuntimeError, "fresh extraction"):
                features("gan_v2_MH", paths, root, root, must_extract)

    def test_publication_gate_resume_and_v1_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, store = Path(tmp) / "repo", Path(tmp) / "drive"
            original = {}
            for cls in RARE:
                files = []
                for i in range(2 if cls == "CSC" else 1):
                    p = root / "data/gan/accepted" / cls / f"old{i}.png"
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_bytes(f"original {cls} {i}".encode())
                    original[p] = p.read_bytes()
                    files.append(str(p))
                if cls != "CSC":
                    checked_copy(Path(files[0]), store / "accepted_v2" / cls / "old0.png")
                    (store / "accepted_v2" / cls / "new.png").write_bytes(b"new accepted")
                    write_json(store / cls / "inputs.json", {"old": inventory(files)})
                    for name in ["sources.json", "plan.json", "pool.csv"]:
                        (store / cls / name).write_text("fixture")
                    qc = store / cls / "qc" / f"qc_{cls}.csv"
                    qc.parent.mkdir()
                    qc.write_text("new QC fixture")
            def fixture_old_pool(root, cls):
                files = sorted(str(p) for p in (root / "data/gan/accepted" / cls).glob("*.png"))
                return files, inventory(files)
            with patch("c5_topup.TARGET", 2), patch("c5_topup.old_pool", fixture_old_pool):
                with self.assertRaisesRegex(ValueError, "incomplete"):
                    publish(root, store)
                self.assertFalse((root / "data/gan/accepted_v2/complete.json").exists())
                for cls in RARE[1:]:
                    write_json(store / cls / "complete.json", inventory((store / "accepted_v2" / cls).glob("*.png")))
                publish(root, store)
                publish(root, store)
            self.assertEqual({p: p.read_bytes() for p in original}, original)
            marker = json.loads((root / "data/gan/accepted_v2/complete.json").read_text())
            self.assertEqual({c: len(v) for c, v in marker.items()}, {c: 2 for c in RARE})
            self.assertTrue((root / "results/stage2/c5_v2/RP/qc_RP.csv").is_file())


if __name__ == "__main__":
    unittest.main()
