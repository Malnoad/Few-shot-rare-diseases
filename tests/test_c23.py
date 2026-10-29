"""Focused training-free tests of C2/C3 selection, caches and resume boundaries."""
import gzip
import hashlib
import json
from pathlib import Path
import random
import ast
import tempfile
import time
import unittest

import numpy as np
import pandas as pd

from c23_experiment import (Sweep, cache_view, major_indices, metric_summary, schedule,
                           selected_gan, selection_record, save_selection, unlock, writer_lock)
from c1_experiment import ARTIFACTS, lock_json
from stage1b_normalize import sha256
from stage3_classifier import MAJOR, RARE, gan_cache_group


class C23Tests(unittest.TestCase):
    def test_notebook_wiring_and_upstream_cells_are_preserved(self):
        root = Path(__file__).resolve().parents[1]
        notebook = json.loads((root / "colab_setup.ipynb").read_text(encoding="utf-8"))
        added_ids = {"c23-plan", "c23-restore", "c23-prepare", "c2-run", "c2-summary", "c3-run", "c3-summary"}
        added = [c for c in notebook["cells"] if c.get("id") in added_ids]
        self.assertEqual(len(added), 7)
        for c in added:
            if c["cell_type"] == "code":
                ast.parse("".join(c["source"]))
                self.assertEqual(c["outputs"], [])
                self.assertIsNone(c["execution_count"])
        code = {c["id"]: "".join(c["source"]) for c in added}
        self.assertIn('"--base-cache"', code["c23-restore"])
        self.assertIn('"--cache"', code["c23-prepare"])
        self.assertIn('"revision"', code["c23-restore"])
        self.assertIn('"checkout", "--detach"', code["c23-restore"])
        self.assertLess(code["c23-restore"].index('"checkout", "--detach"'), code["c23-restore"].index("from c23_experiment import Sweep"))
        self.assertIn('("run", "D", "P", "R")', code["c23-restore"])
        self.assertIn("skip Part1", code["c23-plan"])
        self.assertIn("state are gone", code["c23-plan"])
        for experiment in ("c2", "c3"):
            self.assertIn(f'"{experiment}", trainer="fast"', code[f"{experiment}-run"])
            self.assertIn(f'{experiment}.run()', code[f"{experiment}-run"])
            self.assertIn(f'{experiment}.summarize()', code[f"{experiment}-summary"])
        prior = [c for c in notebook["cells"] if c.get("id") not in added_ids]
        prior[0]["source"] = [s for s in prior[0]["source"] if not s.startswith("| 4b. C2/C3 |")]
        # Canonical JSON of ALL upstream10f2037 cells, including source list
        # representation, metadata, outputs and order. Only index link is allowed.
        digest = hashlib.sha256(json.dumps(prior, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
        self.assertEqual(digest, "fb4de33e95bc07b6f427baf1ec739e42526de42e787f630f84142fceffd8312e")

    def groups(self, n=1000):
        groups = {"major_train": ["a"], "major_val": ["v"], "test": ["t"]}
        for c in RARE:
            groups[f"basic_{c}"] = [f"data/aug_basic/{c}/{i:04d}.png" for i in range(2000)]
            groups[gan_cache_group(c, "v2")] = [f"data/gan/accepted_v2/{c}/{i:04d}.png" for i in range(n)]
        return groups

    def test_fixed_schedules_and_zero_is_basic_pool(self):
        self.assertEqual(len(schedule("c2")), 100)
        self.assertEqual(len(schedule("c3")), 60)
        self.assertEqual({s for s, _, _, _ in schedule("c3")}, set(range(1, 21)))
        self.assertEqual(schedule("c2")[:2], [(1, "n0000", 0, None), (1, "n0500", 500, None)])
        self.assertEqual(schedule("c3")[:3], [(1, "full", 3000, None), (1, "cap10000", 3000, 10000), (1, "cap5000", 3000, 5000)])
        with self.assertRaises(ValueError):
            schedule("c4")

    def test_major_nested_caps_independent_class_streams_and_no_replacement(self):
        labels = np.repeat(MAJOR, [15, 12, 4, 20])
        small, large = (major_indices(labels, cap, 7) for cap in (5, 10))
        self.assertTrue(set(small) < set(large))
        self.assertEqual(len(small), 19)
        self.assertEqual(len(large), 34)
        self.assertEqual(len(set(small)), len(small))
        self.assertTrue(np.all(np.diff(small) > 0))
        np.testing.assert_array_equal(major_indices(labels, None, 7), np.arange(len(labels)))
        np.testing.assert_array_equal(major_indices(labels, 100, 7), np.arange(len(labels)))
        normal = np.flatnonzero(labels == "NORMAL")
        self.assertEqual(set(small) & set(normal), set(major_indices(labels[:15], 5, 7)))
        self.assertNotEqual(small.tolist(), major_indices(labels, 5, 8).tolist())
        with self.assertRaises(ValueError):
            major_indices(["CSC"], 5, 1)

    def test_gan_matches_stage3_existing_non_nested_sampler(self):
        groups = self.groups(3000)
        expected, rng = {}, random.Random(13)
        for c in RARE:
            g = gan_cache_group(c, "v2")
            expected[g] = sorted(rng.sample(groups[g], 500))
        self.assertEqual(selected_gan(groups, 500, 13), expected)
        # The first class may happen to share a prefix; later classes consume
        # a different number of RNG draws at each N, so nesting is not promised.
        self.assertFalse(set(expected["gan_v2_MH"]).issubset(selected_gan(groups, 1000, 13)["gan_v2_MH"]))
        self.assertTrue(all(not p for p in selected_gan(groups, 0, 13).values()))

    def test_exact_pooled_rare_split_and_compressed_inventory(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "data").mkdir()
            pd.DataFrame([dict(path=f"{c}/{i}", label=c, split="train") for c in MAJOR for i in range(12)]).to_csv(root / "data/manifest.csv", index=False)
            groups = self.groups()
            baseline = selection_record(root, groups, 0, None, 3)
            gan = selection_record(root, groups, 500, 5, 3)
            for c in RARE:
                self.assertEqual(len(baseline["rare"][c]["train"]), 1800)
                self.assertEqual(len(baseline["rare"][c]["val"]), 200)
                self.assertEqual(len(gan["rare"][c]["train"]), 2250)
                self.assertEqual(len(gan["rare"][c]["val"]), 250)
                self.assertFalse(set(gan["rare"][c]["train"]) & set(gan["rare"][c]["val"]))
                pool = groups[f"basic_{c}"] + selected_gan(groups, 500, 3)[gan_cache_group(c, "v2")]
                order = np.random.default_rng(3).permutation(2500)
                self.assertEqual(gan["rare"][c]["val"], [pool[i] for i in order[:250]])
            self.assertEqual(len(gan["common_train"]), 20)
            self.assertEqual(gan["common_val"], baseline["common_val"])
            save_selection(root / "selection.json.gz", gan)
            with gzip.open(root / "selection.json.gz", "rt", encoding="utf-8") as f:
                self.assertEqual(json.load(f), gan)

    def test_cache_views_subset_rows_without_mutating_full_bundle(self):
        with tempfile.TemporaryDirectory() as temp:
            cache = Path(temp)
            groups = self.groups()
            # Small width is sufficient for row identity; production validation requires2048.
            for g, paths in groups.items():
                np.savez(cache / f"{g}.npz", paths=np.array(paths), feats=np.arange(len(paths))[:, None])
            before = {g: sha256(cache / f"{g}.npz") for g in groups}
            for n in (0, 500, 1000):
                with cache_view(cache, groups, n, 13) as view:
                    for c in RARE:
                        f = view / f"gan_v2_{c}.npz"
                        self.assertEqual(f.exists(), n > 0)
                        if n:
                            with np.load(f, allow_pickle=False) as z:
                                chosen = selected_gan(groups, n, 13)[f"gan_v2_{c}"]
                                self.assertEqual(z["paths"].tolist(), chosen)
                                self.assertEqual(z["feats"].ravel().tolist(), [groups[f"gan_v2_{c}"].index(p) for p in chosen])
                self.assertFalse(view.exists())
            self.assertEqual(before, {g: sha256(cache / f"{g}.npz") for g in groups})

    def test_resume_refuses_identity_and_artifact_changes(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            lock_json(out / "identity.json", {"trainer": "fast"})
            with self.assertRaisesRegex(ValueError, "Provenance changed"):
                lock_json(out / "identity.json", {"trainer": "keras"})
            s = Sweep.__new__(Sweep)
            s.output, s.plan = out, schedule("c2")
            run = out / "s001/n0000"
            run.mkdir(parents=True)
            files = {}
            for name in (*ARTIFACTS, "run.json", "selection.json.gz"):
                (run / name).write_bytes(b"fixture")
                files[name] = sha256(run / name)
            marker = dict(seed=1, condition="n0000", identity_sha256=sha256(out / "identity.json"), files=files)
            (run / "complete.json").write_text(json.dumps(marker), encoding="utf-8")
            self.assertEqual(set(s.completed()), {(1, "n0000")})
            (run / "head.npz").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "artifact changed"):
                s.completed()

    def test_finish_rejects_partial_history_and_nonfinal_head_without_training(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = Path(__file__).resolve().parents[1]
            (root / "data").mkdir()
            (root / "data/manifest.csv").write_bytes((source / "data/manifest.csv").read_bytes())
            man = pd.read_csv(root / "data/manifest.csv")
            frame = man[man.split == "test"][["path", "label"]].copy()
            frame["path"] = frame.path.str.replace("\\", "/", regex=False)
            frame["pred"] = frame.label
            import stage3_classifier as s3
            for c in s3.CLASSES:
                frame[f"p_{c}"] = (frame.label == c).astype(float)
            out = root / "sweep/s001/cap5000"
            out.mkdir(parents=True)
            frame.to_csv(out / "test_predictions.csv", index=False)
            counts = pd.crosstab(man.label, man.split).reindex(s3.CLASSES)
            counts.loc[MAJOR, "train"] = 5000
            counts.loc[RARE, "train"] = 4500
            counts.loc[RARE, "val"] = 500
            counts.to_csv(out / "data_counts.csv")
            history = pd.DataFrame(dict(epoch=range(250), loss=1., val_loss=1., accuracy=.5, val_accuracy=.5))
            history.to_csv(out / "history.csv", index=False)
            for name in ("head.npz", "last_head.npz"):
                np.savez(out / name, W=np.zeros((2048, 9), np.float32), b=np.zeros(9, np.float32), classes=s3.CLASSES)
            (out / "trainer.json").write_text(json.dumps(dict(trainer="fast")), encoding="utf-8")
            for name in (*ARTIFACTS, "run.json", "selection.json.gz"):
                if not (out / name).exists():
                    (out / name).write_bytes(b"fixture")
            (root / "sweep/identity.json").write_text("{}", encoding="utf-8")
            s = Sweep.__new__(Sweep)
            s.root, s.output, s.test, s.trainer = root, root / "sweep", frame, "fast"
            result = s.finish(out, 1, "cap5000", 3000, 5000, time.monotonic())
            self.assertEqual(result["acc"], 100)
            self.assertEqual(result["mcc"], 1)
            history.iloc[:-1].to_csv(out / "history.csv", index=False)
            with self.assertRaisesRegex(ValueError, "fixed250 history"):
                s.finish(out, 1, "cap5000", 3000, 5000, time.monotonic())
            history.to_csv(out / "history.csv", index=False)
            np.savez(out / "head.npz", W=np.zeros((2048, 9), np.float32), b=np.ones(9, np.float32), classes=s3.CLASSES)
            with self.assertRaisesRegex(ValueError, "FINAL head"):
                s.finish(out, 1, "cap5000", 3000, 5000, time.monotonic())

    def test_lock_and_explicit_stale_recovery(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with writer_lock(root):
                with self.assertRaisesRegex(ValueError, "Writer lock exists"):
                    with writer_lock(root):
                        self.fail("second writer")
                with self.assertRaisesRegex(ValueError, "token mismatch"):
                    unlock(root, "wrong")
            self.assertFalse((root / "writer.lock").exists())
            (root / "writer.lock").write_text(json.dumps(dict(token="known")), encoding="utf-8")
            unlock(root, "known")
            self.assertFalse((root / "writer.lock").exists())

    def test_summary_pairs_only_completed_seeds_and_reports_partial(self):
        records = [dict(seed=s, condition=c, acc=90+s+d, mcc=.8+d/100,
                        rare_macro=.8, common_macro=.9, rare_correct=18)
                   for s in (1, 2, 3) for c, d in (("full", 0), ("cap10000", -1), ("cap5000", -2))]
        runs = pd.DataFrame(records).iloc[:-1]
        summary, pairs, contrasts = metric_summary(runs, schedule("c3"))
        row = contrasts[(contrasts.condition == "cap5000") & (contrasts.metric == "acc")].iloc[0]
        self.assertEqual(row.n_pairs, 2)
        self.assertEqual(row.mean_diff, -2)
        self.assertEqual(row.ci95_lo, -2)
        self.assertFalse(row.complete)
        self.assertFalse(summary.complete.any())
        self.assertEqual(len(pairs[(pairs.condition == "cap5000") & (pairs.metric == "acc")]), 2)


if __name__ == "__main__":
    unittest.main()
