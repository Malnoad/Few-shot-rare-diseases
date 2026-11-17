"""Post hoc rules of protocols.md section 8 on small synthetic inputs (no locked file is read)."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fm_core as fm  # noqa: E402
import fm_posthoc as ph  # noqa: E402


class Rules(unittest.TestCase):
    def test_two_way_ties_go_to_first_class(self):
        scores = np.array([[0.5, 0.5, 0.9], [0.4, 0.6, 0.0]])
        self.assertEqual(ph.two_way_choice(scores, 0, 1).tolist(), [0, 1])  # third column is ignored

    def test_macro_and_present_macro(self):
        correct, labels = np.array([1, 0, 1, 1], bool), ["CSC", "CSC", "MH", "MH"]
        self.assertAlmostEqual(ph.macro(correct, labels, ("CSC", "MH")), 0.75)
        value, present = ph.present_macro(correct[:2], labels[:2], ("CSC", "MH"))
        self.assertEqual((value, present), (0.5, ["CSC"]))  # a class with no query is left out
        self.assertEqual(ph.present_macro(correct[:0], [], ("CSC",)), (None, []))
        with self.assertRaises(ValueError):
            ph.macro(correct, labels, ("CSC", "MH", "RP"))

    def test_k_mean_needs_every_k(self):
        values = {(k, 0): float(k) for k in fm.K_DEV}
        values[1, 1] = 9.0  # draw 1 lacks K = 2, 4, 8
        self.assertEqual(ph.k_mean(values), [3.75])

    def test_zscore_fusion_is_equal_weight(self):
        a = np.array([[0.0, 1.0, 2.0]])
        b = np.array([[300.0, 290.0, 0.0]])  # a raw sum picks class 0; equal weight after z-scoring picks 1
        self.assertEqual(int((a + b).argmax()), 0)
        self.assertEqual(int(ph.zscore_fuse([a, b])[0]), 1)
        self.assertEqual(int(ph.zscore_fuse([a, a * 100])[0]), 2)
        self.assertEqual(int(ph.zscore_fuse([np.array([[0., 3., 2.]]), np.array([[0., 2., 3.1]])])[0]), 2)

    def test_leave_one_out(self):
        labels = ["CSC", "CSC", "MH", "MH"]
        a = np.array([[1, 1, 1, 0]], bool)
        b = np.array([[1, 0, 1, 0]], bool)
        diffs = ph.loo_differences(a, b, labels, ("CSC", "MH"), np.array([1]))
        # leaving out query 1 (a right, b wrong) removes the whole difference
        self.assertAlmostEqual(diffs[1], 0.0)
        self.assertAlmostEqual(diffs[0], 0.75 - 0.25)  # a: CSC 1/1, MH 1/2; b: CSC 0/1, MH 1/2
        self.assertAlmostEqual(diffs[3], 1.0 - 0.75)   # a: CSC 2/2, MH 1/1; b: CSC 1/2, MH 1/1

    def test_top_two_gap_and_cutoff(self):
        top, second, gap = ph.top_two_gap(np.array([[0.1, 0.5, 0.4], [0.9, 0.2, 0.3]]))
        self.assertEqual((top.tolist(), second.tolist()), ([1, 0], [2, 2]))
        np.testing.assert_allclose(gap, [0.1, 0.6])
        cut, below = ph.not_sure_cutoff(np.arange(169, dtype=float))
        self.assertEqual((cut, below), (8.0, 8))  # 8 of 169 lie strictly below, as section 8 says

    def test_patient_recall_weights_patients_equally(self):
        correct = np.array([1, 1, 1, 0], bool)
        labels = ["MH"] * 4
        self.assertAlmostEqual(float(ph.patient_recall(correct, labels, ["a", "a", "a", "b"], ("MH",))[0]), 0.5)

    def test_p3_rosters_nest_and_take_one_image_per_patient(self):
        paths = [f"p{i // 3}_{i % 3}.jpg" for i in range(30)]  # 10 patients, 3 images each
        patients = {p: p.split("_")[0] for p in paths}
        groups = ph.p3_groups(paths, patients)
        self.assertEqual(len(groups), 10)
        lists = ph.p3_supports([groups, ph.p3_groups(paths, None)], 0, 7)
        self.assertEqual(len({patients[p] for p in lists[0]}), 8)  # K = 8 examples, 8 different patients
        self.assertEqual(lists, ph.p3_supports([groups, ph.p3_groups(paths, None)], 0, 7))  # fixed stream
        self.assertNotEqual(lists[0], ph.p3_supports([groups], 1, 7)[0])  # direction changes the stream

    def test_head_prediction_uses_logits(self):
        W, b = np.eye(3), np.array([0.0, 0.0, 10.0])
        self.assertEqual(ph.head_predict(W, b, np.array([[5.0, 0.0, 0.0]])).tolist(), [2])

    def test_parse_cell(self):
        text = "query_id,true_label,predicted_label,correct,cosine_CSC,cosine_MH\na,CSC,MH,0,0.1,0.2\n"
        cell = ph.parse_cell(text)
        self.assertEqual((cell.classes, cell.correct.tolist()), (("CSC", "MH"), [False]))
        logit = "query_id,true_label,predicted_label,correct,logit_CSC,logit_MH,prob_CSC,prob_MH\na,CSC,CSC,1,2,1,0.7,0.3\n"
        self.assertEqual(ph.parse_cell(logit).scores.shape, (1, 2))


class PadToSquare(unittest.TestCase):
    """G1: black padding around the centre, then the arm's own transform (fm_pad_extract.py)."""

    def test_pad_geometry_and_fill(self):
        from PIL import Image
        import fm_pad_extract as pad
        self.assertEqual(pad.pad_box(5, 2), (5, 0, 1))
        image = Image.new("RGBA", (4, 2), (200, 100, 50, 255))
        padded, (side, left, top) = pad.pad_to_square(image)
        self.assertEqual((padded.size, side, left, top), ((4, 4), 4, 0, 1))
        self.assertEqual(padded.getpixel((0, 0)), (0, 0, 0, 255))  # opaque black, not transparent
        self.assertEqual(padded.getpixel((0, 1)), (200, 100, 50, 255))  # the image itself is untouched
        tall, _ = pad.pad_to_square(Image.new("L", (1, 3), 9))
        self.assertEqual([tall.getpixel((x, 1)) for x in range(3)], [0, 9, 0])
        with self.assertRaises(ValueError):
            pad.pad_to_square(Image.new("P", (2, 1)))

    def test_encode_padded_checks_content_and_keeps_order(self):
        import hashlib
        import tempfile
        from PIL import Image
        import fm_extract
        import fm_pad_extract as pad
        with tempfile.TemporaryDirectory() as tmp:
            root, rows = Path(tmp), []
            for i, size in enumerate(((6, 2), (3, 3))):
                Image.new("RGB", size, (i * 50, 0, 0)).save(root / f"im{i}.png")
                rows.append(dict(path=f"im{i}.png", sha256=hashlib.sha256((root / f"im{i}.png").read_bytes()).hexdigest()))
            seen = []

            def encode(images):
                seen.append(images[0].size)
                return np.full((1, fm_extract.DIMENSIONS["MIRAGE"]), 1.0 + len(seen), np.float32)
            encoder = fm_extract.Encoder("MIRAGE", encode, lambda: None, {}, dict)
            log = []
            values = pad.encode_padded(encoder, root, rows, log)
            self.assertEqual(seen, [(6, 6), (3, 3)])  # every image reaches the transform as a square
            self.assertEqual(values[:, 0].tolist(), [2.0, 3.0])
            self.assertEqual([r[5:8] for r in log], [[6, 0, 2], [3, 0, 0]])
            rows[0]["sha256"] = "0" * 64
            with self.assertRaises(ValueError):
                pad.encode_padded(encoder, root, rows, [])


class PadNotebook(unittest.TestCase):
    def test_committed_notebook_is_the_builder_output(self):
        import fm_pad_notebook as builder
        path = builder.HERE / builder.NAME
        self.assertEqual(path.read_text(encoding="utf-8").replace("\r\n", "\n"), builder.build(builder.pinned_commit()))

    def test_extract_command_parses(self):
        import argparse
        import re
        from unittest import mock
        import fm_pad_extract as pad
        import fm_pad_notebook as builder
        flags = re.findall(r'"(--[a-z]+)"', builder.STEP8)
        self.assertEqual(flags, ["--root", "--lock", "--external", "--arm", "--assets", "--out"])
        argv = ["fm_pad_extract.py", "extract", *[x for f in flags for x in (f, "MIRAGE" if f == "--arm" else "x")]]
        with mock.patch.object(sys, "argv", argv), mock.patch.object(pad, "run", side_effect=StopIteration) as run:
            with self.assertRaises(StopIteration):
                pad.main()
        self.assertEqual(run.call_args.args[3], "MIRAGE")


class P9Rules(unittest.TestCase):
    def test_rare_margin_uses_rare_and_common_maxima(self):
        classes = list(fm.NINE)
        row = np.zeros(9)
        row[classes.index("CNV")], row[classes.index("MH")], row[classes.index("RP")] = 0.7, 0.9, 0.5
        self.assertAlmostEqual(float(ph.rare_margin(row[None, :], classes)[0]), 0.2)
        shuffled = classes[::-1]  # column order follows the class names, not a fixed position
        self.assertAlmostEqual(float(ph.rare_margin(row[::-1][None, :], shuffled)[0]), 0.2)

    def test_auroc_counts_ties_as_half(self):
        self.assertEqual(ph.auroc([2.0, 3.0], [0.0, 1.0]), 1.0)
        self.assertEqual(ph.auroc([0.0], [1.0]), 0.0)
        self.assertEqual(ph.auroc([1.0], [1.0]), 0.5)
        self.assertAlmostEqual(ph.auroc([1.0, 2.0], [1.0, 0.0]), 0.875)  # 3 wins + 1 tie of 4 pairs
        with self.assertRaises(ValueError):
            ph.auroc([], [1.0])

    def test_cut_is_read_from_the_normals(self):
        neg = np.arange(100, dtype=float)  # 5% of 100 normals lie above 94
        self.assertEqual(ph.share_above_normal_cut(np.array([94.0, 94.5, 99.5]), neg), 2 / 3)

    def test_resampling_keeps_units_whole(self):
        groups = ph.unit_index(["p1", "p2", "p1", "p3"])
        self.assertEqual([g.tolist() for g in groups], [[0, 2], [1], [3]])
        for seed in range(20):
            draw = ph.resample_units(groups, np.random.default_rng(seed))
            self.assertEqual(int((draw == 0).sum()), int((draw == 2).sum()))  # p1's two scans travel together

    def test_table_and_paired_difference(self):
        arms = ["MIRAGE", "Inception"]
        s = ph.P9Set("toy", "note", ("a", "b", "c"), ("d", "e", "f", "g"),
                     {"MIRAGE": np.array([3., 4., 5.]), "Inception": np.array([0., 4., 5.])},
                     {"MIRAGE": np.array([0., 1., 2., 2.5]), "Inception": np.array([1., 2., 3., 6.])},
                     {a: np.array([True, True, False]) for a in arms}, {a: np.array([False] * 4) for a in arms})
        rows, diffs = ph.p9_table([s], arms, n_boot=50)
        self.assertEqual(rows[0][7], 1.0)
        self.assertEqual((rows[0][8], rows[0][9]), (1.0, 1.0))  # perfect separation in every draw
        self.assertAlmostEqual(rows[1][7], ph.auroc([0., 4., 5.], [1., 2., 3., 6.]))
        self.assertEqual(diffs[0][0], "MIRAGE minus Inception")
        self.assertAlmostEqual(diffs[0][2], 1.0 - rows[1][7])
        again, _ = ph.p9_table([s], arms, n_boot=50)
        self.assertEqual(rows, again)  # fixed seed and namespace: identical draws


if __name__ == "__main__":
    unittest.main()
