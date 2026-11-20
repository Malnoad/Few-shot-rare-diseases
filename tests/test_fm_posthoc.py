"""Post hoc rules of protocols.md section 8 on small synthetic inputs (no locked file is read)."""
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

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


    def test_odd_sizes_centre_with_the_extra_pixel_right_and_bottom(self):
        from PIL import Image
        import fm_pad_extract as pad
        self.assertEqual(pad.pad_box(3, 6), (6, 1, 0))  # 3 columns to add: 1 left, 2 right
        self.assertEqual(pad.pad_box(6, 3), (6, 0, 1))  # 3 rows to add: 1 top, 2 bottom
        self.assertEqual(pad.pad_box(882, 321), (882, 0, 280))  # an OCTDL scan: 280 rows above, 281 below
        wide, (side, left, top) = pad.pad_to_square(Image.new("RGB", (5, 2), (7, 8, 9)))
        self.assertEqual((side, left, top), (5, 0, 1))
        self.assertEqual([wide.getpixel((0, y)) for y in range(5)], [(0, 0, 0), (7, 8, 9), (7, 8, 9), (0, 0, 0), (0, 0, 0)])
        tall, _ = pad.pad_to_square(Image.new("L", (2, 5), 9))
        self.assertEqual([tall.getpixel((x, 0)) for x in range(5)], [0, 9, 9, 0, 0])

    def test_pixels_are_pasted_unchanged_and_squares_are_untouched(self):
        from PIL import Image
        import fm_pad_extract as pad
        rng = np.random.default_rng(0)
        image = Image.fromarray(rng.integers(0, 256, (7, 12, 4), dtype=np.uint8), "RGBA")
        padded, (side, left, top) = pad.pad_to_square(image)
        self.assertEqual(padded.crop((left, top, left + 12, top + 7)).tobytes(), image.tobytes())
        square = Image.fromarray(rng.integers(0, 256, (6, 6), dtype=np.uint8), "L")
        same, box = pad.pad_to_square(square)
        self.assertEqual((box, same.tobytes(), pad.pixel_hash(same)), ((6, 0, 0), square.tobytes(), pad.pixel_hash(square)))

    def test_padding_reaches_the_mirage_transform_before_its_resize(self):
        from PIL import Image
        import fm_extract
        import fm_pad_extract as pad
        scan = Image.fromarray(np.random.default_rng(1).integers(50, 250, (100, 400), dtype=np.uint8), "L")
        padded = fm_extract.mirage_input(pad.pad_to_square(scan)[0])  # the locked 512x512 transform, unchanged
        stretched = fm_extract.mirage_input(scan)
        band = np.flatnonzero(padded.max(axis=1) > 0)
        self.assertEqual(padded.shape, (512, 512))
        self.assertAlmostEqual(len(band), 512 * 100 / 400, delta=2)  # the scan keeps its 4:1 shape
        self.assertAlmostEqual((band[0] + band[-1]) / 2, 255.5, delta=1.5)  # centred
        self.assertTrue((stretched.max(axis=1) > 0).all())  # the locked input fills every row instead

    def test_padded_images_use_the_decoder_and_the_control_the_locked_path(self):
        import hashlib
        import tempfile
        from PIL import Image
        import fm_extract
        import fm_pad_extract as pad
        dim = fm_extract.DIMENSIONS["Inception"]
        calls = []
        encoder = fm_extract.Encoder(
            "Inception", lambda images: calls.append(("encode", images[0].size)) or np.ones((1, dim), np.float32),
            lambda: None, {}, dict,
            stage_path=lambda path: (calls.append(("stage", path.name)), lambda: np.full((1, dim), 2.0, np.float32))[1],
            decode_path=lambda path: calls.append(("decode", path.name)) or Image.new("RGB", (4, 2), (9, 9, 9)))
        with tempfile.TemporaryDirectory() as tmp:
            Image.new("RGB", (4, 2), (9, 9, 9)).save(Path(tmp) / "a.png")
            rows = [dict(path="a.png", sha256=hashlib.sha256((Path(tmp) / "a.png").read_bytes()).hexdigest())]
            pad.encode_padded(encoder, Path(tmp), rows, [])
            self.assertEqual(calls, [("decode", "a.png"), ("encode", (4, 4))])  # Keras' decoder, then the square
            calls.clear()
            self.assertEqual(pad.control(encoder, Path(tmp), rows)[0, 0], 2.0)
            self.assertEqual(calls, [("stage", "a.png")])  # the control is the locked stage3 path, unpadded

    def test_only_the_two_squashing_arms(self):
        import fm_pad_extract as pad
        with self.assertRaises(ValueError):
            pad.run(Path("x"), Path("x"), Path("x"), "DINOv2", Path("x"), Path("x"))


class G1Laptop(unittest.TestCase):
    """The laptop half of G1: reading Colab's archives and the comparison page (no locked file is read)."""

    def test_square_paths(self):
        rows = [dict(path="a", width="5", height="5", side="5", left="0", top="0"),
                dict(path="b", width="5", height="4", side="5", left="0", top="0")]
        self.assertEqual(ph.square_paths(rows), ["a"])

    def test_padded_root_checks_each_archive_against_its_receipt(self):
        import json
        import tempfile
        import zipfile
        import fm_prepare as prep
        with tempfile.TemporaryDirectory() as tmp:
            drive, work = Path(tmp, "g1"), Path(tmp, "work")
            drive.mkdir()
            for arm in ph.G1_ARMS:
                archive = drive / f"g1-{arm}-20261119T000000Z.zip"
                with zipfile.ZipFile(archive, "w") as z:
                    z.writestr("complete.json", "{}")
                Path(str(archive) + ".json").write_text(json.dumps(dict(sha256=prep.sha256(archive))))
            root = ph.padded_root(drive, work)
            self.assertTrue(all((root / arm / "complete.json").exists() for arm in ph.G1_ARMS))
            self.assertEqual(ph.padded_root(root, Path(tmp, "unused")), root)  # unzipped folders are used as given
            receipt = drive / "g1-MIRAGE-20261119T000000Z.zip.json"
            receipt.write_text(json.dumps(dict(sha256="0" * 64)))
            with self.assertRaises(ValueError):
                ph.padded_root(drive, Path(tmp, "work2"))
            (drive / "g1-MIRAGE-20261120T000000Z.zip").write_bytes(b"")
            with self.assertRaisesRegex(ValueError, "expected one"):
                ph.padded_root(drive, Path(tmp, "work3"))

    def test_report_only_rearranges_the_tables(self):
        label = "post hoc, G1 test label"
        g1_rows = []
        for arm, shift in (("MIRAGE", 0.1), ("Inception", -0.05)):
            row = lambda table, k, q, a, b, size="": dict(posthoc=label, arm=arm, table=table, k=k, quantity=q, locked=a,  # noqa: E731
                                                          padded_label="padded", padded=b, size_only_label="", size_only=size)
            g1_rows += [dict(posthoc=label, arm=arm, table="control: unpadded", k="", quantity="max_difference_ratio",
                             locked="2e-06", padded_label="min_cosine", padded="0.99999", size_only_label="", size_only=""),
                        row("development", "1", "macro_recall_mean", "0.5", str(0.5 + shift), "0.578"),
                        row("hold-out", "1", "macro_recall_mean", "0.6", "0.6", "0.645"),
                        row("hold-out", "1", "mean_correct_of_22", "12.0", "13.5"),
                        *[row("nine-way OCTDL NORMAL (n=332)", "", q, "26", "40") for q in
                          ("correct_class", "called_NORMAL", "called_CNV_DME_DRUSEN", "called_any_rare")]]
        p9 = lambda arm, value: dict(arm=arm, set="OCTDL MH against OCTDL NORMAL", auroc=value, boot95_low="0.2",  # noqa: E731
                                     boot95_high="0.9", argmax_rare_called_any_rare="0.1", argmax_normals_called_rare="0.05")
        diff = lambda value: [dict(comparison="MIRAGE minus Inception", set="OCTDL MH against OCTDL NORMAL",  # noqa: E731
                                   auroc_difference=value, boot95_low="-0.3", boot95_high="0.1")]
        page = ph.g1_report(g1_rows, [p9("MIRAGE", "0.357"), p9("Inception", "0.634"), p9("DINOv2", "0.978")],
                            diff("-0.277"), [p9("MIRAGE", "0.700"), p9("Inception", "0.650")], diff("0.05"))
        self.assertIn("never replaces a locked number", page)
        self.assertIn(label, page)
        self.assertIn("| 1 | 0.500 | 0.600 | +0.100 | 0.500 | 0.450 | -0.050 | 0.578 |", page)
        self.assertIn("| 1 | 12.00 | 13.50 | +1.50 | 12.00 | 13.50 | +1.50 |", page)
        self.assertIn("| OCTDL NORMAL (n=332) | MIRAGE | 26 → 40 | 26 → 40 | 26 → 40 | 26 → 40 |", page)
        self.assertIn("| OCTDL MH against OCTDL NORMAL | 0.357 (0.200 to 0.900) | 0.700 (0.200 to 0.900) | "
                      "0.634 (0.200 to 0.900) | 0.650 (0.200 to 0.900) |", page)
        self.assertIn("| OCTDL MH against OCTDL NORMAL | -0.277 (-0.300 to 0.100) | 0.050 (-0.300 to 0.100) |", page)
        self.assertNotIn("DINOv2", page)  # G1 covers MIRAGE and Inception-v3 only
        self.assertIn("max_difference_ratio 2.00e-06", page)


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


def textured(seed: int, h: int = 240, w: int = 320) -> np.ndarray:
    """A greyscale picture with corners and blobs that SIFT can match."""
    import cv2
    rng = np.random.default_rng(seed)
    img = np.full((h, w), 30, np.uint8)
    for _ in range(40):
        x, y = int(rng.integers(0, w - 30)), int(rng.integers(0, h - 30))
        cv2.rectangle(img, (x, y), (x + int(rng.integers(8, 30)), y + int(rng.integers(8, 30))),
                      int(rng.integers(80, 255)), -1)
        cv2.circle(img, (int(rng.integers(0, w)), int(rng.integers(0, h))), int(rng.integers(3, 12)),
                   int(rng.integers(0, 255)), -1)
    return cv2.GaussianBlur(img, (3, 3), 0)


class R2Rules(unittest.TestCase):
    def test_spread_gate_and_paired_difference(self):
        draws = range(4)
        values = {"MIRAGE": {(k, d): 0.5 + 0.01 * k + 0.001 * d for k in fm.K_DEV for d in draws},
                  "Inception": {(k, d): 0.4 + 0.01 * k for k in fm.K_DEV for d in draws}}
        k_scores = {a: {k: float(np.mean([values[a][k, d] for d in draws])) for k in fm.K_DEV} for a in values}
        header, rows = ph.r2_rows(values, k_scores)
        diff = [r for r in rows if r[0] == "MIRAGE minus Inception" and r[2] == 1][0]
        self.assertEqual(diff[3], 4)
        self.assertAlmostEqual(diff[4], 0.1015)  # 0.1 + mean(0, .001, .002, .003)
        self.assertEqual(len(rows), 3 * (len(fm.K_DEV) + 1))  # two arms + one difference, K rows + K mean
        k_scores["MIRAGE"][1] += 1e-6
        with self.assertRaises(ValueError):
            ph.r2_rows(values, k_scores)


class P10SensitivityRules(unittest.TestCase):
    def test_matched_pairs_skip_synthetic_and_follow_unsure(self):
        key = [{"item": "1", "kind": "pair", "left": "a", "right": "b"},
               {"item": "2", "kind": "synthetic", "left": "c", "right": "synthetic:c"},
               {"item": "3", "kind": "decoy", "left": "d", "right": "e"},
               {"item": "4", "kind": "pair", "left": "f", "right": "g"}]
        marks = [{"item": "1", "mark": "same picture"}, {"item": "2", "mark": "same picture"},
                 {"item": "3", "mark": "same case, other slice"}, {"item": "4", "mark": "unsure"}]
        self.assertEqual(ph.p10_matched_pairs(key, marks, False), {frozenset("ab"), frozenset("de")})
        self.assertEqual(ph.p10_matched_pairs(key, marks, True), {frozenset("ab"), frozenset("de"), frozenset("fg")})
        with self.assertRaises(ValueError):
            ph.p10_matched_pairs(key, marks[:3], False)

    def test_development_drops_only_queries_matched_to_their_own_supports(self):
        import fm_evaluate as ev
        rare = fm.RARE
        queries = tuple(f"q{i}" for i in range(5))
        supports = tuple((f"s{i}",) for i in range(5))
        episode = ev.Episode(fold=0, k=1, draw=0, supports=supports, queries=queries, labels=rare)
        cell = ph.Cell(queries=queries, labels=rare, predicted=("CSC",) * 5, classes=rare, scores=np.zeros((5, 5)))
        cells = {(0, 1, 0): cell}
        values, dropped = ph.p10_development([episode], cells, list(queries), list(rare), set())
        self.assertAlmostEqual(values[1, 0], 0.2)  # only CSC correct
        values, dropped = ph.p10_development([episode], cells, list(queries), list(rare), {frozenset(("q0", "s3"))})
        self.assertEqual(dropped, 1)
        self.assertAlmostEqual(values[1, 0], 0.0)  # CSC query removed; four classes left, none correct
        values, dropped = ph.p10_development([episode], cells, list(queries), list(rare), {frozenset(("q0", "zz"))})
        self.assertEqual(dropped, 0)  # a match with an image outside the episode's supports does not count

    def test_holdout_excludes_named_images(self):
        rare = fm.RARE
        cell = ph.Cell(queries=("h0", "h1", "h2", "h3", "h4"), labels=rare, predicted=rare[:4] + ("CSC",),
                       classes=rare, scores=np.zeros((5, 5)))
        out = ph.p10_holdout({(0, 1, 0): cell}, set())
        self.assertAlmostEqual(out[1, 0], 0.8)
        out = ph.p10_holdout({(0, 1, 0): cell}, {"h4"})
        self.assertAlmostEqual(out[1, 0], 1.0)


class P10Rules(unittest.TestCase):
    def test_s1_links_follow_headings_and_join_cut_lines(self):
        text = ["S1. Detailed links (contents)", "S1. Detailed links of collected", "https://data.example/set",
                "CSC", "https://a.example/one", "Macular hole", "https://b.example/very-long-", "path-continues",
                "https://a.example/one", "S2. Detailed CycleGAN", "https://ignored.example"]
        urls = ["https://data.example/set", "https://a.example/one", "https://b.example/very-long-path-continues",
                "https://b.example/very-long-path-continues", "https://a.example/one", "https://ignored.example"]
        got = ph.s1_links(text, urls)
        self.assertEqual(got, [("major", "https://data.example/set"), ("CSC", "https://a.example/one"),
                               ("MH", "https://b.example/very-long-path-continues"), ("MH", "https://a.example/one")])
        header, rows = ph.p10_links(got)
        self.assertEqual([r[3] for r in rows], [1, 2, 1, 2])  # the same link under two diseases counts twice

    def test_link_kind(self):
        self.assertEqual(ph.link_kind("https://x.org/viewimage.asp?img=a_f1.jpg"), "image")
        self.assertEqual(ph.link_kind("https://x.com/articles/10.1186/s1/figures/2"), "figure")
        self.assertEqual(ph.link_kind("https://webeye.org/eyeforum/cases/261-FTMH.htm"), "case")
        self.assertEqual(ph.link_kind("https://www.sciencedirect.com/science/article/pii/S1"), "publication")
        self.assertEqual(ph.link_kind("https://www.clinic.com/central-serous-retinopathy"), "web page")

    def test_sift_finds_a_recropped_copy_and_not_another_picture(self):
        a, other = textured(1), textured(2)
        copy, params = ph.synthetic_positive(a, np.random.default_rng([42, ph.P10_SYNTH_NS]))
        self.assertTrue(0.5 * 0.7 * a.shape[1] - 1 <= copy.shape[1] <= 1.5 * 0.9 * a.shape[1] + 1)
        self.assertTrue(0.70 <= params["keep_w"] <= 0.90 and 0.5 <= params["scale"] <= 1.5)
        same = ph.sift_inliers(ph.sift_keys(a), ph.sift_keys(copy))
        diff = ph.sift_inliers(ph.sift_keys(a), ph.sift_keys(other))
        self.assertGreater(same, 3 * max(diff, 4))
        self.assertEqual(same, ph.sift_inliers(ph.sift_keys(a), ph.sift_keys(copy)))  # seeded RANSAC: repeatable
        self.assertEqual(ph.sift_inliers(ph.sift_keys(np.zeros((50, 50), np.uint8)), ph.sift_keys(a)), 0)

    def test_review_list(self):
        paths = [f"p{i:02d}" for i in range(20)]
        pairs = [(a, b, (i * 7 + j * 3) % 50) for i, a in enumerate(paths) for j, b in enumerate(paths) if i < j]
        review, n_above = ph.p10_review(pairs, 45, paths, np.random.default_rng([42, ph.P10_REVIEW_NS]))
        self.assertEqual(n_above, sum(s >= 45 for _, _, s in pairs))
        kinds = [r[1] for r in review]
        self.assertEqual((kinds.count("synthetic"), kinds.count("decoy")), (ph.P10_SYNTH, ph.P10_DECOYS))
        shown = {frozenset(r[3:5]) for r in review if r[1] == "pair"}
        top = sorted(pairs, key=lambda p: (-p[2], p[0], p[1]))[:ph.P10_TOP]
        self.assertTrue({frozenset(p[:2]) for p in top} <= shown)
        self.assertEqual(len(shown), len({frozenset(p[:2]) for p in pairs if p[2] >= 45} | {frozenset(p[:2]) for p in top}))
        median = np.median([p[2] for p in pairs])
        self.assertTrue(all(r[5] < 45 and r[5] <= median for r in review if r[1] == "decoy"))
        self.assertEqual([r[0] for r in review], list(range(1, len(review) + 1)))
        again, _ = ph.p10_review(pairs, 45, paths, np.random.default_rng([42, ph.P10_REVIEW_NS]))
        self.assertEqual(review, again)

    def test_sheet_shows_no_label_or_score(self):
        review = [[1, "pair", "top 30", "data/raw/rare/CSC/01.PNG", "data/raw/rare/MH/02.PNG", 99]]
        img = textured(3, 40, 60)
        html = ph.p10_sheet(review, {review[0][3]: img, review[0][4]: img})
        import re
        self.assertEqual(html.count("<img"), 2)
        visible = re.sub(r"data:image/png;base64,[A-Za-z0-9+/=]+", "", html)
        for secret in ("CSC", "MH/02", "top 30", "99", "synthetic", "decoy", "rare"):
            self.assertNotIn(secret, visible)


class Phase1Rules(unittest.TestCase):
    def test_success_boundaries_and_pairing(self):
        five = ph.success_rule([0] * 5 + [1] * 95)
        self.assertEqual((five["n_positive"], five["verdict"]), (95, "pass"))
        self.assertAlmostEqual(five["p05"], .95)
        self.assertEqual(ph.success_rule([0] * 6 + [1] * 94)["verdict"], "fail")
        self.assertEqual(ph.success_rule([1] * 99)["verdict"], "not evaluable")
        self.assertEqual(ph.success_rule([1] * 99 + [np.nan])["verdict"], "not evaluable")
        self.assertEqual(ph.success_rule([-100] * 5 + [1] * 95)["verdict"], "fail")  # linear p05 governs
        self.assertEqual(ph.paired({(1, 1): .9, (1, 2): .1}, {(1, 1): .4, (2, 1): .5}), {(1, 1): .5})

    def test_fixed_sentences_use_the_registered_inputs(self):
        yes = ph.success_rule([.125] * 100)
        no = ph.success_rule([-.25] * 100)
        thumb = ph.success_rule([.25] * 100)
        self.assertEqual(
            ph.e1_sentences(yes, .116, thumb),
            "Under the size and framing control, MIRAGE keeps a positive development advantage over Inception-v3 "
            "(mean 0.125, locked 0.116; p05 0.125); it is above the pixel-only reference (mean 0.250).",
        )
        mixed = ph.success_rule([0.] * 6 + [.25] * 94)
        self.assertIn("it is not above the pixel-only reference (mean 0.235)", ph.e1_sentences(yes, .116, mixed))
        for main in (yes, no):
            for other in (yes, no):
                lead = "and also" if main == other else "but not"
                text = ph.e1_sentences(main, .116, thumb, e1l=other, e1c=other, res256_shift=-.031)
                self.assertIn(f", {lead} when effective resolution is evened out (E1-L p05 = {other['p05']:.3f})", text)
                self.assertIn(f", {lead} when the image is only cropped to its content box (E1-C p05 = {other['p05']:.3f})", text)
                if main == no:
                    self.assertIn("K mean by -0.031 (E3 M-res256)", text)
        self.assertEqual(ph.e1_sentences(yes, .116, None), "")
        self.assertEqual(ph.e1_sentences(no, .116, thumb), "")
        self.assertEqual(ph.e1_sentences({"verdict": "not evaluable"}, .116, None), "not evaluable")

    def test_rule_hash_and_changed_label(self):
        section = f"{ph.PHASE1_HEADING}\n\nRule before numbers.\n\n"
        self.assertEqual(ph.section_text("# Start\n" + section + "## Next\n"), section.rstrip() + "\n")
        self.assertEqual(ph.section_sha256(section), ph.section_sha256(section.replace("\n", "\r\n")))
        edited = ph.section_sha256(section.replace("numbers", "outputs"))
        self.assertNotEqual(ph.section_sha256(section), edited)
        commit = "4b4e883fb491b4402738fe47717ceae3b800161d"
        hashes = {"same": False, "commit_sha256": ph.section_sha256(section), "working_sha256": edited}
        self.assertEqual(ph.long_date("2026-11-20"), "20 Nov 2026")
        self.assertIn("commit 4b4e883 (20 Nov 2026), computed 2026-11-21", ph.e_label(1, commit, "20 Nov 2026", "2026-11-21", hashes))
        self.assertIn("WARNING", ph.e_label(4, commit, "20 Nov 2026", "2026-11-21", hashes))

    def test_references_and_matched_queries(self):
        supports = [np.array([[0., 0.], [2., 2.]]), np.array([[3., 3.]])]
        q = np.array([[1., 1.], [2., 2.], [4., 4.]])
        np.testing.assert_array_equal(ph.nearest_mean_predict(supports, q), fm.size_predict(supports, q))
        np.testing.assert_array_equal(ph.nearest_mean_scores(supports, q)[1], [-2, -2])
        cosine = ph.prototype_scores([np.array([[1., 0.]]), np.array([[0., 1.]])], np.array([[3., 4.]]))
        np.testing.assert_allclose(cosine, [[.6, .8]])
        sides = {"a": 1, "b": 2, "c": 4, "d": 2, "e": 4, "f": 8}
        labels = {k: "X" if k < "d" else "Y" for k in sides}
        self.assertEqual(ph.matched_subset(sides, labels), (2, 4, ["b", "c", "d", "e"]))
        self.assertEqual(ph.matched_subset({"a": 1, "b": 3}, {"a": "X", "b": "Y"}), (3, 1, []))

    def test_e2_overlap_and_linear_tertiles(self):
        self.assertIsNone(ph.jaccard(set(), set()))
        self.assertEqual(ph.jaccard({"a"}, set()), 0)
        self.assertEqual(ph.jaccard({"a", "b"}, {"b", "c"}), 1 / 3)
        self.assertEqual(ph.failure_overlap({"A": {"x"}, "B": {"y", "x"}}, ["x", "y", "z"]), [1, 1, 1])
        self.assertEqual(ph.same_class_share(["NORMAL", "CNV"], ["NORMAL", "MH"]), .5)
        self.assertEqual(ph.tertile_shares([1, 1, 1, 4], [1, 0, 1, 0]), ([1., 1.], [3, 0, 1], [2 / 3, None, 0]))
        np.testing.assert_allclose(ph.tertile_shares([1, 2, 3, 4, 5], [0] * 5)[0], [7 / 3, 11 / 3])

    def test_e2d_small_groups_and_cluster_bootstrap(self):
        small = ph.e2d_row(range(6), [False] * 5 + [True], list("abcdef"), None)
        self.assertEqual(len(small), len(ph.E2D_HEADER))
        self.assertEqual(small[:8], [1, 5, 5., None, None, 2., None, None])
        self.assertTrue(all(x is None for x in small[8:]))
        large = ph.e2d_row(range(10), [False] * 5 + [True] * 5, ["a"] * 5 + ["b"] * 5,
                           np.random.default_rng([42, ph.E2_BOOT_NS]), n_boot=8)
        self.assertEqual(large[-4:], [1., 1., 1., True])
        self.assertEqual(ph.quartiles([]), [None, None, None])
        self.assertEqual(ph.quartiles([0, 1, 2]), [1., .5, 1.5])

    def test_e2f_sampling_and_marks(self):
        groups = {("OCTID", "consensus failure"): [f"x{i}" for i in range(24)], ("OCTDL", "consensus normal"): ["y"]}
        anchors = [f"a{i}" for i in range(12)]
        got = ph.contact_sheet_sample(groups, anchors)
        self.assertEqual(got, ph.contact_sheet_sample(dict(reversed(list(groups.items()))), anchors[::-1]))
        self.assertEqual(len(got), 25)
        self.assertEqual(sum(row[3] == "anchor" for row in got), 8)
        self.assertEqual([row[0] for row in got], [f"S{i:03d}" for i in range(1, 26)])
        key = [{"id": "S001", "source": "OCTDL", "group": "consensus normal"}]
        rows = [dict.fromkeys(ph.E2F_MARKS, "yes") | {"id": "S001", "scan_width": "wide"}]
        ph.validate_marks(rows, ["S001"])
        self.assertEqual(len(ph.mark_counts(rows, key)), len(ph.E2F_MARKS))
        self.assertTrue(all(r[-1] == "shown in full, no comparison" for r in ph.mark_counts(rows, key)))
        with self.assertRaises(ValueError):
            ph.validate_marks(rows + rows, ["S001"])

    def test_e3_criteria_and_strict_percentiles(self):
        mirage = {v: [.1] * 100 for v in ph.E3_VARIANTS["MIRAGE"]}
        retfound = {v: [-.1] * 100 for v in ph.E3_VARIANTS["RETFound"]}
        gap = {v: [.2] * 100 for v in retfound}
        result = [r[-1] for r in ph.e3_criteria(mirage | {"M-pad": [-1] * 100}, retfound, gap)]
        self.assertIn("kept within this grid", result)
        self.assertIn("position unchanged within this grid", result)
        mirage["M-res256"] = [0.] * 100
        retfound["R-pad"] = [0.] * 100
        gap["R-squash"] = []
        result = [r[-1] for r in ph.e3_criteria(mirage, retfound, gap)]
        self.assertIn("sensitive to M-res256", result)
        self.assertIn("not evaluable", result)
        self.assertNotIn("position unchanged within this grid", result)
        self.assertTrue(ph.e3_interval_check(5, 10, (.5, .6)))
        self.assertFalse(ph.e3_interval_check(7, 10, (.5, .6)))
        self.assertEqual(ph.ranks_with_ties({"a": 1, "b": 1, "c": 2}), {"a": 1, "b": 1, "c": 3})

    def test_e4_reading_thresholds_and_ceiling(self):
        self.assertEqual([ph.e4_reading(v) for v in (.599, .60, .80, .95)],
                         ["hardly separable", "partly separable", "strongly separable", "near-perfectly separable"])
        self.assertEqual(ph.e4_pair_reading(1., .99, -.02, -.01), "at ceiling; the difference is not read")
        self.assertEqual(ph.e4_pair_reading(.999, .989, -.02, -.01), "the control reduced separability")
        self.assertEqual(ph.e4_pair_reading(.99, .98, -.02, 0.), "no reduction shown")
        self.assertEqual(ph.e4_pair_reading(.95, .90, -.06, -.04, control=False), "descriptive comparison")

    def test_e4_namespaces_folds_draws_and_probes(self):
        self.assertEqual((ph.E2_BOOT_NS, ph.E2_SAMPLE_NS, ph.E2_ORDER_NS), (12101, 12102, 12103))
        self.assertEqual((ph.E4_FOLD_NS, ph.E4_BOOT_NS, ph.E4_FEWSHOT_NS), (14101, 14102, 14103))
        self.assertEqual(len(ph.e4_pairs()), 9)
        self.assertEqual(ph.e4_pairs()[-3:], list(ph.E4_NORMAL_PAIRS))
        y = np.repeat([0, 1], 24)
        units = np.repeat([f"unit-{i}" for i in range(24)], 2)
        X = np.column_stack((1 - y, y, np.ones(len(y))))
        folds = ph.grouped_folds(y, units, 6)
        self.assertEqual(len(folds), 5)
        np.testing.assert_array_equal(np.sort(np.concatenate(folds)), np.arange(len(y)))
        for first, again in zip(folds, ph.grouped_folds(y, units, 6)):
            np.testing.assert_array_equal(first, again)
            self.assertTrue(all(np.count_nonzero(units[first] == u) == 2 for u in units[first]))
        fit, keep, _, _ = ph.fit_logistic_raw(X, y)
        self.assertTrue(fit.converged)
        self.assertEqual(keep.tolist(), [0, 1])
        for normalize in (False, True):
            scores, failed = ph.probe_scores(X, y, folds, normalize)
            self.assertEqual(failed, 0)
            self.assertEqual(ph.auroc(scores[y == 1], scores[y == 0]), 1)
        draws = ph.source_fewshot(X, y, units, 6, ph.prototype_predict, draws=3)
        self.assertEqual(draws, {(k, r): 1. for k in ph.KS for r in range(3)})
        member_ids = [sorted(set(units[y == s])) for s in (0, 1)]
        indices = [{u: np.flatnonzero(units == u).tolist() for u in ids} for ids in member_ids]
        plan = ph.source_draw(member_ids, indices, 6, 1)
        self.assertEqual(plan, ph.source_draw(member_ids, indices, 6, 1))
        self.assertNotEqual(plan, ph.source_draw(member_ids, indices, 6, 2))
        for order, picked in plan:
            self.assertEqual([units[p] for p in picked], order[:8])

    def test_render_layout_and_residuals(self):
        row = {"path": "data/raw/rare/MH/example.png", "sha256": "stub", "box_left": "1", "box_right": "65",
               "box_top": "2", "box_bottom": "34"}
        self.assertEqual(ph.content_sizes(SimpleNamespace(render_rows=lambda: [row])), {row["path"]: (64, 32)})
        self.assertEqual(ph.render_png(Path("render"), "E1-M", row["path"]).as_posix(),
                         "render/E1-M/data/raw/rare/MH/example.png.png")
        edge = {f"{s}_{k}": v for s in ("original", *ph.E_SETS)
                for k, v in (("border_share", ".1"), ("edge_run", ".10"), ("near_white_share", ".12"))}
        values = ph.framing_residual([row | edge], {})
        self.assertEqual(len(values), 20)
        self.assertTrue(all(r[1] == "MH" for r in values))
        self.assertTrue(all(r[4] == 1 for r in values if "> 5%" in r[3]))
        self.assertTrue(all(r[4] == 0 for r in values if "> 10%" in r[3]))
        self.assertEqual(ph.rerender_agreement([(np.array([0, 255]), np.array([255, 0]))]), (1, 1, 255))

    def test_phase1_render_cache_and_receipt(self):
        import zipfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            producer = root / "producer"
            producer.mkdir()
            inventory = [{"path": "original.png", "sha256": "digest"}]
            header = list(ph.RENDER_COLUMNS)
            row = dict.fromkeys(header, "0") | inventory[0]
            ph.prep.write_csv(producer / "render.csv", header, [[row[c] for c in header]])
            ph.prep.write_csv(producer / "pixel_stats.csv",
                              ["path", *[f"{s}_{k}" for s in ("original", "E1-M") for k in ph.PIXEL_STATS]],
                              [["original.png", *range(16)]])
            np.savez(producer / "thumbs16.npz", paths=["original.png"], sha256=["digest"],
                     raw=np.ones((1, 256)) * .5, **{"E1-M": np.ones((1, 256))})
            files = ["render.csv", "pixel_stats.csv", "thumbs16.npz"]
            ph.prep.complete(producer, files)
            archive = root / "phase1-render.zip"
            with zipfile.ZipFile(archive, "w") as stream:
                for name in [*files, "complete.json"]:
                    stream.write(producer / name, name)
            receipt = archive.with_suffix(".zip.json")
            digest = ph.prep.sha256(archive)
            ph.prep.write_json(receipt, {"sha256": digest})
            with patch.object(ph, "safe_unzip", wraps=ph.safe_unzip) as unpack:
                cache = root / "cache"
                _, rows, rendered = ph.phase1_root(root, root / "e1", cache)
                _, other_rows, other_render = ph.phase1_root(root, root / "e2", cache)
                self.assertEqual((other_rows, other_render), (rows, rendered))
                self.assertEqual(rendered, cache / digest)
                self.assertEqual(unpack.call_count, 1)
            self.assertEqual(rows, [[archive.name, digest]])
            phase = ph.Phase1.__new__(ph.Phase1)
            phase.render = ph.render_folder(rendered)
            phase.main_inv, phase.ext_inv = inventory, []
            self.assertEqual(phase.render_rows(), [row])
            np.testing.assert_array_equal(phase.pixel_stats("raw")["original.png"], np.arange(8))
            np.testing.assert_array_equal(phase.pixel_stats("E1-M")["original.png"], np.arange(8, 16))
            np.testing.assert_array_equal(phase.thumbs()["raw"]["original.png"], [.5] * 256)
            with archive.open("ab") as stream:
                stream.write(b"tampered")
            with self.assertRaisesRegex(ValueError, "differs from the SHA256"):
                ph.phase1_root(root, root / "invalid", cache)

    def test_phase1_status_applies_the_right_controls(self):
        phase = ph.Phase1.__new__(ph.Phase1)
        phase.render, phase.dry_run = Path("render"), False
        control, ref, variant = object(), object(), object()
        phase.outputs = {
            ("E1-M", ph.NOMINEE, ""): control, ("E1-M", fm.REFERENCE, ""): control,
            ("E3", ph.NOMINEE, ph.E3_REF): ref, ("E3", ph.NOMINEE, "M-res256"): variant,
        }
        checks = {control: True, ref: True}
        phase.control = lambda out: {"passed": checks[out]}
        self.assertEqual(phase.status(ph.NOMINEE, "locked"), ph.INTERPRETED)
        self.assertEqual(phase.status(ph.NOMINEE, "E1-L"), ph.UNAVAILABLE)
        self.assertEqual(phase.status(ph.NOMINEE, "E1-M"), ph.INTERPRETED)
        checks[ref] = False
        self.assertEqual(phase.status(ph.NOMINEE, "E1-M"), ph.NOT_INTERPRETED)
        self.assertEqual(phase.status(ph.NOMINEE, "E3", "M-res256"), ph.NOT_INTERPRETED)
        self.assertEqual(phase.status(fm.REFERENCE, "E1-M"), ph.INTERPRETED)
        checks[ref], checks[control] = True, False
        self.assertEqual(phase.status(ph.NOMINEE, "E3", "M-res256"), ph.INTERPRETED)
        self.assertEqual(phase.status(ph.NOMINEE, "E1-M"), ph.NOT_INTERPRETED)
        phase.render, checks[control] = None, True
        self.assertEqual(phase.status(ph.NOMINEE, "E1-M"), ph.NOT_INTERPRETED)

    def test_phase1_partial_and_worker_identity(self):
        import fm_control_extract as control

        root = Path("phase1")
        study = SimpleNamespace(lock=root, external=root)
        with patch.object(ph, "phase1_outputs", return_value={}) as outputs, \
             patch.object(ph, "render_folder", return_value=None), \
             patch.object(ph.prep, "sha256", return_value="locked"), \
             patch.object(ph.prep, "read_csv", return_value=[]):
            with self.assertRaisesRegex(ValueError, "Partial Phase-1 delivery .*--partial"):
                ph.Phase1(study, root, [])
            ph.Phase1(study, root, [], partial=True)
            with self.assertRaisesRegex(ValueError, "M-res256 archive exists"):
                ph.Phase1(study, root, [["phase1-E3-MIRAGE-M-res256.zip", "digest"]], partial=True)
            identity = dict(lock_sha256="locked", external_lock_sha256="locked", registry_sha256=control.registry_hash(),
                            script_sha256=ph.prep.text_hash(Path(control.__file__)))
            out = ph.Phase1Output(ph.NOMINEE, "E3", ph.E3_REF, root, identity)
            outputs.return_value = {("E3", ph.NOMINEE, ph.E3_REF): out}
            ph.Phase1(study, root, [], partial=True)
            for key in ("registry_sha256", "script_sha256"):
                with patch.dict(identity, {key: "wrong"}), self.assertRaisesRegex(ValueError, key):
                    ph.Phase1(study, root, [], partial=True)

    def test_e1_assembly_scores_m_res256_inside_e1_and_gates_controls(self):
        def features(arm, feature_set, variant=""):
            if feature_set == "locked":
                return (.75 if arm == ph.NOMINEE else .5, None)
            if feature_set == "E1-M" and arm in (ph.NOMINEE, fm.REFERENCE):
                return (.25 if arm == ph.NOMINEE else .5, None)
            return (.625, None) if variant == "M-res256" else None

        def scores(_study, score):
            return {(k, draw): score.value for k in ph.KS for draw in range(fm.N_DRAWS)}

        def score_features(_study, main, _ext):
            return SimpleNamespace(rep={(k, 0): np.array([True]) for k in ph.KS}, nine={}, value=main)

        phase = SimpleNamespace(features=Mock(side_effect=features), status=Mock(return_value=ph.INTERPRETED),
                                thumbs=lambda: {})
        with patch.multiple(ph, locked_geometry=Mock(return_value={}),
                            score_vectors=Mock(return_value=SimpleNamespace(nine=None, value=.25)),
                            score_feature_set=Mock(side_effect=score_features),
                            dev_macro=Mock(side_effect=scores), rep_macro=Mock(return_value={(k, 0): .5 for k in ph.KS}),
                            nine_way_rows=Mock(return_value=[]), e1_checks=Mock(return_value=[]),
                            content_sizes=Mock(return_value={}), dev_order=Mock(return_value=([], []))):
            # No E3 report or output directory exists. The value comes from archived features.
            drop = {"E1h"}
            header, rows = ph.e1(None, phase, {}, drop=drop)["E1d_success.csv"]
            records = {r["line"]: r for r in (dict(zip(header, row)) for row in rows)}
            self.assertIn("K mean by -0.125 (E3 M-res256)", records["fixed sentence"]["text"])
            phase.features.assert_any_call(ph.NOMINEE, "E3", "M-res256")
            phase.features.reset_mock()
            phase.status.return_value = ph.NOT_INTERPRETED
            _, rows = ph.e1(None, phase, {}, drop=drop)["E1d_success.csv"]
            sentence = next(r for r in rows if r[0] == "fixed sentence")
            self.assertEqual(sentence[-2], "not evaluable")
            self.assertEqual(sentence[-1], "not evaluable")
            self.assertIn("failed its control", sentence[2])
            self.assertFalse(any(len(call.args) == 3 for call in phase.features.call_args_list))

    def test_e3_assembly_counts_m_pad_and_gates_failed_reference(self):
        variant = SimpleNamespace(identity={}, folder=Path("M-crop"))
        ref = SimpleNamespace(identity={}, folder=Path("E3-ref"))
        phase = SimpleNamespace(main_inv=[], ext_inv=[], locked_all=lambda arm: None, features=lambda *args: (1, None),
                                get=lambda name, arm, v: {ph.E3_REF: ref, "M-crop": variant}.get(v)
                                if arm == ph.NOMINEE else None,
                                control=lambda out: dict(passed=False, images=0, max_difference_ratio=.1))
        study = SimpleNamespace(lock=Path("lock"), external=Path("ext"), features=lambda arm: 0,
                                external_features=lambda arm: None)

        def nine(_study, sc):
            count = int(sc == 2)
            return [[name, 10, 10 - count, 10 - count, 0, count, "", 0, 0, .05] for name in ph.NORMAL_SETS]

        dev = {(k, draw): .5 for k in ph.KS for draw in range(fm.N_DRAWS)}
        with patch.multiple(ph, locked_geometry=Mock(return_value={}), nine_way_rows=Mock(side_effect=nine),
                            score_feature_set=Mock(side_effect=lambda study, main, ext:
                                                   SimpleNamespace(nine=main if isinstance(main, int) else 2)),
                            dev_macro=Mock(return_value=dev), rep_macro=Mock(return_value=dev),
                            feature_difference=Mock(return_value=dict(images=1, max_difference_ratio=0,
                                                                     worst_image_difference_ratio=.002))), \
                patch.object(ph.fx, "merge"), patch.object(ph.prep, "verify_complete"), \
                patch.object(ph.prep, "sha256", return_value="lock"), patch.object(ph.prep, "read_csv", return_value=[]), \
                patch.object(ph.prep, "read_json", return_value=dict(arm=ph.NOMINEE, lock_sha256="lock",
                                                                    external_lock_sha256="lock")), \
                patch.object(ph.extract, "load_features"):
            rows = ph.e3(study, phase, padded=Path("padded"))["E3_criteria.csv"][1]
        self.assertTrue(all(ph.M_PAD_NOTE in r[-1] for r in rows if r[:3] == ["(3)", ph.NOMINEE, ph.M_PAD]))
        self.assertTrue(all("not counted" in r[-1] for r in rows if r[:3] == ["(3)", ph.NOMINEE, "M-crop"]))
        summary = next(r for r in rows if r[:2] == ["(3)", "all"])
        self.assertEqual(summary[4], 1)
        self.assertIn(f"{ph.M_PAD} OCTID NORMAL ({ph.M_PAD_NOTE})", summary[-1])

    def test_e2_tables_cover_destinations_transitions_and_patient_units(self):
        def cell(paths, labels, calls):
            return ph.Cell(tuple(paths), tuple(labels), tuple(calls), fm.NINE, np.zeros((len(paths), 9)))

        kermany = ["NORMAL-p1-1.jpeg", "NORMAL-p2-1.jpeg"]
        octid = ["I0", "I1", "I2", "I3"]
        octdl = [f"D{i}" for i in range(11)]
        sources = {"OCTID": (octid, ["NORMAL", "NORMAL", "CSC", "MH"]),
                   "OCTDL": (octdl, ["NORMAL"] * 10 + ["MH"])}
        queries = [{"source": s, "path": p, "label": lab}
                   for s, (paths, labels) in sources.items() for p, lab in zip(paths, labels)]
        patient = {p: str(i // 2) for i, p in enumerate(octdl)}
        study = SimpleNamespace(kermany_normals=lambda: (set(kermany), set(kermany)), queries=queries, patient=patient)
        original = {"Kermany": cell(kermany, ["NORMAL"] * 2, ["CNV", "NORMAL"]),
                    "OCTID": cell(octid, sources["OCTID"][1], ["MH", "NORMAL", "CSC", "NORMAL"]),
                    "OCTDL": cell(octdl, sources["OCTDL"][1], ["NORMAL"] * 5 + ["CNV"] * 3 + ["MH"] * 3)}
        controlled = original | {"OCTDL": cell(octdl, sources["OCTDL"][1], ["NORMAL"] * 11)}
        cells = {(a, "locked"): (ph.INTERPRETED, original) for a in ph.ARMS}
        cells[ph.NOMINEE, "E1-M"] = (ph.NOT_INTERPRETED, controlled)
        properties = {p: {prop: i for prop in ph.E2D_PROPERTIES} for i, p in enumerate(octdl)}
        tables = ph.e2_tables(study, cells, properties, {p: {"mode": "L"} for p in octdl}, {"E2d-intervals"})
        def rows(filename):
            header, records = tables[filename]
            self.assertTrue(all(len(r) == len(header) for r in records))
            return [dict(zip(header, r)) for r in records]
        for name in tables:
            rows(name)
        property_rows = rows("E2d_properties.csv")
        self.assertEqual(sum(row["main_text_table"] for row in property_rows), 28)
        for row in property_rows:
            if row["main_text_table"]:
                self.assertEqual((row["n_failed"], row["failed_common"], row["failed_rare"]), (5, 3, 2))
        patient_rows = rows("E2e_octdl_patients.csv")
        self.assertEqual((patient_rows[0]["all_failed"], patient_rows[0]["some_failed"], patient_rows[0]["none_failed"]), (2, 1, 2))
        transitions = rows("E2c_transitions.csv")
        rows_dl = [row for row in transitions if row["query_set"] == "OCTDL NORMAL"]
        self.assertEqual(sum(row["count"] for row in rows_dl), 10)
        self.assertTrue(all(row["status"] == ph.NOT_INTERPRETED for row in rows_dl))
        groups, anchors = ph.e2f_groups(study, cells)
        self.assertEqual(groups["OCTID", "consensus failure"], ["I0"])
        self.assertEqual(anchors, sorted(kermany))

        def record_seed(values, failed, units, rng):
            return [rng.integers(2**31), *[None] * (len(ph.E2D_HEADER) - 1)]

        with patch.object(ph, "e2d_row", side_effect=record_seed):
            all_rows = ph.e2_tables(study, cells, properties, {}, set())["E2d_properties.csv"][1]
            fewer = {k: v for k, v in cells.items() if k[0] != ph.ARMS[0]}
            fewer_rows = ph.e2_tables(study, fewer, properties, {}, set())["E2d_properties.csv"][1]
        self.assertEqual([r for r in all_rows if r[0] != ph.ARMS[0]], fewer_rows)

    def test_e4_tables_release_failed_control_scores_without_reading(self):
        sources = ["Kermany test", "Yoo rare", "OCTID", "OCTDL"]
        members, vectors = {}, {}
        for i, source in enumerate(sources):
            paths = [f"{source}:{j}" for j in range(12)]
            members[source] = (paths, paths)
            members[source + " NORMAL"] = (paths, paths)
            vectors.update({p: np.eye(4)[i] for p in paths})
        status = {"MIRAGE locked": ph.INTERPRETED, "MIRAGE E1-M": ph.NOT_INTERPRETED}
        tables = ph.e4_tables(members, {"G": vectors}, {"G": list("abcd")},
                             {name: vectors for name in status}, {"raw": vectors, "E1-M": vectors},
                             n_boot=4, draws=2, status=status)
        for header, rows in tables.values():
            self.assertTrue(rows)
            self.assertTrue(all(len(r) == len(header) for r in rows))
        fields, rows = tables["E4a_fewshot_source_test.csv"]
        read = [dict(zip(fields, r)) for r in rows]
        self.assertEqual(len({r["pair"] for r in read}), 3)
        self.assertTrue(all(r["mean"] == 1 for r in read if r["quantity"].startswith("two-way")))
        self.assertTrue(all(r["reading"] == "" for r in read if r["status"] == ph.NOT_INTERPRETED))
        self.assertTrue(any(row[-1] == "different sources: source and disease are mixed" for row in tables["E4c_probe_auroc.csv"][1]))


if __name__ == "__main__":
    unittest.main()
