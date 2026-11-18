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


if __name__ == "__main__":
    unittest.main()
