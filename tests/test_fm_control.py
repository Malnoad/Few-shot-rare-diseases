"""Synthetic Phase-1 render, transform and Colab builder tests."""
from __future__ import annotations

import ast
from contextlib import redirect_stdout
import datetime
from io import StringIO
import json
import pathlib
from pathlib import Path
import shlex
import shutil
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import fm_control_extract as control
import fm_control_notebook as notebook
import fm_core
import fm_extract as extract
import fm_prepare as prep


def scan(width: int, height: int) -> Image.Image:
    """Pattern with varied intensity on every row and column."""
    y, x = np.indices((height, width))
    return Image.fromarray((25 + (x * 97 + y * 43 + x * y) % 182).astype(np.uint8))


class RenderTest(unittest.TestCase):
    def test_iterative_content_box_with_three_pixel_edge_line(self):
        gray = np.zeros((96, 108), dtype=np.uint8)
        gray[:, :3] = 230
        gray[24:72, 3:] = np.asarray(scan(105, 48))
        first, box, flagged, passes = control.content_box(gray)
        self.assertEqual(first, (3, 0, 108, 96))
        self.assertEqual(control.box_share(first, 108, 96), fm_core.border_share(gray))
        self.assertEqual((box, flagged, passes), ((3, 24, 108, 72), False, 3))

    def test_content_boxes_fall_back_when_empty_or_below_32_px(self):
        gray = np.zeros((72, 80), dtype=np.uint8)
        self.assertEqual(control.content_box(gray)[:3], ((80, 72, 80, 72), (0, 0, 80, 72), True))
        gray[20:51, 21:53] = np.asarray(scan(32, 31))
        self.assertEqual(control.content_box(gray)[:3], ((21, 20, 53, 51), (0, 0, 80, 72), True))
        gray[51, 21:53] = np.asarray(scan(32, 1))[0]
        self.assertEqual(control.content_box(gray)[:3], ((21, 20, 53, 52), (21, 20, 53, 52), False))

    def test_square_offsets_floor_in_either_orientation(self):
        self.assertEqual(control.centre_square((2, 7, 52, 48)), (6, 7, 47, 48))
        self.assertEqual(control.centre_square((2, 7, 43, 57)), (2, 11, 43, 52))

    def test_resampling_pixels_and_arguments(self):
        image = scan(804, 521).convert("RGB")
        crop = image.crop((141, 0, 662, 521))
        expected_m = crop.resize((256, 256), Image.Resampling.BICUBIC, reducing_gap=None)
        expected_l = crop.resize((128, 128), Image.Resampling.BOX, reducing_gap=None)
        expected_l = expected_l.resize((256, 256), Image.Resampling.BICUBIC, reducing_gap=None)
        with patch.object(Image.Image, "resize", autospec=True, side_effect=Image.Image.resize) as resize:
            np.testing.assert_array_equal(control.render_e1m(image), expected_m)
            np.testing.assert_array_equal(control.render_e1l(image), expected_l)
        self.assertEqual([(call.args[1:], call.kwargs) for call in resize.call_args_list], [
            (((256, 256), Image.Resampling.BICUBIC), {"reducing_gap": None}),
            (((128, 128), Image.Resampling.BOX), {"reducing_gap": None}),
            (((256, 256), Image.Resampling.BICUBIC), {"reducing_gap": None}),
        ])
        np.testing.assert_array_equal(control.render_e1c(image), image)

    def test_alpha_count_and_lossless_output_on_synthetic_lock_rows(self):
        with TemporaryDirectory() as temp:
            work = Path(temp)
            root, lock, external, output = (work / name for name in ("checkout", "lock", "external", "render"))
            lock.mkdir()
            external.mkdir()
            (lock / "input_lock.json").write_text("{}", encoding="utf-8")
            (external / "external_lock.json").write_text("{}", encoding="utf-8")
            image = scan(57, 42).convert("RGB")
            rgba = image.convert("RGBA")
            rgba.putpixel((0, 0), (*image.getpixel((0, 0)), 23))
            for folder, inventory, source, original in (
                (lock, "extraction_inventory.csv", "data/raw/OCT2017/image.png", image),
                (external, "external_extraction_inventory.csv", "data/external/OCTDL/image.png", rgba),
            ):
                target = root / source
                target.parent.mkdir(parents=True)
                original.save(target)
                prep.write_csv(folder / inventory, ["path", "sha256"], [[source, prep.sha256(target)]])
            self.assertEqual(control.image_meta(target), dict(mode="RGBA", format="PNG", has_alpha=True, min_alpha=23))
            np.testing.assert_array_equal(control.decode(target), image)
            with patch.object(prep, "load_lock", return_value={"code": prep.code_identity()}), \
                    patch("fm_external.verify_external"), patch.object(control, "safe_hardware", return_value={}):
                result = control.render(root, lock, external, output)
            self.assertEqual(result["counts"]["alpha_below_255"], 1)
            self.assertEqual(result["images"], dict(main=1, external=1, decoder=2))
            prep.verify_complete(output)
            for row in prep.read_csv(output / "render.csv"):
                for name in control.E1_SETS:
                    path = control.rendered_path(output, name, row["path"])
                    with Image.open(path) as png:
                        self.assertEqual(png.mode, "RGB")
                        self.assertEqual(row[f"{name}_pixel_sha256"], control.pad.pixel_hash(png))
                    self.assertEqual(row[f"{name}_png_sha256"], prep.sha256(path))
            with Image.open(output / "external640" / (source + ".png")) as copy:
                self.assertEqual(copy.width, 640)
            save_png = control.save_png

            def corrupt_saved_png(path, image):
                digest = save_png(path, image)
                if "E1-M" in path.parts:
                    Image.new("RGB", image.size).save(path)
                return digest

            with patch.object(prep, "load_lock", return_value={"code": prep.code_identity()}), \
                    patch("fm_external.verify_external"), patch.object(control, "save_png", side_effect=corrupt_saved_png):
                with self.assertRaisesRegex(ValueError, "saved PNG .* differs from the repeated render"):
                    control.render(root, lock, external, work / "corrupt")


class VariantTest(unittest.TestCase):
    def test_mirage_overrides_bind_to_locked_hook_and_change_input(self):
        image = scan(61, 39).convert("RGBA")
        baseline = extract.mirage_input
        expected = baseline(image)
        np.testing.assert_array_equal(control.mirage_input_variant("E3-ref")(image), expected)
        for variant in control.E3_VARIANTS["MIRAGE"][1:]:
            with self.subTest(variant=variant), control.applied("MIRAGE", variant):
                self.assertIs(extract.mirage_input, control.mirage_input_variant(variant))
                self.assertFalse(np.array_equal(extract.mirage_input(image), expected))
            self.assertIs(extract.mirage_input, baseline)
        square = scan(71, 71)
        np.testing.assert_array_equal(control.mirage_input_variant("M-crop")(square), baseline(square))

    def test_constant_after_crop_bypasses_minmax_and_is_flagged(self):
        image = Image.new("L", (128, 64), 64)
        image.paste(192, (0, 0, 24, 64))
        pixels = control.mirage_input_variant("M-crop")(image)
        np.testing.assert_array_equal(pixels, np.full((512, 512), np.float32(64 / 255)))
        self.assertEqual(control.input_check("MIRAGE", "M-crop", image), (True, True))
        constant = Image.new("L", (64, 64), 64)
        self.assertEqual(control.input_check("MIRAGE", "E3-ref", constant), (False, True))
        self.assertEqual(control.input_check("RETFound", "R-stretch", constant), (False, True))

    def test_retfound_presteps_and_norm_override(self):
        image = scan(77, 49).convert("RGB")
        self.assertIs(control.retfound_prestep("E3-ref")(image), image)
        self.assertIn("prestep", control.overrides("RETFound", "E3-ref"))
        for variant in ("R-squash", "R-pad", "R-bilinear", "R-stretch", "R-res128"):
            with self.subTest(variant=variant):
                transformed = control.retfound_prestep(variant)(image)
                self.assertFalse(np.array_equal(transformed, image))
        square = scan(61, 61).convert("RGB")
        method = Image.Resampling.BICUBIC
        expected = control.short_edge(square, 256, method)
        for variant in ("R-squash", "R-pad"):
            changed = control.retfound_prestep(variant)(square)
            np.testing.assert_array_equal(control.short_edge(changed, 256, method), expected)
        mean, sd = extract.IMAGENET_MEAN, extract.IMAGENET_STD
        with control.applied("RETFound", "E3-ref"):
            self.assertEqual((extract.IMAGENET_MEAN, extract.IMAGENET_STD), (mean, sd))
        with control.applied("RETFound", "R-norm05"):
            self.assertEqual((extract.IMAGENET_MEAN, extract.IMAGENET_STD), ([.5] * 3, [.5] * 3))
        self.assertIs(extract.IMAGENET_MEAN, mean)
        self.assertIs(extract.IMAGENET_STD, sd)

    def test_locked_loader_still_uses_the_override_hooks(self):
        source = ROOT / "fm_extract.py"
        self.assertEqual(prep.text_hash(source), "34d899be7e14d2820cadb873393b628397aa7573f952fdbb80b92915e3576edc")
        tree = ast.parse(source.read_text(encoding="utf-8"))
        loader = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "load_encoder")
        calls = [node for node in ast.walk(loader) if isinstance(node, ast.Call)]
        self.assertTrue(any(isinstance(call.func, ast.Name) and call.func.id == "mirage_input" for call in calls))
        normalize = next(call for call in calls if isinstance(call.func, ast.Attribute) and call.func.attr == "Normalize")
        self.assertEqual([ast.unparse(arg) for arg in normalize.args], ["IMAGENET_MEAN", "IMAGENET_STD"])

    def test_registry_hash(self):
        self.assertEqual(control.registry_hash(), "f47b84838fb3661932774e74eb366f43514782db1cf0b7346626bc9ee21f96f1")

    def test_resize_identity_reports_mismatch(self):
        tf = SimpleNamespace(
            io=SimpleNamespace(read_file=lambda path: path, decode_image=lambda *a, **kw:
                               SimpleNamespace(numpy=lambda: np.zeros((5, 5, 3)))),
            image=SimpleNamespace(resize=lambda a, size: SimpleNamespace(numpy=lambda: a.astype(np.float32))),
        )
        with patch.dict(sys.modules, tensorflow=tf):
            self.assertEqual(control.tf_resize_identity([Path("bad-size.png")]),
                             {"arrays": 2, "unchanged": False, "failures": 1})


class NotebookTest(unittest.TestCase):
    def test_colab_secret_timeout_is_reported(self):
        class TimeoutException(Exception):
            pass

        def get(_):
            raise TimeoutException("secret service timed out")

        user_data = SimpleNamespace(get=get)
        colab = SimpleNamespace(drive=None, userdata=user_data)
        with patch.dict(sys.modules, {"google.colab": colab}), redirect_stdout(StringIO()) as capture:
            scope = {}
            exec(notebook.PREFLIGHT.split("def access_check(", 1)[0], scope)
            self.assertIsNone(scope["colab_token"]())
        self.assertIn("HF_TOKEN secret could not be read (TimeoutException)", capture.getvalue())

    def test_pin_requires_exact_blobs(self):
        def show(command, **kwargs):
            target = command[-1].split(":")[1]
            return SimpleNamespace(returncode=0, stdout=(ROOT / target).read_bytes())

        with patch.object(notebook.subprocess, "run", side_effect=show) as git, redirect_stdout(StringIO()):
            notebook.check_pin("a" * 40)
        self.assertEqual([c.args[0][-1].split(":")[1] for c in git.call_args_list],
                         ["fm_control_extract.py", "fm_pad_extract.py", "fm_notebook.py"])
        with patch.object(notebook.subprocess, "run", return_value=SimpleNamespace(returncode=128, stdout=b"")):
            with self.assertRaisesRegex(SystemExit, "fm_control_extract.py, fm_pad_extract.py, fm_notebook.py"):
                notebook.check_pin("a" * 40)

    def test_drop_level_filters_actual_restore_selection(self):
        with TemporaryDirectory() as temp, redirect_stdout(StringIO()):
            for session, counts in [(1, [18, 18, 18, 16, 14]), (2, [6, 2, 0, 0, 0])]:
                for level, expected in enumerate(counts):
                    state = dict(SESSION_JOBS={1: notebook.REQUIRED, 2: notebook.OPTIONAL}, SESSION=session,
                                 DROP_LEVEL=level, EXPORT=Path(temp) / "exports", pathlib=pathlib,
                                 OUT=temp, DRIVE_FOLDER="phase1", job_id=lambda *parts: "-".join(filter(None, parts)),
                                 drive=SimpleNamespace(mount=lambda *a, **k: None, flush_and_unmount=lambda: None))
                    exec(notebook.ARCHIVES, state)
                    self.assertEqual(len(state["selected"]), expected, (session, level))

    def test_notebook_matches_builder_and_code_parses(self):
        path = ROOT / notebook.NAME
        text = path.read_text(encoding="utf-8")
        self.assertEqual(text, notebook.build(notebook.pinned_commit(path)))
        cells = [cell for cell in json.loads(text)["cells"] if cell["cell_type"] == "code"]
        for i, cell in enumerate(cells):
            compile("".join(cell["source"]), f"{notebook.NAME} cell {i + 1}", "exec")
        self.assertIn("# Step 0.", "".join(cells[2]["source"]))

    def test_sessions_cover_all_workers_with_protocol_drop_tiers(self):
        self.assertEqual(set(notebook.REQUIRED + notebook.OPTIONAL), {
            ("E1-M", "MIRAGE", "", 0), ("E1-M", "Inception", "", 0),
            ("E1-M", "RETFound", "", 0), ("E1-M", "DINOv2", "", 0),
            ("E1-L", "MIRAGE", "", 0), ("E1-L", "Inception", "", 0),
            ("E1-C", "MIRAGE", "", 2), ("E1-C", "Inception", "", 2),
            ("E1-C", "RETFound", "", 1), ("E1-C", "DINOv2", "", 1),
            ("E1-L", "RETFound", "", 1), ("E1-L", "DINOv2", "", 1),
            ("E3", "MIRAGE", "E3-ref", 0), ("E3", "MIRAGE", "M-crop", 0),
            ("E3", "MIRAGE", "M-bicubic", 0), ("E3", "MIRAGE", "M-nostretch", 0),
            ("E3", "MIRAGE", "M-res256", 0), ("E3", "RETFound", "E3-ref", 0),
            ("E3", "RETFound", "R-squash", 0), ("E3", "RETFound", "R-pad", 0),
            ("E3", "RETFound", "R-bilinear", 3), ("E3", "RETFound", "R-res128", 3),
            ("E3", "RETFound", "R-stretch", 4), ("E3", "RETFound", "R-norm05", 4),
        })
        self.assertTrue(all(name == "E1-C" or arm in ("RETFound", "DINOv2")
                            for name, arm, _, _ in notebook.OPTIONAL))
        self.assertEqual(len(notebook.OPTIONAL), 6)

    def test_workers_save_before_next_run_and_session_two_restores_render(self):
        with TemporaryDirectory() as temp, redirect_stdout(StringIO()):
            work = Path(temp)
            folder = work / "drive" / "phase1"
            folder.mkdir(parents=True)
            state = {}
            seen = []
            saves = []
            (folder / "phase1-E1-M-MIRAGE.zip.json").write_text("orphan", encoding="utf-8")
            (folder / "phase1-E1-M-Inception.zip").write_text("orphan", encoding="utf-8")

            def save_to_drive(_folder, files):
                for path in files:
                    shutil.copyfile(path, folder / path.name)
                    if path.name == "phase1-render.zip.json" and not saves:
                        saves.append(path.name)
                        raise OSError("test Drive disconnect")
                saves.append(files[-1].name)

            def export(command):
                args = shlex.split(command)
                self.assertEqual(args[1:5], ["-E", "-s", "checkout/fm_prepare.py", "export"])
                prep.export(Path(args[args.index("--out") + 1]), Path(args[args.index("--archive") + 1]), work)

            def worker(command, log):
                log = Path(log)
                log.parent.mkdir(parents=True, exist_ok=True)
                log.touch()
                seen.append(log.name)
                if log.name == "E1-M-MIRAGE-try1.log":
                    return 123, 1
                if log.name == "E1-M-Inception-try1.log":
                    self.assertFalse((folder / "phase1-render.zip").exists())
                    self.assertFalse((folder / "phase1-E1-M-MIRAGE.zip.json").exists())
                if log.name == "E1-M-MIRAGE-try2.log":
                    self.assertTrue((folder / "phase1-E1-M-Inception.zip.json").is_file())
                output = Path(command[command.index("--out") + 1])
                output.mkdir(parents=True, exist_ok=True)
                (output / "synthetic.txt").write_text("worker output", encoding="utf-8")
                prep.complete(output, ["synthetic.txt"])
                return 123, 0

            def start(session):
                phase = work / f"runtime{session}"
                state.update(PHASE=phase, EXPORT=phase / "exports", RENDER=phase / "render", LOCK=phase / "lock",
                             EXTERNAL_LOCK=phase / "external", SESSION=session)
                exec(notebook.ARCHIVES, state)
                exec(notebook.RENDER_CELL, state)

            jobs = {1: [("E1-M", "MIRAGE", "", 0), ("E1-M", "Inception", "", 0)],
                    2: [("E1-C", "MIRAGE", "", 2)]}
            exec(notebook.FIRST, state)
            state.update(pathlib=pathlib, json=json, zipfile=zipfile, shutil=shutil, datetime=datetime, shlex=shlex,
                         drive=SimpleNamespace(mount=lambda *a, **kw: None, flush_and_unmount=lambda: None),
                         os=SimpleNamespace(path=SimpleNamespace(ismount=lambda _: False)),
                         SESSION_JOBS=jobs, DROP_LEVEL=0, OUT=work / "drive", DRIVE_FOLDER="phase1",
                         PY="python", ROOT="checkout", ASSETS="assets", SCRIPT="fm_control_extract.py",
                         sha256=prep.sha256, worker=worker, sh=export, save_to_drive=save_to_drive,
                         INTERPRETER={"MIRAGE": "python", "Inception": "python3"}, asset_issues={},
                         RUN_BY="Test", COMMIT="0" * 40, RUNTIME="L4", LEDGER=work / "unused.csv")
            start(1)
            exec(notebook.EXTRACT_CELL, state)
            exec(notebook.EXTRACT_CELL, state)
            self.assertEqual(seen, ["render-try1.log", "E1-M-MIRAGE-try1.log", "E1-M-Inception-try1.log",
                                    "E1-M-MIRAGE-try2.log"])
            self.assertIn("render", state["pending_saves"])
            exec(notebook.STAMP, state)
            self.assertFalse(state["pending_saves"])
            start(2)
            self.assertEqual(prep.verify_complete(state["RENDER"])["files"].keys(), {"synthetic.txt"})
            exec(notebook.EXTRACT_CELL, state)
            self.assertEqual(seen[-1], "E1-C-MIRAGE-try1.log")
            exec(notebook.STAMP, state)
            (log_archive,) = folder.glob("phase1-logs-s2-*.zip")
            with zipfile.ZipFile(log_archive) as archive:
                stamp = archive.read("stamp.txt").decode("utf-8")
            self.assertIn("completed=render, E1-C-MIRAGE | missing=none", stamp)
            self.assertEqual(prep.read_json(log_archive.with_suffix(".zip.json"))["sha256"], prep.sha256(log_archive))


if __name__ == "__main__":
    unittest.main()
