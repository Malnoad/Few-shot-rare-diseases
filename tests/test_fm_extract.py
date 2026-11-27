"""CPU-only contracts and mocked adapters; no framework, checkpoint or real data runs."""
from contextlib import ExitStack, nullcontext
import argparse
import json
from itertools import count
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fm_extract as extract
import fm_prepare as prep


def model_spec(arm):
    return dict(arm=arm, repository=extract.REPOSITORIES[arm], revision=extract.REVISIONS.get(arm, "keras-imagenet"),
                source_revision=extract.SOURCE_REVISIONS.get(arm, ""), terms_evidence="synthetic fixture only",
                files=dict(extract.EXPECTED_ASSET_SHA256[arm]), recipe=extract.recipe(arm))


class CacheAndGeometry(unittest.TestCase):
    def test_same_architecture_wrong_checkpoint_or_source_hash_is_rejected(self):
        for arm in extract.REPOSITORIES:
            original=model_spec(arm)
            for name in original["files"]:
                with self.subTest(arm=arm,asset=name):
                    changed={**original,"files":{**original["files"],name:"0"*64}}
                    with self.assertRaisesRegex(ValueError,"public source pins"):
                        extract.check_spec(changed)

    def test_crop_rounding_matches_each_native_library(self):
        # 256 * 259 / 256 -> 259: (259 - 224) / 2 = 17.5.
        image = Image.new("RGB", (259, 256))
        self.assertEqual(extract.geometry("RETFound", "x", image)[3:9], [259, 256, 18, 16, 224, 224])
        self.assertEqual(extract.geometry("DINOv2", "x", image)[3:9], [259, 256, 17, 16, 224, 224])
        biomed = extract.geometry("BioMedCLIP", "x", Image.new("RGB", (227, 224)))
        self.assertEqual(biomed[3:9], [227, 224, 2, 0, 224, 224])
        self.assertAlmostEqual(biomed[9], 224 / 227)

    def test_resize_truncation_and_full_field_fractions(self):
        image = Image.new("RGBA", (400, 301))
        self.assertEqual(extract.geometry("DINOv2", "x", image)[3:5], [340, 256])
        for arm, size in (("MIRAGE", 512), ("Inception", 299)):
            row = extract.geometry(arm, "x", image)
            self.assertEqual(row[1:9], [400, 301, size, size, 0, 0, size, size])
            self.assertEqual(row[9], 1)
        self.assertEqual(len(extract.GEOMETRY_HEADER), len(row))

    def test_padding_shares_peel_flat_edges_and_count_near_white_pixels(self):
        # A 10x20 scan: 2 black rows on top and 3 near-black columns on the right (range 7 is
        # still flat); content varies along both axes. Lines are full length (the review's rule).
        pixels = (np.arange(20)[None, :] * 5 + np.arange(10)[:, None] * 9).astype(np.uint8)
        pixels[:2] = 0
        pixels[:, -3:] = 0
        pixels[5, -1] = 7
        pixels[5, 5], pixels[6, 6] = 255, 251
        row = extract.geometry("MIRAGE", "x", Image.fromarray(pixels, "L"))
        self.assertAlmostEqual(row[10], 1 - (8 * 17) / 200)
        self.assertAlmostEqual(row[11], 2 / 200)
        # A full-height white band on the right makes the black top rows non-flat: only the band peels.
        pixels[:, -3:] = 252
        self.assertAlmostEqual(extract.geometry("MIRAGE", "x", Image.fromarray(pixels, "L"))[10], 30 / 200)
        self.assertEqual(extract.geometry("MIRAGE", "x", Image.new("L", (4, 4), 9))[10], 1)

    def test_raw_float32_roundtrip_preserves_magnitude_and_order(self):
        values = np.array([[3, 4], [10, -7]], dtype=np.float32)
        rows = [dict(path="a.png", sha256="a" * 64), dict(path="b.png", sha256="b" * 64)]
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "features.npz"
            extract.save_features(target, [r["path"] for r in rows], [r["sha256"] for r in rows], values)
            np.testing.assert_array_equal(extract.load_features(target, rows, 2), values)
            np.testing.assert_array_equal(values, [[3, 4], [10, -7]])
            with self.assertRaisesRegex(ValueError, "ordered inventory"):
                extract.load_features(target, rows[::-1], 2)
            changed = [{**rows[0], "sha256": "c" * 64}, rows[1]]
            with self.assertRaisesRegex(ValueError, "ordered inventory"):
                extract.load_features(target, changed, 2)

    def test_cache_rejects_pickle_extra_keys_wrong_shapes_and_bad_numerics(self):
        row = dict(path="a.png", sha256="a" * 64)
        good = dict(paths=np.array([row["path"]]), sha256=np.array([row["sha256"]]),
                    feats=np.array([[3, 4]], dtype=np.float32))
        cases = [dict(feats=np.ones((1, 2), dtype=np.float16)), dict(feats=np.ones((2, 1), dtype=np.float32)),
                 dict(feats=np.zeros((1, 2), dtype=np.float32)), dict(feats=np.array([[np.nan, 1]], dtype=np.float32)),
                 dict(feats=np.array([[np.inf, 1]], dtype=np.float32)), dict(paths=np.array(["a.png"], dtype=object)),
                 dict(paths=np.array([["a.png"]])), dict(extra=np.array([1]))]
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "features.npz"
            for changes in cases:
                with self.subTest(changes=list(changes)):
                    np.savez(target, **{**good, **changes})
                    with self.assertRaises(ValueError):
                        extract.load_features(target, [row], 2)

    def test_duplicate_or_unsafe_ids_and_bad_hashes_fail_before_saving(self):
        cases = [(["a", "a"], ["a" * 64] * 2), (["../a"], ["a" * 64]),
                 (["a\\b"], ["a" * 64]), (["a"], ["not-a-sha256"]), ([], [])]
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "features.npz"
            for paths, hashes in cases:
                with self.subTest(paths=paths):
                    with self.assertRaises(ValueError):
                        extract.save_features(target, paths, hashes, np.ones((len(paths), 2), dtype=np.float32))
            self.assertFalse(target.exists())

    def test_source_pins_recipe_and_hashes_are_not_interchangeable(self):
        for arm in extract.REPOSITORIES:
            spec = model_spec(arm)
            extract.check_spec(spec)
            for key, value in (("recipe", {**spec["recipe"], "dtype": "float16"}),
                               ("files", {name: "bad" for name in spec["files"]})):
                with self.subTest(arm=arm, key=key), self.assertRaises(ValueError):
                    extract.check_spec({**spec, key: value})
            if arm != "Inception":
                with self.assertRaisesRegex(ValueError, "source pin"):
                    extract.check_spec({**spec, "revision": "f" * 40})

    def test_description_and_cpu_import_do_not_load_frameworks(self):
        code = "import sys, fm_extract; fm_extract.recipe('DINOv2'); assert 'torch' not in sys.modules; assert 'tensorflow' not in sys.modules"
        result = subprocess.run([sys.executable, "-c", code], cwd=Path(extract.__file__).parent,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_direct_loader_refuses_laptop_before_framework_import(self):
        with patch.object(extract.platform, "system", return_value="Windows"), \
                self.assertRaisesRegex(ValueError, "not on the laptop"):
            extract.load_encoder(model_spec("DINOv2"), Path("nonexistent"))


class Timing(unittest.TestCase):
    def test_fixed_order_batch_one_validation_outside_synchronized_intervals(self):
        clock, seen, synchronizations = [0.], [], []
        images = [Image.new("L", (1, 1), i) for i in range(100)]
        pass_seconds = [9, 1, 2, 5, 3, 4, 50, 50]  # the last two: untimed R3 synthetic repeats

        def encode(batch):
            self.assertEqual(len(batch), 1)
            clock[0] += pass_seconds[min(len(seen) // 100, 6 + (len(seen) - 600) // 5 if len(seen) >= 600 else 5)]
            seen.append(batch[0].getpixel((0, 0)))
            return np.ones((1, 768), dtype=np.float32)

        def synchronize():
            synchronizations.append(len(seen))

        def validate(*args):
            # Very expensive CPU validation must not contaminate the five means.
            clock[0] += 1000

        encoder = extract.Encoder("DINOv2", encode, synchronize, {}, lambda: {})
        with patch.object(extract.time, "perf_counter", side_effect=lambda: clock[0]), \
                patch.object(extract.time, "monotonic_ns", side_effect=lambda: int(clock[0] * 1e9) + 1), \
                patch.object(extract, "validate_features", side_effect=validate):
            timing = extract.warm_timing(encoder, images)
        self.assertEqual(seen[:600], list(range(100)) * 6)
        self.assertEqual(len(seen), 610)  # R3: five synthetic images, twice, after the timed passes
        self.assertEqual(synchronizations, [0, 100, 100, 200, 200, 300, 300, 400, 400, 500, 500, 600])
        self.assertEqual(timing["seconds_per_image"], [1, 2, 5, 3, 4])
        self.assertEqual(timing["median_seconds_per_image"], 3)
        # 600 timed-pass validations, 10 synthetic validations, and the untimed synthetic encodes.
        self.assertEqual(timing["validation_seconds"], 610000 + 10 * 50)
        self.assertAlmostEqual(timing["numerical"]["real"]["min_cosine"], 1)
        extract.validate_timing(timing)

    def test_repeat_checks_stop_a_nondeterministic_encoder_with_its_values(self):
        calls = count()

        def drifting(batch):  # stable warm/timed passes 0-4, a different last pass
            n = next(calls)
            values = np.ones((1, 768), dtype=np.float32)
            if 500 <= n < 600:
                values[0, :384] = -1
            return values

        encoder = extract.Encoder("DINOv2", drifting, lambda: None, {}, lambda: {})
        with self.assertRaisesRegex(ValueError, "numerical repeat check failed.*min_cosine"):
            extract.warm_timing(encoder, [Image.new("L", (1, 1))] * 100)
        stable = extract.Encoder("DINOv2", MagicMock(return_value=np.ones((1, 768), dtype=np.float32)),
                                 lambda: None, {}, lambda: {})
        timing = extract.warm_timing(stable, [Image.new("L", (1, 1))] * 100)
        for change in ({"tolerance": 1e-3}, {"real": {**timing["numerical"]["real"], "images": 5}}):
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "fixed R3/A-6 rule"):
                extract.validate_timing({**timing, "numerical": {**timing["numerical"], **change}})
        with self.assertRaisesRegex(ValueError, "lacks the numerical"):
            extract.validate_timing({k: v for k, v in timing.items() if k != "numerical"})

    def test_synthetic_repeat_images_are_fixed(self):
        first, second = extract.synthetic_images(), extract.synthetic_images()
        self.assertEqual([im.size for im in first], [(512, 496), (768, 496), (1536, 496), (414, 413), (327, 326)])
        self.assertTrue(all(a.tobytes() == b.tobytes() for a, b in zip(first, second)))

    def test_bad_warm_output_stops_before_timed_passes(self):
        encode = MagicMock(return_value=np.zeros((1, 768), dtype=np.float32))
        encoder = extract.Encoder("DINOv2", encode, lambda: None, {}, lambda: {})
        with self.assertRaises(ValueError):
            extract.warm_timing(encoder, [Image.new("L", (1, 1))] * 100)
        self.assertEqual(encode.call_count, 100)
        with self.assertRaisesRegex(ValueError, "exactly 100"):
            extract.warm_timing(encoder, [])


class Operations(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.root, self.folder, self.out = self.base / "root", self.base / "prepared", self.base / "out"
        self.root.mkdir()
        self.folder.mkdir()
        self.rows = []
        for i in range(2):
            path = self.root / f"image{i}.png"
            Image.new("RGBA", (259 + i, 256), (i + 1, 2, 3, 7)).save(path)
            self.rows.append(dict(path=path.name, sha256=prep.sha256(path)))
        prep.write_csv(self.folder / "extraction_inventory.csv", ["path", "sha256"],
                       [[r["path"], r["sha256"]] for r in self.rows])
        prep.write_csv(self.folder / "timing_inventory.csv", ["path"], [[r["path"]] for r in self.rows] * 50)
        prep.write_json(self.folder / "freeze_summary.json", dict(files={
            name: prep.sha256(self.folder / name) for name in ("extraction_inventory.csv", "timing_inventory.csv")}))
        # A mock lock for synthetic inputs only; no project input lock is created.
        (self.folder / "input_lock.json").write_text("synthetic only", encoding="utf-8")
        self.spec = model_spec("DINOv2")
        self.current = dict(spec=self.spec, code={"mock": "code"}, environment={"mock": "environment"})
        self.lock = dict(arms={"DINOv2": dict(status="available", identity=self.current)})
        self.encode = MagicMock(side_effect=lambda images: np.full((1, 768), 3 + images[0].getpixel((0, 0))[0], dtype=np.float32))
        self.encoder = extract.Encoder("DINOv2", self.encode, lambda: None, {"mock": True},
                                       lambda: dict(peak_vram_bytes=123))
        self.stack = ExitStack()
        self.stack.enter_context(patch.object(extract, "require_colab"))
        self.stack.enter_context(patch.object(extract, "runtime_out", side_effect=lambda p: p))
        # Full preparation semantics are exercised in test_fm_prepare; this fixture
        # isolates extraction receipts, cache integrity and interruption behavior.
        self.stack.enter_context(patch.object(extract,"prepared",return_value=prep.read_json(self.folder / "freeze_summary.json")))
        self.stack.enter_context(patch.object(extract, "environment", return_value=self.current["environment"]))
        self.stack.enter_context(patch.object(extract, "hardware", return_value={"host": "mock", "boot_id": "1", "gpu": "mock"}))
        self.stack.enter_context(patch.object(prep, "code_identity", return_value=self.current["code"]))
        self.stack.enter_context(patch.object(prep, "load_lock", return_value=self.lock))
        self.stack.enter_context(patch.object(extract, "check_spec"))
        self.loader = self.stack.enter_context(patch.object(extract, "load_encoder", return_value=self.encoder))

    def tearDown(self):
        self.stack.close()
        self.tmp.cleanup()

    def run_operation(self, step="extract", session=""):
        extract.run(step, self.root, self.folder, self.spec, self.base / "no-weights", self.out, session)

    def test_extract_decodes_once_keeps_native_mode_and_separates_costs(self):
        with patch.object(Image, "open", wraps=Image.open) as opened:
            self.run_operation()
        self.assertEqual(opened.call_count, 2)
        self.assertEqual([call.args[0][0].mode for call in self.encode.call_args_list], ["RGBA", "RGBA"])
        values = extract.load_features(self.out / "features.npz", self.rows, 768)
        np.testing.assert_array_equal(values[:, 0], [4, 5])
        costs = prep.read_json(self.out / "costs.json")
        for key in ("extraction_seconds", "staging_seconds", "serialization_seconds", "validation_seconds"):
            self.assertGreaterEqual(costs[key], 0)
        self.assertNotIn("extraction_and_serialization_seconds", costs)
        extract.verify_operation(self.out, "extract", self.rows, "DINOv2")
        prep.verify_complete(self.out)

    def test_expensive_serialization_and_validation_do_not_become_extraction_costs(self):
        clock = [0.]
        original_open, original_encode = Image.open, self.encoder.encode
        original_save, original_validate = np.savez, extract.validate_features

        def delay(function, seconds):
            def delayed(*args, **kwargs):
                clock[0] += seconds
                return function(*args, **kwargs)
            return delayed

        self.encoder.encode = delay(original_encode, 3)
        with patch.object(extract.time, "perf_counter", side_effect=lambda: clock[0]), \
                patch.object(Image, "open", side_effect=delay(original_open, 2)), \
                patch.object(np, "savez", side_effect=delay(original_save, 5)), \
                patch.object(extract, "validate_features", side_effect=delay(original_validate, 7)):
            self.run_operation()
        costs = prep.read_json(self.out / "costs.json")
        self.assertEqual(costs["extraction_seconds"], 6)
        self.assertEqual(costs["staging_seconds"], 4)
        self.assertEqual(costs["serialization_seconds"], 5)
        self.assertEqual(costs["validation_seconds"], 28)

    def test_resume_after_receipt_preserves_costs_without_loading_model(self):
        with patch.object(prep, "complete", side_effect=KeyboardInterrupt), self.assertRaises(KeyboardInterrupt):
            self.run_operation()
        before = (self.out / "costs.json").read_bytes()
        self.loader.reset_mock()
        self.run_operation()
        self.loader.assert_not_called()
        self.assertEqual((self.out / "costs.json").read_bytes(), before)
        self.run_operation()
        self.loader.assert_not_called()

    def test_unreceipted_cache_is_recomputed_and_compared(self):
        original = prep.write_csv

        def interrupt(path, *args):
            if path.name == "geometry.csv":
                raise KeyboardInterrupt
            return original(path, *args)

        with patch.object(prep, "write_csv", side_effect=interrupt), self.assertRaises(KeyboardInterrupt):
            self.run_operation()
        self.assertTrue((self.out / "features.npz").exists())
        self.assertFalse((self.out / "costs.json").exists())
        self.run_operation()
        self.assertEqual(self.encode.call_count, 4)
        prep.verify_complete(self.out)

    def test_completed_artifact_and_input_tampering_both_stop(self):
        self.run_operation()
        with (self.out / "features.npz").open("ab") as stream:
            stream.write(b"corruption")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            self.run_operation()
        (self.root / self.rows[0]["path"]).write_bytes(b"changed input")
        with self.assertRaisesRegex(ValueError, "input image content differs"):
            self.run_operation()

    def test_lock_environment_mismatch_and_unavailable_arm_stop_before_loading(self):
        with patch.object(extract, "environment", return_value={"changed": True}), \
                self.assertRaisesRegex(ValueError, "differs from the input lock"):
            self.run_operation()
        self.lock["arms"]["DINOv2"]["status"] = "unavailable"
        with self.assertRaisesRegex(ValueError, "unavailable at input lock"):
            self.run_operation()
        self.loader.assert_not_called()
        self.assertFalse(self.out.exists())

    def test_pilot_realized_ids_and_operation_windows_are_archived(self):
        with patch.object(extract.time,"monotonic_ns",side_effect=count(1000,100)):
            self.run_operation("pilot", "synthetic-session")
        timing = prep.read_json(self.out / "timing.json")
        self.assertEqual(timing["image_ids"], [r["path"] for r in self.rows] * 50)
        self.assertEqual(self.encode.call_count, 610)  # 600 timing calls + 10 R3 synthetic repeats
        extract.verify_operation(self.out, "pilot", prep.read_csv(self.folder / "timing_inventory.csv"), "DINOv2")
        self.assertEqual(timing["costs"]["image_count"], 100)

    def test_missing_pilot_session_is_rejected_before_loading_or_writing(self):
        with self.assertRaisesRegex(ValueError, "timing-session ID"):
            self.run_operation("pilot")
        self.loader.assert_not_called()
        self.assertFalse(self.out.exists())


class Tensor:
    """Small NumPy-backed test double; never a real torch tensor or encoder."""
    def __init__(self, values):
        self.values = np.asarray(values)
        self.shape, self.dtype = self.values.shape, self.values.dtype

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        return Tensor(self.values[index])

    def mean(self, dim):
        return Tensor(self.values.mean(axis=dim))

    def to(self, **kwargs):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.values


class MockedLoaders(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.assets = Path(self.tmp.name)
        self.stack = ExitStack()
        self.stack.enter_context(patch.object(extract, "require_colab"))
        self.stack.enter_context(patch.object(extract, "check_spec"))
        self.stack.enter_context(patch.dict(extract.os.environ))
        self.torch = MagicMock()
        self.torch.cuda.is_available.return_value = True
        self.torch.float32 = np.dtype("float32")
        self.torch.stack.side_effect = lambda values: Tensor(np.stack([v.values for v in values]))
        self.torch.from_numpy.side_effect = Tensor
        self.torch.inference_mode.side_effect = nullcontext
        self.torch.autocast.side_effect = lambda **kwargs: nullcontext()
        self.transform = MagicMock(return_value=Tensor(np.ones((3, 224, 224), dtype=np.float32)))
        self.transforms = MagicMock()
        self.transforms.Compose.return_value = self.transform
        self.stack.enter_context(patch.dict(sys.modules, {"torch": self.torch,
            "torchvision": SimpleNamespace(transforms=self.transforms)}))
        self.model = MagicMock()
        self.model.float.return_value = self.model
        self.model.cuda.return_value = self.model
        self.model.eval.return_value = self.model
        self.model.requires_grad_.return_value = self.model

    def tearDown(self):
        self.stack.close()
        self.tmp.cleanup()

    def write_config(self, name, value):
        (self.assets / name).write_text(json.dumps(value), encoding="utf-8")

    def load(self, arm):
        return extract.load_encoder(model_spec(arm), self.assets)

    def test_retfound_checks_complete_norm_pair_and_all_encoder_keys(self):
        source = SimpleNamespace(RETFound_mae=MagicMock(return_value=self.model))
        self.stack.enter_context(patch.object(extract, "import_source", return_value=source))
        self.torch.load.return_value = {"model": {"patch_embed.weight": "weights", "head.weight": "unused", "head.bias": "unused"}}
        head = ["head.weight", "head.bias"]
        for missing, extra in ((head + ["fc_norm.weight"], []),
                               (head + ["blocks.0.attn.qkv.weight"], []),
                               (head, ["unknown.weight"]), (head, ["norm.weight"])):
            self.model.load_state_dict.return_value = SimpleNamespace(missing_keys=missing, unexpected_keys=extra)
            with self.subTest(missing=missing, extra=extra), self.assertRaises(ValueError):
                self.load("RETFound")
        self.model.load_state_dict.return_value = SimpleNamespace(missing_keys=head + ["fc_norm.weight", "fc_norm.bias"], unexpected_keys=[])
        self.model.forward_features.return_value = Tensor(np.full((1, 1, 1024), 7, dtype=np.float32))
        encoder = self.load("RETFound")
        source.RETFound_mae.assert_called_with(global_pool=True)
        self.assertEqual(encoder.encode([Image.new("RGBA", (8, 8))]).shape, (1, 1024))
        self.assertEqual(self.transform.call_args.args[0].mode, "RGB")
        np.testing.assert_array_equal(encoder.encode([Image.new("L", (8, 8))]), np.full((1, 1024), 7))
        self.model.forward_features.return_value = Tensor(np.ones((1, 1024), dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "forward_features"):
            encoder.encode([Image.new("RGB", (8, 8))])
        self.torch.load.assert_called_with(self.assets / "RETFound_mae_natureOCT.pth", map_location="cpu", weights_only=True)

    def test_retfound_metadata_permission_is_scoped_and_preserves_existing_allowlist(self):
        source=SimpleNamespace(RETFound_mae=MagicMock(return_value=self.model))
        self.stack.enter_context(patch.object(extract,"import_source",return_value=source))
        self.torch.load.return_value={"model":{}}
        self.model.load_state_dict.return_value=SimpleNamespace(
            missing_keys=["head.weight","head.bias"],unexpected_keys=[])
        for existing,expected in (([],[argparse.Namespace]),([argparse.Namespace],[])):
            with self.subTest(existing=existing):
                self.torch.serialization.get_safe_globals.return_value=existing
                scope=self.torch.serialization.safe_globals.return_value
                scope.reset_mock()
                encoder=self.load("RETFound")
                self.torch.serialization.safe_globals.assert_called_with(expected)
                scope.__enter__.assert_called_once()
                scope.__exit__.assert_called_once_with(None,None,None)
                self.assertEqual(encoder.load_report["checkpoint_metadata_allowlist"],["argparse.Namespace"])
        self.torch.load.side_effect=RuntimeError("synthetic deserialization failure")
        scope=self.torch.serialization.safe_globals.return_value
        scope.reset_mock()
        with self.assertRaisesRegex(RuntimeError,"synthetic deserialization"):
            self.load("RETFound")
        scope.__exit__.assert_called_once()
        self.assertIs(scope.__exit__.call_args.args[0],RuntimeError)
        self.torch.load.assert_called_with(self.assets / "RETFound_mae_natureOCT.pth",map_location="cpu",weights_only=True)

    def test_mirage_normalizes_each_patch_before_averaging_and_drops_global(self):
        self.write_config("config.json", dict(input_size=512, modalities="bscan-slo", patch_size=32, size="base"))
        source = SimpleNamespace(MIRAGEWrapper=MagicMock(return_value=self.model))
        self.stack.enter_context(patch.object(extract, "import_source", return_value=source))
        self.model.state_dict.return_value = {"model.weight": None}
        self.model.model.state_dict.return_value = {"weight": None}
        self.model.args.num_global_tokens = 1
        load_file = MagicMock(return_value={"weight": "pinned"})
        self.stack.enter_context(patch.dict(sys.modules, {"safetensors.torch": SimpleNamespace(load_file=load_file)}))
        grid = np.linspace(-2, 2, 768, dtype=np.float32)
        tokens = np.empty((1, 257, 768), dtype=np.float32)
        tokens[:, :128] = grid
        tokens[:, 128:256] = grid ** 2 * 7 + 3
        tokens[:, 256] = 1e8  # must never enter pooling
        self.model.return_value = Tensor(tokens)
        norm = MagicMock()
        for method in ("float", "cuda", "eval", "requires_grad_"):
            getattr(norm, method).return_value = norm
        norm.side_effect = lambda x: Tensor((x.values - x.values.mean(-1, keepdims=True)) /
                                           np.sqrt(x.values.var(-1, keepdims=True) + 1e-6))
        self.torch.nn.LayerNorm.return_value = norm
        encoder = self.load("MIRAGE")
        pixels = np.array([[[255, 0, 0, 0], [0, 255, 0, 255]], [[0, 0, 255, 255], [255, 255, 255, 20]]], dtype=np.uint8)
        image = Image.fromarray(pixels, "RGBA")
        result = encoder.encode([image])
        expected = ((tokens[:, :256] - tokens[:, :256].mean(-1, keepdims=True)) /
                    np.sqrt(tokens[:, :256].var(-1, keepdims=True) + 1e-6)).mean(1)
        np.testing.assert_allclose(result, expected, atol=1e-6)
        actual_input = self.model.call_args.args[0]
        self.assertEqual(set(actual_input), {"bscan"})
        # A2 A-11: /255, then per-image min-max, as MIRAGE's published frozen probe.
        scaled = np.asarray(image.resize((512, 512), Image.Resampling.BILINEAR).convert("L"), dtype=np.float32) / 255
        expected_pixels = (scaled - scaled.min()) / (scaled.max() - scaled.min())
        np.testing.assert_array_equal(actual_input["bscan"].values[0, 0], expected_pixels)
        dim = Image.fromarray(np.tile(np.arange(50, 150, dtype=np.uint8), (8, 1)), "L")  # never reaches 0 or 255
        stretched = extract.mirage_input(dim)
        self.assertEqual((stretched.min(), stretched.max()), (0, 1))
        self.assertGreater(np.abs(stretched - np.asarray(dim.resize((512, 512), Image.Resampling.BILINEAR),
                                                          dtype=np.float32) / 255).max(), .1)
        with self.assertRaisesRegex(ValueError, "nonconstant"):
            encoder.encode([Image.new("L", (8, 8), 77)])
        self.torch.nn.LayerNorm.assert_called_once_with(768, eps=1e-6)
        self.model.model.load_state_dict.assert_called_once_with({"weight": "pinned"}, strict=True)
        load_file.return_value = {"unexpected": "weights"}
        with self.assertRaisesRegex(ValueError, "state keys differ"):
            self.load("MIRAGE")

    def test_dinov2_uses_local_slow_processor_pooler_and_rejects_loading_mismatch(self):
        self.write_config("preprocessor_config.json", dict(crop_size=dict(height=224, width=224),
            do_center_crop=True, do_convert_rgb=True, do_normalize=True, do_rescale=True, do_resize=True,
            image_mean=[.485, .456, .406], image_std=[.229, .224, .225], image_processor_type="BitImageProcessor",
            resample=3, rescale_factor=1 / 255, size=dict(shortest_edge=256)))
        processor = MagicMock(return_value={"pixel_values": Tensor(np.ones((1, 3, 224, 224), dtype=np.float32))})
        processor_factory = MagicMock()
        processor_factory.from_pretrained.return_value = processor
        model_factory = MagicMock()
        model_factory.from_pretrained.return_value = self.model, {}
        self.model.return_value = SimpleNamespace(pooler_output=Tensor(np.full((1, 768), 9, dtype=np.float32)))
        self.stack.enter_context(patch.dict(sys.modules, {"transformers": SimpleNamespace(
            AutoImageProcessor=processor_factory, Dinov2Model=model_factory)}))
        encoder = self.load("DINOv2")
        np.testing.assert_array_equal(encoder.encode([Image.new("L", (10, 10))]), np.full((1, 768), 9))
        processor_factory.from_pretrained.assert_called_once_with(str(self.assets), local_files_only=True, use_fast=False)
        self.assertTrue(model_factory.from_pretrained.call_args.kwargs["local_files_only"])
        self.assertTrue(model_factory.from_pretrained.call_args.kwargs["use_safetensors"])
        self.assertEqual(extract.os.environ["USE_TF"], "0")
        self.torch.autocast.assert_called_once_with(device_type="cuda", enabled=False)
        model_factory.from_pretrained.return_value = self.model, {"missing_keys": ["encoder.weight"]}
        with self.assertRaisesRegex(ValueError, "loading mismatch"):
            self.load("DINOv2")

    def test_biomedclip_only_constructs_strict_local_image_tower_with_native_transform(self):
        self.write_config("open_clip_config.json", dict(model_cfg=dict(embed_dim=512, vision_cfg=dict(
            timm_model_name="vit_base_patch16_224", timm_model_pretrained=False, timm_pool="", timm_proj="linear", image_size=224),
            text_cfg=dict(hf_model_name="must-not-be-constructed")), preprocess_cfg=dict(
            mean=[.48145466, .4578275, .40821073], std=[.26862954, .26130258, .27577711])))
        tower = MagicMock(return_value=self.model)
        native_transform = MagicMock(return_value=Tensor(np.ones((3, 224, 224), dtype=np.float32)))
        transform_factory = MagicMock(return_value=native_transform)
        self.stack.enter_context(patch.dict(sys.modules, {
            "open_clip.timm_model": SimpleNamespace(TimmModel=tower), "open_clip.transform": SimpleNamespace(
                PreprocessCfg=lambda **kwargs: SimpleNamespace(**kwargs), image_transform_v2=transform_factory)}))
        self.torch.load.return_value = {"visual.trunk.weight": "image", "text.weight": "ignored"}
        self.model.return_value = Tensor(np.full((1, 512), 5, dtype=np.float32))
        encoder = self.load("BioMedCLIP")
        image = Image.new("RGBA", (8, 8))
        np.testing.assert_array_equal(encoder.encode([image]), np.full((1, 512), 5))
        tower.assert_called_once_with(model_name="vit_base_patch16_224", embed_dim=512, image_size=224,
                                      pool="", proj="linear", pretrained=False)
        self.model.load_state_dict.assert_called_once_with({"trunk.weight": "image"}, strict=True)
        self.assertIs(native_transform.call_args.args[0], image)
        self.assertFalse(transform_factory.call_args.kwargs["is_train"])

    def test_framework_isolation_rejects_a_mixed_process(self):
        with self.assertRaisesRegex(ValueError, "without PyTorch"):
            self.load("Inception")
        with patch.dict(sys.modules, {"tensorflow": SimpleNamespace()}), \
                self.assertRaisesRegex(ValueError, "without TensorFlow"):
            self.load("DINOv2")


class KerasReference(unittest.TestCase):
    def test_reference_stages_through_stage3_and_pilot_uses_its_native_decoder(self):
        tf, model = MagicMock(), MagicMock()
        tf.config.list_physical_devices.return_value = ["synthetic GPU"]
        pixels = np.full((17, 19, 3), 73, dtype=np.uint8)
        tf.io.decode_image.return_value = Tensor(pixels)
        tf.image.resize.side_effect = lambda pixels, size: np.full((*size, 3), 73, dtype=np.float32)
        tf.stack.side_effect = np.stack
        model.return_value = Tensor(np.full((1, 2048), 11, dtype=np.float32))
        stage3 = SimpleNamespace(need_tf=MagicMock(return_value=tf), backbone=MagicMock(return_value=model),
                                 load_image=MagicMock(return_value=np.ones((299, 299, 3), dtype=np.float32)))
        with patch.object(extract, "require_colab"), patch.object(extract, "check_spec"), \
                patch.dict(sys.modules, {"stage3_classifier": stage3}):
            # This test never imports or executes TensorFlow/PyTorch.
            self.assertNotIn("torch", sys.modules)
            encoder = extract.load_encoder(model_spec("Inception"), Path("synthetic-assets"))
            infer = encoder.stage_path(Path("synthetic.jpg"))
            model.assert_not_called()
            stage3.load_image.assert_called_once_with("synthetic.jpg")
            np.testing.assert_array_equal(infer(), np.full((1, 2048), 11))
            decoded = encoder.decode_path(Path("synthetic.jpg"))
            np.testing.assert_array_equal(np.array(decoded), pixels)
            tf.io.decode_image.assert_called_once_with(tf.io.read_file.return_value, channels=3, expand_animations=False)
            np.testing.assert_array_equal(encoder.encode([decoded]), np.full((1, 2048), 11))
            self.assertEqual([call.args[1] for call in tf.image.resize.call_args_list], [(256, 256), (299, 299)])
            np.testing.assert_array_equal(tf.image.resize.call_args_list[0].args[0], pixels)
            self.assertIs(encoder.synchronize, tf.test.experimental.sync_devices)
            stage3.backbone.assert_called_once_with(weights="None")
            tf.keras.mixed_precision.set_global_policy.assert_called_once_with("float32")
            tf.config.experimental.enable_tensor_float_32_execution.assert_called_once_with(False)
            tf.config.experimental.reset_memory_stats.assert_called_once_with("GPU:0")
            model.load_weights.assert_called_once_with(str(Path("synthetic-assets") / extract.REQUIRED_ASSETS["Inception"][0]))


if __name__ == "__main__":
    unittest.main()
