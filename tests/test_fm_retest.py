"""Re-tests RT1-RT2 (protocols.md section 13, draft) on stand-in encoders and tiny images.

No model weight, locked file or real feature is read, and no real score is computed: the encoders are NumPy
stand-ins shaped like the upstream RETFound class and the Keras path of the locked loader."""
from __future__ import annotations

from contextlib import nullcontext, redirect_stdout
import datetime
from io import BytesIO, StringIO
import json
import pathlib
from pathlib import Path
import shlex
import shutil
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, Mock, patch
import zipfile

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import fm_control_extract as control  # noqa: E402
import fm_core as fm  # noqa: E402
import fm_extract as extract  # noqa: E402
import fm_posthoc as ph  # noqa: E402
import fm_prepare as prep  # noqa: E402
import fm_retest_extract as retest  # noqa: E402
import fm_retest_notebook as notebook  # noqa: E402

DIM, PATCH = extract.DIMENSIONS["RETFound"], 4
TRAINED = np.random.default_rng(7)
TRAINED_WEIGHT, TRAINED_BIAS = TRAINED.uniform(.5, 1.5, DIM), TRAINED.normal(size=DIM)


class Tensor(np.ndarray):
    """NumPy stand-in for a framework tensor: .mean(dim=, keepdim=) as the read-outs call it."""

    def mean(self, dim, keepdim=False):
        return np.asarray(self).mean(axis=dim, keepdims=keepdim).view(Tensor)

    def to(self, **kwargs):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return np.asarray(self)


def layer_norm(weight, bias, eps=1e-6):
    def norm(x):
        a = np.asarray(x, dtype=np.float64)
        scaled = (a - a.mean(axis=-1, keepdims=True)) / np.sqrt(a.var(axis=-1, keepdims=True) + eps)
        return (scaled * weight + bias).astype(np.float32).view(Tensor)
    return norm


class Block:
    def __init__(self, matrix):
        self.matrix, self.hooks = matrix, []

    def register_forward_hook(self, hook):
        self.hooks.append(hook)

    def __call__(self, x):
        out = np.tanh(np.asarray(x) @ self.matrix).astype(np.float32).view(Tensor)
        for hook in self.hooks:
            hook(self, (x,), out)
        return out


class StandIn:
    """The upstream VisionTransformer.forward_features in miniature (models_vit.py at the pinned revision):
    patch embedding, class token first, blocks, then fc_norm(patch mean) or norm(tokens)[:, 0]."""

    def __init__(self, global_pool=False):
        rng = np.random.default_rng(3)
        self.global_pool = global_pool
        self.embed = rng.normal(size=(PATCH * PATCH * 3, DIM)) / 4
        self.cls = rng.normal(size=(1, 1, DIM))
        self.blocks = [Block(rng.normal(size=(DIM, DIM)) / np.sqrt(DIM)) for _ in range(2)]
        if global_pool:
            self.fc_norm = layer_norm(np.ones(DIM), np.zeros(DIM))  # the locked, untrained fc_norm
        else:
            self.norm = layer_norm(TRAINED_WEIGHT, TRAINED_BIAS)  # the checkpoint's trained norm

    def forward_features(self, x):
        batch, _, height, width = x.shape
        grid = x.reshape(batch, 3, height // PATCH, PATCH, width // PATCH, PATCH).transpose(0, 2, 4, 1, 3, 5)
        patches = grid.reshape(batch, -1, PATCH * PATCH * 3) @ self.embed
        tokens = np.concatenate([np.broadcast_to(self.cls, (batch, 1, DIM)), patches], axis=1).view(Tensor)
        for block in self.blocks:
            tokens = block(tokens)
        if self.global_pool:
            return self.fc_norm(tokens[:, 1:, :].mean(dim=1, keepdim=True))
        return self.norm(tokens)[:, 0]

    # What the locked loader asks of a model besides forward_features.
    def load_state_dict(self, state, strict=True):
        return SimpleNamespace(missing_keys=["head.weight", "head.bias", "fc_norm.weight", "fc_norm.bias"],
                               unexpected_keys=[key for key in state if key.startswith("norm.")])

    def float(self):
        return self

    cuda = eval = float

    def requires_grad_(self, flag):
        return self


def prepare(images) -> np.ndarray:
    return np.stack([np.asarray(im.convert("RGB").resize((8, 8)), dtype=np.float64).transpose(2, 0, 1) / 255 for im in images])


def standin_loader(spec, assets):
    """Stands in for fm_extract.load_encoder: builds the model through fm_extract.import_source and reads
    model.forward_features at call time, as the locked RETFound closure does."""
    model = extract.import_source(assets / "models_vit.py").RETFound_mae(global_pool=True)

    def encode(images):
        values = model.forward_features(prepare(images))
        if values.shape != (len(images), 1, DIM):
            raise ValueError("RETFound forward_features must return [batch, 1, 1024]")
        return np.asarray(values[:, 0, :], dtype=np.float32)

    return extract.Encoder("RETFound", encode, lambda: None, {"standin": True}, dict)


def trained(_assets):
    return layer_norm(TRAINED_WEIGHT, TRAINED_BIAS), {"available": True}


def no_norm(_assets):
    return None, {"available": False, "reason": "the checkpoint holds no norm.weight and norm.bias"}


def upstream_false(_assets, capture):
    return capture.build(global_pool=False).forward_features, {"missing_keys": ["head.weight", "head.bias"]}


def picture(seed: int, size=(24, 16), mode="RGB") -> Image.Image:
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, (size[1], size[0], len(mode)), dtype=np.uint8), mode)


class Bench:
    """A temporary checkout with tiny images, both inventories and a stand-in input lock."""

    def __init__(self, test: unittest.TestCase, main: int = 6, external: int = 2):
        temp = TemporaryDirectory()
        test.addCleanup(temp.cleanup)
        self.work = Path(temp.name)
        self.root, self.lock, self.external, self.assets = (self.work / n for n in ("checkout", "lock", "external", "assets"))
        for folder in (self.lock, self.external, self.assets):
            folder.mkdir()
        (self.lock / "input_lock.json").write_text("{}", encoding="utf-8")
        (self.external / "external_lock.json").write_text("{}", encoding="utf-8")
        (self.assets / "models_vit.py").write_text(
            f"import sys\nRETFound_mae = sys.modules[{__name__!r}].StandIn\n", encoding="utf-8")
        self.rows = {}
        for folder, inventory, count, stem in ((self.lock, "extraction_inventory.csv", main, "data/raw/rare/CSC/im{}.png"),
                                               (self.external, "external_extraction_inventory.csv", external,
                                                "data/external/OCTID/im{}.png")):
            rows = []
            for i in range(count):
                target = self.root / stem.format(i)
                target.parent.mkdir(parents=True, exist_ok=True)
                picture(len(rows) + 10 * count, mode="RGBA" if i == 1 else "RGB").save(target)
                rows.append([stem.format(i), prep.sha256(target)])
            prep.write_csv(folder / inventory, ["path", "sha256"], rows)
            self.rows[inventory] = prep.read_csv(folder / inventory)
        identity = dict(spec={}, code=prep.code_identity(), environment=extract.environment())
        lock = dict(code=prep.code_identity(), arms={arm: dict(status="available", identity=identity)
                                                   for arm in ("RETFound", "Inception")})
        for target in (patch.object(prep, "load_lock", return_value=lock), patch("fm_external.verify_external"),
                       patch.object(control, "safe_hardware", return_value={})):
            target.start()
            test.addCleanup(target.stop)

    def rt1(self, variant, norm_of=trained, name=None):
        load = lambda spec, assets, v: retest.load_readout(spec, assets, v, standin_loader, norm_of, upstream_false)  # noqa: E731
        out = self.work / "out" / (name or variant)
        return out, retest.run_rt1(self.root, self.lock, self.external, variant, self.assets, out, load=load)

    def features(self, folder: Path, name="features.npz", inventory="extraction_inventory.csv", dim=DIM):
        return extract.load_features(folder / name, self.rows[inventory], dim)


class ReadoutTest(unittest.TestCase):
    def setUp(self):
        self.x = prepare([picture(i) for i in range(3)])
        self.norm = layer_norm(TRAINED_WEIGHT, TRAINED_BIAS)

    def readout(self, variant, norm=None):
        model = StandIn(global_pool=True)
        record = retest.attach_readout(model, variant, norm)
        return np.asarray(model.forward_features(self.x)), record

    def test_each_readout_has_the_locked_shape_and_differs_from_the_others(self):
        outputs = {v: self.readout(v, self.norm)[0] for v in retest.READOUTS}
        for variant, values in outputs.items():
            self.assertEqual(values.shape, (3, 1, DIM), variant)
        names = list(outputs)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                self.assertFalse(np.allclose(outputs[a], outputs[b], atol=1e-3), (a, b))

    def test_r_ref_is_the_locked_path_and_every_call_checks_the_hook(self):
        locked = np.asarray(StandIn(global_pool=True).forward_features(self.x))
        values, record = self.readout("R-ref")
        np.testing.assert_array_equal(values, locked)
        for variant in ("R-meanraw", "R-clsnorm"):
            _, record = self.readout(variant, self.norm)
            self.assertEqual(retest.hook_check(record)["max_difference_ratio"], 0)
            self.assertEqual(record["calls"], 1)

    def test_norm_readouts_follow_the_upstream_definitions(self):
        tokens = {}
        model = StandIn(global_pool=True)
        model.blocks[-1].register_forward_hook(lambda module, inputs, output: tokens.update(last=output))
        model.forward_features(self.x)
        last = tokens["last"]
        self.assertEqual(last.shape, (3, 5, DIM))  # class token and four patches
        # R-clsnorm is upstream global_pool=False: norm on every token, then the class token.
        np.testing.assert_array_equal(self.readout("R-clsnorm", self.norm)[0][:, 0, :],
                                      StandIn(global_pool=False).forward_features(self.x))
        # R-meannorm normalises each token first; normalising the mean instead is another read-out.
        meannorm = self.readout("R-meannorm", self.norm)[0]
        np.testing.assert_allclose(meannorm, np.asarray(self.norm(last))[:, 1:, :].mean(axis=1, keepdims=True), rtol=1e-6)
        self.assertFalse(np.allclose(meannorm, self.norm(retest.patch_mean(last)), atol=1e-3))
        np.testing.assert_allclose(self.readout("R-meanraw")[0][:, 0, :], np.asarray(last)[:, 1:, :].mean(axis=1), rtol=1e-6)

    def test_a_missing_trained_norm_is_never_substituted(self):
        for variant in retest.NEEDS_NORM:
            with self.assertRaisesRegex(ValueError, "never substituted"):
                retest.attach_readout(StandIn(global_pool=True), variant, None)
        with self.assertRaisesRegex(ValueError, "no read-out"):
            retest.attach_readout(StandIn(global_pool=True), "R-best", self.norm)

    def test_probe_keeps_upstream_and_our_outputs_and_returns_the_locked_one(self):
        model = StandIn(global_pool=True)
        locked = np.asarray(model.forward_features(self.x))
        kept = retest.attach_probe(model, self.norm, StandIn(global_pool=False).forward_features)
        np.testing.assert_array_equal(model.forward_features(self.x), locked)
        self.assertEqual({name: values[0].shape for name, values in kept.items()}, dict.fromkeys(retest.PROBE, (3, DIM)))
        np.testing.assert_array_equal(kept["upstream_global_pool_true"][0], kept["R-ref"][0])
        np.testing.assert_array_equal(kept["upstream_global_pool_false"][0], kept["R-clsnorm"][0])
        self.assertFalse(np.allclose(kept["R-ref"][0], kept["R-clsnorm"][0], atol=1e-3))
        with self.assertRaisesRegex(ValueError, "both"):
            retest.attach_probe(StandIn(global_pool=True), self.norm, None)


class LockedLoaderTest(unittest.TestCase):
    def test_readouts_ride_on_the_real_locked_loader(self):
        """fm_extract.load_encoder itself, with torch mocked as tests/test_fm_extract.py mocks it: the wrapped
        import_source reaches the model the locked code builds, and the locked closure returns our read-out."""
        bench = Bench(self)
        torch = MagicMock()
        torch.cuda.is_available.return_value = True
        torch.float32 = np.dtype("float32")
        torch.stack.side_effect = lambda values: np.stack(values).view(Tensor)
        torch.inference_mode.side_effect = nullcontext
        torch.autocast.side_effect = lambda **kwargs: nullcontext()
        torch.load.return_value = {"model": {"norm.weight": "trained", "norm.bias": "trained", "head.weight": "unused"}}
        transforms = MagicMock()
        transforms.Compose.return_value = lambda image: prepare([image])[0].view(Tensor)
        image, spec = picture(4), {"arm": "RETFound"}
        with patch.object(extract, "require_colab"), patch.object(extract, "check_spec"), patch.dict(extract.os.environ), \
                patch.dict(sys.modules, {"torch": torch, "torchvision": SimpleNamespace(transforms=transforms)}):
            sys.modules.pop("tensorflow", None)  # the locked loader wants a process without TensorFlow
            plain = extract.load_encoder(spec, bench.assets)
            self.assertEqual(plain.load_report["unexpected_keys"], ["norm.weight", "norm.bias"])
            locked = plain.encode([image])
            reference, _ = retest.load_readout(spec, bench.assets, "R-ref", norm_of=trained)
            np.testing.assert_array_equal(reference.encode([image]), locked)
            encoder, report = retest.load_readout(spec, bench.assets, "R-clsnorm", norm_of=trained)
            values = encoder.encode([image])
        self.assertEqual((values.shape, values.dtype), ((1, DIM), np.float32))
        np.testing.assert_array_equal(values, StandIn(global_pool=False).forward_features(prepare([image])))
        self.assertFalse(np.allclose(values, locked, atol=1e-3))
        self.assertEqual(retest.hook_check(report["record"]), dict(
            calls=1, max_difference_ratio=0, compares=retest.hook_check(report["record"])["compares"]))
        self.assertEqual(report["overrides"]["model.forward_features"], "fm_retest_extract._r_clsnorm")


class WorkerTest(unittest.TestCase):
    def test_rt1_reference_reproduces_the_locked_stand_in_and_records_its_overrides(self):
        bench = Bench(self)
        loader = extract.import_source
        out, identity = bench.rt1("R-ref")
        self.assertIs(extract.import_source, loader)  # the override ends with the load
        plain = standin_loader({}, bench.assets)  # the locked path: no read-out attached
        locked = np.concatenate([plain.encode([Image.open(bench.root / r["path"])]) for r in bench.rows["extraction_inventory.csv"]])
        np.testing.assert_array_equal(bench.features(out), locked)
        self.assertEqual(bench.features(out, "features_external.npz", "external_extraction_inventory.csv").shape, (2, DIM))
        prep.verify_complete(out)
        self.assertEqual(set(identity["overrides"]), {"fm_extract.import_source", "model.forward_features"})
        self.assertEqual(identity["overrides"]["model.forward_features"], "fm_retest_extract._r_ref")
        self.assertEqual((identity["retest"], identity["variant"], identity["available"]), ("RT1", "R-ref", True))
        self.assertEqual(identity["hook_check"]["max_difference_ratio"], 0)
        self.assertEqual(identity["hook_check"]["calls"], 6 + 2 + 5)  # main, external and the five repeats
        self.assertEqual(identity["repeat_check"]["max_difference_ratio"], 0)
        self.assertEqual((identity["registry_sha256"], identity["script_sha256"]),
                         (retest.registry_hash(), prep.text_hash(ROOT / "fm_retest_extract.py")))
        self.assertTrue(identity["code_matches_lock"] and identity["environment_matches_lock"])
        other, _ = bench.rt1("R-meannorm")
        self.assertFalse(np.allclose(bench.features(other), locked, atol=1e-3))
        with redirect_stdout(StringIO()):
            again = retest.run_rt1(bench.root, bench.lock, bench.external, "R-ref", bench.assets, out,
                                   load=Mock(side_effect=AssertionError("a complete output is not re-extracted")))
        self.assertEqual(again, identity)

    def test_rt1_norm_variant_without_a_trained_norm_is_reported_not_available(self):
        bench = Bench(self)
        out, identity = bench.rt1("R-clsnorm", no_norm)
        self.assertFalse(identity["available"])
        self.assertIn("no norm.weight", identity["reason"])
        self.assertEqual(set(prep.verify_complete(out)["files"]), {"identity.json"})  # no feature file is written

    def test_upstream_check_compares_forty_rows_at_most_and_degrades_without_the_norm(self):
        bench = Bench(self, main=22, external=2)
        out, identity = bench.rt1(retest.UPSTREAM)
        self.assertEqual(identity["images"], dict(main=20, external=2))
        self.assertEqual({k: v["max_difference_ratio"] for k, v in identity["upstream_check"].items()},
                         {"global_pool_true_vs_R-ref": 0, "global_pool_false_vs_R-clsnorm": 0})
        self.assertEqual(set(prep.verify_complete(out)["files"]), {*(f"{name}.npz" for name in retest.PROBE), "identity.json"})
        out, identity = bench.rt1(retest.UPSTREAM, no_norm, name="no-norm")
        self.assertEqual(identity["upstream_check"]["global_pool_false_vs_R-clsnorm"], "not available")
        self.assertEqual(set(prep.verify_complete(out)["files"]),
                         {"upstream_global_pool_true.npz", "R-ref.npz", "identity.json"})

    def test_rt2_changes_only_the_file_the_locked_loader_reads(self):
        bench = Bench(self)
        dim = extract.DIMENSIONS["Inception"]

        def decoded(path):
            with Image.open(path) as image:
                return np.asarray(image.convert("RGB"))

        read = []

        def keras(spec, assets):  # the locked loader's Keras shape: stage_path(path)() returns the features
            def stage_path(path):
                read.append(Path(path))
                return lambda: (np.resize(decoded(path).astype(np.float32).ravel(), dim) + 1)[None]
            return extract.Encoder("Inception", None, lambda: None, {"standin": True}, dict, stage_path)

        pixels = lambda a, b: retest.pixel_difference(decoded(BytesIO(a)), decoded(BytesIO(b)))  # noqa: E731
        render, png = bench.work / "render", bench.work / "png"
        receipt = retest.render_pil(bench.root, bench.lock, bench.external, render, png)
        self.assertEqual((receipt["images"], receipt["counts"]["alpha_below_255"]), (dict(main=6, external=2), 2))
        self.assertEqual(set(prep.verify_complete(render)["files"]), {"render.csv", "identity.json"})  # PNGs are not archived
        first = bench.rows["extraction_inventory.csv"][1]
        with Image.open(retest.png_path(png, first["path"])) as saved, Image.open(bench.root / first["path"]) as original:
            self.assertEqual((saved.mode, saved.size, original.mode), ("RGB", original.size, "RGBA"))  # no crop, no resize
        run = lambda variant, name, **kw: (bench.work / name, retest.run_rt2(  # noqa: E731
            bench.root, bench.lock, bench.external, kw.pop("render", None), kw.pop("png", None), variant, bench.assets,
            bench.work / name, load=keras, pixels=pixels, **kw))
        ref, ref_identity = run("I-ref", "ref")
        self.assertTrue(all(bench.root in path.parents for path in read))  # the control reads the originals
        del read[:]
        pil, identity = run("I-pil", "pil", render=render, png=png)
        self.assertTrue(all(png / "I-pil" in path.parents and path.suffix == ".png" for path in read))
        self.assertEqual(len(read), 6 + 2 + 5)  # main, external and the five repeats
        np.testing.assert_array_equal(bench.features(pil, dim=dim), bench.features(ref, dim=dim))  # PNG originals: lossless
        self.assertEqual(identity["decoder_pixels"], {"rare": dict(images=6, identical=6), "OCTID": dict(images=2, identical=2)})
        self.assertEqual((identity["overrides"].keys(), ref_identity["overrides"]), ({"input file"}, {}))
        rows = prep.read_csv(pil / "decoder_pixels.csv")
        self.assertEqual({(r["identical"], r["max_abs_difference"]) for r in rows}, {("True", "0")})
        short, identity = run("I-ref", "short", main_only=True)
        self.assertEqual((identity["main_only"], identity["images"]), (True, dict(main=6, external=0)))
        self.assertNotIn("features_external.npz", prep.verify_complete(short)["files"])
        with self.assertRaisesRegex(ValueError, "only for I-pil"):
            run("I-ref", "wrong", render=render, png=png)
        # Receipts restored without their PNGs: the render rebuilds them and refuses one that differs.
        shutil.rmtree(png)
        with redirect_stdout(StringIO()):
            retest.render_pil(bench.root, bench.lock, bench.external, render, png)
        self.assertTrue(retest.checked_png(png, first, retest.read_receipts(render, bench.lock, bench.external)).is_file())
        retest.png_path(png, first["path"]).write_bytes(b"other")
        with self.assertRaisesRegex(ValueError, "differs from its receipt"):
            retest.checked_png(png, first, retest.read_receipts(render, bench.lock, bench.external))

    def test_pixel_difference(self):
        a = np.zeros((2, 3, 3), dtype=np.uint8)
        b = a.copy()
        b[0, 0, 0] = 255
        self.assertEqual(retest.pixel_difference(a, a), (True, True, 0, 0.0))
        self.assertEqual(retest.pixel_difference(a, b)[:3], (True, False, 255))
        self.assertEqual(retest.pixel_difference(a, a[:1]), (False, False, None, None))

    def test_laptop_pixel_count_reads_rare_rows_only_and_stays_outside_the_checkouts(self):
        bench = Bench(self)
        prep.write_csv(bench.work / "both.csv", ["path", "sha256"], [
            *[[r["path"], r["sha256"]] for r in bench.rows["extraction_inventory.csv"][:2]],
            ["data/raw/OCT2017/test/NORMAL/NORMAL-1-1.jpeg", "0" * 64]])
        seen = []

        def pixels(original, png):
            seen.append((original[1:4], png[1:4]))
            return True, len(seen) > 1, 0, 0.0

        out = bench.work / "scratch" / "pixels.csv"
        self.assertEqual(retest.decoder_pixels(bench.root, bench.work / "both.csv", out, pixels=pixels),
                         {"rare": dict(images=2, identical=1)})  # the Kermany row is not read without --kermany
        self.assertEqual(seen, [(b"PNG", b"PNG")] * 2)
        self.assertEqual([r["source"] for r in prep.read_csv(out)], ["rare", "rare"])
        with self.assertRaisesRegex(ValueError, "outside all project checkouts"):
            retest.decoder_pixels(bench.root, bench.work / "both.csv", ROOT / "results" / "pixels.csv", pixels=pixels)

    def test_tensorflow_and_pil_agree_on_synthetic_png_and_the_check_sees_a_changed_pixel(self):
        try:
            import tensorflow  # noqa: F401
        except ImportError:
            self.skipTest("TensorFlow is not installed")
        for mode in ("RGBA", "RGB", "L"):  # RGBA: both decoders drop alpha, values below 255 included
            buffer = BytesIO()
            original = picture(5).convert("L") if mode == "L" else picture(5, mode=mode)
            original.save(buffer, format="PNG")
            png = control.png_bytes(original.convert("RGB"))
            self.assertEqual(retest.tf_pixels(buffer.getvalue(), png), (True, True, 0, 0.0), mode)
        other = original.convert("RGB")
        other.putpixel((0, 0), tuple(255 - value for value in other.getpixel((0, 0))))
        self.assertEqual(retest.tf_pixels(buffer.getvalue(), control.png_bytes(other))[:2], (True, False))

    def test_registry_covers_every_job_and_is_hashed(self):
        self.assertEqual(retest.RT1_VARIANTS, ("R-ref", "R-meanraw", "R-clsnorm", "R-meannorm", "upstream-check"))
        self.assertEqual(retest.RT2_VARIANTS, ("I-pil", "I-ref"))
        self.assertEqual([retest.retest_of(v) for v in ("R-clsnorm", "I-pil")], ["RT1", "RT2"])
        before = retest.registry_hash()
        with patch.dict(retest.VARIANTS, {"RT1/R-ref": {}}):
            self.assertNotEqual(retest.registry_hash(), before)  # identity.json binds each output to these definitions


class NotebookTest(unittest.TestCase):
    def test_builder_is_deterministic_and_the_committed_notebook_is_its_draft_output(self):
        path = ROOT / notebook.NAME
        pins = notebook.pinned(path)
        self.assertEqual(pins, (notebook.DRAFT_PIN, notebook.DRAFT_PIN))  # a draft rule: the notebook cannot run
        text = notebook.build(*pins)
        self.assertEqual(text, notebook.build(*pins))
        self.assertEqual(path.read_text(encoding="utf-8").replace("\r\n", "\n"), text)
        cells = [cell for cell in json.loads(text)["cells"] if cell["cell_type"] == "code"]
        for i, cell in enumerate(cells):
            compile("".join(cell["source"]), f"{notebook.NAME} cell {i + 1}", "exec")
        source = "\n".join("".join(cell["source"]) for cell in cells)
        self.assertIn('userdata.get("HF_TOKEN")', source)  # the token comes from Colab Secrets only
        self.assertNotRegex(source, r"hf_[A-Za-z0-9]{20,}")
        self.assertLess(source.index("# Step 2a. Rules before numbers"), source.index("# Step 3."))
        self.assertEqual([job[0] for job in notebook.JOBS], ["RT2"] * 2 + ["RT1"] * 5)  # RT2 first

    def test_a_runnable_notebook_needs_the_approved_pushed_rule(self):
        commit, rule = "a" * 40, "b" * 40
        heading = "## 13. Re-tests RT1–RT2 (rules fixed 25 Nov 2026, before any RT number)"

        def git(protocol, added):
            def run(*args):
                if args[0] == "log":
                    return SimpleNamespace(returncode=0, stdout=added.encode())
                name = args[1].split(":")[1]
                data = protocol.encode("utf-8") if name == "protocols.md" else (ROOT / name).read_bytes()
                return SimpleNamespace(returncode=0, stdout=data)
            return run

        with redirect_stdout(StringIO()):
            with patch.object(notebook, "_git", side_effect=git(f"# x\n{heading}\n", rule + "\n")):
                notebook.check_pin(commit, rule)
            with patch.object(notebook, "_git", side_effect=git("## 13. Re-tests RT1–RT2 (DRAFT, not approved)\n", rule)):
                with self.assertRaisesRegex(SystemExit, "still a DRAFT"):
                    notebook.check_pin(commit, rule)
            with patch.object(notebook, "_git", side_effect=git(f"{heading}\n", "c" * 40)):
                with self.assertRaisesRegex(SystemExit, "not the commit that fixed"):
                    notebook.check_pin(commit, rule)
            with self.assertRaisesRegex(SystemExit, "or neither"):
                notebook.check_pin(commit, notebook.DRAFT_PIN)
            notebook.check_pin(notebook.DRAFT_PIN, notebook.DRAFT_PIN)
        with self.assertRaisesRegex(ValueError, "reused cell changed"):
            notebook.swap("one two two", ("two", "three"))

    def test_session_runs_rt2_then_rt1_and_saves_each_worker(self):
        with TemporaryDirectory() as temp, redirect_stdout(StringIO()):
            work = Path(temp)
            folder = work / "drive" / "retest"
            folder.mkdir(parents=True)
            commands = []

            def save_to_drive(_folder, files):
                for path in files:
                    shutil.copyfile(path, folder / path.name)

            def export(command):
                args = shlex.split(command)
                prep.export(Path(args[args.index("--out") + 1]), Path(args[args.index("--archive") + 1]), work)

            def worker(command, log):
                Path(log).parent.mkdir(parents=True, exist_ok=True)
                Path(log).touch()
                commands.append(command)
                output = Path(command[command.index("--out") + 1])
                output.mkdir(parents=True, exist_ok=True)
                (output / "synthetic.txt").write_text("worker output", encoding="utf-8")
                prep.complete(output, ["synthetic.txt"])
                return 123, 0

            phase = work / "runtime"
            state = dict(pathlib=pathlib, json=json, zipfile=zipfile, shutil=shutil, datetime=datetime, shlex=shlex,
                         drive=SimpleNamespace(mount=lambda *a, **kw: None, flush_and_unmount=lambda: None),
                         os=SimpleNamespace(path=SimpleNamespace(ismount=lambda _: False)),
                         SESSION=1, SESSION_JOBS={1: notebook.JOBS}, OUT=work / "drive", DRIVE_FOLDER="retest",
                         PHASE=phase, EXPORT=phase / "exports", RENDER=phase / "render", PNG=phase / "png",
                         PNG_READY=phase / "png.ok", LOCK=phase / "lock", EXTERNAL_LOCK=phase / "external",
                         PY="python", ROOT="checkout", ASSETS="assets", SCRIPT="fm_retest_extract.py", sha256=prep.sha256,
                         worker=worker, sh=export, save_to_drive=save_to_drive, asset_issues={},
                         INTERPRETER={"RETFound": "python", "Inception": "python3"}, RUNTIME="L4", COMMIT="0" * 40,
                         LEDGER=work / "unused.csv")
            exec(notebook.swap(notebook.FIRST, ("@RULE@", "1" * 40)), state)
            state.update(RUN_BY="Test", DROP_LEVEL=1)
            for cell in (notebook.ARCHIVES, notebook.RENDER_CELL, notebook.EXTRACT_CELL, notebook.STAMP):
                exec(cell, state)
            self.assertEqual(commands[0][3:5], ["fm_retest_extract.py", "render"])
            self.assertIn("--png", commands[0])
            ran = [(c[0], c[c.index("--variant") + 1], "--render" in c, "--main-only" in c) for c in commands[1:]]
            self.assertEqual(ran, [("python3", "I-pil", True, True), ("python3", "I-ref", False, True),
                                   ("python", "R-ref", False, False), ("python", "upstream-check", False, False),
                                   ("python", "R-meanraw", False, False), ("python", "R-clsnorm", False, False),
                                   ("python", "R-meannorm", False, False)])
            self.assertTrue((folder / "retest-RT1-RETFound-R-clsnorm.zip.json").is_file())
            self.assertTrue((folder / "retest-render.zip").is_file())
            (logs,) = folder.glob("retest-logs-*.zip")
            with zipfile.ZipFile(logs) as archive:
                stamp = archive.read("stamp.txt").decode("utf-8")
            self.assertRegex(stamp, r"^RETEST STAMP \| \d{8}T\d{6}Z \| drop_level=1 \| by=Test ")
            self.assertIn(f"rule={'1' * 40} | completed=render, RT2-Inception-I-pil", stamp)
            self.assertIn("missing=none", stamp)


def draws(value: float) -> dict[tuple[int, int], float]:
    return {(k, draw): value for k in (*ph.KS, 16) for draw in range(fm.N_DRAWS)}


SECTION = dict(same=True, working_sha256="a" * 64, commit_sha256="a" * 64)


class Named(str):
    """A feature set known by its name, holding one rare image (stands in for fm_evaluate.Features)."""
    index = {"data/raw/rare/CSC/a.png": 0}


class ScoringTest(unittest.TestCase):
    def test_refuses_under_a_draft_rule_before_anything_is_read(self):
        approved = "## 13. Re-tests RT1–RT2 (rules fixed 25 Nov 2026, before any RT number)"
        self.assertEqual(ph.rt_heading(f"## 12. Phase 2\n\n{approved}\n\ntext\n"), approved)
        with self.assertRaisesRegex(SystemExit, "still a DRAFT"):
            ph.rt_heading("## 13. Re-tests RT1–RT2 (DRAFT, not approved)\n")
        with self.assertRaisesRegex(SystemExit, "exactly one"):
            ph.rt_heading("## 12. Phase 2\n")
        # The committed rule is a draft: the command stops before it opens the attempt, dry runs included.
        for dry_run in (False, True):
            args = SimpleNamespace(date="2026-11-23", dry_run=dry_run, remote="origin/t3/fm-code-prep", retest=Path("none"),
                                   attempt=Path("no-such-attempt"), partial=False)
            with patch.object(ph, "Study", side_effect=AssertionError("the attempt must not be opened")), \
                    self.assertRaisesRegex(SystemExit, "still a DRAFT"):
                ph.run_retest(args, Path("no-such-out"))
        # An approved heading that no commit holds is refused by rule 1 (rule_commit), as for E1-E4.
        with patch.object(ph, "rt_heading", return_value=approved), self.assertRaisesRegex(SystemExit, "rule 1"):
            ph.rt_rule("2026-11-23", False)

    def test_label_carries_the_rule_text_and_marks_dry_runs(self):
        label = ph.rt_label(1, "c0ffee0" + "0" * 33, "25 Nov 2026", "2026-11-26", SECTION)
        self.assertEqual(label, "post hoc, re-test rule RT1 timestamped by commit c0ffee0 (25 Nov 2026), written after the "
                                "Phase-1 results and the mentor's review of 23 Nov; descriptive; cannot change the E1 verdict "
                                "or any locked result; computed 2026-11-26")
        dry = ph.rt_label(2, "c0ffee0" + "0" * 33, "25 Nov 2026", "2026-11-26", SECTION, dry_run=True)
        self.assertIn("dry run", dry)  # the string the paper's table builder refuses (make_tables.e_guard)
        self.assertIn("WARNING", ph.rt_label(1, "c" * 40, "25 Nov 2026", "2026-11-26", {**SECTION, "same": False}))
        args = SimpleNamespace(date="2026-11-26", dry_run=True, remote="origin", retest=None, attempt=Path("x"), partial=False)
        with patch.object(ph, "rt_rule", return_value=({}, SECTION)), self.assertRaisesRegex(ValueError, "outside all project"):
            ph.run_retest(args, ROOT / "results" / "fm_v1" / "posthoc")  # a dry run never writes into a checkout

    def test_fixed_reading_and_sentences(self):
        self.assertEqual(ph.rt_shift([.01] * 100)["reading"], "above")
        self.assertEqual(ph.rt_shift([-.01] * 100)["reading"], "below")
        self.assertEqual(ph.rt_shift([-.01] * 6 + [.01] * 94)["reading"], "not separated")
        self.assertEqual(ph.rt_shift([.01] * 99)["reading"], "not evaluable")
        below, above = ph.rt_shift([-.128] * 100), ph.rt_shift([.044] * 100)
        upstream = {"images": 40, "upstream_global_pool_true vs R-ref": {"max_difference_ratio": 0.0},
                    "upstream_global_pool_false vs R-clsnorm": {"max_difference_ratio": .5}}
        stays, check = ph.rt1_sentences(below, {"R-meanraw": below, "R-clsnorm": below}, {"R-meannorm": "not available"}, upstream)
        self.assertTrue(stays.startswith("RETFound stays below ImageNet Inception-v3 on development under every read-out tested"))
        self.assertIn("R-meanraw: mean -0.128, p05 -0.128, p95 -0.128", stays)
        self.assertTrue(stays.endswith("; R-meannorm: not available."))
        self.assertEqual(check, "On 40 images: R-ref equals the upstream class with global_pool=True (largest difference ratio "
                                "0.00e+00); R-clsnorm differs from the upstream class with global_pool=False (largest "
                                "difference ratio 5.00e-01).")
        depends = ph.rt1_sentences(below, {"R-meanraw": below, "R-clsnorm": above}, {}, None)
        self.assertIn("depends on the read-out: R-meanraw is below Inception-v3", depends[0])
        self.assertIn("R-clsnorm is above Inception-v3 (mean +0.044", depends[0])
        self.assertEqual(depends[1], "The upstream check is unavailable.")
        self.assertEqual(ph.rt1_sentences(below, {}, {"R-meanraw": ph.NOT_INTERPRETED}, None)[0],
                         f"RT1 is not evaluable; R-meanraw: {ph.NOT_INTERPRETED}.")
        same = ph.rt2_sentences(ph.rt_shift([0.] * 100), (146, 146), 0., [("OCTID NORMAL", 15, 206, 15, True)], "")
        self.assertIn("decode all 146 rare images to the same pixels", same[0])
        self.assertIn(f"the decoder is not part of its {ph.RT_E1M_SHIFT} change under E1-M", same[0])
        self.assertEqual(same[1], "Normal scans called rare by Inception-v3 under I-pil: OCTID NORMAL 15/206 (locked 15/206, "
                                  "inside the locked 95% interval).")
        moved = ph.rt2_sentences(above, (140, 146), .02, None, "")
        self.assertIn("PIL decoding alone raises it above the locked Inception-v3 development K mean (mean +0.044", moved[0])
        self.assertIn("6 of 146 rare images change pixels", moved[0])
        self.assertEqual(moved[1], "RT2 nine-way is not scored: OCTID and OCTDL were not extracted.")
        self.assertEqual(ph.rt2_sentences(None, None, None, None, ph.NOT_INTERPRETED), [f"RT2 is not evaluable ({ph.NOT_INTERPRETED})."])

    def test_archives_must_match_their_receipts(self):
        with TemporaryDirectory() as temp:
            folder = Path(temp)
            archive = folder / "retest-RT1-RETFound-R-ref.zip"
            with zipfile.ZipFile(archive, "w") as stream:
                stream.writestr("identity.json", "{}")
            (folder / "retest-logs-1.zip").write_text("not read", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "no .json receipt"):
                ph.rt_root(folder, folder / "work")
            prep.write_json(folder / (archive.name + ".json"), {"sha256": prep.sha256(archive)})
            root, rows = ph.rt_root(folder, folder / "work")
            self.assertEqual(rows, [[archive.name, prep.sha256(archive)]])
            self.assertTrue((root / archive.stem / "identity.json").is_file())
            with archive.open("ab") as stream:
                stream.write(b"tampered")
            with self.assertRaisesRegex(ValueError, "differs from the SHA256"):
                ph.rt_root(folder, folder / "other")

    def test_outputs_are_bound_to_the_runner_the_lock_and_the_render(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            study = SimpleNamespace(lock=root, external=root)
            for name in ("input_lock.json", "external_lock.json"):
                (root / name).write_text("{}", encoding="utf-8")
            prep.write_csv(root / "extraction_inventory.csv", ["path", "sha256"], [])
            prep.write_csv(root / "external_extraction_inventory.csv", ["path", "sha256"], [])
            base = dict(script_sha256=prep.text_hash(ROOT / "fm_retest_extract.py"), registry_sha256=retest.registry_hash(),
                        lock_sha256=prep.sha256(root / "input_lock.json"), external_lock_sha256=prep.sha256(root / "input_lock.json"))

            def folder(name, **identity):
                target = root / "out" / name
                target.mkdir(parents=True)
                prep.write_json(target / "identity.json", {**base, **identity})
                prep.complete(target, ["identity.json"])
                return target

            folder("R-ref", step="retest_extract", retest="RT1", variant="R-ref")
            with self.assertRaisesRegex(ValueError, "Partial re-test delivery \\(no I-ref\\)"):
                ph.Retest(study, root / "out")
            render = folder("render", step="retest_render", retest="RT2")
            folder("I-ref", step="retest_extract", retest="RT2", variant="I-ref")
            folder("I-pil", step="retest_extract", retest="RT2", variant="I-pil", render_identity_sha256="other render")
            with self.assertRaisesRegex(ValueError, "render whose receipts are not in this delivery"):
                ph.Retest(study, root / "out")
            shutil.rmtree(root / "out" / "I-pil")
            folder("I-pil", step="retest_extract", retest="RT2", variant="I-pil",
                   render_identity_sha256=prep.sha256(render / "identity.json"))
            self.assertEqual(set(ph.Retest(study, root / "out").outputs),
                             {("RT1", "R-ref"), ("RT2", "I-ref"), ("RT2", "I-pil"), ("RT2", "render")})
            folder("stale", step="retest_extract", retest="RT1", variant="R-meanraw", registry_sha256="older registry")
            with self.assertRaisesRegex(ValueError, "registry_sha256 differs"):
                ph.Retest(study, root / "out")

    def tables(self, name, rt, values, nine_counts=None):
        """rt_tables with the episode scoring replaced: `values[feature name]` is its development macro recall."""
        study = SimpleNamespace(features=lambda arm: f"locked {arm}", external_features=lambda arm: "external")

        def nine_rows(_study, cells):
            count = (nine_counts or {}).get(cells, 2)
            rows = [[name_, 10, 10 - count, 10 - count, 0, count, "false_rare_rate", count / 10, 0., .3, "exact", 1 - count / 10, 0, 1]
                    for name_ in ph.NORMAL_SETS]
            return rows + [[name_, 4, 2, 0, 0, 2, "share", .5, None, None, "", None, None, None]
                           for name_ in (*ph.RARE_EXTERNAL, "Legacy 22")]

        with patch.multiple(ph, rt_scored=Mock(side_effect=lambda _s, main, ext: SimpleNamespace(
                                value=values[main], nine=None if ext is None else main)),
                            dev_macro=Mock(side_effect=lambda _s, sc: draws(sc.value)),
                            rep_macro=Mock(side_effect=lambda _s, sc: draws(sc.value)),
                            nine_way_rows=Mock(side_effect=nine_rows)):
            tables = ph.rt_tables(study, rt, name, SECTION)
        return {key: [dict(zip(header, row)) for row in rows] for key, (header, rows) in tables.items()}

    def fake(self, name, outputs, control_ratio=0., external=True, effect=.5, pixels=()):
        """A Retest stand-in: `outputs[variant]` is the identity of a delivered folder."""
        return SimpleNamespace(
            receipts=[[f"retest-{name}-x.zip", "digest"]], dry_run=False, get=lambda n, v: (
                SimpleNamespace(identity=outputs[v]) if v in outputs else None),
            features=lambda n, v: None if outputs.get(v, {}).get("available") is False else (
                Named(v), SimpleNamespace(index={"data/external/OCTID/a.png": 0}) if external else None),
            control=lambda n: dict(images=3, max_difference_ratio=control_ratio, passed=control_ratio <= ph.CONTROL_TOL),
            difference=lambda n, v, paths=None: dict(images=3, worst_image_difference_ratio=effect),
            upstream=lambda: None, pixels=lambda: list(pixels))

    def test_rt1_tables_gate_on_the_control_the_hook_and_availability(self):
        hook = dict(hook_check=dict(max_difference_ratio=0.), repeat_check=dict(images=5, max_difference_ratio=0.))
        outputs = {"R-ref": hook, "R-meanraw": hook, "R-clsnorm": {**hook, "hook_check": dict(max_difference_ratio=.1)},
                   "R-meannorm": dict(available=False, reason="no norm")}
        values = {"locked RETFound": .4, "locked Inception": .5, "R-ref": .4, "R-meanraw": .625, "R-clsnorm": .3}
        tables = self.tables("RT1", self.fake("RT1", outputs), values)
        reading = tables["RT1_reading.csv"]
        states = {(r["variant"], r["statistic"]): r["result"] for r in reading if r["item"].startswith("development")}
        self.assertEqual(states, {("locked", "minus Inception-v3 locked"): "below",
                                  ("R-meanraw", "minus Inception-v3 locked"): "above",
                                  ("R-meanraw", "minus RETFound locked"): "above"})
        self.assertEqual({r["variant"]: r["result"] for r in reading if r["item"] == "not read"},
                         {"R-clsnorm": "read-out check failed; not interpreted", "R-meannorm": ph.RT_NOT_AVAILABLE})
        sentence = next(r["result"] for r in reading if r["item"] == "fixed sentence")
        self.assertIn("depends on the read-out: R-meanraw is above Inception-v3 (mean +0.125", sentence)
        self.assertTrue(any(r["item"].startswith("false-rare") and r["variant"] == "R-meanraw" and "inside" in r["result"]
                            for r in reading))
        development = tables["RT1_development.csv"]
        self.assertEqual({r["status"] for r in development if r["variant"] == "R-ref"}, {ph.RT_CONTROL})
        row = next(r for r in development if (r["variant"], r["quantity"], r["k"]) == ("R-meanraw", "minus RETFound locked", "mean_k1_2_4_8"))
        self.assertEqual((row["n_draws"], round(row["mean"], 6)), (100, .225))
        self.assertEqual({r["direction"] for r in tables["RT1_holdout.csv"] if r["k"] == 16}, {"shown, never read"})
        self.assertFalse(any(r["variant"] == "R-meannorm" for r in development))  # nothing is substituted
        # A failed control: every variant is released as not interpreted and none is read.
        failed = self.tables("RT1", self.fake("RT1", outputs, control_ratio=.01), values)
        self.assertEqual({r["status"] for r in failed["RT1_development.csv"] if r["variant"] == "R-meanraw"}, {ph.NOT_INTERPRETED})
        self.assertEqual(next(r["result"] for r in failed["RT1_reading.csv"] if r["item"] == "fixed sentence"),
                         f"RT1 is not evaluable; R-meanraw: {ph.NOT_INTERPRETED}; R-clsnorm: {ph.NOT_INTERPRETED}; "
                         f"R-meannorm: {ph.RT_NOT_AVAILABLE}.")
        # A read-out that leaves the features unchanged was not applied.
        flat = self.tables("RT1", self.fake("RT1", outputs, effect=0.), values)
        self.assertIn("no effect, change not applied; not interpreted",
                      {r["result"] for r in flat["RT1_reading.csv"] if r["variant"] == "R-meanraw"})

    def test_rt2_tables_read_the_pixel_check_and_skip_the_nine_way_without_external_images(self):
        outputs = {"I-ref": {}, "I-pil": dict(main_only=False)}
        values = {"locked Inception": .5, "I-ref": .5, "I-pil": .5}
        pixels = [dict(source="OCTID", identical="False", max_abs_difference="3")]
        tables = self.tables("RT2", self.fake("RT2", outputs, pixels=pixels), values, {"I-pil": 9})
        reading = tables["RT2_reading.csv"]
        self.assertEqual([r["statistic"] for r in reading if r["item"].startswith("development")], ["minus Inception-v3 locked"])
        outside = [r for r in reading if r["item"].startswith("false-rare")]
        self.assertEqual(len(outside), 3)
        self.assertTrue(all(r["result"].startswith("9/10 (locked 2/10); outside") for r in outside))
        self.assertIn("OCTID: images whose pixels PIL decoding leaves unchanged, of 1",
                      {r["check"] for r in tables["RT2_checks.csv"]})
        sentences = [r["result"] for r in reading if r["item"] == "fixed sentence"]
        self.assertIn("does not separate it from the locked Inception-v3 development K mean", sentences[0])
        self.assertIn("OCTID NORMAL 9/10 (locked 2/10, outside the locked 95% interval)", sentences[1])
        short = self.tables("RT2", self.fake("RT2", outputs, external=False), values)
        self.assertEqual([r for r in short["RT2_nine_way.csv"] if r["variant"] == "I-pil"], [])
        self.assertIn("not scored: OCTID and OCTDL were not extracted", {r["status"] for r in short["RT2_checks.csv"]})
        self.assertEqual([r["result"] for r in short["RT2_reading.csv"] if r["item"] == "fixed sentence"][1],
                         "RT2 nine-way is not scored: OCTID and OCTDL were not extracted.")
        page = ph.rt_report({name: [{"posthoc": "label; dry run", **{k: "" if v is None else str(v) for k, v in row.items()}}
                                    for row in rows] for name, rows in tables.items()})
        self.assertIn("## RT1: RETFound read-out (token and normalisation choice)\n\nNot scored.", page)
        self.assertIn("Label of every row: label; dry run.", page)
        self.assertIn("**Sentence.** PIL decoding alone does not separate it", page)

    def test_dry_run_uses_the_locked_features_as_stand_ins(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            prep.write_csv(root / "extraction_inventory.csv", ["path", "sha256"], [])
            prep.write_csv(root / "external_extraction_inventory.csv", ["path", "sha256"], [])
            study = SimpleNamespace(lock=root, external=root, features=lambda arm: f"locked {arm}",
                                    external_features=lambda arm: f"external {arm}")
            rt = ph.Retest(study, None)
            self.assertTrue(rt.dry_run)
            self.assertEqual(rt.features("RT1", "R-clsnorm"), ("locked RETFound", "external RETFound"))
            self.assertEqual(rt.features("RT2", "I-pil"), ("locked Inception", "external Inception"))
            self.assertEqual((rt.upstream(), rt.pixels()), (None, []))


if __name__ == "__main__":
    unittest.main()
