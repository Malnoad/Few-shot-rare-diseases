"""Pure provenance and notebook wiring checks; numerical gate is benchmark_head.py."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from c0_experiment import FEATURE_COMPATIBLE, compatible_feature_code, guard
from stage3_classifier import lock_trainer


class TrainerTests(unittest.TestCase):
    def test_c0_sweep_lock_rejects_another_trainer_for_new_seeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "training_identity.json"
            guard(path, dict(trainer="fast", scripts={"head": "same"}))
            guard(path, dict(trainer="fast", scripts={"head": "same"}))
            with self.assertRaisesRegex(ValueError, "Provenance mismatch"):
                guard(path, dict(trainer="keras", scripts={"head": "same"}))

    def test_resume_rejects_mixed_trainers_and_protocols_without_overwriting(self):
        with tempfile.TemporaryDirectory() as tmp:
            identity = dict(trainer="fast", epochs=250, restore="last", seed=1)
            lock_trainer(tmp, identity)
            path = Path(tmp) / "trainer.json"
            before = path.read_bytes()
            lock_trainer(tmp, identity)
            for changed in [dict(trainer="keras"), dict(epochs=25), dict(restore="best")]:
                with self.assertRaisesRegex(ValueError, "Trainer/configuration"):
                    lock_trainer(tmp, dict(identity, **changed))
                self.assertEqual(path.read_bytes(), before)

    def test_only_inspected_feature_code_can_resume(self):
        current = dict(scripts={k: "new" for k in FEATURE_COMPATIBLE} | {"stage1b_normalize.py": "recipe"},
                       tensorflow="same", numpy="same")
        old = copy.deepcopy(current)
        old["scripts"].update({k: next(iter(v)) for k, v in FEATURE_COMPATIBLE.items()})
        self.assertTrue(compatible_feature_code(old, current))
        for field in ("stage1b_normalize.py", "stage3_classifier.py"):
            bad = copy.deepcopy(old)
            bad["scripts"][field] = "unknown"
            self.assertFalse(compatible_feature_code(bad, current))
        old["tensorflow"] = "changed"
        self.assertFalse(compatible_feature_code(old, current))

    def test_fast_cannot_overwrite_unlabelled_historical_head(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "head.npz").write_bytes(b"historical")
            with self.assertRaisesRegex(ValueError, "Legacy head"):
                lock_trainer(tmp, dict(trainer="fast"))
            self.assertEqual((Path(tmp) / "head.npz").read_bytes(), b"historical")

    def test_notebook_drive_shortcuts_and_trainer_wiring(self):
        notebook = json.loads(Path("colab_setup.ipynb").read_text(encoding="utf-8"))
        code = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
        setup = next(s for s in code if s.startswith("# 4.1"))
        assertion = next(s for s in setup.splitlines() if s.startswith("assert C1_DRIVE"))
        shortcut = Path("/content/drive/.shortcut-targets-by-id/shared/octfsl")
        import pathlib
        exec(assertion, dict(C1_DRIVE=shortcut / "c1/v1/current", D=str(shortcut), pathlib=pathlib))
        with self.assertRaises(AssertionError):
            exec(assertion, dict(C1_DRIVE=shortcut.parent / "elsewhere", D=str(shortcut), pathlib=pathlib))
        self.assertIn('trainer="fast"', setup)
        self.assertIn('INPUT_ARM = "current"  # D20=A', setup)
        heads = next(s for s in code if s.startswith("# C0.4"))
        self.assertIn('"--trainer", "fast"', heads)


if __name__ == "__main__":
    unittest.main()
