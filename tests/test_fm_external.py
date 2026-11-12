"""External-source audit on tiny synthetic data: no encoder, network or real score."""
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fm_core as fm
import fm_evaluate as ev
import fm_external as ext
import fm_extract as extract
import fm_prepare as prep


class Candidates(unittest.TestCase):
    octid = [dict(doi="doi:10.5683/SP/WLW4ZT", filename="NORMAL1.jpeg", md5="a"),
             dict(doi="doi:10.5683/SP/WLW4ZT", filename="readme.txt", md5="b"),
             dict(doi="doi:10.5683/SP/MBMQGD", filename="MH1.jpeg", md5="c"),
             dict(doi="doi:10.5683/SP/FVLZIL", filename="CSR1.jpeg", md5="d"),
             dict(doi="doi:10.5683/SP/YEM3RA", filename="AMRD1.jpeg", md5="e")]
    octdl = [dict(file_name="no_1_1", disease="NO", subcategory="none", patient_id="1"),
             dict(file_name="vid_2_1", disease="VID", subcategory="MH", patient_id="2"),
             dict(file_name="vid_3_1", disease="VID", subcategory="LH", patient_id="3"),
             dict(file_name="amd_4_1", disease="AMD", subcategory="late", patient_id="4")]

    def test_the_a1_cohorts_are_fixed_in_code_and_octdl_mh_uses_subcategory(self):
        rows = ext.candidates(self.octid, self.octdl)
        self.assertEqual([r[:3] for r in rows], [
            ["OCTDL", "data/external/OCTDL/NO/no_1_1.jpg", "NORMAL"], ["OCTDL", "data/external/OCTDL/VID/vid_2_1.jpg", "MH"],
            ["OCTID", "data/external/OCTID/CSR/CSR1.jpeg", "CSC"], ["OCTID", "data/external/OCTID/MH/MH1.jpeg", "MH"],
            ["OCTID", "data/external/OCTID/NORMAL/NORMAL1.jpeg", "NORMAL"]])  # LH, AMD and non-images never enter
        self.assertEqual({(r[0], r[2]): r[4] for r in rows if r[2] != "NORMAL"}, ext.A1_COHORTS)

    def test_no_command_line_option_can_add_or_drop_a_cohort(self):
        result = subprocess.run([sys.executable, "fm_external.py", "candidates", "--help"], cwd=Path(ext.__file__).parent,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("--mapping", result.stdout)
        lock = dict(external_sources=dict(OCTID=dict(status="planned")))
        normal = dict(source="OCTID", path="data/external/n.png", label="NORMAL", patient_id="", mapping_evidence="", md5="")
        for evidence in ("clinician note", ""):
            rare = dict(normal, path="data/external/r.png", label="CSC", mapping_evidence=evidence)
            with self.subTest(evidence=evidence), self.assertRaisesRegex(ValueError, "A1 cohort"):
                ext.check_candidates([normal, rare], lock, set())
        ext.check_candidates([normal, dict(normal, path="data/external/r.png", label="CSC",
                                           mapping_evidence=ext.A1_COHORTS["OCTID", "CSC"])], lock, set())


class ExternalInputs(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.root, self.lock_folder, self.out = self.base / "root", self.base / "lock", self.base / "external"
        self.lock_folder.mkdir()
        self.rng = np.random.default_rng(0)
        self.locked = {f"data/raw/fit{i}.png": "fitting" for i in range(len(fm.NINE))} | {f"data/raw/query{i}.png": "query" for i in range(2)}
        for path in self.locked:
            self.image(path)
        prep.write_csv(self.lock_folder / "nine_way_roster.csv", ["path", "label", "role"],
                       [[p, fm.NINE[i % len(fm.NINE)], role] for i, (p, role) in enumerate(self.locked.items())])
        prep.write_csv(self.lock_folder / "extraction_inventory.csv", ["path", "sha256"],
                       [[p, prep.sha256(self.root / p)] for p in sorted(self.locked)])
        prep.write_json(self.lock_folder / "input_lock.json", {"synthetic": True})
        self.lock = dict(external_sources=dict(OCTID=dict(status="planned"), OCTDL=dict(status="planned")))
        loader = patch.object(prep, "load_lock", return_value=self.lock)
        loader.start()
        self.addCleanup(loader.stop)
        self.rows = []

    def image(self, path, pixels=None):
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        pixels = self.rng.integers(0, 256, (32, 32, 3), dtype=np.uint8) if pixels is None else pixels
        Image.fromarray(pixels, "RGB").save(target)
        return target

    def add(self, source, name, label="NORMAL", patient="", evidence="", copy_of=None, pixels=None, md5=""):
        path = f"data/external/{source}/{name}.png"
        if copy_of:
            target = self.root / path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(self.root / copy_of, target)
        else:
            self.image(path, pixels)
        self.rows.append([source, path, label, patient, evidence, md5])
        return path

    def both_sources(self):
        return self.add("OCTID", "a"), self.add("OCTDL", "b")

    def near_copy_of(self, path):
        with Image.open(self.root / path) as image:
            pixels = np.array(image)
        pixels[0, 0, 0] ^= 1
        return pixels

    def screen(self):
        csv_path = self.base / "candidates.csv"
        prep.write_csv(csv_path, ext.CANDIDATE_HEADER, self.rows)
        return ext.screen(self.root, self.lock_folder, csv_path, self.out)

    def reviews(self, rows):
        path = self.base / "reviews.csv"
        prep.write_csv(path, ["a", "b", "decision"], rows)
        return path

    def queries(self):
        return prep.read_csv(self.out / "external_queries.csv")

    def exclusions(self):
        return prep.read_csv(self.out / "external_exclusions.csv")


class Screening(ExternalInputs):
    def test_clean_sources_lock_every_query_and_rerun_is_byte_identical(self):
        self.both_sources()
        self.screen()
        identity = ext.freeze(self.root, self.lock_folder, self.out)
        self.assertEqual({k: v["queries"] for k, v in identity["sources"].items()}, {"OCTDL": 1, "OCTID": 1})
        self.assertEqual(identity["input_lock_sha256"], prep.sha256(self.lock_folder / "input_lock.json"))
        self.assertEqual(ext.verify_external(self.out, self.lock_folder)["sources"], identity["sources"])
        first = {p.name: p.read_bytes() for p in self.out.iterdir()}
        ext.freeze(self.root, self.lock_folder, self.out)
        self.assertEqual(first, {p.name: p.read_bytes() for p in self.out.iterdir()})

    def test_copy_of_fitting_row_removes_only_the_external_image(self):
        self.both_sources()
        copied = self.add("OCTID", "copy", copy_of="data/raw/fit0.png")
        roster = (self.lock_folder / "nine_way_roster.csv").read_bytes()
        summary = self.screen()
        self.assertEqual(summary["pairs_by_block"]["external-fitting"]["file"], 1)
        ext.freeze(self.root, self.lock_folder, self.out)
        self.assertNotIn(copied, [q["path"] for q in self.queries()])
        self.assertEqual([(e["path"], e["reason"]) for e in self.exclusions()], [(copied, "linked_to_fitting")])
        self.assertEqual(roster, (self.lock_folder / "nine_way_roster.csv").read_bytes())

    def test_copy_of_legacy_query_is_removed_with_its_own_reason(self):
        self.both_sources()
        self.add("OCTDL", "copy", copy_of="data/raw/query1.png")
        self.screen()
        ext.freeze(self.root, self.lock_folder, self.out)
        self.assertEqual([e["reason"] for e in self.exclusions()], ["linked_to_legacy_query"])

    def test_every_member_of_a_linked_component_is_removed(self):
        self.both_sources()
        one = self.add("OCTID", "one", copy_of="data/raw/fit1.png")
        two = self.add("OCTDL", "two", copy_of="data/raw/fit1.png")
        self.screen()
        ext.freeze(self.root, self.lock_folder, self.out)
        self.assertEqual({e["path"] for e in self.exclusions()}, {one, two})

    def test_near_duplicate_needs_exactly_one_decision(self):
        self.both_sources()
        near = self.add("OCTID", "near", pixels=self.near_copy_of("data/raw/fit2.png"))
        self.screen()
        with self.assertRaisesRegex(ValueError, "no review decision"):
            ext.freeze(self.root, self.lock_folder, self.out)
        ext.freeze(self.root, self.lock_folder, self.out, self.reviews([[near, "data/raw/fit2.png", "rejected"]]))
        self.assertIn(near, [q["path"] for q in self.queries()])

    def test_unresolved_near_duplicate_is_removed_conservatively(self):
        self.both_sources()
        near = self.add("OCTID", "near", pixels=self.near_copy_of("data/raw/fit2.png"))
        self.screen()
        ext.freeze(self.root, self.lock_folder, self.out, self.reviews([[near, "data/raw/fit2.png", "unresolved"]]))
        self.assertEqual([e["path"] for e in self.exclusions()], [near])

    def test_within_query_duplicates_stay_visible_in_one_dependence_group(self):
        a, _ = self.both_sources()
        twin = self.add("OCTID", "twin", copy_of=a)
        self.screen()
        identity = ext.freeze(self.root, self.lock_folder, self.out)
        groups = {q["path"]: q["dependence_group"] for q in self.queries()}
        self.assertEqual(groups[a], groups[twin])
        self.assertEqual(self.exclusions(), [])
        self.assertEqual(identity["sources"]["OCTID"]["queries"], 2)
        self.assertEqual(identity["sources"]["OCTID"]["dependence_groups"], 1)

    def test_shared_patient_groups_rows_without_a_label_conflict(self):
        self.add("OCTID", "a")
        normal = self.add("OCTDL", "b", patient="7")
        rare = self.add("OCTDL", "c", label="MH", patient="7", evidence=ext.A1_COHORTS["OCTDL", "MH"])
        other = self.add("OCTID", "d", patient="7")  # IDs are namespaced by dataset
        self.screen()
        ext.freeze(self.root, self.lock_folder, self.out)
        groups = {q["path"]: q["dependence_group"] for q in self.queries()}
        self.assertEqual(groups[normal], groups[rare])
        self.assertNotEqual(groups[normal], groups[other])

    def test_duplicates_with_conflicting_labels_stop_before_inference(self):
        a, _ = self.both_sources()
        self.add("OCTID", "twin", label="MH", evidence=ext.A1_COHORTS["OCTID", "MH"], copy_of=a)
        self.screen()
        with self.assertRaisesRegex(ValueError, "conflicting labels"):
            ext.freeze(self.root, self.lock_folder, self.out)

    def test_rows_outside_the_frozen_decisions_stop_before_reading_images(self):
        cases = {
            "not planned": ["Other", "data/external/x.png", "NORMAL", "", "", ""],
            "eligible external class": ["OCTID", "data/external/x.png", "CNV", "", "", ""],
            "A1 cohort": ["OCTID", "data/external/x.png", "MH", "", " ", ""],
            "outside the locked roster": ["OCTID", "data/raw/fit0.png", "NORMAL", "", "", ""],
        }
        for message, row in cases.items():
            rows = [dict(zip(ext.CANDIDATE_HEADER, r)) for r in (
                ["OCTID", "data/external/a.png", "NORMAL", "", "", ""], ["OCTDL", "data/external/b.png", "NORMAL", "", "", ""], row)]
            with self.subTest(message), self.assertRaisesRegex(ValueError, message):
                ext.check_candidates(rows, self.lock, set(self.locked))

    def test_planned_source_without_normal_rows_stops(self):
        self.add("OCTID", "a")
        with self.assertRaisesRegex(ValueError, "no NORMAL rows"):
            self.screen()

    def test_unavailable_source_is_not_required(self):
        self.lock["external_sources"]["OCTDL"] = dict(status="unavailable", reason="access failed")
        self.add("OCTID", "a")
        self.screen()
        self.assertEqual(list(ext.freeze(self.root, self.lock_folder, self.out)["sources"]), ["OCTID"])

    def test_source_with_no_normal_left_after_screening_stops(self):
        self.add("OCTID", "a")
        self.add("OCTDL", "copy", copy_of="data/raw/fit0.png")
        self.screen()
        with self.assertRaisesRegex(ValueError, "no NORMAL query after screening"):
            ext.freeze(self.root, self.lock_folder, self.out)

    def test_wrong_published_md5_and_changed_locked_image_stop(self):
        self.both_sources()
        self.rows[0][5] = "0" * 32
        with self.assertRaisesRegex(ValueError, "published MD5"):
            self.screen()
        self.rows[0][5] = ""
        (self.base / "candidates.csv").unlink()
        self.image("data/raw/fit3.png")
        with self.assertRaisesRegex(ValueError, "locked image changed"):
            self.screen()

    def test_freeze_needs_the_screening_decoder_and_input_lock(self):
        self.both_sources()
        self.screen()
        changed = {**prep.versions(), "pillow": "other"}
        with patch.object(prep, "versions", return_value=changed), self.assertRaisesRegex(ValueError, "decoder"):
            ext.freeze(self.root, self.lock_folder, self.out)
        (self.lock_folder / "input_lock.json").write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "changed since"):
            ext.freeze(self.root, self.lock_folder, self.out)

    def locked_folder(self):
        self.both_sources()
        self.add("OCTID", "copy", copy_of="data/raw/fit0.png")
        self.screen()
        return ext.freeze(self.root, self.lock_folder, self.out)

    def reseal(self, identity):
        """Forge a consistent hash trail, so only the evidence check can catch the edit."""
        identity["files"]["external_queries.csv"] = prep.sha256(self.out / "external_queries.csv")
        (self.out / "external_lock.json").write_text(prep.json.dumps(identity), encoding="utf-8")
        names = [*identity["files"], "external_lock.json"]
        (self.out / "complete.json").write_text(prep.json.dumps(
            {"files": {name: prep.sha256(self.out / name) for name in names}}), encoding="utf-8")

    def test_edited_query_list_fails_even_with_a_forged_hash_trail(self):
        identity = self.locked_folder()
        path = self.out / "external_queries.csv"
        text = path.read_text(encoding="utf-8").splitlines()
        path.write_text("\n".join([*text, "OCTID,data/raw/fit0.png,NORMAL," + "0" * 64 + ",x"]) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            ext.verify_external(self.out, self.lock_folder)
        self.reseal(identity)
        with self.assertRaisesRegex(ValueError, "contradict"):
            ext.verify_external(self.out, self.lock_folder)

    def test_query_lock_is_bound_to_one_input_lock(self):
        self.locked_folder()
        (self.lock_folder / "input_lock.json").write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "different input lock"):
            ext.verify_external(self.out, self.lock_folder)


class RealQueryLock(ExternalInputs):
    def test_a_real_query_lock_drives_a_complete_report(self):
        self.add("OCTID", "a")
        self.add("OCTDL", "b")
        rare = self.add("OCTID", "c", label="CSC", evidence=ext.A1_COHORTS["OCTID", "CSC"])
        removed = self.add("OCTDL", "copy", copy_of="data/raw/fit0.png")
        self.screen()
        ext.freeze(self.root, self.lock_folder, self.out)
        self.lock.update(evaluation_environment=prep.math_environment(), arms={"DINOv2": dict(status="available")})
        fitting = [f"data/raw/fit{i}.png" for i in range(9)]
        main = ev.Features(np.eye(9, dtype=np.float32), {p: i for i, p in enumerate(fitting)})
        queries = [q["path"] for q in prep.read_csv(self.out / "external_queries.csv")]
        self.assertNotIn(removed, queries)
        other = ev.Features(np.tile(np.eye(9, dtype=np.float32)[4], (len(queries), 1)), {p: i for i, p in enumerate(queries)})
        planned = [ev.Episode(0, 1, 0, tuple((p,) for p in fitting[4:]), ("legacy",), ("CSC",))]
        nomination, cache, external_cache = self.base / "nomination", self.base / "cache", self.base / "external_cache"
        for folder in (nomination, cache, external_cache):
            folder.mkdir()
            prep.write_json(folder / "nomination.json", {"synthetic": True})
            prep.write_json(folder / "complete.json", {"synthetic": True})
        out = self.base / "run"
        with patch.object(ev, "nomination_decision"), patch.object(ev, "episodes", return_value=planned), \
                patch.object(ev, "open_features", return_value=main), patch.object(ext, "external_features", return_value=other), \
                patch.object(extract, "environment", return_value={"mock": True}):
            ext.run_report(self.lock_folder, cache, external_cache, self.out, "DINOv2", nomination, out)
        prep.verify_complete(out)
        audit = prep.read_json(out / "OCTID" / "nine_way" / "source_audit.json")
        self.assertEqual((audit["false_rare_n"], audit["normal_n"], audit["dependence_groups"]), (1, 1, 2))
        self.assertEqual(prep.read_csv(out / "OCTID" / "prototype" / "fold0_k01_draw000" / "predictions.csv")[0]["query_id"], rare)
        self.assertFalse((out / "OCTDL" / "prototype").exists())


class Extraction(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        base = Path(self.tmp.name)
        self.root, self.folder, self.external, self.out = base / "root", base / "lock", base / "external", base / "features"
        for folder in (self.root, self.folder, self.external):
            folder.mkdir()
        self.rows = []
        for i in range(2):
            path = self.root / f"external{i}.png"
            Image.new("RGB", (300, 260), (i + 1, 2, 3)).save(path)
            self.rows.append(dict(path=path.name, sha256=prep.sha256(path)))
        prep.write_csv(self.folder / "extraction_inventory.csv", ["path", "sha256"], [["main.png", "0" * 64]])
        prep.write_csv(self.external / "external_extraction_inventory.csv", ["path", "sha256"], [[r["path"], r["sha256"]] for r in self.rows])
        prep.write_json(self.folder / "freeze_summary.json", {"files": {}})
        prep.write_json(self.folder / "input_lock.json", {"synthetic": True})
        prep.write_json(self.external / "external_lock.json", {"synthetic": True})
        self.spec = dict(arm="DINOv2")
        self.current = dict(spec=self.spec, code={"mock": "code"}, environment={"mock": "environment"})
        self.lock = dict(arms={"DINOv2": dict(status="available", identity=self.current)})
        self.encoder = extract.Encoder("DINOv2", MagicMock(side_effect=lambda images: np.full((1, 768), 3 + images[0].getpixel((0, 0))[0], dtype=np.float32)),
                                       lambda: None, {"mock": True}, lambda: dict(peak_vram_bytes=1))
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        for name, kwargs in (("require_colab", {}), ("runtime_out", dict(side_effect=lambda p: p)), ("check_spec", {}),
                             ("prepared", dict(return_value={"files": {}})), ("environment", dict(return_value=self.current["environment"])),
                             ("hardware", dict(return_value={"gpu": "mock"}))):
            self.stack.enter_context(patch.object(extract, name, **kwargs))
        self.stack.enter_context(patch.object(prep, "code_identity", return_value=self.current["code"]))
        self.stack.enter_context(patch.object(prep, "load_lock", return_value=self.lock))
        self.loader = self.stack.enter_context(patch.object(extract, "load_encoder", return_value=self.encoder))
        self.verify = self.stack.enter_context(patch.object(ext, "verify_external"))

    def extract(self):
        extract.run("extract", self.root, self.folder, self.spec, self.root, self.out, "", self.external)

    def test_only_locked_queries_are_extracted_under_both_locks(self):
        self.extract()
        self.verify.assert_called_once_with(self.external, self.folder)
        values = extract.load_features(self.out / "features.npz", self.rows, 768)
        np.testing.assert_array_equal(values[:, 0], [4, 5])
        identity = prep.read_json(self.out / "identity.json")
        self.assertEqual(identity["external_lock_sha256"], prep.sha256(self.external / "external_lock.json"))
        features = ext.external_features(self.folder, self.out, self.external, "DINOv2")
        self.assertEqual(sorted(features.index), [r["path"] for r in self.rows])

    def test_failed_query_lock_stops_before_any_model_loads(self):
        self.verify.side_effect = ValueError("bad query lock")
        with self.assertRaisesRegex(ValueError, "bad query lock"):
            self.extract()
        self.loader.assert_not_called()

    def test_pilot_cannot_touch_external_queries(self):
        with self.assertRaisesRegex(ValueError, "only under"):
            extract.run("pilot", self.root, self.folder, self.spec, self.root, self.out, "session", self.external)
        self.loader.assert_not_called()

    def test_changed_environment_stops_external_extraction(self):
        self.lock["arms"]["DINOv2"]["identity"] = {**self.current, "environment": {"other": True}}
        with self.assertRaisesRegex(ValueError, "differs from the input lock"):
            self.extract()
        self.loader.assert_not_called()

    def test_features_from_another_query_lock_are_rejected(self):
        self.extract()
        (self.external / "external_lock.json").write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "input/query locks"):
            ext.external_features(self.folder, self.out, self.external, "DINOv2")

    def test_unavailable_arm_is_rejected(self):
        self.extract()
        self.lock["arms"]["DINOv2"] = dict(status="unavailable", reason="x")
        with self.assertRaisesRegex(ValueError, "unavailable"):
            ext.external_features(self.folder, self.out, self.external, "DINOv2")


class Reporting(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        base = Path(self.tmp.name)
        self.base = base
        self.folder, self.external, self.nomination = base / "lock", base / "external", base / "nomination"
        self.cache, self.external_cache, self.runs = base / "cache", base / "external_cache", base / "runs"
        for folder in (self.folder, self.external, self.nomination, self.cache, self.external_cache):
            folder.mkdir()
        prep.write_csv(self.folder / "nine_way_roster.csv", ["path", "label", "role"], [[f"fit/{c}", c, "fitting"] for c in fm.NINE])
        prep.write_json(self.folder / "input_lock.json", {"synthetic": True})
        prep.write_json(self.external / "external_lock.json", {"synthetic": True})
        prep.write_json(self.nomination / "nomination.json", {"synthetic": True})
        for folder in (self.cache, self.external_cache):
            prep.write_json(folder / "complete.json", {"synthetic": True})
        queries = [("OCTID", "q/false_rare", "NORMAL", "RP"), ("OCTID", "q/common", "NORMAL", "CNV"), ("OCTID", "q/normal", "NORMAL", "NORMAL"),
                   ("OCTID", "q/csc", "CSC", "CSC"), ("OCTID", "q/mh_wrong", "MH", "RP"), ("OCTDL", "q/octdl", "NORMAL", "NORMAL")]
        self.targets = {path: target for _, path, _, target in queries}
        prep.write_csv(self.external / "external_queries.csv", ["source", "path", "label", "sha256", "dependence_group"],
                       [[s, p, c, "0" * 64, p] for s, p, c, _ in queries])
        # Original sizes for the size-only reference: OCTID's CSC/MH queries match their class's supports.
        prep.write_csv(self.folder / "image_inventory.csv", ["path", "width", "height"],
                       [[f"fit/{c}", 300 + 50 * i, 300] for i, c in enumerate(fm.RARE)])
        prep.write_csv(self.external / "external_inventory.csv", ["path", "width", "height"],
                       [[p, {"q/csc": 310, "q/mh_wrong": 345}.get(p, 512), 300] for _, p, _, _ in queries])
        eye = np.eye(9, dtype=np.float32)
        supports = [f"fit/{c}" for c in fm.RARE]
        self.main = ev.Features(eye, {f"fit/{c}": i for i, c in enumerate(fm.NINE)})
        self.planned = [ev.Episode(0, k, draw, tuple((p,) * k for p in supports), ("legacy",), ("CSC",))
                        for k in (1,) for draw in range(30)]
        self.lock = dict(evaluation_environment=prep.math_environment(), arms={
            "DINOv2": dict(status="available"), "MIRAGE": dict(status="available"), "Inception": dict(status="available")})
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(prep, "load_lock", return_value=self.lock))
        self.decide = self.stack.enter_context(patch.object(ev, "nomination_decision", return_value=dict(nominee="DINOv2", reporting_scope="primary nominee plus all secondary arms")))
        self.stack.enter_context(patch.object(ext, "verify_external"))
        self.stack.enter_context(patch.object(ev, "episodes", return_value=self.planned))
        self.open_main = self.stack.enter_context(patch.object(ev, "open_features", return_value=self.main))
        self.open_external = self.stack.enter_context(patch.object(ext, "external_features", side_effect=lambda *a: self.queries_features()))
        self.stack.enter_context(patch.object(extract, "environment", return_value={"mock": True}))

    def queries_features(self):
        paths = list(self.targets)
        values = np.stack([np.eye(9, dtype=np.float32)[fm.NINE.index(self.targets[p])] for p in paths])
        return ev.Features(values, {p: i for i, p in enumerate(paths)})

    def report(self, arm="DINOv2", reason=""):
        out = self.runs / arm / "external"
        ext.run_report(self.folder, self.cache, self.external_cache, self.external, arm, self.nomination, out, reason)
        return out

    def test_false_rare_rate_is_per_source_and_uses_nine_outputs(self):
        out = self.report()
        octid = prep.read_json(out / "OCTID" / "nine_way" / "source_audit.json")
        self.assertEqual((octid["false_rare_n"], octid["normal_n"]), (1, 3))
        self.assertEqual(octid["class_counts"], {"NORMAL": dict(correct=1, total=3), "CSC": dict(correct=1, total=1), "MH": dict(correct=0, total=1)})
        octdl = prep.read_json(out / "OCTDL" / "nine_way" / "source_audit.json")
        self.assertEqual((octdl["false_rare_n"], octdl["normal_n"]), (0, 1))
        rows = prep.read_csv(out / "OCTID" / "nine_way" / "predictions.csv")
        self.assertTrue(all(f"cosine_{c}" in rows[0] for c in fm.NINE))
        self.assertIsNone(prep.read_json(out / "OCTID" / "nine_way" / "metrics.json")["macro_recall"])

    def test_transfer_keeps_all_five_outputs_and_never_reports_macro(self):
        out = self.report()
        cell = out / "OCTID" / "prototype" / "fold0_k01_draw000"
        metric = prep.read_json(cell / "metrics.json")
        self.assertIsNone(metric["macro_recall"])
        self.assertEqual(metric["totals"], [1, 1, 0, 0, 0])
        self.assertEqual(metric["confusion"][1], [0, 0, 0, 0, 1])
        rows = prep.read_csv(cell / "predictions.csv")
        self.assertEqual([r["query_id"] for r in rows], ["q/csc", "q/mh_wrong"])
        self.assertTrue(all(f"cosine_{c}" in rows[0] for c in fm.RARE))
        self.assertFalse(any(f"cosine_{c}" in rows[0] for c in fm.NINE[:4]))

    def test_logistic_runs_for_the_first_25_draws_and_normal_only_sources_have_no_transfer(self):
        out = self.report()
        self.assertEqual(len(list((out / "OCTID" / "prototype").iterdir())), 30)
        self.assertEqual(len(list((out / "OCTID" / "logistic").iterdir())), 25)
        self.assertFalse((out / "OCTDL" / "prototype").exists())
        self.assertEqual(prep.read_json(out / "summary.json")["sources"]["OCTID"]["episodes"], dict(prototype=[30, 0], logistic=[25, 0]))

    def test_disclosed_omission_skips_only_the_secondary_probe(self):
        out = self.report(reason="time limit, disclosed")
        self.assertTrue((out / "OCTID" / "prototype").exists())
        self.assertFalse((out / "OCTID" / "logistic").exists())

    def test_nominee_must_be_sealed_before_any_feature_is_opened(self):
        self.decide.side_effect = ValueError("missing binding")
        with self.assertRaisesRegex(ValueError, "missing binding"):
            self.report()
        self.open_main.assert_not_called()
        self.open_external.assert_not_called()

    def test_changed_arithmetic_environment_stops(self):
        self.lock["evaluation_environment"] = dict(numpy="other")
        with self.assertRaisesRegex(ValueError, "differs"):
            self.report()
        self.open_external.assert_not_called()

    def test_overlapping_inventories_stop(self):
        with self.assertRaisesRegex(ValueError, "overlap"):
            ext.merge(self.main, ev.Features(np.eye(9, dtype=np.float32)[:1], {"fit/NORMAL": 0}))

    def test_resume_verifies_and_keeps_every_output(self):
        out = self.report()
        before = {p: p.read_bytes() for p in out.rglob("*") if p.is_file()}
        self.report()
        self.assertEqual(before, {p: p.read_bytes() for p in out.rglob("*") if p.is_file()})
        (out / "OCTID" / "nine_way" / "metrics.json").write_text("{}", encoding="utf-8")
        with self.assertRaises(ValueError):
            self.report()

    def summarize(self):
        out = self.base / "tables"
        ext.summarize(self.folder, self.external, self.runs, self.nomination, out)
        return out

    def test_summary_is_per_source_present_class_with_a_primary_contrast(self):
        self.report("DINOv2")
        self.targets["q/csc"] = "MH"
        self.report("Inception")
        out = self.summarize()
        audit = prep.read_csv(out / "external_source_audit.csv")
        frr = {(r["arm"], r["source"]): (r["numerator"], r["denominator"]) for r in audit if r["endpoint"] == "false_rare_rate"}
        self.assertEqual(frr["DINOv2", "OCTID"], ("1", "3"))
        self.assertEqual(frr["Inception", "OCTDL"], ("0", "1"))
        drawn = prep.read_csv(out / "external_transfer_summary.csv")
        self.assertEqual({r["class"] for r in drawn}, {"CSC", "MH"})
        self.assertEqual({r["source"] for r in drawn}, {"OCTID"})
        prototype = next(r for r in drawn if (r["arm"], r["classifier"], r["class"]) == ("DINOv2", "prototype", "CSC"))
        self.assertEqual((prototype["available"], prototype["planned"], float(prototype["mean"])), ("30", "100", 1.))
        contrast = next(r for r in prep.read_csv(out / "external_paired_differences.csv")
                        if (r["arm_a"], r["arm_b"], r["classifier"], r["class"]) == ("DINOv2", "Inception", "prototype", "CSC"))
        self.assertEqual((contrast["role"], float(contrast["mean_a_minus_b"]), contrast["available_pairs"]), ("primary", 1., "30"))
        logistic = next(r for r in prep.read_csv(out / "external_paired_differences.csv") if r["classifier"] == "logistic")
        self.assertEqual((logistic["role"], logistic["planned_pairs"]), ("secondary", "25"))
        outputs = [r for r in prep.read_csv(out / "external_transfer_outputs.csv")
                   if (r["arm"], r["classifier"], r["true_class"]) == ("Inception", "prototype", "CSC")]
        self.assertEqual([r["predicted_class"] for r in outputs], list(fm.RARE))
        self.assertEqual(float(next(r for r in outputs if r["predicted_class"] == "MH")["mean_count"]), 1.)
        self.assertEqual(prep.read_json(out / "interpretation.json")["incomplete_arms"], ["MIRAGE"])
        # Only the two-class OCTID test has a macro; partial five-class cohorts never do.
        text = "".join(p.read_text(encoding="utf-8") for p in out.iterdir() if p.suffix == ".csv" and p.name != "external_two_way.csv")
        self.assertNotIn("macro", text)
        prep.verify_complete(out)

    def test_octid_two_way_uses_the_csc_and_mh_cosines_with_a_size_only_row(self):
        self.report("DINOv2")
        self.targets["q/csc"] = "MH"
        self.report("Inception")
        rows = {(r["arm"], r["k"]): r for r in prep.read_csv(self.summarize() / "external_two_way.csv")}
        # DINOv2: CSC right; the MH query's RP output ties CSC and MH at 0 -> CSC. Every query called CSC.
        dino = rows["DINOv2", "1"]
        self.assertEqual((dino["role"], dino["available"], dino["planned"], dino["n_CSC"], dino["n_MH"]),
                         ("headline", "30", "100", "1", "1"))
        self.assertEqual((float(dino["macro_mean"]), float(dino["MH_as_CSC"]), dino["collapsed_draws"]), (.5, 1., "30"))
        reference = rows["Inception", "1"]
        self.assertEqual((reference["role"], float(reference["macro_mean"]), reference["collapsed_draws"]), ("secondary", 0., "0"))
        size = rows["size_only_reference", "1"]
        self.assertEqual((size["role"], size["available"], float(size["macro_mean"])), ("size_only_reference", "30", 1.))
        self.assertEqual(rows["DINOv2", ev.K_MEAN_LABEL]["available"], "0")  # only K=1 was planned here

    def test_summary_rejects_cells_built_on_other_supports(self):
        self.report()
        changed = [replace(e, supports=tuple(reversed(e.supports))) for e in self.planned]
        with patch.object(ev, "episodes", return_value=changed), self.assertRaisesRegex(ValueError, "identity differs"):
            self.summarize()

    def test_failed_logistic_fit_stays_missing_and_is_counted(self):
        real, calls = fm.fit_logistic, []

        def first_fails(*args, **kwargs):
            calls.append(1)
            fit = real(*args, **kwargs)
            return fit if len(calls) > 1 else fm.LogisticFit(fit.W, fit.b, False, fit.objective, fit.max_abs_gradient, "iteration limit")

        with patch.object(fm, "fit_logistic", side_effect=first_fails):
            out = self.report()
        self.assertEqual(prep.read_json(out / "summary.json")["sources"]["OCTID"]["episodes"]["logistic"], [24, 1])
        row = next(r for r in prep.read_csv(self.summarize() / "external_transfer_summary.csv") if (r["classifier"], r["class"]) == ("logistic", "CSC"))
        self.assertEqual((row["available"], row["planned"]), ("24", "25"))

    def legacy_run(self, arm):
        run = self.runs / arm / "reporting"
        (run / "nine_way").mkdir(parents=True)
        identity = dict(arm=arm, phase="reporting", lock_sha256=prep.sha256(self.folder / "input_lock.json"),
                        nomination_sha256=prep.sha256(self.nomination / "nomination.json"), code=prep.code_identity())
        prep.write_json(run / "identity.json", identity)
        prep.write_json(run / "nine_way" / "source_audit.json", dict(
            false_rare_n=2, normal_n=250, false_rare_rate=.008,
            intervals=dict(method=ev.CLUSTER_METHOD, false_rare=[.0, .02], recall=dict(NORMAL=[.93, .98], CSC=[.2, 1.])),
            patient_disjoint=dict(false_rare_n=1, normal_n=169, false_rare_rate=1 / 169, interval=[0., .02])))
        prep.write_json(run / "nine_way" / "metrics.json", dict(classes=list(fm.NINE), correct=[240, 0, 0, 0, 3, 0, 0, 0, 0], totals=[250, 0, 0, 0, 5, 0, 0, 0, 0]))
        prep.complete(run, ["identity.json", "nine_way/source_audit.json", "nine_way/metrics.json"])

    def test_source_table_puts_kermany_beside_each_external_source(self):
        self.report()
        self.legacy_run("DINOv2")
        audit = prep.read_csv(self.summarize() / "external_source_audit.csv")
        kermany = [r for r in audit if r["source"] == "Kermany"]
        self.assertEqual([(r["endpoint"], r["numerator"], r["denominator"]) for r in kermany],
                         [("false_rare_rate_patient_disjoint", "1", "169"), ("false_rare_rate_all", "2", "250"),
                          ("recall_NORMAL", "240", "250"), ("recall_CSC", "3", "5")])
        self.assertEqual((kermany[0]["interval_high"], kermany[2]["interval_low"]), ("0.02", "0.93"))
        octid = next(r for r in audit if (r["source"], r["endpoint"]) == ("OCTID", "false_rare_rate"))
        self.assertEqual(octid["interval_method"], ev.EXACT_METHOD)  # A2 B-3: no patient IDs in OCTID
        octdl = next(r for r in audit if (r["source"], r["endpoint"]) == ("OCTDL", "false_rare_rate"))
        self.assertEqual(octdl["interval_method"], ev.CLUSTER_METHOD)
        self.assertEqual(prep.read_json(self.base / "tables" / "interpretation.json")["reporting_scope"], "primary nominee plus all secondary arms")

    def test_legacy_run_from_another_lock_cannot_enter_the_source_table(self):
        self.report()
        self.legacy_run("DINOv2")
        run = self.runs / "DINOv2" / "reporting"
        identity = prep.read_json(run / "identity.json")
        identity["lock_sha256"] = "0" * 64
        (run / "identity.json").write_text(prep.json.dumps(identity), encoding="utf-8")
        (run / "complete.json").unlink()
        prep.complete(run, ["identity.json", "nine_way/source_audit.json", "nine_way/metrics.json"])
        with self.assertRaisesRegex(ValueError, "legacy run belongs"):
            self.summarize()

    def test_summary_rejects_a_run_from_another_nomination(self):
        self.report("DINOv2")
        (self.nomination / "nomination.json").write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "different lock/nomination/code"):
            self.summarize()

    def test_summary_without_completed_runs_stops(self):
        with self.assertRaisesRegex(ValueError, "no completed external runs"):
            self.summarize()


if __name__ == "__main__":
    unittest.main()
