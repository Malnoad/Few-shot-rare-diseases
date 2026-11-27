"""Input-preparation gates on a tiny synthetic dataset (FM_PROTOCOL.md sections 4, 6, 8)."""
import csv
from copy import deepcopy
import hashlib
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import zipfile

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fm_core as fm
import fm_prepare as prep


class SyntheticInputs(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "root"
        self.out = Path(self.tmp.name) / "out"
        self.rng = np.random.default_rng(0)
        self.rows = []
        for label in fm.RARE:
            for i in range(12):
                self.add(label, "train", i)
            self.add(label, "test", 99)
        for label in fm.NINE[:4]:
            for i in range(3):
                self.add(label, "train", i)
            self.add(label, "test", 99)
        self.write_manifest()

    def path(self, label, split, i):
        return f"data/raw/{label}/{split}_{i:02d}.png"

    def add(self, label, split, i, pixels=None):
        path = self.path(label, split, i)
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        pixels = self.rng.integers(0, 256, (32, 32, 4), dtype=np.uint8) if pixels is None else pixels
        Image.fromarray(pixels, "RGBA").save(target)
        self.rows.append([path.replace("/", "\\"), label, f"{label}_{i}", split])
        return target

    def csv_fixture(self, name, header, rows):
        path = Path(self.tmp.name) / name
        with path.open("w", newline="", encoding="utf-8") as stream:
            csv.writer(stream).writerows([header, *rows])
        return path

    def patients(self, rows):
        return self.csv_fixture("patients.csv", ["path", "dataset", "patient_id", "evidence"], rows)

    def json_fixture(self, path, value):
        path.write_text(json.dumps(value, sort_keys=True, allow_nan=False), encoding="utf-8")

    def digest(self, path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def write_manifest(self):
        with (self.root / "data/manifest.csv").open("w", newline="", encoding="utf-8") as stream:
            csv.writer(stream).writerows([["path", "label", "pid", "split"], *self.rows])

    def read(self, name):
        return prep.read_csv(self.out / name)

    def run_both(self, reviews=None, patients=None):
        prep.screen(self.root, self.out)
        return prep.freeze(self.root, self.out, reviews, patients)

    def near_copy(self):
        with Image.open(self.root / self.path("RP", "train", 0)) as image:
            pixels = np.array(image)
        pixels[0, 0, 0] ^= 1
        self.add("RP", "train", 51, pixels)
        self.write_manifest()
        prep.screen(self.root, self.out)
        pairs = self.read("screen_pairs.csv")
        self.assertEqual(len(pairs), 1, "fixture must have exactly one reviewed candidate")
        self.assertEqual(pairs[0]["kind"], "dhash")
        return pairs[0]


class Prepare(SyntheticInputs):

    def test_clean_run_writes_full_rosters(self):
        result = self.run_both()
        self.assertEqual(set(result["groups_per_class"].values()), {12})
        self.assertEqual(result["report_k"], [1, 2, 4, 8])
        development = self.read("episodes_development.csv")
        self.assertEqual(len(development), 5 * 5 * 100)
        self.assertTrue(all(row["s08"] for row in development))
        roles = [r["role"] for r in self.read("nine_way_roster.csv")]
        self.assertEqual((roles.count("fitting"), roles.count("query")), (12 + 60, 9))

    def test_screen_publishes_pair_counts_for_every_split_block_including_zeros(self):
        prep.screen(self.root,self.out)
        clean=prep.read_json(self.out / "screen_summary.json")
        self.assertEqual(clean["pairs_by_block"],{block:{kind:0 for kind in ("file","rgb","dhash")}
                                               for block in ("fitting-fitting","fitting-legacy","legacy-legacy")})
        shutil.copy(self.root/self.path("CSC","train",0),self.root/self.path("CSC","train",1))
        shutil.copy(self.root/self.path("CSC","test",99),self.root/self.path("CSC","train",2))
        with Image.open(self.root/self.path("CSC","test",99)) as image:
            self.add("CSC","test",100,np.array(image))
        self.write_manifest()
        self.out=Path(self.tmp.name)/"block-counts"
        summary=prep.screen(self.root,self.out)
        self.assertEqual(summary["pairs_by_block"],{
            "fitting-fitting":{"file":1,"rgb":0,"dhash":0},
            "fitting-legacy":{"file":2,"rgb":0,"dhash":0},
            "legacy-legacy":{"file":1,"rgb":0,"dhash":0}})
        self.assertEqual(summary["pairs"],{"file":4,"rgb":0,"dhash":0})

    def test_rerun_is_byte_identical(self):
        manifest = (self.root / "data/manifest.csv").read_bytes()
        self.run_both()
        first = {p.name: p.read_bytes() for p in self.out.iterdir()}
        self.run_both()  # unchanged retries must succeed too
        self.assertEqual(first, {p.name: p.read_bytes() for p in self.out.iterdir()})
        self.out = Path(self.tmp.name) / "second"
        self.run_both()
        self.assertEqual(first, {p.name: p.read_bytes() for p in self.out.iterdir()})
        self.assertEqual((self.root / "data/manifest.csv").read_bytes(), manifest)

    def test_changed_output_is_not_overwritten(self):
        self.run_both()
        target = self.out / "groups.csv"
        target.write_bytes(target.read_bytes() + b"tampered\n")
        with self.assertRaises(ValueError):
            prep.freeze(self.root, self.out)

    def test_alpha_only_copy_shares_a_group(self):
        original = self.root / self.path("CSC", "train", 0)
        with Image.open(original) as image:
            pixels = np.array(image)
        pixels[..., 3] = 7
        self.add("CSC", "train", 50, pixels)
        self.write_manifest()
        self.assertEqual(self.run_both()["groups_per_class"]["CSC"], 12)
        self.assertEqual(self.read("groups.csv")[0]["member"], self.path("CSC", "train", 0))

    def test_fitting_copy_of_a_legacy_query_is_excluded_but_test_stays(self):
        shutil.copy(self.root / self.path("MH", "test", 99), self.root / self.path("MH", "train", 0))
        result = self.run_both()
        self.assertEqual(result["excluded_rows"], 1)
        self.assertEqual(result["groups_per_class"]["MH"], 11)
        self.assertEqual(self.read("exclusions.csv")[0]["path"], self.path("MH", "train", 0))
        self.assertEqual(result["query_rows"], 9)
        self.assertNotIn(self.path("MH","train",0),[r["path"] for r in self.read("extraction_inventory.csv")])

    def test_flagged_pairs_need_a_decision_and_rejected_stay_apart(self):
        pair = self.near_copy()
        with self.assertRaises(ValueError):
            prep.freeze(self.root, self.out)
        reviews = self.csv_fixture("reviews.csv", ["a", "b", "decision"], [[pair["a"], pair["b"], "rejected"]])
        self.assertEqual(prep.freeze(self.root, self.out, reviews)["groups_per_class"]["RP"], 13)

    def test_screen_artifact_tampering_stops_before_rosters(self):
        prep.screen(self.root, self.out)
        for name in ("image_inventory.csv", "screen_pairs.csv"):
            with self.subTest(artifact=name):
                target = self.out / name
                original = target.read_bytes()
                target.write_bytes(original + b"\n")
                with self.assertRaisesRegex(ValueError, "hash mismatch"):
                    prep.freeze(self.root, self.out)
                self.assertFalse((self.out / "groups.csv").exists())
                target.write_bytes(original)

    def test_changed_image_bytes_stop_even_when_decoded_pixels_are_unchanged(self):
        prep.screen(self.root, self.out)
        target = self.root / self.path("CSC", "train", 0)
        with Image.open(target) as image:
            original_pixels = image.tobytes()
        target.write_bytes(target.read_bytes() + b"synthetic trailing bytes")
        with Image.open(target) as image:
            self.assertEqual(image.tobytes(), original_pixels)
        with self.assertRaisesRegex(ValueError, "image changed"):
            prep.freeze(self.root, self.out)
        self.assertFalse((self.out / "groups.csv").exists())

    def test_manifest_change_stops_even_when_image_paths_and_labels_are_unchanged(self):
        prep.screen(self.root, self.out)
        self.rows[0][2] = "changed-patient-metadata"
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "manifest changed"):
            prep.freeze(self.root, self.out)
        self.assertFalse((self.out / "groups.csv").exists())

    def test_crlf_conversion_does_not_change_the_manifest_identity(self):
        prep.screen(self.root, self.out)
        manifest = self.root / "data/manifest.csv"
        manifest.write_bytes(manifest.read_bytes().replace(b"\r\n", b"\n"))
        self.assertEqual(prep.freeze(self.root, self.out)["groups_per_class"]["CSC"], 12)

    def test_decoder_environment_change_stops_before_rosters(self):
        prep.screen(self.root, self.out)
        with patch.object(prep, "versions", return_value={**prep.versions(), "pillow": "changed"}):
            with self.assertRaisesRegex(ValueError, "environment changed"):
                prep.freeze(self.root, self.out)
        self.assertFalse((self.out / "groups.csv").exists())

    def test_rehashed_inventory_cannot_omit_a_manifest_image(self):
        prep.screen(self.root, self.out)
        target = self.out / "image_inventory.csv"
        target.write_bytes(b"\n".join(target.read_bytes().splitlines()[:-1]) + b"\n")
        summary = prep.read_json(self.out / "screen_summary.json")
        summary["files"]["image_inventory.csv"] = self.digest(target)
        self.json_fixture(self.out / "screen_summary.json", summary)
        with self.assertRaisesRegex(ValueError, "inventory differs"):
            prep.freeze(self.root, self.out)

    def test_screen_uses_inclusive_four_bit_candidate_boundary(self):
        # Native 9x8 grayscale makes each chosen horizontal comparison one dHash bit.
        for index, descending in enumerate((0, 4, 5), start=50):
            levels = [128]
            for column in range(8):
                levels.append(levels[-1] + (-12 if column < descending else 12))
            pixels = np.full((8, 9, 4), 255, dtype=np.uint8)
            gray = np.tile(np.arange(9, dtype=np.uint8) * 12 + 128, (8, 1))
            gray[0] = levels
            pixels[..., :3] = gray[..., None]
            self.add("CSC", "train", index, pixels)
        self.write_manifest()
        summary = prep.screen(self.root, self.out)
        pairs = {(r["a"], r["b"]): r for r in self.read("screen_pairs.csv")}
        a, b, c = (self.path("CSC", "train", i) for i in (50, 51, 52))
        self.assertEqual((pairs[a, b]["kind"], pairs[a, b]["hamming"]), ("dhash", "4"))
        self.assertNotIn((a, c), pairs)
        self.assertEqual(summary["pairs"]["dhash"], 2)

    def test_every_candidate_needs_exactly_one_valid_decision(self):
        pair = self.near_copy()
        row = [pair["a"], pair["b"], "confirmed"]
        for rows in ([row, row], [[pair["a"], pair["b"], "unknown"]],
                     [[pair["a"], self.path("MH", "train", 0), "rejected"]]):
            with self.subTest(reviews=rows):
                reviews = self.csv_fixture("reviews.csv", ["a", "b", "decision"], rows)
                with self.assertRaisesRegex(ValueError, "review"):
                    prep.freeze(self.root, self.out, reviews)
                self.assertFalse((self.out / "groups.csv").exists())

    def test_review_decisions_and_image_component_sizes_are_saved(self):
        pair = self.near_copy()
        screened = self.out
        for decision, group_count in (("confirmed", 12), ("unresolved", 12), ("rejected", 13)):
            with self.subTest(decision=decision):
                self.out = Path(self.tmp.name) / decision
                shutil.copytree(screened, self.out)
                reviews = self.csv_fixture("reviews.csv", ["a", "b", "decision"], [[pair["a"], pair["b"], decision]])
                result = prep.freeze(self.root, self.out, reviews)
                self.assertEqual(result["groups_per_class"]["RP"], group_count)
                self.assertEqual(self.read("duplicate_decisions.csv"), [dict(a=pair["a"], b=pair["b"], decision=decision)])
                component = [r for r in self.read("components.csv") if r["member"] == pair["a"]][0]
                self.assertEqual(component["size"], "1" if decision == "rejected" else "2")
                self.assertEqual(component["has_legacy_query"], "0")

    def test_only_verified_namespaced_patients_join_development_groups(self):
        a, b, c = self.path("CSC", "train", 0), self.path("CSC", "train", 1), self.path("MH", "train", 0)
        # Manifest pid is deliberately unverified and must not link the rare images.
        for row in self.rows:
            row[2] = "same-unverified-id"
        self.write_manifest()
        patients = self.patients([[a, "dataset-A", "p1", "synthetic source record 1"],
                                  [b, "dataset-A", "p1", "synthetic source record 2"],
                                  [c, "dataset-B", "p1", "synthetic source record 3"]])
        result = self.run_both(patients=patients)
        self.assertEqual(result["groups_per_class"], {label: 11 if label == "CSC" else 12 for label in fm.RARE})
        saved = {r["member"]: r for r in self.read("groups.csv")}
        self.assertEqual((saved[a]["group_id"], saved[b]["group_id"]), (a, a))
        self.assertEqual(saved[a]["fold"], saved[b]["fold"])
        self.assertEqual(saved[c]["group_id"], c)
        self.assertEqual(self.read("verified_patients.csv"), prep.read_csv(patients))
        # An image-duplicate component must not be relabelled as a patient group.
        self.assertEqual({r["size"] for r in self.read("components.csv")}, {"1"})

    def test_image_and_verified_patient_links_exclude_transitively_from_fitting_only(self):
        a, b, c = (self.path("CSC", "train", i) for i in (0, 1, 2))
        query = self.path("CSC", "test", 99)
        shutil.copyfile(self.root / query, self.root / a)
        shutil.copyfile(self.root / b, self.root / c)
        self.add("CSC", "train", 12)  # 13 minus the three linked rows leaves the exact 10-group boundary.
        self.write_manifest()
        patients = self.patients([[a, "dataset-A", "p1", "record 1"], [b, "dataset-A", "p1", "record 2"]])
        manifest = (self.root / "data/manifest.csv").read_bytes()
        result = self.run_both(patients=patients)
        self.assertEqual((result["excluded_rows"], result["groups_per_class"]["CSC"]), (3, 10))
        self.assertEqual({r["path"] for r in self.read("exclusions.csv")}, {a, b, c})
        self.assertEqual({r["component"] for r in self.read("exclusions.csv")}, {query})
        self.assertEqual({r["reason"] for r in self.read("exclusions.csv")}, {"linked_to_legacy_query"})
        roster = self.read("nine_way_roster.csv")
        self.assertTrue({a, b, c}.isdisjoint(r["path"] for r in roster if r["role"] == "fitting"))
        self.assertIn(dict(path=query, label="CSC", role="query"), roster)
        extraction={r["path"] for r in self.read("extraction_inventory.csv")}
        self.assertTrue({a,b,c}.isdisjoint(extraction))
        self.assertIn(query,extraction)
        self.assertEqual((self.root / "data/manifest.csv").read_bytes(), manifest)

    def test_cross_label_verified_patient_stops_before_any_episode(self):
        patients = self.patients([[self.path(label, "train", 0), "dataset-A", "p1", "source record"]
                                  for label in ("CSC", "MH")])
        prep.screen(self.root, self.out)
        with self.assertRaisesRegex(ValueError, "patient spans disease labels"):
            prep.freeze(self.root, self.out, patients=patients)
        self.assertFalse((self.out / "groups.csv").exists())
        self.assertFalse((self.out / "episodes_development.csv").exists())

    def test_patient_evidence_requires_unique_known_paths_and_nonempty_provenance(self):
        a = self.path("CSC", "train", 0)
        valid = [a, "dataset-A", "p1", "source record"]
        invalid = [[valid, valid], [[a, "dataset-A", "p1", " "]],
                   [["data/raw/absent.png", "dataset-A", "p1", "source record"]],
                   [[a, "", "p1", "source record"]], [[a, "dataset-A", "", "source record"]]]
        prep.screen(self.root, self.out)
        for rows in invalid:
            with self.subTest(patients=rows):
                with self.assertRaisesRegex(ValueError, "verified patient rows"):
                    prep.freeze(self.root, self.out, patients=self.patients(rows))
                self.assertFalse((self.out / "groups.csv").exists())

    def test_saved_query_ids_fold_counts_and_supports_describe_the_same_groups(self):
        with Image.open(self.root / self.path("CSC", "train", 0)) as image:
            pixels = np.array(image)
        pixels[..., 3] = 7
        self.add("CSC", "train", 50, pixels)
        self.write_manifest()
        self.run_both()
        groups = {r["member"]: r for r in self.read("groups.csv")}
        queries = self.read("queries_development.csv")
        self.assertEqual(len(queries), len(groups))
        self.assertEqual({r["path"] for r in queries}, set(groups))
        for row in queries:
            self.assertEqual((row["fold"], row["label"]), (groups[row["path"]]["fold"], groups[row["path"]]["label"]))
        counts = self.read("fold_counts.csv")
        self.assertEqual(len(counts), 25)
        self.assertEqual({(row["label"], int(row["fold"])) for row in counts}, {(c, f) for c in fm.RARE for f in range(5)})
        for row in counts:
            query_members = {p for p, g in groups.items() if g["label"] == row["label"] and g["fold"] == row["fold"]}
            fitting_members = {p for p, g in groups.items() if g["label"] == row["label"] and g["fold"] != row["fold"]}
            self.assertEqual(int(row["query_images"]), len(query_members))
            self.assertEqual(int(row["query_groups"]), len({groups[p]["group_id"] for p in query_members}))
            self.assertEqual(int(row["fitting_images"]), len(fitting_members))
            self.assertEqual(int(row["fitting_groups"]), len({groups[p]["group_id"] for p in fitting_members}))
            self.assertGreaterEqual(int(row["fitting_groups"]), 8)
            self.assertGreaterEqual(int(row["query_groups"]), 1)
        slots = [f"s{i:02d}" for i in range(1, 9)]
        episodes = self.read("episodes_development.csv")
        self.assertEqual({(r["fold"], r["label"], int(r["draw"])) for r in episodes},
                         {(str(f), c, d) for f in range(5) for c in fm.RARE for d in range(100)})
        self.assertEqual(len(episodes), 2500)
        self.assertEqual(set(episodes[0]), {"fold", "label", "draw", *slots})
        for row in episodes:
            supports = [groups[row[slot]] for slot in slots]
            self.assertEqual(len({g["group_id"] for g in supports}), 8)
            self.assertTrue(all(g["label"] == row["label"] and g["fold"] != row["fold"] for g in supports))
        legacy = {r[0].replace("\\", "/") for r in self.rows if r[3] == "test"}
        self.assertEqual({r["path"] for r in self.read("nine_way_roster.csv") if r["role"] == "query"}, legacy)

    def test_persisted_episode_ids_match_fixed_protocol_streams(self):
        self.run_both()
        development = next(r for r in self.read("episodes_development.csv")
                           if (r["fold"], r["label"], r["draw"]) == ("0", "CSC", "7"))
        reporting = next(r for r in self.read("episodes_reporting.csv") if (r["label"], r["draw"]) == ("CSC", "7"))
        self.assertEqual({r["path"] for r in self.read("queries_development.csv") if (r["fold"], r["label"]) == ("0", "CSC")},
                         {self.path("CSC", "train", rank) for rank in (3, 6, 10)})
        # Independently realized PCG64 streams [42,0,0,7,6101] and [42,0,0,7,7101].
        for row, ranks in ((development, [1, 2, 11, 8, 9, 5, 4, 0]), (reporting, [4, 8, 6, 2, 5, 1, 3, 10])):
            self.assertEqual([row[f"s{i:02d}"] for i in range(1, 9)], [self.path("CSC", "train", rank) for rank in ranks])
        reports = self.read("episodes_reporting.csv")
        self.assertEqual(len(reports), 500)
        self.assertEqual({(r["label"], int(r["draw"])) for r in reports}, {(c, d) for c in fm.RARE for d in range(100)})
        eligible = sorted(r["member"] for r in self.read("groups.csv"))
        self.assertEqual([r["path"] for r in self.read("timing_inventory.csv")], [eligible[i % len(eligible)] for i in range(100)])

    def test_nine_way_cap_maps_common_frame_positions_back_to_canonical_paths(self):
        from stage3_classifier import major_indices
        rows = []
        for i in range(1002):
            rows += [dict(path=f"common/{c}/{i}.png", label=c, split="train") for c in fm.NINE[:4]]
            rows.append(dict(path=f"validation/{i}.png", label="NORMAL", split="val"))
        rare = dict(path="rare/eligible.png", label="CSC", split="train")
        query = dict(path="legacy/query.png", label="CSC", split="test")
        rows = [rare, *rows, query]
        common = [r for r in rows if r["split"] == "train" and r["label"] in fm.NINE[:4]]
        selected = major_indices([r["label"] for r in common], 1000, 42)
        expected = [[common[int(i)]["path"], common[int(i)]["label"], "fitting"] for i in selected]
        roster = prep.nine_way_roster(rows, [rare["path"]])
        self.assertEqual(roster, [*expected, [rare["path"], "CSC", "fitting"], [query["path"], "CSC", "query"]])
        self.assertEqual(len(expected), 4000)

    def test_reporting_sixteen_is_saved_only_when_every_class_has_sixteen_groups(self):
        for label in fm.RARE:
            for index in range(12, 16):
                self.add(label, "train", index)
        self.write_manifest()
        result = self.run_both()
        reporting = self.read("episodes_reporting.csv")
        self.assertEqual(result["report_k"], [1, 2, 4, 8, 16])
        self.assertTrue(all(row["s16"] for row in reporting))
        self.assertTrue(all(len({row[f"s{i:02d}"] for i in range(1, 17)}) == 16 for row in reporting))
        self.rows = [row for row in self.rows if row[0].replace("\\", "/") != self.path("RP", "train", 15)]
        self.write_manifest()
        self.out = Path(self.tmp.name) / "one-class-has-fifteen"
        result = self.run_both()
        self.assertEqual(result["report_k"], [1, 2, 4, 8])
        self.assertEqual(set(self.read("episodes_reporting.csv")[0]), {"label", "draw", *[f"s{i:02d}" for i in range(1, 9)]})

    def test_too_few_groups_stops_before_any_roster(self):
        self.rows = [r for r in self.rows if not (r[1] == "RP" and r[3] == "train" and r[0].endswith(("_09.png", "_10.png", "_11.png")))]
        self.write_manifest()
        with self.assertRaises(ValueError):
            self.run_both()
        self.assertFalse((self.out / "groups.csv").exists())

    def test_cross_disease_duplicate_stops(self):
        shutil.copy(self.root / self.path("CSC", "train", 0), self.root / self.path("MH", "train", 0))
        with self.assertRaises(ValueError):
            self.run_both()

    def test_outputs_may_not_go_under_data(self):
        with self.assertRaises(ValueError):
            prep.screen(self.root, self.root / "data/out")


class Archives(SyntheticInputs):
    def test_parent_completion_and_export_cannot_omit_a_completed_child(self):
        run=Path(self.tmp.name)/"parent"
        child=run/"prototype/cell"
        child.mkdir(parents=True)
        (run/"identity.json").write_text('{}',encoding="utf-8")
        (child/"predictions.csv").write_text('synthetic\n',encoding="utf-8")
        prep.complete(child,["predictions.csv"])
        with self.assertRaisesRegex(ValueError,"child"):
            prep.complete(run,["identity.json"])
        self.assertFalse((run/"complete.json").exists())
        self.json_fixture(run/"complete.json",dict(files={"identity.json":self.digest(run/"identity.json")}))
        archive=Path(self.tmp.name)/"incomplete-export.zip"
        with self.assertRaisesRegex(ValueError,"child"):
            prep.verify_complete(run)
        with self.assertRaisesRegex(ValueError,"child"):
            prep.export(run,archive)
        self.assertFalse(archive.exists())

    def test_export_cli_cannot_write_archive_temporary_or_sidecar_into_raw(self):
        run=Path(self.tmp.name)/"completed-run"
        run.mkdir()
        (run/"predictions.csv").write_bytes(b"synthetic\n")
        prep.complete(run,["predictions.csv"])
        archive=self.root/"data/raw/forbidden-export.zip"
        with patch.object(sys,"argv",["fm_prepare.py","export","--root",str(self.root),"--out",str(run),
                                      "--archive",str(archive)]):
            with self.assertRaisesRegex(ValueError,"outside data"):
                prep.main()
        for suffix in ("",".tmp",".json",".json.tmp"):
            self.assertFalse(Path(str(archive)+suffix).exists())

    def test_staging_preserves_original_bytes_and_contains_no_extra_images(self):
        self.run_both()
        unrelated = self.root / "data/raw/not_in_manifest.png"
        unrelated.write_bytes(b"synthetic file that must never be staged")
        (self.out / "unrelated.txt").write_text("must not be packed", encoding="utf-8")
        archive = Path(self.tmp.name) / "stage.zip"
        result = prep.pack(self.root, self.out, archive)
        inventory = self.read("extraction_inventory.csv")
        expected = {row["path"]: (self.root / row["path"]).read_bytes() for row in inventory}
        with zipfile.ZipFile(archive) as staged:
            self.assertEqual(staged.namelist(), [row["path"] for row in inventory])
            self.assertEqual({name: staged.read(name) for name in staged.namelist()}, expected)
        self.assertEqual(result["sha256"], self.digest(archive))
        self.assertEqual(result["images"], len(expected))
        self.assertEqual(result["inventory_sha256"], self.digest(self.out / "extraction_inventory.csv"))
        second = Path(self.tmp.name) / "stage-again.zip"
        self.assertEqual(prep.pack(self.root, self.out, second), result)
        self.assertEqual(second.read_bytes(), archive.read_bytes())
        self.assertEqual(prep.pack(self.root, self.out, archive), result)

    def test_staging_rejects_changed_sources_and_changed_frozen_rosters(self):
        self.run_both()
        archive = Path(self.tmp.name) / "stage.zip"
        for target in (self.root / self.path("NORMAL", "train", 0), self.out / "extraction_inventory.csv"):
            with self.subTest(target=target.name):
                original = target.read_bytes()
                target.write_bytes(original + b"changed")
                with self.assertRaises(ValueError):
                    prep.pack(self.root, self.out, archive)
                self.assertFalse(archive.exists())
                target.write_bytes(original)

    def test_export_only_contains_completed_hashed_artifacts_and_rejects_tampering(self):
        run = Path(self.tmp.name) / "synthetic-run"
        run.mkdir()
        (run / "predictions.csv").write_bytes(b"image,prediction\nsynthetic.png,CSC\n")
        (run / "uncommitted.tmp").write_bytes(b"must not be exported")
        prep.complete(run, ["predictions.csv"])
        archive = Path(self.tmp.name) / "results.zip"
        result = prep.export(run, archive)
        with zipfile.ZipFile(archive) as exported:
            self.assertEqual(set(exported.namelist()), {"predictions.csv", "complete.json"})
            for name in exported.namelist():
                self.assertEqual(exported.read(name), (run / name).read_bytes())
        self.assertEqual((result["sha256"], result["files"]), (self.digest(archive), 2))
        (run / "predictions.csv").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            prep.export(run, Path(self.tmp.name) / "bad-results.zip")


class WriterLocks(SyntheticInputs):
    def test_second_writer_is_blocked_and_exception_releases_its_own_lock(self):
        with self.assertRaisesRegex(RuntimeError, "synthetic interruption"):
            with prep.writer_lock(self.out):
                original = (self.out / "writer.lock").read_bytes()
                with self.assertRaisesRegex(ValueError, "writer lock exists"):
                    with prep.writer_lock(self.out):
                        self.fail("a second writer acquired the same folder")
                self.assertEqual((self.out / "writer.lock").read_bytes(), original)
                raise RuntimeError("synthetic interruption")
        self.assertFalse((self.out / "writer.lock").exists())
        with prep.writer_lock(self.out):
            self.assertTrue((self.out / "writer.lock").exists())

    def test_stale_token_cannot_unlock_or_clean_up_a_replacement_writer(self):
        with prep.writer_lock(self.out):
            old = prep.read_json(self.out / "writer.lock")
            with self.assertRaisesRegex(ValueError, "token mismatch"):
                prep.unlock(self.out, "wrong-token")
            self.assertEqual(prep.read_json(self.out / "writer.lock"), old)
            prep.unlock(self.out, old["token"])
            replacement = {**old, "token": "synthetic-replacement-token", "pid": 123456789}
            self.json_fixture(self.out / "writer.lock", replacement)
            with self.assertRaisesRegex(ValueError, "token mismatch"):
                prep.unlock(self.out, old["token"])
        self.assertEqual(prep.read_json(self.out / "writer.lock"), replacement)
        prep.unlock(self.out, replacement["token"])
        self.assertFalse((self.out / "writer.lock").exists())


class InputLocks(SyntheticInputs):
    """Structural input-lock tests: all images, weights, timing and pilot runs are synthetic."""

    def setUp(self):
        super().setUp()
        import fm_extract as extract
        self.extract = extract
        self.summary = self.run_both()
        self.code = prep.code_identity()
        self.environment = dict(python="synthetic-python", platform="synthetic-runtime",
                                packages={"numpy": "synthetic-numpy", "pillow": "synthetic-pillow"})
        self.pilot_root = Path(self.tmp.name) / "synthetic-pilots"
        self.pilots = {arm: self.make_pilot(arm) for arm in ("RETFound", "DINOv2", "MIRAGE", "Inception")}
        self.gate = dict(arms=[dict(name=arm, status="available", pilot=str(folder)) for arm, folder in self.pilots.items()],
                         external_sources={source: dict(status="unavailable", reason="synthetic no-data fixture")
                                           for source in ("OCTID", "OCTDL")})
        self.gate_path = Path(self.tmp.name) / "synthetic-gate.json"
        self.archive = Path(self.tmp.name) / "synthetic-input-lock.zip"

    def make_pilot(self, arm):
        # Only public metadata is used; no real weights or model loader exists in
        # this synthetic pilot. Asset-byte validation is tested at the spec boundary.
        spec=dict(arm=arm,repository=self.extract.REPOSITORIES[arm],revision=self.extract.REVISIONS[arm],
                  source_revision=self.extract.SOURCE_REVISIONS.get(arm,""),terms_evidence="synthetic terms evidence",
                  files=dict(self.extract.EXPECTED_ASSET_SHA256[arm]),recipe=self.extract.recipe(arm))
        self.extract.check_spec(spec)
        folder = self.pilot_root / arm
        folder.mkdir(parents=True)
        identity = dict(step="pilot", spec=spec, code=self.code, input_files=self.summary["files"],
                        preparation_sha256=self.digest(self.out / "freeze_summary.json"),
                        environment=deepcopy(self.environment), lock_sha256=None,
                        timing_session="synthetic-session-1",
                        hardware=dict(host="synthetic-host", boot_id="synthetic-boot", gpu="synthetic-gpu"))
        self.json_fixture(folder / "identity.json", identity)
        slot={"RETFound":0,"BioMedCLIP":0,"DINOv2":1,"MIRAGE":2,"Inception":3}[arm]
        started=1000+slot*1000
        timing=dict(batch_size=1,images=100,warm_passes=1,timed_passes=5,
                    seconds_per_image=[.01,.02,.03,.04,.05],median_seconds_per_image=.03,
                    process_id=100+slot,started_monotonic_ns=started+10,finished_monotonic_ns=started+20,
                    operation_started_monotonic_ns=started,operation_finished_monotonic_ns=started+30,
                    image_ids=[r["path"] for r in self.read("timing_inventory.csv")],
                    numerical=dict(rule="synthetic", tolerance=self.extract.REPEAT_TOLERANCE,
                                   min_cosine=self.extract.REPEAT_COSINE,
                                   real=dict(images=100, max_difference_ratio=0., worst_image_difference_ratio=0., min_cosine=1.),
                                   synthetic=dict(images=5, max_difference_ratio=0., worst_image_difference_ratio=0., min_cosine=1.)),
                    costs={**{key:0. for key in ("model_loading_seconds","staging_seconds","extraction_seconds",
                        "validation_seconds","serialization_seconds","input_verification_seconds","wall_seconds")},
                        "image_count":100})
        self.json_fixture(folder / "timing.json",timing)
        self.json_fixture(folder / "load_report.json", dict(arm=arm, synthetic=True))
        with (folder / "geometry.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(self.extract.GEOMETRY_HEADER)
            for row in self.read("timing_inventory.csv"):
                with Image.open(self.root / row["path"]) as image:
                    writer.writerow(self.extract.geometry(arm, row["path"], image))
        self.seal_pilot(folder)
        return folder

    def seal_pilot(self, folder):
        timing=prep.read_json(folder / "timing.json")
        timing["files"]={name:self.digest(folder / name) for name in ("identity.json","load_report.json","geometry.csv")}
        self.json_fixture(folder / "timing.json",timing)
        self.json_fixture(folder / "complete.json", dict(files={name: self.digest(folder / name) for name in
                          ("identity.json", "timing.json", "load_report.json", "geometry.csv")}))

    def change_pilot(self, arm, name, change):
        folder = self.pilots[arm]
        value = prep.read_json(folder / name)
        change(value)
        self.json_fixture(folder / name, value)
        self.seal_pilot(folder)

    def lock(self):
        self.json_fixture(self.gate_path, self.gate)
        return prep.lock_inputs(self.out, self.gate_path, self.archive,self.root)

    def assert_gate_rejects(self):
        with self.assertRaises(ValueError):
            self.lock()
        self.assertFalse((self.out / "input_lock.json").exists())
        self.assertFalse(self.archive.exists())

    def test_synthetic_lock_is_archived_portable_and_unchanged_retry_is_identical(self):
        (self.out / "not-locked.txt").write_bytes(b"must not enter the archive")
        identity = self.lock()
        self.assertEqual(prep.load_lock(self.out), identity)
        self.assertEqual(set(identity["arms"]), {"RETFound", "DINOv2", "MIRAGE", "Inception"})
        self.assertEqual({r["status"] for r in identity["arms"].values()}, {"available"})
        self.assertEqual(identity["endpoints"]["development_k"], [1, 2, 4, 8])
        self.assertEqual(identity["endpoints"]["reporting_k"], [1, 2, 4, 8])
        self.assertEqual(identity["endpoints"]["draws"], 100)
        receipt = prep.read_json(self.out / "lock_archive.json")
        self.assertEqual(receipt["sha256"], self.digest(self.archive))
        with zipfile.ZipFile(self.archive) as archive:
            self.assertEqual(archive.namelist(), sorted(receipt["files"]))
            for name in archive.namelist():
                self.assertEqual(archive.read(name), (self.out / name).read_bytes())
                self.assertEqual(hashlib.sha256(archive.read(name)).hexdigest(), receipt["files"][name])
        self.assertNotIn(json.dumps(str(self.pilot_root))[1:-1], (self.out / "input_lock.json").read_text(encoding="utf-8"))
        first_archive = self.archive.read_bytes()
        first_lock = (self.out / "input_lock.json").read_bytes()
        self.assertEqual(self.lock(), identity)
        self.assertEqual((self.out / "input_lock.json").read_bytes(), first_lock)
        self.assertEqual(self.archive.read_bytes(), first_archive)
        moved = Path(self.tmp.name) / "copied-portable-lock"
        shutil.copytree(self.out, moved)
        self.out.rename(Path(self.tmp.name) / "original-location-moved")
        self.assertEqual(prep.load_lock(moved), identity)

    def test_lock_rejects_corrupted_completed_pilot_bytes(self):
        target = self.pilots["DINOv2"] / "timing.json"
        target.write_bytes(target.read_bytes() + b"\n")
        self.assert_gate_rejects()

    def test_lock_requires_completion_hashes_to_cover_the_pilot_identity(self):
        marker = self.pilots["DINOv2"] / "complete.json"
        original = prep.read_json(marker)
        for missing in ("identity.json", "timing.json", "load_report.json", "geometry.csv"):
            with self.subTest(missing=missing):
                complete = deepcopy(original)
                del complete["files"][missing]
                self.json_fixture(marker, complete)
                self.assert_gate_rejects()

    def test_lock_rejects_pilot_code_and_realized_roster_mismatches(self):
        original = prep.read_json(self.pilots["DINOv2"] / "identity.json")
        for key, entry in (("code", "fm_core.py"), ("input_files", "episodes_development.csv")):
            with self.subTest(identity_field=key):
                changed = deepcopy(original)
                changed[key][entry] = "0" * 64
                self.json_fixture(self.pilots["DINOv2"] / "identity.json", changed)
                self.seal_pilot(self.pilots["DINOv2"])
                self.assert_gate_rejects()

    def test_lock_rejects_different_environments_with_the_same_session_and_gpu(self):
        self.change_pilot("DINOv2", "identity.json",
                          lambda value: value["environment"]["packages"].update(numpy="different-numpy"))
        self.assert_gate_rejects()

    def test_lock_rejects_different_or_missing_sessions_and_hardware_changes(self):
        original = prep.read_json(self.pilots["DINOv2"] / "identity.json")
        for session, boot in (("", "synthetic-boot"), ("different-session", "synthetic-boot"),
                              ("synthetic-session-1", "different-boot")):
            with self.subTest(session=session, boot=boot):
                changed = deepcopy(original)
                changed["timing_session"] = session
                changed["hardware"]["boot_id"] = boot
                self.json_fixture(self.pilots["DINOv2"] / "identity.json", changed)
                self.seal_pilot(self.pilots["DINOv2"])
                self.assert_gate_rejects()

    def test_lock_rejects_wrong_timing_recipe_nonpositive_passes_and_wrong_median(self):
        original = prep.read_json(self.pilots["DINOv2"] / "timing.json")
        changes = [dict(images=99), dict(batch_size=2), dict(warm_passes=0), dict(timed_passes=4),
                   dict(seconds_per_image=[0.01] * 4), dict(seconds_per_image=[0.01, 0.02, 0, 0.04, 0.05]),
                   dict(median_seconds_per_image=0.031)]
        for change in changes:
            with self.subTest(timing=change):
                self.json_fixture(self.pilots["DINOv2"] / "timing.json", {**original, **change})
                self.seal_pilot(self.pilots["DINOv2"])
                self.assert_gate_rejects()

    def test_unavailable_arm_is_recorded_and_cannot_join_on_a_later_retry(self):
        self.gate["arms"][2] = dict(name="MIRAGE", status="unavailable", reason="synthetic access failure")
        identity = self.lock()
        self.assertEqual(identity["arms"]["MIRAGE"], dict(status="unavailable", reason="synthetic access failure"))
        self.assertEqual(set(identity["arms"]), {"RETFound", "DINOv2", "MIRAGE", "Inception"})
        self.gate["arms"][2] = dict(name="MIRAGE", status="available", pilot=str(self.pilots["MIRAGE"]))
        with self.assertRaises(ValueError):
            self.lock()
        self.assertEqual(prep.load_lock(self.out), identity)

    def test_every_fixed_arm_needs_a_decision_and_inception_must_be_available(self):
        original = deepcopy(self.gate["arms"])
        for arms in (original[:-1], [*original, original[0]],
                     [*original[:3], dict(name="Inception", status="unavailable", reason="synthetic failure")],
                     [*original[:2], dict(name="MIRAGE", status="unavailable", reason=" "), original[3]]):
            with self.subTest(arms=arms):
                self.gate["arms"] = arms
                self.assert_gate_rejects()

    def test_fallback_requires_a_pre_score_failure_and_stays_in_retfounds_slot(self):
        fallback = self.make_pilot("BioMedCLIP")
        self.gate["arms"][0] = dict(name="BioMedCLIP", status="available", pilot=str(fallback))
        self.assert_gate_rejects()
        self.gate["retfound_failure"] = "synthetic RETFound access/terms failure before any scores"
        identity = self.lock()
        self.assertEqual(set(identity["arms"]), {"BioMedCLIP", "DINOv2", "MIRAGE", "Inception"})
        self.assertEqual(identity["retfound_failure"], self.gate["retfound_failure"])

    def test_planned_external_sources_need_release_terms_eligibility_and_label_inventory(self):
        complete = dict(status="planned", release="synthetic release", release_sha256="a" * 64,
                        terms_evidence="synthetic terms", eligibility="synthetic NORMAL eligibility",
                        label_inventory="NORMAL only; disease mapping not certified")
        for missing in ("release", "release_sha256", "terms_evidence", "eligibility", "label_inventory"):
            with self.subTest(missing=missing):
                self.gate["external_sources"]["OCTID"] = {**complete, missing: ""}
                self.assert_gate_rejects()
        self.gate["external_sources"]["OCTID"] = complete
        self.assertEqual(self.lock()["external_sources"]["OCTID"], complete)

    def test_pilot_windows_must_be_sequential_and_in_prespecified_arm_order(self):
        original=prep.read_json(self.pilots["DINOv2"] / "timing.json")
        for change in (
                {"operation_started_monotonic_ns":1001,"started_monotonic_ns":1011,"finished_monotonic_ns":1021,
                 "operation_finished_monotonic_ns":1031},
                {"operation_started_monotonic_ns":1,"started_monotonic_ns":11,"finished_monotonic_ns":21,
                 "operation_finished_monotonic_ns":31},
                {"process_id":100}):
            with self.subTest(change=change):
                self.json_fixture(self.pilots["DINOv2"] / "timing.json",{**original,**change})
                self.seal_pilot(self.pilots["DINOv2"])
                self.assert_gate_rejects()
        self.json_fixture(self.pilots["DINOv2"] / "timing.json",original)
        self.seal_pilot(self.pilots["DINOv2"])
        self.lock()

    def test_pilot_realized_timing_ids_must_match_frozen_inventory(self):
        self.change_pilot("DINOv2","timing.json",lambda x:x["image_ids"].reverse())
        self.assert_gate_rejects()

    def test_loading_a_lock_rejects_current_code_changes(self):
        self.lock()
        with patch.object(prep, "code_identity", return_value={**self.code, "fm_core.py": "0" * 64}):
            with self.assertRaisesRegex(ValueError, "code identity differs"):
                prep.load_lock(self.out)

    def test_final_lock_rejects_changed_summary_endpoints_after_pilots(self):
        target=self.out / "freeze_summary.json"
        value=prep.read_json(target)
        self.json_fixture(target,{**value,"report_k":[1,2]})
        self.assert_gate_rejects()

    def test_final_lock_rebuilds_grouping_from_patient_evidence_even_when_all_hashes_match(self):
        patient_file=self.out/"verified_patients.csv"
        a,b=self.path("CSC","train",0),self.path("CSC","test",99)
        with patient_file.open("w",newline="",encoding="utf-8") as stream:
            csv.writer(stream,lineterminator="\n").writerows([
                ["path","dataset","patient_id","evidence"],
                [a,"dataset-A","verified-1","synthetic record"],[b,"dataset-A","verified-1","synthetic record"]])
        self.summary["files"]["verified_patients.csv"]=self.digest(patient_file)
        self.json_fixture(self.out/"freeze_summary.json",self.summary)
        for arm in self.pilots:
            self.change_pilot(arm,"identity.json",lambda x:x.update(input_files=self.summary["files"],
                preparation_sha256=self.digest(self.out/"freeze_summary.json")))
        with self.assertRaisesRegex(ValueError,"duplicate/patient evidence"):
            self.extract.prepared(self.out)
        self.assert_gate_rejects()

    def test_archive_and_external_digest_restore_a_lock_without_the_original_folder(self):
        expected=self.lock()
        digest=prep.read_json(self.archive.with_name(self.archive.name+".json"))["sha256"]
        source=Path(self.tmp.name)/"backup.zip"
        shutil.copyfile(self.archive,source)
        self.out.rename(Path(self.tmp.name)/"original-now-unavailable")
        restored=Path(self.tmp.name)/"restored"
        self.assertEqual(prep.restore_lock(source,restored,digest,self.root),expected)
        self.assertEqual(prep.load_lock(restored),expected)
        self.assertEqual(prep.read_json(restored/"lock_archive.json")["sha256"],digest)
        forbidden=Path(self.tmp.name)/"bad-restore"
        with self.assertRaisesRegex(ValueError,"saved SHA256"):
            prep.restore_lock(source,forbidden,"0"*64,self.root)
        self.assertFalse(forbidden.exists())

    def test_final_lock_rejects_changed_source_manifest_after_freezing(self):
        with (self.root / "data/manifest.csv").open("ab") as stream:
            stream.write(b"\n")
        self.assert_gate_rejects()

    def test_old_namespace_preparation_cannot_enter_current_pilots_or_lock(self):
        old=Path(self.tmp.name) / "old-namespace"
        with patch.object(fm,"DEV_NS",9991):
            prep.screen(self.root,old)
            prep.freeze(self.root,old)
        with self.assertRaisesRegex(ValueError,"code/recipe"):
            self.extract.prepared(old)

    def test_pilot_binds_entire_preparation_summary_not_just_its_file_map(self):
        target=self.out / "freeze_summary.json"
        value=prep.read_json(target)
        value["versions"]["pillow"]="a-different-recorded-decoder"
        self.json_fixture(target,value)
        self.assert_gate_rejects()

    def test_loading_a_lock_rejects_changed_rosters_and_archive_receipts(self):
        self.lock()
        for name in ("queries_development.csv", "lock_archive.json"):
            with self.subTest(artifact=name):
                target = self.out / name
                original = target.read_bytes()
                if name.endswith(".csv"):
                    target.write_bytes(original + b"changed")
                else:
                    receipt = prep.read_json(target)
                    receipt["files"]["input_lock.json"] = "0" * 64
                    self.json_fixture(target, receipt)
                with self.assertRaises(ValueError):
                    prep.load_lock(self.out)
                target.write_bytes(original)


if __name__ == "__main__":
    unittest.main()
