"""Synthetic evaluation contracts: pairing, numerical failures and score gates."""
import csv
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fm_core as fm
import fm_evaluate as ev
import fm_prepare as prep
import fm_extract as extract


def replace_json(path, value):
    """Deliberately alter a synthetic artifact to exercise fail-closed readers."""
    path.write_text(json.dumps(value), encoding="utf-8")


def reseal(folder, names):
    replace_json(folder / "complete.json", {"files": {name: prep.sha256(folder / name) for name in names}})


class Episodes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.plan = Path(cls.tmp.name) / "plan"
        cls.plan.mkdir()
        groups = {c: [fm.Group(f"{c}/{i:02d}.png", c, i, (f"{c}/{i:02d}.png",))
                      for i in range(12)] for c in fm.RARE}
        folds = {c: fm.assign_folds(groups[c], i) for i, c in enumerate(fm.RARE)}
        prep.write_csv(cls.plan / "groups.csv", ["group_id","label","rank","fold","member"],
                       [[g.id,c,g.rank,folds[c][g.id],g.id] for c in fm.RARE for g in groups[c]])
        prep.write_csv(cls.plan / "queries_development.csv", ["fold","label","path"],
                       [[f,c,g.id] for f in range(5) for c in fm.RARE for g in groups[c] if folds[c][g.id] == f])
        prep.write_csv(cls.plan / "episodes_development.csv", ["fold","label","draw",*[f"s{i:02d}" for i in range(1,9)]],
                       [[f,c,r,*fm.draw_supports([g for g in groups[c] if folds[c][g.id] != f], f,ci,r,6101)[:8]]
                        for f in range(5) for ci,c in enumerate(fm.RARE) for r in range(100)])
        prep.write_csv(cls.plan / "episodes_reporting.csv", ["label","draw",*[f"s{i:02d}" for i in range(1,9)]],
                       [[c,r,*fm.draw_supports(groups[c],0,ci,r,7101)[:8]]
                        for ci,c in enumerate(fm.RARE) for r in range(100)])
        prep.write_csv(cls.plan / "nine_way_roster.csv", ["path","label","role"],
                       [[f"legacy/{c}.png",c,"query"] for c in fm.RARE])
        prep.write_json(cls.plan / "freeze_summary.json", {"report_k":[1,2,4,8]})
        prep.write_json(cls.plan / "input_lock.json", {"synthetic":True})

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_all_required_episodes_are_present_nested_and_paired(self):
        planned = ev.episodes(self.plan, "development")
        self.assertEqual(len(planned), 5*100*4)
        by_key = {(e.fold,e.draw,e.k): e for e in planned}
        for e in planned:
            largest = by_key[e.fold,e.draw,8]
            self.assertEqual(e.supports, tuple(p[:e.k] for p in largest.supports))
            self.assertEqual(e.queries, largest.queries)
            self.assertTrue(set(e.queries).isdisjoint(p for ps in e.supports for p in ps))
        self.assertEqual(sum(ev.planned_probe(e,"development") for e in planned), 5*25)
        reports = ev.episodes(self.plan, "reporting")
        self.assertEqual(len(reports), 100*4)
        self.assertEqual(sum(ev.planned_probe(e,"reporting") for e in reports), 25*4)
        self.assertTrue(all(e.queries == tuple(f"legacy/{c}.png" for c in fm.RARE) for e in reports))

    def test_one_missing_class_draw_stops_before_scoring(self):
        target = Path(self.tmp.name) / "missing"
        shutil.copytree(self.plan,target)
        rows = prep.read_csv(target / "episodes_development.csv")
        with (target / "episodes_development.csv").open("w",newline="",encoding="utf-8") as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows[:-1])
        with self.assertRaisesRegex(ValueError,"incomplete"):
            ev.episodes(target,"development")

    def test_support_from_query_fold_is_rejected(self):
        target=Path(self.tmp.name)/"overlap"
        shutil.copytree(self.plan,target)
        rows=prep.read_csv(target/"episodes_development.csv")
        held=next(r["path"] for r in prep.read_csv(target/"queries_development.csv")
                  if r["label"]==rows[0]["label"] and r["fold"]==rows[0]["fold"])
        rows[0]["s01"]=held
        with (target/"episodes_development.csv").open("w",newline="",encoding="utf-8") as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        with self.assertRaisesRegex(ValueError,"class/fold"):
            ev.episodes(target,"development")

    def test_incomplete_run_cannot_receive_partial_nomination_score(self):
        empty=Path(self.tmp.name)/"empty"
        empty.mkdir()
        hits, queries, status=ev.collect_development(self.plan,empty,"DINOv2")
        self.assertIsNone(hits)
        self.assertEqual(status["available"],0)
        self.assertEqual(status["planned"],2000)
        self.assertEqual(queries.sum(axis=0).tolist(),[12]*5)


class Predictions(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.out=Path(self.tmp.name)
        paths=tuple(f"support/{c}" for c in fm.RARE)+tuple(f"query/{c}" for c in fm.RARE)
        self.features=ev.Features(np.tile(np.eye(5,dtype=np.float32),(2,1)),{p:i for i,p in enumerate(paths)})
        self.episode=ev.Episode(0,1,0,tuple((paths[i],) for i in range(5)),paths[5:],fm.RARE)

    def tearDown(self):
        self.tmp.cleanup()

    def test_prototype_saves_cosines_fixed_order_counts_and_ids(self):
        ev.evaluate_episode(self.features,self.episode,"prototype",self.out,"synthetic")
        rows=prep.read_csv(self.out/"predictions.csv")
        self.assertEqual([r["query_id"] for r in rows],list(self.episode.queries))
        self.assertEqual([r["predicted_label"] for r in rows],list(fm.RARE))
        self.assertTrue(all(f"cosine_{c}" in rows[0] for c in fm.RARE))
        self.assertFalse(any(k.startswith("prob_") for k in rows[0]))
        self.assertEqual(prep.read_json(self.out/"metrics.json")["confusion"],np.eye(5,dtype=int).tolist())
        self.assertEqual(prep.read_json(self.out/"metrics.json")["macro_recall"],1.)
        prep.verify_complete(self.out)

    def test_logistic_saves_probabilities_in_fixed_class_order(self):
        ev.evaluate_episode(self.features,self.episode,"logistic",self.out,"synthetic")
        rows=prep.read_csv(self.out/"predictions.csv")
        self.assertEqual([r["predicted_label"] for r in rows],list(fm.RARE))
        self.assertTrue(all(f"logit_{c}" in rows[0] for c in fm.RARE))
        self.assertAlmostEqual(sum(float(rows[0][f"prob_{c}"]) for c in fm.RARE),1.)
        self.assertTrue(prep.read_json(self.out/"fit.json")["converged"])

    def test_failed_fit_saves_reason_without_touching_query_features(self):
        failed=fm.LogisticFit(np.zeros((5,5)),np.zeros(5),False,1.,.001,"iteration limit")
        with patch.object(fm,"fit_logistic",return_value=failed), patch.object(self.features,"select",wraps=self.features.select) as select:
            ev.evaluate_episode(self.features,self.episode,"logistic",self.out,"synthetic")
        self.assertEqual(select.call_count,5)
        self.assertEqual(prep.read_json(self.out/"fit.json")["termination_reason"],"iteration limit")
        self.assertFalse((self.out/"predictions.csv").exists())
        prep.verify_complete(self.out)

    def test_resume_verifies_artifacts_and_support_identity(self):
        ev.evaluate_episode(self.features,self.episode,"prototype",self.out,"synthetic")
        first=(self.out/"predictions.csv").read_bytes()
        ev.evaluate_episode(self.features,self.episode,"prototype",self.out,"synthetic")
        self.assertEqual(first,(self.out/"predictions.csv").read_bytes())
        with self.assertRaises(ValueError):
            ev.evaluate_episode(self.features,self.episode,"prototype",self.out,"changed")
        (self.out/"metrics.json").write_text('{}',encoding="utf-8")
        with self.assertRaises(ValueError):
            ev.evaluate_episode(self.features,self.episode,"prototype",self.out,"synthetic")

    def test_partial_class_cohort_has_no_macro_and_retains_all_outputs(self):
        metric=ev.classification_metrics(["CSC","CSC","MH"],["RP","CSC","MH"],fm.RARE)
        self.assertIsNone(metric["macro_recall"])
        self.assertEqual(metric["totals"],[2,1,0,0,0])
        self.assertEqual(metric["confusion"][0],[1,0,0,0,1])

    def test_nine_way_normal_false_rare_rate_uses_nine_outputs(self):
        fitting=[[f"fit/{c}",c,"fitting"] for c in fm.NINE]
        queries=[["q/rp","NORMAL","query"],["q/cnv","NORMAL","query"],["q/normal","NORMAL","query"],
                 ["q/rare","CSC","query"]]
        plan=self.out/"plan"; plan.mkdir()
        prep.write_csv(plan/"nine_way_roster.csv",["path","label","role"],fitting+queries)
        values=np.vstack((np.eye(9),np.eye(9)[[8,1,0,4]])).astype(np.float32)
        features=ev.Features(values,{row[0]:i for i,row in enumerate(fitting+queries)})
        result=self.out/"nine"
        ev.nine_way(plan,features,result,"synthetic")
        source=prep.read_json(result/"source_audit.json")
        self.assertEqual((source["false_rare_n"],source["normal_n"]),(1,3))
        self.assertEqual(prep.read_json(result/"metrics.json")["per_class_recall"]["NORMAL"],1/3)
        self.assertEqual(len(prep.read_json(result/"metrics.json")["confusion"]),9)

    def test_reporting_gate_precedes_feature_access(self):
        with patch.object(prep,"load_lock",return_value={}), patch.object(ev,"open_features") as load:
            with self.assertRaisesRegex(ValueError,"prior nomination"):
                ev.run_arm(self.out,self.out,"DINOv2",self.out/"reports","reporting")
        load.assert_not_called()

    def test_draw_summary_uses_sample_sd_and_percentiles_without_sem(self):
        mean,sd,low,high=ev.draw_summary([0.,.5,1.])
        self.assertEqual((mean,sd),(.5,.5))
        self.assertAlmostEqual(low,.05)
        self.assertAlmostEqual(high,.95)
        self.assertEqual(ev.draw_summary([]),[None]*4)

    def test_feature_cache_boundary_preserves_raw_fp32_and_rejects_stale_preparation(self):
        folder=self.out/"lock"; cache=self.out/"cache"
        folder.mkdir(); cache.mkdir()
        inventory=[dict(path="synthetic.png",sha256="a"*64)]
        prep.write_csv(folder/"extraction_inventory.csv",["path","sha256"],[["synthetic.png","a"*64]])
        prep.write_json(folder/"freeze_summary.json",dict(files={"extraction_inventory.csv":prep.sha256(folder/"extraction_inventory.csv")}))
        prep.write_json(folder/"input_lock.json",dict(synthetic=True))
        current=dict(spec={"arm":"DINOv2"},code=prep.code_identity(),environment={"synthetic":True})
        lock=dict(arms={"DINOv2":dict(status="available",identity=current)})
        identity=dict(**current,step="extract",lock_sha256=prep.sha256(folder/"input_lock.json"),
                      input_files=prep.read_json(folder/"freeze_summary.json")["files"],
                      preparation_sha256=prep.sha256(folder/"freeze_summary.json"))
        prep.write_json(cache/"identity.json",identity)
        prep.write_json(cache/"load_report.json",dict(synthetic=True))
        prep.write_csv(cache/"geometry.csv",extract.GEOMETRY_HEADER,[["synthetic.png",32,32,256,256,16,16,224,224,.765625]])
        extract.save_features(cache/"features.npz",["synthetic.png"],["a"*64],np.full((1,768),3,dtype=np.float32))
        payload=dict(image_count=1,**{key:0. for key in ("model_loading_seconds","staging_seconds","extraction_seconds",
                     "validation_seconds","serialization_seconds","input_verification_seconds","wall_seconds")})
        payload["files"]={name:prep.sha256(cache/name) for name in extract.operation_files("extract")[:-1]}
        prep.write_json(cache/"costs.json",payload)
        prep.complete(cache,extract.operation_files("extract"))
        with patch.object(prep,"load_lock",return_value=lock):
            actual=ev.open_features(folder,cache,"DINOv2")
        self.assertEqual(actual.values.dtype,np.float32)
        np.testing.assert_array_equal(actual.values,np.full((1,768),3,dtype=np.float32))
        replace_json(cache/"identity.json",{**identity,"preparation_sha256":"0"*64})
        reseal(cache,extract.operation_files("extract"))
        with patch.object(prep,"load_lock",return_value=lock),self.assertRaisesRegex(ValueError,"identity differs"):
            ev.open_features(folder,cache,"DINOv2")


class SyntheticArtifacts(unittest.TestCase):
    """Small real artifact bundles; the input-lock validator alone is mocked."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.plan = self.root / "plan"
        self.plan.mkdir()
        prep.write_json(self.plan / "input_lock.json", {"synthetic": True})
        self.code = {"synthetic_evaluator": "v1"}
        code = patch.object(prep, "code_identity", return_value=self.code)
        code.start()
        self.addCleanup(code.stop)
        self.lock = dict(evaluation_environment=prep.math_environment(), arms={
            "RETFound": dict(status="available", timing=dict(median_seconds_per_image=1.)),
            "DINOv2": dict(status="available", timing=dict(median_seconds_per_image=.1)),
            "MIRAGE": dict(status="available", timing=dict(median_seconds_per_image=.001)),
            "Inception": dict(status="available"),
        })
        load = patch.object(prep, "load_lock", return_value=self.lock)
        load.start()
        self.addCleanup(load.stop)
        self.runs = self.root / "runs"
        self.nomination = self.root / "nomination"
        self.supports = tuple(tuple(f"fit/{c}/{i}" for i in range(8)) for c in fm.RARE)
        self.report_episode = ev.Episode(0, 1, 0, tuple(p[:1] for p in self.supports),
                                        tuple(f"query/{c}" for c in fm.RARE), fm.RARE)

    def save_decision(self, nominee="DINOv2"):
        self.nomination.mkdir()
        decision = dict(
            lock_sha256=prep.sha256(self.plan / "input_lock.json"), code=self.code, nominee=nominee,
            primary_fm=nominee,
            reporting_scope="primary nominee plus all secondary arms" if nominee else
                            "exploratory feasibility report; no comparative winner")
        payload = dict(lock_sha256=prep.sha256(self.plan / "input_lock.json"), decision=decision)
        binding = dict(**payload, binding_sha256=prep.object_hash(payload))
        prep.write_json(self.plan / "nomination_binding.json", binding)
        prep.write_json(self.nomination / "nomination_binding.json", binding)
        prep.write_json(self.nomination / "nomination.json", decision)
        prep.complete(self.nomination, ["nomination.json", "nomination_binding.json"])

    def make_run(self, arm, phase):
        out = self.runs / arm / phase
        out.mkdir(parents=True)
        prep.write_json(out / "identity.json", dict(
            arm=arm, phase=phase, code=self.code, environment={"synthetic": True},
            lock_sha256=prep.sha256(self.plan / "input_lock.json"), omit_logistic_reason="",
            nomination_sha256=prep.sha256(self.nomination / "nomination.json") if phase == "reporting" else None))
        return out

    def save_cell(self, run, episode, correct, classifier="prototype", failed=False):
        out = run / classifier / episode.name
        out.mkdir(parents=True)
        prep.write_json(out / "episode.json", dict(
            run_identity_sha256=prep.sha256(run / "identity.json"), classifier=classifier,
            fold=episode.fold, k=episode.k, draw=episode.draw, supports=[list(p) for p in episode.supports],
            queries=list(episode.queries), labels=list(episode.labels)))
        prep.write_json(out / "fit.json", dict(state="fit_failed" if failed else "complete",
                                              termination_reason="synthetic convergence failure" if failed else "synthetic"))
        names = ["episode.json", "fit.json", "costs.json"]
        if not failed:
            seen = [0] * 5
            scores = np.zeros((len(episode.queries), 5), dtype=np.float64)
            for i, label in enumerate(episode.labels):
                c = fm.RARE.index(label)
                scores[i, c if seen[c] < correct[c] else (c + 1) % 5] = 1.
                seen[c] += 1
            ev.write_predictions(out, episode.queries, episode.labels, scores, fm.RARE,
                                 "logit" if classifier == "logistic" else "cosine")
            names.extend(["predictions.csv", "metrics.json"])
        prep.write_json(out / "costs.json", dict(classifier_seconds=.01, serialization_seconds=.01))
        prep.complete(out, names)
        return out

    def nine_way_features(self):
        fitting = [[f"fit/{c}/0", c, "fitting"] for c in fm.NINE]
        queries = [[f"query/{c}", c, "query"] for c in fm.NINE]
        prep.write_csv(self.plan / "nine_way_roster.csv", ["path", "label", "role"], fitting + queries)
        return ev.Features(np.tile(np.eye(9, dtype=np.float32), (2, 1)),
                           {r[0]: i for i, r in enumerate(fitting + queries)})


    def run_report(self, out, environment=None, omission="synthetic dated omission"):
        cache = self.root / "cache"
        cache.mkdir(exist_ok=True)
        prep.write_json(cache / "complete.json", {"synthetic": True})
        with patch.object(ev, "episodes", return_value=[self.report_episode]), \
                patch.object(ev, "open_features", return_value=self.features), \
                patch.object(ev.extract, "environment", return_value=environment or {"synthetic": True}):
            ev.run_arm(self.plan, cache, "DINOv2", out, "reporting", omission, self.nomination)


class NominationAndSummaries(SyntheticArtifacts):
    def setUp(self):
        super().setUp()
        # Two unequal folds and two draws exercise the full collectors without 2,000 fits/files per arm.
        self.query_counts = np.array([[1, 2, 1, 2, 1], [9, 8, 9, 8, 9]])
        self.development = []
        for fold, counts in enumerate(self.query_counts):
            labels = tuple(c for c, n in zip(fm.RARE, counts) for _ in range(n))
            queries = tuple(f"fold{fold}/{c}/{i}" for c, n in zip(fm.RARE, counts) for i in range(n))
            for draw in range(2):
                for k in fm.K_DEV:
                    self.development.append(ev.Episode(fold, k, draw, tuple(p[:k] for p in self.supports),
                                                       queries, labels))
        self.reporting = [ev.Episode(0, k, r, tuple(p[:k] for p in self.supports),
                                     self.report_episode.queries, fm.RARE) for k in (1, 4) for r in range(3)]
        for attribute, value in (("N_FOLDS", 2), ("N_DRAWS", 2)):
            constant = patch.object(fm, attribute, value)
            constant.start()
            self.addCleanup(constant.stop)
        plan = patch.object(ev, "episodes", side_effect=lambda folder, phase:
                            self.development if phase == "development" else self.reporting)
        plan.start()
        self.addCleanup(plan.stop)

    def complete_development(self, arm, totals, missing=False):
        run = self.make_run(arm, "development")
        for episode in self.development[:-1] if missing else self.development:
            total = totals[fm.K_DEV.index(episode.k)]
            pooled = np.clip(total - np.arange(5) * 10, 0, 10)
            # Correctness is concentrated in the larger fold: averaging fold recalls is wrong.
            hits = np.minimum(pooled, self.query_counts[1]) if episode.fold else \
                np.maximum(pooled - self.query_counts[1], 0)
            self.save_cell(run, episode, hits)
        return run

    def test_pooled_nomination_and_joint_legacy_draws_end_to_end(self):
        self.complete_development("RETFound", [45, 41, 39, 35])  # .90, .82, .78, .70
        self.complete_development("DINOv2", [44, 40, 38, 35])   # .88, .80, .76, .70
        self.complete_development("MIRAGE", [50] * 4, missing=True)
        self.complete_development("Inception", [50] * 4)
        result = ev.nominate(self.plan, self.runs, self.nomination)
        self.assertEqual(result["nominee"], "DINOv2")  # Faster near-best, not the highest raw score.
        self.assertAlmostEqual(result["arms"]["RETFound"]["score"], .8)
        self.assertAlmostEqual(result["arms"]["DINOv2"]["score"], .785)
        self.assertEqual(result["arms"]["RETFound"]["per_class_queries"], [10] * 5)
        for k, expected in zip(fm.K_DEV, [.9, .82, .78, .7]):
            self.assertAlmostEqual(result["arms"]["RETFound"]["k_scores"][str(k)], expected)
        incomplete = result["arms"]["MIRAGE"]
        self.assertEqual((incomplete["available"], incomplete["planned"]), (15, 16))
        self.assertNotIn("score", incomplete)
        self.assertEqual(incomplete["missing_or_failed"], [self.development[-1].name])
        self.assertEqual(result["arms"]["Inception"]["score"], 1.)  # Excluded from S_max.
        prep.verify_complete(self.nomination)

        dino = self.make_run("DINOv2", "reporting")
        reference = self.make_run("Inception", "reporting")
        retfound = self.make_run("RETFound", "reporting")
        for episode in self.reporting:
            if episode.k != 1:
                continue
            self.save_cell(dino, episode, ([1] * 5, [1, 1, 0, 0, 0], [1, 1, 1, 1, 0])[episode.draw])
            if episode.draw != 1:
                self.save_cell(reference, episode, [1, 1, 0, 0, 0] if episode.draw == 0 else [1, 0, 0, 0, 0])
            if episode.draw == 1:
                self.save_cell(retfound, episode, [1, 0, 0, 0, 0])
            self.save_cell(dino, episode, [1, 1, 0, 0, 0], "logistic", failed=episode.draw == 1)
            self.save_cell(reference, episode, [1, 0, 0, 0, 0], "logistic", failed=episode.draw == 0)
        out = self.root / "summary"
        ev.summarize(self.plan, self.runs, self.nomination, out)
        summaries = {(r["arm"], r["classifier"], int(r["k"])): r
                     for r in prep.read_csv(out / "support_draw_summary.csv")}
        dino_proto = summaries["DINOv2", "prototype", 1]
        self.assertEqual((dino_proto["available"], dino_proto["planned"]), ("3", "100"))
        self.assertAlmostEqual(float(dino_proto["mean"]), (1. + .4 + .8) / 3)
        self.assertEqual(summaries["DINOv2", "logistic", 1]["available"], "2")
        self.assertEqual(summaries["DINOv2", "logistic", 1]["planned"], "25")
        for classifier, planned in (("prototype", "100"), ("logistic", "25")):
            for arm, k in (("MIRAGE", 1), ("DINOv2", 4)):
                row = summaries[arm, classifier, k]
                self.assertEqual((row["available"], row["planned"], row["mean"]), ("0", planned, ""))
        contrasts = {(r["arm_a"], r["arm_b"], r["classifier"], int(r["k"])): r
                     for r in prep.read_csv(out / "paired_differences.csv")}
        primary = contrasts["DINOv2", "Inception", "prototype", 1]
        self.assertEqual((primary["role"], primary["available_pairs"], primary["planned_pairs"]),
                         ("primary", "2", "100"))
        self.assertAlmostEqual(float(primary["mean_a_minus_b"]), .6)
        logistic = contrasts["DINOv2", "Inception", "logistic", 1]
        self.assertEqual((logistic["role"], logistic["available_pairs"], logistic["planned_pairs"]),
                         ("secondary", "1", "25"))
        self.assertAlmostEqual(float(logistic["mean_a_minus_b"]), .2)
        interpretation = prep.read_json(out / "interpretation.json")
        self.assertEqual(interpretation["nominee"], "DINOv2")
        self.assertIn("not run", interpretation["external"])
        self.assertIn("not new-patient confidence intervals", interpretation["variability"])
        prep.verify_complete(out)

    def test_one_complete_fm_is_feasibility_without_comparative_winner(self):
        self.complete_development("DINOv2", [40] * 4)
        self.complete_development("Inception", [50] * 4)
        self.lock["arms"]["MIRAGE"] = dict(status="unavailable", reason="synthetic access failure")
        result = ev.nominate(self.plan, self.runs, self.nomination)
        self.assertIsNone(result["nominee"])
        self.assertIsNone(result["primary_fm"])
        self.assertIn("feasibility", result["reporting_scope"])
        self.assertNotIn("score", result["arms"]["RETFound"])
        self.assertEqual(result["arms"]["MIRAGE"]["reason"], "synthetic access failure")
        self.complete_development("RETFound", [50] * 4)
        copied = self.root / "late_nomination"
        self.assertEqual(ev.nominate(self.plan, self.runs, copied), result)
        self.assertIsNone(ev.nomination_decision(self.plan, copied)["nominee"])
        report = self.make_run("DINOv2", "reporting")
        self.save_cell(report, self.reporting[0], [1] * 5)
        ev.summarize(self.plan, self.runs, self.nomination, self.root / "summary")
        self.assertIn("exploratory", prep.read_json(self.root / "summary/interpretation.json")["reporting_scope"])
        self.assertTrue(all(r["role"] == "secondary"
                            for r in prep.read_csv(self.root / "summary/paired_differences.csv")))

    def test_all_unstarted_paths_stop_without_sealing_an_empty_study(self):
        with self.assertRaisesRegex(ValueError, "no development runs.*started"):
            ev.nominate(self.plan, self.runs, self.nomination)
        self.assertFalse((self.nomination / "complete.json").exists())
        self.assertFalse((self.plan / "nomination_binding.json").exists())
        self.save_decision()
        with self.assertRaisesRegex(ValueError, "no reporting runs.*started"):
            ev.summarize(self.plan, self.runs, self.nomination, self.root / "summary")
        self.assertFalse((self.root / "summary/complete.json").exists())

    def test_actual_failed_run_remains_reportable_with_zero_available_cells(self):
        run = self.make_run("DINOv2", "development")
        self.save_cell(run, self.development[0], [0] * 5, failed=True)
        result = ev.nominate(self.plan, self.runs, self.nomination)
        self.assertIsNone(result["nominee"])
        self.assertEqual(result["arms"]["DINOv2"]["available"], 0)
        self.assertNotIn("score", result["arms"]["DINOv2"])
        run = self.make_run("DINOv2", "reporting")
        self.save_cell(run, self.reporting[0], [0] * 5, "logistic", failed=True)
        ev.summarize(self.plan, self.runs, self.nomination, self.root / "summary")
        rows = prep.read_csv(self.root / "summary/support_draw_summary.csv")
        self.assertTrue(all(r["available"] == "0" and r["mean"] == "" for r in rows))

    def test_additional_reporting_cells_require_a_new_summary_version(self):
        self.save_decision()
        decision = (self.nomination / "nomination.json").read_bytes()
        run = self.make_run("DINOv2", "reporting")
        self.save_cell(run, self.reporting[0], [1] * 5)
        first = self.root / "summary_v1"
        ev.summarize(self.plan, self.runs, self.nomination, first)
        summary = (first / "support_draw_summary.csv").read_bytes()
        self.save_cell(run, self.reporting[1], [0] * 5)
        with self.assertRaisesRegex(ValueError, "overwrite different output"):
            ev.summarize(self.plan, self.runs, self.nomination, first)
        self.assertEqual((first / "support_draw_summary.csv").read_bytes(), summary)
        prep.verify_complete(first)
        ev.summarize(self.plan, self.runs, self.nomination, self.root / "summary_v2")
        self.assertEqual((self.nomination / "nomination.json").read_bytes(), decision)
        second = prep.read_csv(self.root / "summary_v2/support_draw_summary.csv")
        self.assertTrue(any(r["arm"] == "DINOv2" and r["available"] == "2" and float(r["mean"]) == .5
                            for r in second))

    def assert_late_arm_cannot_reselect(self, stage):
        self.complete_development("RETFound", [45, 41, 39, 35])
        self.complete_development("DINOv2", [44, 40, 38, 35])
        late = self.complete_development("MIRAGE", [50] * 4, missing=True)
        original_lock = (self.plan / "input_lock.json").read_bytes()
        original = ev.nominate(self.plan, self.runs, self.nomination)
        self.assertEqual(original["nominee"], "DINOv2")
        binding_bytes = (self.plan / "nomination_binding.json").read_bytes()
        decision_bytes = (self.nomination / "nomination.json").read_bytes()
        self.features = self.nine_way_features()
        if stage in ("legacy", "summary"):
            self.run_report(self.runs / "DINOv2" / "reporting")
        if stage == "summary":
            ev.summarize(self.plan, self.runs, self.nomination, self.root / "summary_v1")
        last = self.development[-1]
        self.save_cell(late, last, self.query_counts[last.fold])
        hits, queries, _ = ev.collect_development(self.plan, late, "MIRAGE")
        self.assertEqual(fm.nomination_score(hits, queries), 1.)  # Would win a recomputed decision.
        copied = self.root / "nomination_v2"
        with patch.object(ev, "collect_development", side_effect=AssertionError("must not recompute")), \
                patch.object(fm, "nominate", side_effect=AssertionError("must not reselect")):
            self.assertEqual(ev.nominate(self.plan, self.runs, copied), original)
            self.assertEqual(ev.nominate(self.plan, self.runs, self.nomination), original)
        self.assertEqual(ev.nomination_decision(self.plan, copied), original)
        self.assertEqual((copied / "nomination.json").read_bytes(), decision_bytes)
        self.assertEqual((self.plan / "nomination_binding.json").read_bytes(), binding_bytes)
        self.assertEqual((self.plan / "input_lock.json").read_bytes(), original_lock)
        self.assertEqual(original["arms"]["MIRAGE"]["available"], 15)
        archive = self.root / "nomination.zip"
        prep.export(copied, archive)
        with zipfile.ZipFile(archive) as saved:
            self.assertEqual(saved.read("nomination_binding.json"), binding_bytes)
            self.assertEqual(saved.read("nomination.json"), decision_bytes)
        if stage == "summary":
            report = self.make_run("MIRAGE", "reporting")
            self.save_cell(report, self.reporting[0], [1] * 5)
            ev.summarize(self.plan, self.runs, copied, self.root / "summary_v2")
            for version in ("summary_v1", "summary_v2"):
                summary = self.root / version
                marker = prep.verify_complete(summary)
                self.assertTrue({"nomination.json", "nomination_binding.json"} <= marker["files"].keys())
                self.assertEqual((summary / "nomination.json").read_bytes(), decision_bytes)
                self.assertEqual(ev.nomination_decision(self.plan, summary), original)
            contrasts = prep.read_csv(self.root / "summary_v2/paired_differences.csv")
            self.assertTrue(all(row["role"] == "secondary" for row in contrasts
                                if "MIRAGE" in (row["arm_a"], row["arm_b"])))
            late_rows = prep.read_csv(self.root / "summary_v2/support_draw_summary.csv")
            self.assertTrue(any(row["arm"] == "MIRAGE" and row["available"] == "1" for row in late_rows))

    def test_new_nomination_output_copies_original_after_late_development_completion(self):
        self.assert_late_arm_cannot_reselect("development")

    def test_new_nomination_output_copies_original_after_legacy_reporting(self):
        self.assert_late_arm_cannot_reselect("legacy")

    def test_versioned_summary_keeps_original_nominee_and_late_arms_secondary(self):
        self.assert_late_arm_cannot_reselect("summary")

    def test_study_writer_lock_serializes_nominations_across_output_directories(self):
        self.save_decision()
        copied = self.root / "another_nomination"
        with prep.writer_lock(self.plan), patch.object(ev, "collect_development") as collect:
            with self.assertRaisesRegex(ValueError, "writer lock exists"):
                ev.nominate(self.plan, self.runs, copied)
        collect.assert_not_called()
        self.assertFalse(copied.exists())

    def test_missing_binding_cannot_adopt_an_existing_unbound_decision(self):
        self.save_decision()
        (self.plan / "nomination_binding.json").unlink()
        with patch.object(ev, "collect_development") as collect:
            with self.assertRaisesRegex(ValueError, "cannot adopt or replace an unbound decision"):
                ev.nominate(self.plan, self.runs, self.nomination)
        collect.assert_not_called()
        self.assertFalse((self.plan / "nomination_binding.json").exists())

    def test_invalid_binding_stops_before_creating_a_new_nomination_output(self):
        self.save_decision()
        binding = prep.read_json(self.plan / "nomination_binding.json")
        replace_json(self.plan / "nomination_binding.json", {**binding, "binding_sha256": "0" * 64})
        copied = self.root / "another_nomination"
        with patch.object(ev, "collect_development") as collect:
            with self.assertRaisesRegex(ValueError, "invalid or mismatched study-level nomination binding"):
                ev.nominate(self.plan, self.runs, copied)
        collect.assert_not_called()
        self.assertFalse(copied.exists())

    def test_interrupted_nomination_output_retries_the_bound_decision_without_reselection(self):
        self.complete_development("RETFound", [45, 41, 39, 35])
        self.complete_development("DINOv2", [44, 40, 38, 35])
        late = self.complete_development("MIRAGE", [50] * 4, missing=True)
        write = prep.write_json

        def interrupt_output(path, value):
            if path.resolve() == (self.nomination / "nomination.json").resolve():
                raise OSError("synthetic interrupted output")
            write(path, value)

        with patch.object(prep, "write_json", side_effect=interrupt_output):
            with self.assertRaisesRegex(OSError, "synthetic interrupted output"):
                ev.nominate(self.plan, self.runs, self.nomination)
        binding = prep.read_json(self.plan / "nomination_binding.json")
        self.assertEqual(binding["decision"]["nominee"], "DINOv2")
        self.assertFalse((self.nomination / "complete.json").exists())
        last = self.development[-1]
        self.save_cell(late, last, self.query_counts[last.fold])
        with patch.object(ev, "collect_development", side_effect=AssertionError("must not recompute")):
            retried = ev.nominate(self.plan, self.runs, self.nomination)
        self.assertEqual(retried, binding["decision"])
        self.assertEqual(ev.nomination_decision(self.plan, self.nomination), retried)

    def test_cross_arm_development_directory_is_rejected(self):
        run = self.make_run("RETFound", "development")
        with self.assertRaisesRegex(ValueError, "development run identity"):
            ev.collect_development(self.plan, run, "DINOv2")

    def test_collectors_reject_counts_that_numpy_would_broadcast_or_truncate(self):
        self.save_decision()
        for phase in ("development", "reporting"):
            run = self.make_run("DINOv2", phase)
            episode = self.development[0] if phase == "development" else self.reporting[0]
            cell = self.save_cell(run, episode, [0] * 5)
            metric = prep.read_json(cell / "metrics.json")
            names = list(prep.verify_complete(cell)["files"])
            for correct in ([1], [.5] * 5, [True] * 5, [-1] * 5, [100] * 5):
                with self.subTest(phase=phase, correct=correct):
                    replace_json(cell / "metrics.json", {**metric, "correct": correct})
                    reseal(cell, names)
                    with self.assertRaisesRegex(ValueError, "class counts"):
                        if phase == "development":
                            ev.collect_development(self.plan, run, "DINOv2")
                        else:
                            ev.reporting_metrics(self.plan, run, self.reporting, "DINOv2",
                                                 prep.sha256(self.nomination / "nomination.json"))

    def test_collectors_verify_present_root_markers_even_for_unfinished_runs(self):
        self.save_decision()
        for phase in ("development", "reporting"):
            with self.subTest(phase=phase):
                run = self.make_run("DINOv2", phase)
                prep.write_json(run / "summary.json", {"synthetic": True})
                prep.complete(run, ["identity.json", "summary.json"])
                replace_json(run / "summary.json", {"tampered": True})
                with self.assertRaisesRegex(ValueError, "artifact hash mismatch"):
                    if phase == "development":
                        ev.collect_development(self.plan, run, "DINOv2")
                    else:
                        ev.reporting_metrics(self.plan, run, self.reporting, "DINOv2",
                                             prep.sha256(self.nomination / "nomination.json"))

    def test_completed_root_cannot_hide_a_missing_or_unsealed_identity(self):
        self.save_decision()
        for phase in ("development", "reporting"):
            for identity_present in (False, True):
                with self.subTest(phase=phase, identity_present=identity_present):
                    arm = "DINOv2" if identity_present else "RETFound"
                    run = self.make_run(arm, phase)
                    if not identity_present:
                        (run / "identity.json").unlink()
                    prep.write_json(run / "summary.json", {"synthetic": True})
                    prep.complete(run, ["summary.json"])
                    with self.assertRaisesRegex(ValueError, "completion.*identity"):
                        if phase == "development":
                            ev.collect_development(self.plan, run, arm)
                        else:
                            ev.reporting_metrics(self.plan, run, self.reporting, arm,
                                                 prep.sha256(self.nomination / "nomination.json"))

    def test_completed_collectors_require_sealed_summary_costs_and_valid_aggregate_costs(self):
        self.save_decision()
        cases = [("unsealed_summary", {"classifiers_and_serialization_seconds": 0.}, "summary.json"),
                 ("unsealed_costs", {"classifiers_and_serialization_seconds": 0.}, "costs.json"),
                 ("missing_cost", {}, None),
                 ("negative_cost", {"classifiers_and_serialization_seconds": -1.}, None),
                 ("bool_cost", {"classifiers_and_serialization_seconds": True}, None),
                 ("text_cost", {"classifiers_and_serialization_seconds": "0"}, None)]
        for phase in ("development", "reporting"):
            for name, costs, unsealed in cases:
                with self.subTest(phase=phase, case=name):
                    run = self.root / f"{phase}_{name}"
                    run.mkdir()
                    prep.write_json(run / "identity.json", {"synthetic": True})
                    prep.write_json(run / "summary.json", {"synthetic": True})
                    prep.write_json(run / "costs.json", costs)
                    prep.complete(run, [n for n in ("identity.json", "summary.json", "costs.json") if n != unsealed])
                    expected = "completion.*summary/costs" if unsealed else "invalid.*costs"
                    with self.assertRaisesRegex(ValueError, expected):
                        if phase == "development":
                            ev.collect_development(self.plan, run, "DINOv2")
                        else:
                            ev.reporting_metrics(self.plan, run, self.reporting, "DINOv2",
                                                 prep.sha256(self.nomination / "nomination.json"))


class ResumeAndCompletion(SyntheticArtifacts):
    def setUp(self):
        super().setUp()
        self.features = self.nine_way_features()
        self.nine = self.root / "nine"

    def test_nine_way_resume_requires_every_artifact_and_matching_run(self):
        ev.nine_way(self.plan, self.features, self.nine, "synthetic")
        marker = prep.verify_complete(self.nine)
        for missing in ("head.json", "metrics.json", "predictions.csv", "source_audit.json", "costs.json"):
            with self.subTest(missing=missing):
                reseal(self.nine, [name for name in marker["files"] if name != missing])
                with patch.object(self.features, "select") as select:
                    with self.assertRaisesRegex(ValueError, "nine-way completion/identity"):
                        ev.nine_way(self.plan, self.features, self.nine, "synthetic")
                select.assert_not_called()
        replace_json(self.nine / "complete.json", marker)
        with self.assertRaisesRegex(ValueError, "different run"):
            ev.verify_nine_way(self.nine, "different-run")
        with patch.object(self.features, "select") as select:
            ev.nine_way(self.plan, self.features, self.nine, "synthetic")
        select.assert_not_called()

    def test_nine_way_rejects_duplicate_overlap_and_missing_fitting_class_before_features(self):
        roster = prep.read_csv(self.plan / "nine_way_roster.csv")
        invalid = {
            "duplicate fitting": [*roster, roster[0]],
            "fitting query overlap": [*roster, {**roster[0], "role": "query"}],
            "missing class": roster[1:],
            "unknown role": [*roster, dict(path="bad", label="NORMAL", role="validation")],
        }
        for name, rows in invalid.items():
            with self.subTest(name=name):
                with (self.plan / "nine_way_roster.csv").open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.DictWriter(stream, fieldnames=["path", "label", "role"])
                    writer.writeheader()
                    writer.writerows(rows)
                with patch.object(self.features, "select") as select:
                    with self.assertRaisesRegex(ValueError, "unique disjoint.*every fitting class"):
                        ev.nine_way(self.plan, self.features, self.nine, "synthetic")
                select.assert_not_called()

    def test_corrupt_preexisting_costs_cannot_be_sealed_on_retry(self):
        for phase in ("episode", "nine-way"):
            for costs in ({}, {"classifier_seconds": -1, "serialization_seconds": 0},
                          {"classifier_seconds": "0", "serialization_seconds": 0},
                          {"classifier_seconds": True, "serialization_seconds": 0},
                          {"classifier_seconds": 0, "serialization_seconds": float("nan")}):
                with self.subTest(phase=phase, costs=costs):
                    out = self.root / f"cost{phase}{len(list(self.root.glob('cost*')))}"
                    out.mkdir()
                    replace_json(out / "costs.json", costs)
                    with self.assertRaisesRegex(ValueError, "invalid.*costs"):
                        if phase == "episode":
                            ev.evaluate_episode(self.features, self.report_episode, "prototype", out, "synthetic")
                        else:
                            ev.nine_way(self.plan, self.features, out, "synthetic")
                    self.assertFalse((out / "complete.json").exists())

    def test_episode_resume_requires_fit_costs_and_successful_predictions(self):
        out = self.root / "episode"
        ev.evaluate_episode(self.features, self.report_episode, "prototype", out, "synthetic")
        marker = prep.verify_complete(out)
        for missing in ("episode.json", "fit.json", "costs.json", "predictions.csv", "metrics.json"):
            with self.subTest(missing=missing):
                reseal(out, [name for name in marker["files"] if name != missing])
                with patch.object(self.features, "select") as select:
                    with self.assertRaisesRegex(ValueError, "completion.*omits"):
                        ev.evaluate_episode(self.features, self.report_episode, "prototype", out, "synthetic")
                select.assert_not_called()

    def test_reporting_rejects_stale_saved_nomination_before_features(self):
        self.save_decision()
        decision = prep.read_json(self.nomination / "nomination.json")
        replace_json(self.nomination / "nomination.json", {**decision, "lock_sha256": "different-lock"})
        reseal(self.nomination, ["nomination.json"])
        with patch.object(ev, "open_features") as load:
            with self.assertRaisesRegex(ValueError, "different input lock/code"):
                ev.run_arm(self.plan, self.root / "cache", "DINOv2", self.root / "report", "reporting",
                           nomination=self.nomination)
        load.assert_not_called()

    def test_missing_or_mismatched_study_binding_stops_reports_and_summaries_before_features(self):
        self.save_decision()
        original_binding = prep.read_json(self.plan / "nomination_binding.json")
        original_decision = prep.read_json(self.nomination / "nomination.json")
        for case in ("missing", "bad_hash", "wrong_lock", "different_decision", "missing_copy", "wrong_copy"):
            with self.subTest(case=case):
                replace_json(self.plan / "nomination_binding.json", original_binding)
                replace_json(self.nomination / "nomination_binding.json", original_binding)
                replace_json(self.nomination / "nomination.json", original_decision)
                names = ["nomination.json", "nomination_binding.json"]
                if case == "missing":
                    (self.plan / "nomination_binding.json").unlink()
                elif case == "bad_hash":
                    replace_json(self.plan / "nomination_binding.json", {**original_binding, "binding_sha256": "0" * 64})
                elif case == "wrong_lock":
                    payload = dict(lock_sha256="different-lock", decision=original_decision)
                    replace_json(self.plan / "nomination_binding.json", dict(**payload, binding_sha256=prep.object_hash(payload)))
                elif case == "different_decision":
                    forged = {**original_decision, "nominee": "MIRAGE", "primary_fm": "MIRAGE"}
                    payload = dict(lock_sha256=original_binding["lock_sha256"], decision=forged)
                    replace_json(self.nomination / "nomination.json", forged)
                    replace_json(self.nomination / "nomination_binding.json", dict(**payload, binding_sha256=prep.object_hash(payload)))
                elif case == "missing_copy":
                    names.remove("nomination_binding.json")
                else:
                    replace_json(self.nomination / "nomination_binding.json", {**original_binding, "binding_sha256": "0" * 64})
                reseal(self.nomination, names)
                with patch.object(ev, "open_features") as features, patch.object(ev, "reporting_metrics") as metrics:
                    with self.assertRaisesRegex(ValueError, "nomination binding|study binding"):
                        ev.run_arm(self.plan, self.root / "cache", "DINOv2", self.root / "report", "reporting",
                                   nomination=self.nomination)
                    with self.assertRaisesRegex(ValueError, "nomination binding|study binding"):
                        ev.summarize(self.plan, self.runs, self.nomination, self.root / "summary")
                features.assert_not_called()
                metrics.assert_not_called()
                self.assertFalse((self.root / "report").exists())
                self.assertFalse((self.root / "summary").exists())

    def test_canonical_binding_accepts_an_original_decision_with_different_json_formatting(self):
        self.save_decision()
        copied = self.root / "copied_nomination"
        shutil.copytree(self.nomination, copied)
        decision = prep.read_json(copied / "nomination.json")
        replace_json(copied / "nomination.json", decision)
        reseal(copied, ["nomination.json", "nomination_binding.json"])
        self.assertNotEqual(prep.sha256(copied / "nomination.json"), prep.sha256(self.nomination / "nomination.json"))
        self.assertEqual(ev.nomination_decision(self.plan, copied), decision)

    def test_run_completion_and_export_cover_query_predictions_and_nine_way(self):
        self.save_decision()
        out = self.root / "report"
        self.run_report(out, omission="")
        marker = prep.verify_complete(out)
        predictions = f"prototype/{self.report_episode.name}/predictions.csv"
        probe_predictions = f"logistic/{self.report_episode.name}/predictions.csv"
        self.assertTrue({predictions, probe_predictions, "nine_way/predictions.csv", "nine_way/complete.json"}
                        <= marker["files"].keys())
        identity_hash = prep.sha256(out / "identity.json")
        summary = ev.verify_run(out, [self.report_episode], "", identity_hash)
        self.assertEqual(summary["episodes"]["prototype"], dict(planned=1, complete=1, failed=0))
        self.assertEqual(summary["episodes"]["logistic"], dict(planned=1, complete=1, failed=0))
        archive = self.root / "export.zip"
        prep.export(out, archive)
        with zipfile.ZipFile(archive) as saved:
            self.assertEqual(saved.read(predictions), (out / predictions).read_bytes())
            self.assertEqual(saved.read(probe_predictions), (out / probe_predictions).read_bytes())
            self.assertEqual(saved.read("nine_way/predictions.csv"), (out / "nine_way/predictions.csv").read_bytes())
        self.run_report(out, omission="")
        with self.assertRaisesRegex(ValueError, "overwrite different output"):
            self.run_report(out, environment={"synthetic": "changed full environment"}, omission="")
        reseal(out / "nine_way", ["head.json"])
        with self.assertRaisesRegex(ValueError, "nine-way completion/identity"):
            ev.verify_run(out, [self.report_episode], "", identity_hash)

    def test_corrupt_aggregate_costs_cannot_be_sealed_by_run_retry(self):
        self.save_decision()
        for i, costs in enumerate(({}, {"classifiers_and_serialization_seconds": -1},
                                   {"classifiers_and_serialization_seconds": False})):
            with self.subTest(costs=costs):
                out = self.root / f"report{i}"
                out.mkdir()
                replace_json(out / "costs.json", costs)
                with self.assertRaisesRegex(ValueError, "invalid.*costs"):
                    self.run_report(out)
                self.assertFalse((out / "complete.json").exists())


if __name__=="__main__":
    unittest.main()
