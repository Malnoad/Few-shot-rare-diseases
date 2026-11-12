"""Protocol rules that would silently change the study if they broke (FM_PROTOCOL.md v1.0)."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image
from scipy.optimize import check_grad

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fm_core as fm


def groups_of(label, n, members=1):
    names = [tuple(f"{label}/{i:02d}_{m}.png" for m in range(members)) for i in range(n)]
    return [fm.Group(g[0], label, rank, g) for rank, g in enumerate(names)]


class Screen(unittest.TestCase):
    def test_alpha_is_ignored_but_pixels_and_size_are_not(self):
        a = Image.new("RGBA", (4, 3), (10, 20, 30, 255))
        self.assertEqual(fm.rgb_fingerprint(a), fm.rgb_fingerprint(Image.new("RGBA", (4, 3), (10, 20, 30, 7))))
        self.assertNotEqual(fm.rgb_fingerprint(a), fm.rgb_fingerprint(Image.new("RGBA", (3, 4), (10, 20, 30, 255))))

    def test_dhash_sets_bit_where_left_is_brighter_row_major(self):
        gray = np.zeros((8, 9), dtype=np.uint8)
        gray[0, 0] = 200  # only the first comparison of the first row is left > right
        self.assertEqual(fm.dhash64(Image.fromarray(gray, "L").convert("RGB")), 1 << 63)

    def test_flagging_uses_inclusive_limit(self):
        self.assertEqual(fm.flag_near_duplicates({"a": 0b1111, "b": 0, "c": 0b11111}, 4), [("a", "b", 4), ("a", "c", 1)])

    def test_mixed_label_component_stops(self):
        with self.assertRaises(ValueError):
            fm.build_groups({"x": "CSC", "y": "MH"}, [("x", "y")])

    def test_group_id_is_minimum_path_and_exclusion_removes_component(self):
        labels = {"b": "CSC", "a": "CSC", "c": "CSC", "d": "CSC"}
        out = fm.build_groups(labels, [("a", "b"), ("c", "d")], excluded=["d"])["CSC"]
        self.assertEqual([(g.id, g.rank, g.members) for g in out], [("a", 0, ("a", "b"))])


class Folds(unittest.TestCase):
    def test_protocol_fold_stream_and_sorted_input_have_fixed_assignments(self):
        # Independently realized from SeedSequence([42, 0, 5101]), PCG64.
        order = [6, 5, 1, 7, 0, 3, 2, 9, 11, 4, 10, 8]
        expected = {f"CSC/{rank:02d}_0.png": i % 5 for i, rank in enumerate(order)}
        self.assertEqual(fm.assign_folds(list(reversed(groups_of("CSC", 12))), 0), expected)

    def test_round_robin_sizes(self):
        fold = fm.assign_folds(groups_of("CSC", 12), 0)
        self.assertEqual(sorted(np.bincount(list(fold.values())).tolist()), [2, 2, 2, 3, 3])

    def test_gate_needs_ten_groups_for_every_class(self):
        enough = {c: groups_of(c, 10) for c in fm.RARE}
        self.assertEqual(set(fm.check_feasible(enough).values()), {10})
        with self.assertRaises(ValueError):
            fm.check_feasible({**enough, "RP": groups_of("RP", 9)})

    def test_k16_is_all_or_nothing(self):
        counts = {c: groups_of(c, 16) for c in fm.RARE}
        self.assertEqual(fm.report_k_values(counts), fm.K_REPORT)
        self.assertEqual(fm.report_k_values({**counts, "MH": groups_of("MH", 15)}), fm.K_DEV)


class Supports(unittest.TestCase):
    def test_exact_development_and_reporting_streams(self):
        groups = groups_of("CSC", 12, members=3)
        expected = {
            6101: [(7,1),(1,2),(6,0),(2,0),(10,0),(11,0),(9,1),(3,1),(0,1),(4,2),(8,0),(5,2)],
            7101: [(4,0),(8,0),(6,2),(2,0),(5,0),(1,1),(3,0),(10,1),(0,2),(11,1),(7,0),(9,1)],
        }
        for ns, choices in expected.items():
            self.assertEqual(fm.draw_supports(groups, 0, 0, 7, ns),
                             [f"CSC/{rank:02d}_{member}.png" for rank, member in choices])

    def test_deterministic_one_image_per_group(self):
        groups = groups_of("CSC", 20, members=3)
        roster = fm.draw_supports(groups, 1, 0, 7, fm.DEV_NS)
        self.assertEqual(roster, fm.draw_supports(groups, 1, 0, 7, fm.DEV_NS))
        self.assertEqual(len({path.split("_")[0] for path in roster}), 20)

    def test_member_choice_depends_on_inventory_rank_not_permuted_position(self):
        full = groups_of("CSC", 20, members=5)
        whole = {p.split("_")[0]: p for p in fm.draw_supports(full, 2, 0, 3, fm.DEV_NS)}
        part = {p.split("_")[0]: p for p in fm.draw_supports(full[5:], 2, 0, 3, fm.DEV_NS)}
        self.assertTrue(all(whole[key] == value for key, value in part.items()))

    def test_namespaces_differ(self):
        groups = groups_of("MH", 20)
        self.assertNotEqual(fm.draw_supports(groups, 0, 1, 0, fm.DEV_NS), fm.draw_supports(groups, 0, 1, 0, fm.REPORT_NS))


class Classifiers(unittest.TestCase):
    def test_supports_are_normalized_before_averaging(self):
        supports = np.array([[100.0, 0.0], [0.0, 1.0]])
        self.assertTrue(np.allclose(fm.centroids([supports])[0], np.array([1, 1]) / np.sqrt(2)))

    def test_exact_tie_takes_first_class(self):
        self.assertEqual(fm.predict_prototype(np.eye(2), np.array([[1.0, 1.0]])).tolist(), [0])

    def test_bad_features_stop(self):
        for bad in ([[0.0, 0.0]], [[np.nan, 1.0]], [[np.inf, 1.0]], []):
            with self.assertRaises(ValueError):
                fm.unit(np.array(bad))

    def test_cancelling_centroid_stops(self):
        with self.assertRaises(ValueError):
            fm.centroids([np.array([[1., 0.], [-1., 0.]])])

    def test_unit_is_float64_and_handles_extreme_finite_magnitudes(self):
        for scale in (1e-300, 1e300):
            result = fm.unit(np.array([[scale, scale]]))
            self.assertEqual(result.dtype, np.float64)
            np.testing.assert_allclose(result, [[2 ** -0.5, 2 ** -0.5]])

    def test_mean_loss_and_fixed_penalty_are_independent_of_support_count(self):
        X, y = np.eye(2), np.arange(2)
        theta = np.array([1., 0., 0., 1., 0., 0.])
        value, grad = fm.logistic_objective(theta, X, y, 2)
        self.assertAlmostEqual(value, np.log1p(np.exp(-1)) + .01)
        repeated = fm.logistic_objective(theta, np.tile(X, (3, 1)), np.tile(y, 3), 2)
        self.assertAlmostEqual(value, repeated[0])
        np.testing.assert_allclose(grad, repeated[1])

    def test_optimizer_settings_initialization_and_gradient_gate(self):
        fake = SimpleNamespace(x=np.zeros(30), message="reported success", success=True)
        with patch.object(fm, "minimize", return_value=fake) as optimizer:
            fit = fm.fit_logistic(np.eye(5), np.arange(5))
        call = optimizer.call_args
        np.testing.assert_array_equal(call.args[1], np.zeros(30))
        self.assertEqual(call.kwargs["method"], "L-BFGS-B")
        self.assertIsNone(call.kwargs["bounds"])
        self.assertTrue(call.kwargs["jac"])
        self.assertEqual(call.kwargs["options"], dict(maxiter=1000, maxfun=20000, maxls=20,
                                                     maxcor=10, gtol=1e-6, ftol=0.0))
        self.assertFalse(fit.converged)  # optimizer.success cannot replace the gradient test
        self.assertEqual(fit.message, "reported success")

    def test_gradient_matches_finite_differences(self):
        rng = np.random.default_rng(0)
        X, y = fm.unit(rng.normal(size=(10, 4))), np.arange(10) % 5
        theta = rng.normal(size=25)
        error = check_grad(lambda t: fm.logistic_objective(t, X, y, 5)[0], lambda t: fm.logistic_objective(t, X, y, 5)[1], theta)
        self.assertLess(error, 1e-5)

    def test_penalty_covers_weights_only(self):
        X, y = fm.unit(np.eye(5)), np.arange(5)
        base = np.zeros(30)
        shifted = base.copy()
        shifted[25:] = 3.0  # the same intercept on every class leaves logit differences unchanged
        self.assertAlmostEqual(fm.logistic_objective(base, X, y, 5)[0], fm.logistic_objective(shifted, X, y, 5)[0])

    def test_fit_converges_to_stated_tolerance(self):
        fit = fm.fit_logistic(np.eye(5) + 0.1, np.arange(5))
        self.assertTrue(fit.converged, fit.message)
        self.assertLessEqual(fit.max_abs_gradient, 1e-6)


class Endpoints(unittest.TestCase):
    def test_nomination_pools_within_class_before_averaging(self):
        hits = np.zeros((1, 1, 2, 5))
        queries = np.ones((2, 5))
        queries[0, 0], queries[1, 0] = 1, 3
        hits[0, 0, 0, 0] = 1  # class 0: 1/4 pooled, whereas averaging folds would give 1/2
        self.assertAlmostEqual(fm.nomination_score(hits, queries), 0.25 / 5)

    def test_nominee_is_fastest_within_two_points_inclusive(self):
        scores = {"RETFound": 0.80, "DINOv2": 0.785, "MIRAGE": 0.70, "Inception": 0.99}
        seconds = {"RETFound": 0.9, "DINOv2": 0.1, "MIRAGE": 0.01}
        self.assertEqual(fm.nominate(scores, seconds), "DINOv2")

    def test_exact_two_point_boundary(self):
        for hundredths in range(2,101):
            with self.subTest(best=hundredths/100):
                scores={"RETFound":hundredths/100,"DINOv2":(hundredths-2)/100}
                self.assertEqual(fm.nominate(scores,{"RETFound":1.,"DINOv2":.5}),"DINOv2")
        self.assertEqual(fm.nominate({"RETFound": .8, "DINOv2": .78-1e-12},
                                     {"RETFound":1.,"DINOv2":.5}),"RETFound")

    def test_tie_order_and_fallback_slot(self):
        self.assertEqual(fm.nominate({"BioMedCLIP": 0.8, "DINOv2": 0.8}, {"BioMedCLIP": 1, "DINOv2": 1}), "BioMedCLIP")

    def test_fewer_than_two_complete_models_names_no_winner(self):
        self.assertIsNone(fm.nominate({"DINOv2": 0.8, "Inception": 0.9}, {"DINOv2": 1}))

    def test_false_rare_rate_counts_only_true_normals(self):
        self.assertEqual(fm.false_rare_rate(["NORMAL", "NORMAL", "NORMAL", "CNV"], ["RP", "CNV", "NORMAL", "CSC"]), (1, 3))


class AmendmentA2(unittest.TestCase):
    def test_size_reference_uses_mean_support_size_and_first_class_on_ties(self):
        supports = [np.array([[400, 400], [420, 400]]), np.array([[300, 300]]), np.array([[300, 300]])]
        # (305, 300) is nearest the 300x300 classes, which tie: the first wins.
        self.assertEqual(fm.size_predict(supports, np.array([[410, 401], [305, 300]])).tolist(), [0, 1])
        with self.assertRaises(ValueError):
            fm.size_predict([np.empty((0, 2)), supports[1]], np.array([[1, 1]]))

    def test_exact_interval_matches_known_clopper_pearson_values(self):
        low, high = fm.clopper_pearson(0, 10)
        self.assertEqual(low, 0)
        self.assertAlmostEqual(high, 1 - .025 ** (1 / 10))
        low, high = fm.clopper_pearson(5, 10)
        self.assertAlmostEqual(low, .187086, places=5)
        self.assertAlmostEqual(high, .812914, places=5)
        self.assertEqual(fm.clopper_pearson(10, 10)[1], 1)

    def test_cluster_interval_resamples_whole_clusters_with_a_fixed_seed(self):
        hits, clusters = [1, 1, 0, 0, 1, 0], ["a", "a", "b", "b", "c", "d"]
        first = fm.cluster_interval(hits, clusters)
        self.assertEqual(first, fm.cluster_interval(hits, clusters))
        self.assertTrue(0 <= first[0] <= .5 <= first[1] <= 1)
        # One cluster with every image correct cannot vary.
        self.assertEqual(fm.cluster_interval([1, 1], ["a", "a"]), (1., 1.))
        # Same draws as analysis.py: one integers(0, G, G) call per draw from default_rng(0).
        rng, totals = np.random.default_rng(0), np.array([[2, 2], [0, 2], [1, 1], [0, 1]])
        values = [totals[rng.integers(0, 4, 4)].sum(axis=0) for _ in range(5000)]
        expected = np.percentile([v[0] / v[1] for v in values], [2.5, 97.5])
        np.testing.assert_allclose(first, expected)

    def test_padding_shares_need_a_grayscale_image(self):
        with self.assertRaises(ValueError):
            fm.border_share(np.zeros((2, 2, 3), dtype=np.uint8))
        self.assertEqual(fm.near_white_share(np.array([[250, 249], [255, 0]], dtype=np.uint8)), .5)


if __name__ == "__main__":
    unittest.main()
