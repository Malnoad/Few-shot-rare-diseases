"""Focused cluster-bootstrap checks against explicit whole-patient image resampling."""
import unittest

import numpy as np
from numpy.testing import assert_allclose
from sklearn.metrics import matthews_corrcoef

from analysis import cluster_bootstrap, paired_cluster_bootstrap, patient_cluster


class ClusterBootstrapTests(unittest.TestCase):
    def test_patient_keys_span_classes_and_keep_rare_paths_distinct(self):
        self.assertEqual(patient_cluster(r"data\CNV-123-1.jpeg", "CNV"),
                         patient_cluster("data/DME-123-7.jpeg", "DME"))
        self.assertNotEqual(patient_cluster("CSC/01.PNG", "CSC"),
                            patient_cluster("MH/01.PNG", "MH"))
        with self.assertRaises(ValueError):
            patient_cluster("CNV-123-1.jpeg", "DME")
        with self.assertRaises(ValueError):
            patient_cluster("unrecognized.jpeg", "CNV")

    def test_intervals_match_explicit_resampling_of_unequal_cross_class_clusters(self):
        # Patient a spans two classes; averaging per-patient accuracy would give a different result.
        y = np.array([0, 1, 1, 2, 2, 0, 2])
        pa = np.array([0, 0, 1, 2, 0, 0, 1])
        pb = np.array([1, 1, 1, 0, 2, 0, 2])
        groups = np.array(["a", "a", "a", "b", "c", "c", "d"])
        keys = np.unique(groups)
        rng = np.random.default_rng(19)
        scores, differences = [], []
        for _ in range(200):
            draw = rng.integers(0, len(keys), len(keys))
            idx = np.concatenate([np.flatnonzero(groups == keys[i]) for i in draw])
            a = np.array([(y[idx] == pa[idx]).mean() * 100, matthews_corrcoef(y[idx], pa[idx])])
            b = np.array([(y[idx] == pb[idx]).mean() * 100, matthews_corrcoef(y[idx], pb[idx])])
            scores.append(a)
            differences.append(a - b)
        single = cluster_bootstrap(y, pa, groups, n_boot=200, seed=19)
        paired = paired_cluster_bootstrap(y, pa, pb, groups, n_boot=200, seed=19)
        for i, metric in enumerate(["acc", "mcc"]):
            assert_allclose(single[metric], np.percentile(np.array(scores)[:, i], [2.5, 97.5]))
            assert_allclose(paired[metric], np.percentile(np.array(differences)[:, i], [2.5, 97.5]))

    def test_singletons_reduce_to_image_resampling_and_pairing_is_preserved(self):
        y = np.array([0, 1, 1, 2, 2, 0])
        pred = np.array([0, 0, 1, 1, 2, 0])
        rng = np.random.default_rng(3)
        boot = []
        for _ in range(100):
            idx = rng.integers(0, len(y), len(y))
            boot.append((y[idx] == pred[idx]).mean() * 100)
        result = cluster_bootstrap(y, pred, np.arange(len(y)), n_boot=100, seed=3)
        assert_allclose(result["acc"], np.percentile(boot, [2.5, 97.5]))
        identical = paired_cluster_bootstrap(y, pred, pred, [0, 0, 1, 2, 2, 2], n_boot=100)
        assert_allclose(list(identical.values()), 0)

    def test_one_cluster_and_constant_predictions(self):
        y, pred = np.array([0, 0, 1]), np.array([0, 0, 0])
        result = cluster_bootstrap(y, pred, ["one"] * 3, n_boot=20)
        assert_allclose(result["acc"], [200 / 3, 200 / 3])
        assert_allclose(result["mcc"], [0, 0])
        assert_allclose(cluster_bootstrap([0], [0], [0], n_boot=20)["mcc"], [0, 0])
        with self.assertRaises(ValueError):
            cluster_bootstrap([], [], [])
        with self.assertRaises(ValueError):
            cluster_bootstrap([0, 1], [0], [0, 1])


if __name__ == "__main__":
    unittest.main()
