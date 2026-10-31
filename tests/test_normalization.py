import tempfile
import unittest
from pathlib import Path

import numpy as np

from mmt.data.normalization import (
    NormalizerError,
    fit_normalizer,
    load_normalizer,
    save_normalizer,
)
from mmt.data.selection import select_observations


def source(values, ids, mask=None):
    values = np.asarray(values, dtype=np.float32)
    if mask is None:
        mask = np.ones(values.shape, dtype=bool)
    times = np.asarray(ids, dtype=np.float64)
    return {
        "observation_ids": np.asarray(ids, dtype=np.int64),
        "values": values,
        "valid_mask": np.asarray(mask, dtype=bool),
        "event_start": times,
        "event_end": times,
        "available_at": times,
    }


class NormalizationTests(unittest.TestCase):
    def test_fast_and_unsorted_paths_match_selector_with_delay_and_crossing_intervals(self):
        raw = source([[1.0], [3.0], [5.0], [7.0]], [1, 2, 3, 4])
        raw["event_start"] = np.asarray([0.0, 1.0, 2.0, 3.0])
        raw["available_at"] = np.asarray([1.0, 2.0, 5.0, 4.0])
        decision = {
            "sequence_id": "s1",
            "decision_time": 4.0,
            "support_start": 1.0,
            "support_end": 4.0,
            "split": "train",
        }
        reference = select_observations(raw, decision)
        self.assertEqual(reference.observation_ids, (2, 4))
        identity = {"manifest_hash": "m", "training_membership_hash": "t"}
        for permutation in (np.arange(4), np.asarray([3, 1, 0, 2])):
            permuted = {key: values[permutation] for key, values in raw.items()}
            normalizer = fit_normalizer({("s1", "imu"): permuted}, [decision], ["imu"], identity)
            self.assertEqual(normalizer.counts["imu"].tolist(), [2])
            np.testing.assert_allclose(normalizer.means["imu"], [5.0])
            np.testing.assert_allclose(normalizer.scales["imu"], [2.0])
        for bounds in ({"support_end": 5.0}, {"support_start": float("nan")}):
            with self.assertRaises(NormalizerError):
                fit_normalizer({("s1", "imu"): raw}, [decision | bounds], ["imu"], identity)

    def test_unique_observations_match_population_statistics_and_overlap_does_not_duplicate(self):
        values = np.asarray([[1.0], [3.0], [5.0]], dtype=np.float32)
        data = {("s1", "imu"): source(values, [1, 2, 3])}
        decisions = [
            {
                "sequence_id": "s1",
                "decision_id": "d1",
                "decision_time": 3.0,
                "support_start": 1.0,
                "support_end": 3.0,
                "split": "train",
            },
            {
                "sequence_id": "s1",
                "decision_id": "d2",
                "decision_time": 3.0,
                "support_start": 2.0,
                "support_end": 3.0,
                "split": "train",
            },
        ]

        normalizer = fit_normalizer(
            data,
            decisions,
            ["imu"],
            {
                "manifest_hash": "manifest",
                "training_membership_hash": "train-groups",
            },
        )

        self.assertEqual(normalizer.counts["imu"].tolist(), [3])
        self.assertEqual(normalizer.means["imu"][0], 3.0)
        self.assertAlmostEqual(normalizer.scales["imu"][0], np.sqrt(8.0 / 3.0))

    def test_invalid_values_are_excluded_and_transformed_invalids_stay_zero(self):
        data = {
            ("s1", "imu"): source([[0.0, 99.0], [2.0, 0.0]], [1, 2], [[True, False], [True, True]])
        }
        decision = {
            "sequence_id": "s1",
            "decision_time": 2.0,
            "support_start": 1.0,
            "support_end": 2.0,
            "split": "train",
        }
        normalizer = fit_normalizer(
            data,
            [decision],
            ["imu"],
            {
                "manifest_hash": "m",
                "training_membership_hash": "t",
            },
        )

        transformed, mask = normalizer.transform(
            "imu",
            np.asarray([[0.0, 99.0], [0.0, 0.0]]),
            np.asarray([[True, False], [False, False]]),
        )

        self.assertEqual(normalizer.counts["imu"].tolist(), [2, 1])
        self.assertEqual(normalizer.means["imu"].tolist(), [1.0, 0.0])
        self.assertEqual(float(transformed[0, 1]), 0.0)
        self.assertFalse(bool(mask[0, 1]))
        self.assertTrue(np.isfinite(transformed).all())

    def test_rejects_validation_rows_and_missing_training_source(self):
        data = {("s1", "imu"): source([[1.0]], [1])}
        valid_decision = {
            "sequence_id": "s1",
            "decision_time": 1.0,
            "support_start": 1.0,
            "support_end": 1.0,
            "split": "validation",
        }
        identity = {"manifest_hash": "m", "training_membership_hash": "t"}
        with self.assertRaises(NormalizerError):
            fit_normalizer(data, [valid_decision], ["imu"], identity)
        with self.assertRaises(NormalizerError):
            fit_normalizer(data, [valid_decision | {"split": "train"}], ["missing"], identity)

    def test_save_load_checks_identity_and_preserves_transform_semantics(self):
        data = {("s1", "imu"): source([[1.0], [3.0], [5.0]], [1, 2, 3])}
        decision = {
            "sequence_id": "s1",
            "decision_time": 3.0,
            "support_start": 1.0,
            "support_end": 3.0,
            "split": "train",
        }
        identity = {"manifest_hash": "m", "training_membership_hash": "t"}
        normalizer = fit_normalizer(data, [decision], ["imu"], identity)
        with tempfile.TemporaryDirectory() as directory:
            path = save_normalizer(Path(directory) / "normalizer.npz", normalizer)
            loaded = load_normalizer(path, identity)
            with self.assertRaises(NormalizerError):
                load_normalizer(path, {"manifest_hash": "other", "training_membership_hash": "t"})

        transformed, mask = loaded.transform(
            "imu", np.asarray([[1.0], [0.0]]), np.asarray([[True], [False]])
        )
        self.assertAlmostEqual(float(transformed[0, 0]), -2.0 / np.sqrt(8.0 / 3.0))
        self.assertEqual(float(transformed[1, 0]), 0.0)
        self.assertEqual(mask.tolist(), [[True], [False]])


if __name__ == "__main__":
    unittest.main()
