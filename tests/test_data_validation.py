import tempfile
import unittest

import numpy as np
import pandas as pd

from mmt.services.data_validation import validate_prepared_splits
from mmt.storage import write_prepared


def source(value, observation_id):
    return {
        "values": np.asarray([[value]], dtype=np.float32),
        "valid_mask": np.asarray([[True]], dtype=bool),
        "observation_ids": np.asarray([observation_id], dtype=np.int64),
        "event_start": np.asarray([0.0]),
        "event_end": np.asarray([0.0]),
        "available_at": np.asarray([0.0]),
    }


class DataValidationTests(unittest.TestCase):
    def test_prepared_split_validation_checks_ids_support_and_hashes(self):
        membership = pd.DataFrame(
            {
                "sequence_id": ["s1", "s2", "s3", "s4"],
                "group_id": ["g1", "g2", "g3", "g4"],
                "split": ["train", "validation", "calibration", "test"],
            }
        )
        decisions = pd.DataFrame(
            {
                "decision_id": ["d1", "d2", "d3", "d4"],
                "sequence_id": ["s1", "s2", "s3", "s4"],
                "decision_time": [0.0] * 4,
                "support_start": [0.0] * 4,
                "support_end": [0.0] * 4,
            }
        )
        targets = pd.DataFrame(
            {
                "decision_id": ["d1", "d2", "d3", "d4"],
                "target_class": [0, 0, 0, 0],
                "target_time": [0.0] * 4,
                "score_eligible": [True] * 4,
                "exclusion_reason": [None] * 4,
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            prepared = write_prepared(
                directory,
                "toy",
                "p1",
                {f"s{i}": {"imu": source(float(i), i)} for i in range(1, 5)},
                {"imu": {"feature_names": ["x"]}},
                extra_manifest={"class_map": {"1": 0}},
                tables={
                    "partitions.csv": membership,
                    "decisions.csv": decisions,
                    "targets.csv": targets,
                },
            )
            report = validate_prepared_splits(prepared / "manifest.json")

        self.assertEqual(report["counts"]["train"]["eligible"], 1)
        self.assertEqual(report["class_support"]["train"], {0: 1})
        self.assertEqual(len(report["manifest_hash"]), 64)


if __name__ == "__main__":
    unittest.main()
