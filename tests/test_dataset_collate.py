import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from mmt.data.collate import collate_decisions
from mmt.data.dataset import DecisionDataset
from mmt.data.normalization import fit_normalizer
from mmt.storage import write_prepared


def source(values, ids, mask=None):
    values = np.asarray(values, dtype=np.float32)
    if mask is None:
        mask = np.ones(values.shape, dtype=bool)
    times = np.asarray(ids, dtype=np.float64)
    return {
        "values": values,
        "valid_mask": np.asarray(mask, dtype=bool),
        "observation_ids": np.asarray(ids, dtype=np.int64),
        "event_start": times,
        "event_end": times,
        "available_at": times,
    }


class DatasetAndCollateTests(unittest.TestCase):
    def test_dataset_uses_shared_selector_and_collates_missing_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared = write_prepared(
                root / "prepared",
                "toy",
                "prep-1",
                {
                    "s1": {
                        "imu": source([[0.0], [2.0]], [1, 2]),
                        "audio": source([[10.0, 20.0], [30.0, 40.0]], [1, 2]),
                    },
                    "s2": {"imu": source([[3.0]], [2])},
                },
                {"imu": {"feature_names": ["x"]}, "audio": {"feature_names": ["a", "b"]}},
            )
            decisions = pd.DataFrame(
                [
                    {
                        "decision_id": "d1",
                        "sequence_id": "s1",
                        "decision_time": 2.0,
                        "support_start": 1.0,
                        "support_end": 2.0,
                    },
                    {
                        "decision_id": "d2",
                        "sequence_id": "s2",
                        "decision_time": 2.0,
                        "support_start": 2.0,
                        "support_end": 2.0,
                    },
                ]
            )
            membership = pd.DataFrame(
                [
                    {"sequence_id": "s1", "group_id": "g1", "split": "train"},
                    {"sequence_id": "s2", "group_id": "g2", "split": "validation"},
                ]
            )
            targets = pd.DataFrame(
                [
                    {"decision_id": "d1", "target_class": 1, "score_eligible": True},
                    {"decision_id": "d2", "target_class": np.nan, "score_eligible": False},
                ]
            )
            arrays = {
                (sequence_id, source_id): data
                for sequence_id, sources in {
                    "s1": {
                        "imu": source([[0.0], [2.0]], [1, 2]),
                        "audio": source([[10.0, 20.0], [30.0, 40.0]], [1, 2]),
                    },
                    "s2": {"imu": source([[3.0]], [2])},
                }.items()
                for source_id, data in sources.items()
            }
            manifest_hash = hashlib.sha256((prepared / "manifest.json").read_bytes()).hexdigest()
            normalizer = fit_normalizer(
                arrays,
                [decisions.iloc[0].to_dict()],
                ["imu", "audio"],
                {
                    "manifest_hash": manifest_hash,
                    "training_membership_hash": "train-groups",
                },
            )
            dataset = DecisionDataset(
                prepared, decisions, membership, normalizer, ["imu", "audio"], targets
            )

            train_batch = collate_decisions([dataset[0]])
            validation_batch = collate_decisions([dataset[1]])
            loader_batch = next(
                iter(DataLoader(dataset, batch_size=2, collate_fn=collate_decisions))
            )

        self.assertEqual(train_batch.decision_ids, ("d1",))
        self.assertEqual(train_batch.target_valid.tolist(), [True])
        self.assertEqual(train_batch.targets.tolist(), [1])
        self.assertEqual(train_batch.inputs["imu"].values.shape, (1, 2, 1))
        self.assertEqual(train_batch.metadata[0], {"group_id": "g1", "split": "train"})
        self.assertEqual(validation_batch.inputs["audio"].lengths.tolist(), [0])
        self.assertEqual(validation_batch.inputs["audio"].values.shape, (1, 1, 2))
        self.assertEqual(validation_batch.inputs["audio"].padding_mask.tolist(), [[True]])
        self.assertFalse(validation_batch.inputs["audio"].feature_mask.any())
        self.assertFalse(validation_batch.target_valid[0])
        self.assertNotIn("group_id", validation_batch.inputs)
        self.assertEqual(loader_batch.targets.tolist(), [1, 0])
        self.assertEqual(loader_batch.inputs["audio"].lengths.tolist(), [2, 0])


if __name__ == "__main__":
    unittest.main()
