import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from mmt.data.preparation import prepare_pamap2_dataset
from mmt.storage import load_manifest, load_sequence


class Pamap2PreparationServiceTests(unittest.TestCase):
    def test_preparation_connects_audit_adapter_splits_decisions_and_storage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw_root = root / "raw"
            protocol_dir = raw_root / "Protocol"
            protocol_dir.mkdir(parents=True)
            membership = pd.DataFrame(
                {
                    "sequence_id": ["s1", "s2", "s3", "s4"],
                    "group_id": ["p1", "p2", "p3", "p4"],
                    "split": ["train", "validation", "calibration", "test"],
                }
            )
            paths = {}
            provenance = {}
            for sequence_id in membership.sequence_id:
                values = np.zeros((6, 54), dtype=np.float64)
                values[:, 0] = np.arange(6, dtype=np.float64)
                values[:, 1] = 1
                for column in range(54):
                    if column not in (0, 1):
                        values[:, column] = column + np.arange(6)
                path = protocol_dir / f"{sequence_id}.dat"
                np.savetxt(path, values)
                paths[sequence_id] = path.relative_to(raw_root)
                provenance[sequence_id] = {
                    "provider_reference": "synthetic-fixture",
                    "retrieved_at": "2026-10-25T00:00:00Z",
                    "declared_version": "fixture-v1",
                    "reader_status": "synthetic-test-only",
                }

            result = prepare_pamap2_dataset(
                raw_root,
                root / "prepared",
                root / "audit" / "prep-1",
                paths,
                membership,
                provenance,
                "prep-1",
                chunksize=2,
            )

            manifest = load_manifest(result.prepared_path)
            recording = load_sequence(result.prepared_path, "s1")
            decisions = pd.read_csv(result.prepared_path / "decisions.csv")
            targets = pd.read_csv(result.prepared_path / "targets.csv")
            audit_inventory_exists = (result.audit_dir / "inventory.csv").is_file()

        self.assertEqual(manifest["application"], "pamap2")
        self.assertEqual(manifest["availability_mode"], "simulated")
        self.assertEqual(manifest["source_order"], ["hand", "chest", "ankle", "heart_rate"])
        self.assertEqual(
            manifest["source_dims"], {"heart_rate": 1, "hand": 6, "chest": 6, "ankle": 6}
        )
        self.assertEqual(recording["hand"]["observation_ids"].tolist(), list(range(6)))
        self.assertEqual(
            decisions.groupby("sequence_id").size().to_dict(), {key: 1 for key in paths}
        )
        self.assertEqual(targets.target_class.tolist(), [0, 0, 0, 0])
        self.assertEqual(result.audit["status"], "audit_pending_review")
        self.assertTrue(audit_inventory_exists)


if __name__ == "__main__":
    unittest.main()
