import unittest
from datetime import datetime, timezone
from pathlib import Path

from pydantic import ValidationError

from mmt.contracts import ArtifactRef, DataConfig, PredictionRecord, RuntimeConfig


class ContractTests(unittest.TestCase):
    def test_data_config_rejects_profile_mismatch_and_unknown_fields(self):
        common = {
            "application": "pamap2",
            "manifest_path": "manifest.json",
            "partitions_path": "partitions.csv",
            "source_schema": {"hand": 6},
            "decision_policy": "segment",
            "availability_mode": "simulated",
            "feature_order": ("acc_x",),
        }

        with self.assertRaises(ValidationError):
            DataConfig(**common)
        with self.assertRaises(ValidationError):
            RuntimeConfig(max_buffer_bytes=0)

    def test_prediction_contract_rejects_nan_unknown_fields_and_fake_unavailable_scores(self):
        base = {
            "run_id": "r1",
            "scenario_id": "clean",
            "sequence_id": "s1",
            "decision_id": "d1",
            "decision_time": 1.0,
            "computable": True,
            "status": "accepted",
            "score_kind": "logits",
            "scores": (1.0, 0.0),
        }
        with self.assertRaises(ValidationError):
            PredictionRecord(**base, unrecognized="no")
        with self.assertRaises(ValidationError):
            PredictionRecord(**{**base, "scores": (float("nan"), 0.0)})
        with self.assertRaises(ValidationError):
            PredictionRecord(**{**base, "computable": False, "status": "unavailable_no_data"})

    def test_artifact_requires_utc_and_sha256(self):
        base = {
            "artifact_type": "weights",
            "path": Path("weights.pt"),
            "sha256": "a" * 64,
            "schema_version": "1.0",
            "producer": "test",
        }
        with self.assertRaises(ValidationError):
            ArtifactRef(**base, created_at_utc=datetime(2026, 1, 1))
        artifact = ArtifactRef(**base, created_at_utc=datetime(2026, 1, 1, tzinfo=timezone.utc))
        self.assertEqual(artifact.schema_version, "1.0")


if __name__ == "__main__":
    unittest.main()
