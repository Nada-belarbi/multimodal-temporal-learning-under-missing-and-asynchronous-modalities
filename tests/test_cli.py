import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import yaml
from typer.testing import CliRunner

from mmt.cli import app
from mmt.storage import load_manifest


class CliSmokeTests(unittest.TestCase):
    def test_help_is_available(self):
        result = CliRunner().invoke(app, ["--help"])

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("Multimodal temporal learning tools", " ".join(result.output.split()))

    def test_version_is_available(self):
        result = CliRunner().invoke(app, ["--version"])

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(result.output.strip(), "0.1.0")

    def test_config_validate_reports_a_machine_readable_success(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "pyproject.toml").write_text("[project]\nname='fixture'\n")
            (root / "configs").mkdir()
            (root / "data").mkdir()
            (root / "data" / "manifest.json").write_text("{}")
            (root / "data" / "partitions.csv").write_text("sequence_id,split\ns1,train\n")
            config = {
                "data": {
                    "application": "pamap2",
                    "manifest_path": "../data/manifest.json",
                    "partitions_path": "../data/partitions.csv",
                    "source_schema": {"hand": 6},
                    "decision_policy": "window",
                    "availability_mode": "simulated",
                    "feature_order": ["acc_x"],
                },
                "run": {
                    "run_id": "r1",
                    "model_name": "v0",
                    "seed": 11,
                    "regime": "clean",
                    "partition_hashes": {"train": "a" * 64},
                    "output_root": "runs/r1",
                },
            }
            config_path = root / "configs" / "fixture.yaml"
            config_path.write_text(yaml.safe_dump(config))

            result = CliRunner().invoke(app, ["config", "validate", "--config", str(config_path)])

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn('"status": "valid"', result.output)

    def test_pamap2_audit_then_prepare_cli_uses_separate_immutable_outputs(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "pyproject.toml").write_text("[project]\nname='fixture'\n")
            (root / "configs").mkdir()
            raw_root = root / "data" / "raw"
            protocol = raw_root / "Protocol"
            protocol.mkdir(parents=True)
            for index, sequence_id in enumerate(("s1", "s2", "s3", "s4")):
                values = np.zeros((6, 54), dtype=np.float64)
                values[:, 0] = np.arange(6)
                values[:, 1] = 1
                np.savetxt(protocol / f"{sequence_id}.dat", values)
            membership_path = root / "data" / "partitions.csv"
            membership_path.parent.mkdir(exist_ok=True)
            pd.DataFrame(
                {
                    "sequence_id": ["s1", "s2", "s3", "s4"],
                    "group_id": ["p1", "p2", "p3", "p4"],
                    "split": ["train", "validation", "calibration", "test"],
                }
            ).to_csv(membership_path, index=False)
            config = {
                "application": "pamap2",
                "raw_root": "../data/raw",
                "membership_path": "../data/partitions.csv",
                "prepared_root": "../data/prepared",
                "audit_root": "../runs/audit",
                "preparation_id": "fixture-v1",
                "recordings": [
                    {
                        "sequence_id": sequence_id,
                        "relative_path": f"Protocol/{sequence_id}.dat",
                        "provider_reference": "synthetic-test-fixture",
                        "retrieved_at": "2026-10-25T00:00:00Z",
                        "declared_version": "fixture-v1",
                        "reader_status": "synthetic-only",
                    }
                    for sequence_id in ("s1", "s2", "s3", "s4")
                ],
                "chunksize": 2,
            }
            config_path = root / "configs" / "pamap2.yaml"
            config_path.write_text(yaml.safe_dump(config))

            audit_result = CliRunner().invoke(app, ["audit", "--config", str(config_path)])
            prepare_result = CliRunner().invoke(app, ["prepare", "--config", str(config_path)])

            self.assertEqual(audit_result.exit_code, 0, audit_result.output)
            self.assertEqual(prepare_result.exit_code, 0, prepare_result.output)
            self.assertTrue((root / "runs/audit/audit-fixture-v1/inventory.csv").is_file())
            prepared = root / "data/prepared/pamap2/fixture-v1"
            self.assertEqual(load_manifest(prepared)["availability_mode"], "simulated")


if __name__ == "__main__":
    unittest.main()
