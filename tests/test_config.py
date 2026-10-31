import json
import tempfile
import unittest
from pathlib import Path

import yaml

from mmt.config import ALLOWED_OVERRIDES, ConfigError, resolve_config, save_resolved_config


class ConfigResolverTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "pyproject.toml").write_text("[project]\nname='fixture'\n")
        (self.root / "configs").mkdir()
        (self.root / "data").mkdir()
        (self.root / "data" / "manifest.json").write_text('{"schema_version":"1.0"}')
        (self.root / "data" / "partitions.csv").write_text("sequence_id,split\ns1,train\n")
        self.config_path = self.root / "configs" / "fixture.yaml"
        self.config_data = {
            "schema_version": "1.0",
            "data": {
                "application": "pamap2",
                "manifest_path": "../data/manifest.json",
                "partitions_path": "../data/partitions.csv",
                "source_schema": {"hand": 6},
                "decision_policy": "window",
                "availability_mode": "simulated",
                "feature_order": ["acc_x", "acc_y"],
            },
            "run": {
                "run_id": "fixture-run",
                "model_name": "v0",
                "seed": 11,
                "regime": "clean",
                "partition_hashes": {"train": "a" * 64},
                "model_config": {},
                "training": {},
                "output_root": "runs/fixture-run",
            },
            "runtime": {"max_active_sequences": 1},
        }
        self.write_config()

    def tearDown(self):
        self.temp.cleanup()

    def write_config(self, payload=None):
        self.config_path.write_text(yaml.safe_dump(payload or self.config_data, sort_keys=False))

    def test_resolves_relative_paths_and_hashes_inputs(self):
        result = resolve_config(self.config_path)

        self.assertEqual(
            result.config.data.manifest_path, (self.root / "data" / "manifest.json").resolve()
        )
        self.assertEqual(set(result.input_hashes), {"data.manifest_path", "data.partitions_path"})
        self.assertEqual(len(result.config_sha256), 64)

    def test_declared_overrides_change_hash_and_unknown_overrides_fail(self):
        changed = resolve_config(self.config_path, {"run.seed": 22})
        original = resolve_config(self.config_path)

        self.assertNotEqual(changed.config_sha256, original.config_sha256)
        self.assertEqual(changed.config.run.seed, 22)
        renamed = resolve_config(self.config_path, {"run.run_id": "new-immutable-run"})
        self.assertEqual(renamed.config.run.run_id, "new-immutable-run")
        self.assertNotEqual(renamed.config_sha256, original.config_sha256)
        self.assertEqual(
            set(ALLOWED_OVERRIDES),
            {
                "run.model_name",
                "run.seed",
                "run.run_id",
                "run.regime",
                "run.output_root",
                "runtime.strict_errors",
            },
        )
        with self.assertRaises(ConfigError):
            resolve_config(self.config_path, {"run.unrecognized": True})

    def test_unknown_yaml_fields_and_missing_inputs_fail_before_use(self):
        invalid = dict(self.config_data)
        invalid["surprise"] = True
        self.write_config(invalid)
        with self.assertRaises(ConfigError):
            resolve_config(self.config_path)

        self.write_config()
        (self.root / "data" / "manifest.json").unlink()
        with self.assertRaises(ConfigError):
            resolve_config(self.config_path)

    def test_paths_outside_project_need_explicit_input_root(self):
        outside = Path(tempfile.gettempdir()) / f"mmt-input-{self.root.name}.json"
        outside.write_text("{}")
        try:
            invalid = dict(self.config_data)
            invalid["data"] = dict(invalid["data"], manifest_path=str(outside))
            self.write_config(invalid)
            with self.assertRaises(ConfigError):
                resolve_config(self.config_path)
        finally:
            outside.unlink(missing_ok=True)

    def test_saves_resolved_config_and_hash(self):
        result = resolve_config(self.config_path)
        output = self.root / "runs" / "fixture-run"

        config_path, hash_path = save_resolved_config(result, output)

        payload = json.loads(result.canonical_json)
        self.assertTrue(config_path.is_file())
        self.assertEqual(yaml.safe_load(config_path.read_text()), payload)
        self.assertIn(result.config_sha256, hash_path.read_text())


if __name__ == "__main__":
    unittest.main()
