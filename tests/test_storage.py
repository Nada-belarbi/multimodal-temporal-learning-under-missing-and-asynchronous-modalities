import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from mmt.data.selection import select_observations
from mmt.storage import PreparedError, load_manifest, load_sequence, write_prepared


def make_source(n=5, features=3):
    times = np.arange(n, dtype=float)
    mask = np.ones((n, features), dtype=bool)
    mask[1, 0] = False
    values = np.zeros((n, features), dtype=np.float32)
    values[3, features - 1] = 7.0
    values[1, 0] = np.nan
    return {
        "values": values,
        "valid_mask": mask,
        "observation_ids": np.arange(n),
        "event_start": times,
        "event_end": times + 0.5,
        "available_at": times + 0.5,
    }


METADATA = {"imu": {"modality": "imu"}}


def write(root, preparation_id="p1", source=None, sequence_id="s1", **kwargs):
    return write_prepared(
        root,
        "toy",
        preparation_id,
        {sequence_id: {"imu": source if source is not None else make_source()}},
        METADATA,
        **kwargs,
    )


class PreparedStorageTests(unittest.TestCase):
    def test_round_trip_preserves_measured_zero_and_missing_mask(self):
        with TemporaryDirectory() as directory:
            base = write(directory)
            manifest = load_manifest(base)
            source = load_sequence(base, "s1")["imu"]

            self.assertEqual(manifest["source_dims"], {"imu": 3})
            self.assertTrue(source["valid_mask"][2, 1])
            self.assertFalse(source["valid_mask"][1, 0])
            self.assertEqual(source["values"][1, 0], 0.0)
            self.assertEqual(manifest["sequence_stats"]["s1"]["imu"]["n_nonfinite_replaced"], 1)

    def test_loaded_sequence_can_be_passed_to_member4_selector(self):
        with TemporaryDirectory() as directory:
            base = write(directory)
            source = load_sequence(base, "s1")["imu"]

            selected = select_observations(
                source, {"decision_time": 2.5, "support_start": 0.0, "support_end": 2.5}
            )

            self.assertEqual(selected.observation_ids, (0, 1, 2))
            self.assertEqual(selected.source_status, "present")
            self.assertEqual(selected.values[1][0], 0.0)
            self.assertFalse(selected.valid_mask[1][0])

    def test_rejects_causality_violation_and_cleans_temporary_directory(self):
        with TemporaryDirectory() as directory:
            source = make_source()
            source["available_at"] = source["event_end"] - 1
            with self.assertRaises(PreparedError):
                write(directory, "p2", source)
            self.assertFalse((Path(directory) / "toy" / "p2").exists())
            self.assertEqual(list((Path(directory) / "toy").glob(".*tmp*")), [])

    def test_rejects_duplicate_observation_ids(self):
        with TemporaryDirectory() as directory:
            source = make_source()
            source["observation_ids"] = np.zeros(5)
            with self.assertRaises(PreparedError):
                write(directory, "p3", source)

    def test_rejects_mixed_dimensions_across_sequences(self):
        with TemporaryDirectory() as directory:
            with self.assertRaises(PreparedError):
                write_prepared(
                    directory,
                    "toy",
                    "p4",
                    {"a": {"imu": make_source(features=3)}, "b": {"imu": make_source(features=2)}},
                    METADATA,
                )

    def test_detects_tampered_array(self):
        with TemporaryDirectory() as directory:
            base = write(directory, "p5")
            values_path = base / "sequences" / "s1" / "imu" / "values.npy"
            values_path.write_bytes(values_path.read_bytes() + b"x")
            with self.assertRaises(PreparedError):
                load_manifest(base)

    def test_rejects_incompatible_schema_version(self):
        with TemporaryDirectory() as directory:
            base = write(directory, "p6")
            manifest_path = base / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["schema_version"] = "0.1"
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaises(PreparedError):
                load_manifest(base)

    def test_refuses_to_overwrite_existing_preparation(self):
        with TemporaryDirectory() as directory:
            write(directory, "p7")
            with self.assertRaises(FileExistsError):
                write(directory, "p7")

    def test_rejects_unsafe_identifiers_table_paths_and_manifest_overrides(self):
        with TemporaryDirectory() as directory:
            with self.assertRaises(PreparedError):
                write_prepared(directory, "../escape", "p8", {}, METADATA)
            with self.assertRaises(PreparedError):
                write_prepared(
                    directory, "toy", "p9", {"s1": {"../escape": make_source()}}, {"../escape": {}}
                )
            with self.assertRaises(PreparedError):
                write(directory, "p10", tables={"../escape.csv": object()})
            with self.assertRaises(PreparedError):
                write(directory, "p11", extra_manifest={"schema_version": "spoofed"})

    def test_rejects_checksum_paths_that_escape_prepared_root(self):
        with TemporaryDirectory() as directory:
            base = write(directory)
            manifest_path = base / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["checksums"]["../outside"] = "unused"
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaises(PreparedError):
                load_manifest(base)


if __name__ == "__main__":
    unittest.main()
