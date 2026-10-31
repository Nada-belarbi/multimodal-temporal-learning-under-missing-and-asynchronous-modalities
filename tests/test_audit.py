import csv
import hashlib
import json
import runpy
import tempfile
import unittest
from pathlib import Path

from mmt.data.audit import AuditError, inventory_dataset_files, write_audit_bundle


class Pamap2PartitionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.script = runpy.run_path(str(Path(__file__).resolve().parents[1] / "audit_pamap2.py"))

    def audit_fixture(self):
        classes = self.script["EXPECTED_CLASSES"]
        return {
            "recordings": {
                f"subject{index}": {
                    "class_support": {str(class_id): index for class_id in classes},
                    "row_count": index * len(classes),
                }
                for index in range(101, 105)
            }
        }

    def test_partition_is_deterministic_disjoint_and_class_complete(self):
        audit = self.audit_fixture()
        sizes = dict.fromkeys(("train", "validation", "calibration", "test"), 1)
        groups, report = self.script["choose_partition"](audit, sizes)
        reversed_audit = {"recordings": dict(reversed(list(audit["recordings"].items())))}
        repeated, _ = self.script["choose_partition"](reversed_audit, sizes)
        self.assertEqual(groups, repeated)
        subjects = [subject for group in groups.values() for subject in group]
        self.assertEqual(len(subjects), len(set(subjects)))
        self.assertEqual(set(subjects), set(audit["recordings"]))
        self.assertEqual(report["selection"]["evaluated_assignments"], 24)
        self.assertEqual(report["selection"]["feasible_assignments"], 24)
        self.assertTrue(
            all(
                count > 0
                for counts in report["class_support_by_split"].values()
                for count in counts.values()
            )
        )

    def test_impossible_coverage_and_invalid_sizes_fail(self):
        audit = self.audit_fixture()
        sizes = dict.fromkeys(("train", "validation", "calibration", "test"), 1)
        for subject in ("subject102", "subject103", "subject104"):
            audit["recordings"][subject]["class_support"]["24"] = 0
        with self.assertRaises(self.script["AuditError"]):
            self.script["choose_partition"](audit, sizes)
        with self.assertRaises(self.script["AuditError"]):
            self.script["choose_partition"](self.audit_fixture(), sizes | {"train": 2})

    def test_incomplete_class_dictionary_fails(self):
        audit = self.audit_fixture()
        del audit["recordings"]["subject101"]["class_support"]["24"]
        with self.assertRaises(self.script["AuditError"]):
            self.script["choose_partition"](
                audit, dict.fromkeys(("train", "validation", "calibration", "test"), 1)
            )


class DatasetAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "Protocol").mkdir()
        (self.root / "Protocol" / "subject.dat").write_bytes(b"fixture bytes")
        self.input = {
            "relative_path": "Protocol/subject.dat",
            "provider_reference": "https://example.invalid/dataset",
            "retrieved_at": "2026-10-25T00:00:00Z",
            "declared_version": "fixture-1",
            "reader_status": "pending-reader-audit",
        }

    def tearDown(self):
        self.temp.cleanup()

    def test_inventory_records_only_declared_paths_and_hashes_content(self):
        inventory = inventory_dataset_files(self.root, [self.input])

        self.assertEqual(inventory[0]["relative_path"], "Protocol/subject.dat")
        self.assertEqual(inventory[0]["bytes"], len(b"fixture bytes"))
        self.assertEqual(inventory[0]["sha256"], hashlib.sha256(b"fixture bytes").hexdigest())

    def test_rejects_traversal_duplicates_missing_files_and_incomplete_provenance(self):
        with self.assertRaises(AuditError):
            inventory_dataset_files(self.root, [{**self.input, "relative_path": "../secret"}])
        with self.assertRaises(AuditError):
            inventory_dataset_files(self.root, [self.input, self.input])
        with self.assertRaises(AuditError):
            inventory_dataset_files(self.root, [{**self.input, "relative_path": "missing.dat"}])
        with self.assertRaises(AuditError):
            inventory_dataset_files(self.root, [{**self.input, "provider_reference": ""}])

    def test_publishes_inventory_and_audit_atomically_without_overwrite(self):
        inventory = inventory_dataset_files(self.root, [self.input])
        output = self.root / "audit" / "run-1"

        write_audit_bundle(output, inventory, {"application": "pamap2", "qualified": False})

        with (output / "inventory.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(rows[0]["relative_path"], "Protocol/subject.dat")
        self.assertFalse(json.loads((output / "audit.json").read_text())["qualified"])
        with self.assertRaises(FileExistsError):
            write_audit_bundle(output, inventory, {})


if __name__ == "__main__":
    unittest.main()
