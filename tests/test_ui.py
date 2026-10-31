import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mmt.ui.app import (
    _completed_replays,
    _completed_runs,
    _load_predictions,
    _prepared_datasets,
    _read_manifest,
    _replay_command,
    _replay_export_rows,
    _startup_paths,
)


class ReviewerUiTests(unittest.TestCase):
    def test_corrupt_metrics_and_non_object_manifests_do_not_crash_discovery(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "metrics.json").write_text("not valid JSON")
            (root / "status.json").write_text('{"status":"completed"}')
            self.assertEqual(_completed_runs(root), {})
            (root / "manifest.json").write_text("[]")
            manifest, error = _read_manifest(root)
            self.assertIsNone(manifest)
            self.assertIn("JSON object", error)

    def test_replay_navigation_updates_button_boundaries_immediately(self):
        from streamlit.testing.v1 import AppTest

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            replay = root / "replay"
            replay.mkdir()
            (replay / "status.json").write_text('{"status":"completed"}')
            records = [
                {
                    "sequence_id": "s1",
                    "decision_id": f"d{i}",
                    "decision_time": float(i),
                    "status": "accepted",
                    "confidence": 0.5,
                }
                for i in range(2)
            ]
            (replay / "predictions.jsonl").write_text("\n".join(json.dumps(row) for row in records))
            path = Path(__file__).resolve().parents[1] / "src/mmt/ui/app.py"
            with patch(
                "sys.argv",
                ["app.py", "--runs-root", str(root), "--prepared-root", str(root / "data")],
            ):
                app = AppTest.from_file(str(path), default_timeout=30).run()

                def button(label):
                    return next(value for value in app.button if value.label == label)

                self.assertFalse(app.exception)
                self.assertTrue(button("Previous").disabled)
                button("Next").click().run()
                self.assertFalse(button("Previous").disabled)
                self.assertTrue(button("Next").disabled)
                button("Previous").click().run()
                self.assertTrue(button("Previous").disabled)
                button("Next").click().run()
                button("Reset").click().run()
                self.assertTrue(button("Previous").disabled)
                self.assertFalse(app.exception)

    def test_discovers_nested_completed_artifacts_and_preparations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            completed = root / "campaign/results/run/test"
            completed.mkdir(parents=True)
            (completed / "metrics.json").write_text("{}")
            (completed / "status.json").write_text('{"status":"completed"}')
            hidden = root / ".partial/results/run"
            hidden.mkdir(parents=True)
            (hidden / "metrics.json").write_text("{}")
            (hidden / "status.json").write_text('{"status":"completed"}')
            prepared = root / "data/pamap2/pamap2/prep"
            prepared.mkdir(parents=True)
            (prepared / "manifest.json").write_text(
                '{"source_dims":{"hand":6},"class_map":{"1":0,"2":1}}'
            )
            self.assertEqual(list(_completed_runs(root)), ["campaign/results/run/test"])
            self.assertEqual(list(_prepared_datasets(root / "data")), ["pamap2/pamap2/prep"])

    def test_replay_export_excludes_targets_and_preserves_selected_ids(self):
        import pandas as pd

        rows = _replay_export_rows(
            pd.DataFrame(
                [
                    {
                        "decision_id": "d1",
                        "sequence_id": "s1",
                        "target_class": 1,
                        "raw_sentiment": 2.0,
                        "group_id": "speaker",
                        "confidence": float("nan"),
                    }
                ]
            ),
            {"d1": {"audio": [1, 2]}},
        )
        self.assertNotIn("target_class", rows[0])
        self.assertNotIn("raw_sentiment", rows[0])
        self.assertNotIn("group_id", rows[0])
        self.assertIsNone(rows[0]["confidence"])
        self.assertEqual(rows[0]["selected_ids"], {"audio": [1, 2]})

    def test_ui_command_path_options_are_used_as_sidebar_defaults(self):
        with patch(
            "mmt.ui.app.sys.argv",
            [
                "app.py",
                "--runs-root",
                "/tmp/project-runs",
                "--prepared-root",
                "/tmp/project-data",
            ],
        ):
            self.assertEqual(_startup_paths(), ("/tmp/project-runs", "/tmp/project-data"))

    def test_export_command_requires_a_saved_schedule_for_non_clean_replay(self):
        prediction = {"run_id": "r1", "sequence_id": "s1", "scenario_id": "outage"}

        self.assertIsNone(
            _replay_command("configs/pamap2.yaml", Path("runs/r1"), prediction, "runs/new")
        )
        command = _replay_command(
            "configs/pamap2.yaml", Path("runs/r1"), prediction, "runs/new", "scenarios/outage.json"
        )

        self.assertIn("--schedule scenarios/outage.json", command)
        calibrated = _replay_command(
            "configs/pamap2.yaml",
            Path("runs/r1"),
            prediction,
            "runs/new",
            "scenarios/outage.json",
            Path("calibration.json"),
            Path("policy.json"),
        )
        self.assertIn("--calibration calibration.json", calibrated)
        self.assertIn("--policy policy.json", calibrated)

    def test_reviewer_loads_only_completed_replay_jsonl(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            complete = root / "replay-complete"
            complete.mkdir()
            (complete / "status.json").write_text(json.dumps({"status": "completed"}))
            (complete / "predictions.jsonl").write_text(
                json.dumps(
                    {
                        "sequence_id": "s1",
                        "decision_id": "d1",
                        "status": "accepted",
                        "target_class": 1,
                    }
                )
                + "\n"
            )
            incomplete = root / "replay-incomplete"
            incomplete.mkdir()
            (incomplete / "status.json").write_text(json.dumps({"status": "incomplete"}))
            (incomplete / "predictions.jsonl").write_text("{}\n")

            replays = _completed_replays(root)
            predictions = _load_predictions(replays["replay-complete"])

        self.assertEqual(list(replays), ["replay-complete"])
        self.assertEqual(predictions.loc[0, "decision_id"], "d1")


if __name__ == "__main__":
    unittest.main()
