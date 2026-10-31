import json
import runpy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from mmt.services.runs import RunDirectory, RunLifecycleError, write_sha256sums


class RunLifecycleTests(unittest.TestCase):
    def test_replay_comparison_uses_saved_batch_scores_and_rejects_different_status(self):
        script = runpy.run_path(str(Path(__file__).resolve().parents[1] / "replay_study.py"))
        record = SimpleNamespace(
            decision_id="decision",
            computable=True,
            status="accepted",
            scores=[0.0, 0.0],
            probabilities=[0.5, 0.5],
            predicted_class=0,
        )
        expected = {"computable": True, "status": "accepted", "scores": "0;0", "predicted_class": 0}
        script["compare_exported_prediction"](record, expected, 1.0)
        with self.assertRaises(AssertionError):
            script["compare_exported_prediction"](
                record, expected | {"status": "unavailable_no_data"}, 1.0
            )

    def test_partial_robustness_catalogue_has_shared_declared_conditions(self):
        script = runpy.run_path(str(Path(__file__).resolve().parents[1] / "robustness_study.py"))
        mosi = script["catalogue"](["language", "audio", "visual"], "mosi")
        pamap = script["catalogue"](["hand", "chest", "ankle", "heart_rate"], "pamap2")
        self.assertEqual(len(mosi), 11)
        self.assertEqual(len(pamap), 12)
        self.assertEqual(len({row["id"] for row in mosi}), 11)
        self.assertTrue(all(row["R"] == "segment_duration" for row in mosi))
        self.assertTrue(all(row["R"] == 5.0 for row in pamap))
        self.assertEqual({row["kind"] for row in mosi}, {"clean", "subset", "removal", "delay"})

    def test_campaign_summary_keeps_real_metrics_and_all_seed_results(self):
        script = runpy.run_path(str(Path(__file__).resolve().parents[1] / "run_experiments.py"))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entries = []
            for seed, f1 in ((11, 0.4), (22, 0.6), (33, 0.8)):
                run_id = f"mosi-v1-{seed}"
                entries.append(
                    {"application": "mosi", "model": "v1", "seed": seed, "run_id": run_id}
                )
                output = root / "results" / run_id / "validation"
                output.mkdir(parents=True)
                (output / "metrics.json").write_text(
                    json.dumps(
                        {
                            "quality_S": {"macro_f1": f1, "balanced_accuracy": f1},
                            "quality_A": {"macro_f1": f1},
                            "counts": {"eligible": 10},
                            "coverage": {"availability_coverage": 1.0},
                            "probability": {"NLL": 1.0, "status": "ok"},
                        }
                    )
                )
            script["summarize"](root, {"entries": entries})
            summary = json.loads((root / "summary.json").read_text())
            self.assertEqual(len(summary["rows"]), 3)
            aggregate = summary["aggregates"][0]
            self.assertEqual(aggregate["completed_seeds"], 3)
            self.assertAlmostEqual(aggregate["quality_S.macro_f1"]["mean"], 0.6)
            self.assertAlmostEqual(aggregate["quality_S.macro_f1"]["sample_std"], 0.2)

    def test_complete_run_publishes_atomically_and_is_immutable(self):
        with tempfile.TemporaryDirectory() as directory:
            run = RunDirectory(directory, "run-1")
            work = run.start()
            (work / "metrics.json").write_text('{"ok":true}')
            final = run.complete(["metrics.json"])
            status = json.loads((final / "status.json").read_text())

            self.assertEqual(status["status"], "completed")
            self.assertEqual(sorted(path.name for path in Path(directory).iterdir()), ["run-1"])
            with self.assertRaises(FileExistsError):
                RunDirectory(directory, "run-1").start()

    def test_failed_run_is_retained_with_structured_status(self):
        with tempfile.TemporaryDirectory() as directory:
            run = RunDirectory(directory, "failed-run")
            work = run.start()
            (work / "partial.txt").write_text("partial")
            final = run.fail(ValueError("broken model"))
            status = json.loads((final / "status.json").read_text())
            self.assertEqual(status["status"], "failed")
            self.assertEqual(status["error"]["type"], "ValueError")
            self.assertTrue((final / "partial.txt").is_file())

    def test_missing_required_artifact_does_not_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            run = RunDirectory(directory, "incomplete-run")
            run.start()
            with self.assertRaises(RunLifecycleError):
                run.complete(["metrics.json"])
            final = run.fail(RuntimeError("missing metrics"), incomplete=True)
            self.assertEqual(
                json.loads((final / "status.json").read_text())["status"], "incomplete"
            )

    def test_hash_manifest_uses_relative_files_and_rejects_traversal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "weights.pt").write_bytes(b"weights")
            hashes = write_sha256sums(root, ["weights.pt"])
            self.assertEqual(json.loads(hashes.read_text()).keys(), {"weights.pt"})
            with self.assertRaises(RunLifecycleError):
                write_sha256sums(root, ["../outside"])


if __name__ == "__main__":
    unittest.main()
