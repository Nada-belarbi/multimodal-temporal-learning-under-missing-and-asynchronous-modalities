import json
import tempfile
import unittest
from pathlib import Path

from mmt.runtime.engine import InferenceEngine
from mmt.runtime.sink import replay_to_directory


def observation(observation_id, available_at):
    return {
        "sequence_id": "s1",
        "source_id": "imu",
        "observation_id": observation_id,
        "values": [1.0],
        "valid_mask": [True],
        "event_start": available_at,
        "event_end": available_at,
        "available_at": available_at,
    }


def decision(decision_id, time):
    return {
        "sequence_id": "s1",
        "decision_id": decision_id,
        "decision_time": time,
        "support_start": 0.0,
        "support_end": time,
    }


def engine():
    return InferenceEngine(
        source_ids=("imu",),
        batch_builder=lambda selections: selections,
        predict_fn=lambda _: {"scores": [[1.0, 0.0]], "computable": [True], "score_kind": "logits"},
        n_classes=2,
        run_id="run-1",
        scenario_id="clean",
    )


class ReplaySinkTests(unittest.TestCase):
    def test_complete_replay_publishes_incremental_predictions_and_traces(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            result = replay_to_directory(
                engine(),
                [observation(1, 1.0), observation(2, 2.0)],
                [decision("d1", 1.0), decision("d2", 2.0)],
                output,
                expected_count=2,
            )

            status = json.loads((result / "status.json").read_text())
            rows = [
                json.loads(line) for line in (result / "predictions.jsonl").read_text().splitlines()
            ]
            traces = [
                json.loads(line)
                for line in (result / "selected_ids.jsonl").read_text().splitlines()
            ]

        self.assertEqual(status["status"], "completed")
        self.assertEqual(status["rows_written"], 2)
        self.assertEqual([row["decision_id"] for row in rows], ["d1", "d2"])
        self.assertEqual(traces[1]["selected_ids"]["imu"], [1, 2])

    def test_failed_replay_leaves_an_incomplete_record(self):
        broken = InferenceEngine(
            source_ids=("imu",),
            batch_builder=lambda selections: selections,
            predict_fn=lambda _: (_ for _ in ()).throw(ValueError("model failed")),
            n_classes=2,
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            with self.assertRaisesRegex(RuntimeError, "model_execution_failed"):
                replay_to_directory(broken, [observation(1, 1.0)], [decision("d1", 1.0)], output)
            partial_dirs = list(Path(directory).glob(".run.incomplete-*"))
            self.assertEqual(len(partial_dirs), 1)
            status = json.loads((partial_dirs[0] / "status.json").read_text())

        self.assertEqual(status["status"], "incomplete")
        self.assertEqual(status["rows_written"], 0)
        self.assertIn("model_execution_failed", status["error"])

    def test_wrong_expected_count_never_publishes_completed_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            with self.assertRaises(ValueError):
                replay_to_directory(
                    engine(), [observation(1, 1.0)], [decision("d1", 1.0)], output, expected_count=2
                )
            self.assertFalse(output.exists())
            partial = list(Path(directory).glob(".run.incomplete-*"))[0]
            self.assertEqual(
                json.loads((partial / "status.json").read_text())["status"], "incomplete"
            )


if __name__ == "__main__":
    unittest.main()
