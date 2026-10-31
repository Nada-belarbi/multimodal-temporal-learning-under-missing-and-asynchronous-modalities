import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from mmt.evaluation.runner import IdentityError, ModelFailure, evaluate


def one_source():
    return {
        ("s1", "imu"): SimpleNamespace(
            observation_ids=np.asarray([1], dtype=np.int64),
            values=np.asarray([[2.0]], dtype=np.float32),
            valid_mask=np.asarray([[True]], dtype=bool),
            event_start=np.asarray([1.0]),
            event_end=np.asarray([1.0]),
            available_at=np.asarray([1.0]),
        )
    }


def registry():
    return pd.DataFrame(
        [
            {
                "decision_id": "d1",
                "sequence_id": "s1",
                "group_id": "g1",
                "split": "test",
                "decision_time": 1.0,
                "support_start": 0.0,
                "support_end": 1.0,
                "target_class": 1,
                "score_eligible": True,
            }
        ]
    )


class EvaluationRunnerTests(unittest.TestCase):
    def test_runner_never_passes_labels_to_predictor_and_writes_complete_result(self):
        seen_rows = []

        def predict(row, selected_sources):
            seen_rows.append(dict(row))
            return {
                "computable": True,
                "scores": [0.0, 2.0],
                "predicted_class": 1,
                "presence": {"imu": True},
            }

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run-1"
            metrics, joined = evaluate(
                registry(),
                one_source(),
                None,
                predict,
                2,
                output,
                "run-1",
                "clean",
                "logits",
                partition="test",
            )

            self.assertTrue((output / "metrics.json").is_file())
            self.assertTrue((output / "status.json").is_file())
            saved = json.loads((output / "metrics.json").read_text())
            self.assertEqual(saved["counts"]["eligible"], 1)
            self.assertEqual(saved["quality_S"]["accuracy"], 1.0)
            self.assertEqual(metrics["coverage"]["availability_coverage"], 1.0)
            self.assertTrue(joined.loc[0, "computable"])

        self.assertEqual(len(seen_rows), 1)
        self.assertNotIn("target_class", seen_rows[0])
        self.assertNotIn("score_eligible", seen_rows[0])

    def test_identity_mismatch_fails_before_prediction(self):
        def should_not_run(*_args):
            self.fail("predictor must not run after identity mismatch")

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(IdentityError):
                evaluate(
                    registry(),
                    one_source(),
                    None,
                    should_not_run,
                    2,
                    Path(directory) / "run",
                    "r",
                    "clean",
                    "logits",
                    expected_ids={"checkpoint": "expected"},
                    actual_ids={"checkpoint": "other"},
                )
            self.assertFalse((Path(directory) / "run").exists())

    def test_model_failure_is_not_written_as_a_successful_run(self):
        def broken(*_args):
            raise RuntimeError("synthetic model failure")

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            with self.assertRaises(ModelFailure):
                evaluate(registry(), one_source(), None, broken, 2, output, "r", "clean", "logits")
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
