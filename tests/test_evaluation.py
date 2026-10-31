import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from mmt.evaluation.abstention import apply_policy, select_policy
from mmt.evaluation.bootstrap import (
    group_confusions,
    intersection_ids,
    paired_bootstrap,
    seed_summary,
)
from mmt.evaluation.calibration import fit_temperature
from mmt.evaluation.metrics import compute_metrics, join_registry, probability_metrics
from mmt.evaluation.perturbations import apply_schedule, build_schedule
from mmt.evaluation.protocol import ProtocolError, validate_protocol
from mmt.evaluation.report import (
    evaluate_acceptance,
    final_status,
    freeze_manifest,
    render_report,
    verify_frozen,
)


class EvaluationTests(unittest.TestCase):
    def test_analytical_probability_metrics_match_specification(self):
        probabilities = np.asarray([[0.8, 0.2], [0.3, 0.7]], dtype=np.float64)
        labels = np.asarray([0, 1])

        metrics, bins = probability_metrics(np.log(probabilities), labels)

        self.assertAlmostEqual(metrics["nll"], 0.2899092476264711, places=12)
        self.assertAlmostEqual(metrics["brier"], 0.13, places=12)
        self.assertAlmostEqual(metrics["ece"], 0.25, places=12)
        self.assertEqual(sum(row["count"] for row in bins), 2)

    def test_registry_join_and_metrics_keep_eligible_denominators(self):
        registry = pd.DataFrame(
            {
                "decision_id": ["d1", "d2", "d3"],
                "sequence_id": ["s1"] * 3,
                "group_id": ["g1", "g1", "g2"],
                "target_class": [0, 1, 1],
                "score_eligible": [True, True, True],
                "exclusion_reason": [None] * 3,
            }
        )
        predictions = pd.DataFrame(
            {
                "decision_id": ["d1", "d2", "d3"],
                "computable": [True, True, False],
                "status": ["accepted", "accepted", "unavailable_no_data"],
                "scores": [np.log([0.8, 0.2]), np.log([0.3, 0.7]), None],
                "predicted_class": [0, 1, None],
                "accepted": [True, True, False],
            }
        )

        joined = join_registry(registry, predictions, 2)
        metrics = compute_metrics(joined, 2, "log_probabilities")

        self.assertEqual(metrics["counts"]["scheduled"], 3)
        self.assertEqual(metrics["counts"]["eligible"], 3)
        self.assertEqual(metrics["counts"]["computable"], 2)
        self.assertAlmostEqual(metrics["coverage"]["availability_coverage"], 2 / 3)
        self.assertEqual(metrics["coverage"]["selective_risk"], 0.0)

    def test_registry_join_rejects_missing_duplicate_and_unknown_decisions(self):
        registry = pd.DataFrame(
            {
                "decision_id": ["d1"],
                "sequence_id": ["s1"],
                "target_class": [0],
                "score_eligible": [True],
            }
        )
        valid = {"decision_id": "d1", "computable": True, "status": "accepted"}

        with self.assertRaises(ValueError):
            join_registry(registry, pd.DataFrame([valid, valid]), 2)
        with self.assertRaises(ValueError):
            join_registry(registry, pd.DataFrame([{**valid, "decision_id": "unknown"}]), 2)
        with self.assertRaises(ValueError):
            join_registry(
                registry, pd.DataFrame(columns=["decision_id", "computable", "status"]), 2
            )

    def test_empty_prediction_cases_return_null_quality(self):
        registry = pd.DataFrame(
            {
                "decision_id": ["d1"],
                "sequence_id": ["s1"],
                "target_class": [1],
                "score_eligible": [True],
            }
        )
        predictions = pd.DataFrame(
            {
                "decision_id": ["d1"],
                "computable": [False],
                "status": ["unavailable_no_data"],
                "scores": [None],
            }
        )
        result = compute_metrics(join_registry(registry, predictions, 2), 2)

        self.assertEqual(result["coverage"]["availability_coverage"], 0.0)
        self.assertIsNone(result["quality_S"]["macro_f1"])
        self.assertEqual(result["probability"]["status"], "not_available")

    def test_temperature_fit_is_finite_and_does_not_worsen_nll(self):
        scores = np.asarray([[3, 0], [0, 3], [2, 0], [2, 0], [0, 3], [2, 0]], dtype=float)
        labels = np.asarray([0, 1, 0, 1, 1, 0])
        artifact = fit_temperature(scores, labels, 2, {"checkpoint": "a" * 64}, "logits")

        self.assertEqual(artifact["status"], "ok")
        self.assertGreater(artifact["T"], 0)
        self.assertLessEqual(
            artifact["diagnostics"]["final_nll"], artifact["diagnostics"]["initial_nll"] + 1e-8
        )
        self.assertTrue(artifact["diagnostics"]["converged"])
        np.testing.assert_array_equal(scores.argmax(1), (scores / artifact["T"]).argmax(1))
        self.assertEqual(artifact["diagnostics"]["n"], len(labels))

    def test_temperature_fit_reports_invalid_inputs(self):
        result = fit_temperature(np.empty((0, 2)), np.empty((0,), dtype=int), 2, {}, "logits")

        self.assertEqual(result["status"], "fit_failed")
        self.assertEqual(result["T"], 1.0)

    def test_abstention_threshold_keeps_ties_and_uses_eligible_coverage(self):
        policy, curve = select_policy([0.9, 0.8, 0.8], [True, False, False], 4, risk_limit=0.1)

        self.assertEqual(policy["status"], "selected")
        self.assertEqual(policy["tau"], 0.9)
        self.assertEqual(policy["selected"]["accepted"], 1)
        self.assertEqual(policy["selected"]["coverage"], 0.25)
        self.assertEqual(apply_policy([0.9, 0.8], [True, True], 0.9).tolist(), [True, False])
        self.assertEqual(curve[-1]["accepted"], 1)
        self.assertEqual(curve[-1]["risk"], 0.0)

    def test_unset_risk_limit_does_not_invent_a_policy(self):
        policy, _ = select_policy([0.9], [True], 1)

        self.assertEqual(policy["status"], "disabled_no_constraints")
        self.assertIsNone(policy["tau"])

    def test_protocol_rejects_group_leakage_before_experiment(self):
        membership = pd.DataFrame(
            {
                "sequence_id": ["s1", "s2", "s3", "s4"],
                "group_id": ["g1", "g1", "g3", "g4"],
                "split": ["train", "test", "validation", "calibration"],
            }
        )
        registry = pd.DataFrame(
            {
                "decision_id": ["d1", "d2", "d3", "d4"],
                "sequence_id": ["s1", "s2", "s3", "s4"],
                "target_class": [0, 0, 0, 0],
                "score_eligible": [True] * 4,
            }
        )

        with self.assertRaises(ProtocolError):
            validate_protocol(membership, registry, {1: 0}, require_train_classes=True)

    def test_saved_scenario_is_reproducible_and_delay_preserves_event_times(self):
        source = SimpleNamespace(
            observation_ids=np.arange(8, dtype=np.int64),
            values=np.arange(8, dtype=np.float32).reshape(-1, 1),
            valid_mask=np.ones((8, 1), dtype=bool),
            event_start=np.arange(8, dtype=np.float64),
            event_end=np.arange(8, dtype=np.float64),
            available_at=np.arange(8, dtype=np.float64),
        )
        sources = {("s1", "imu"): source}
        scenario = {
            "id": "delay-test",
            "kind": "delay",
            "sources": ["imu"],
            "param": 0.25,
            "R": 4.0,
            "profile": "window",
        }
        first = build_schedule(sources, scenario, seed=19, protocol_id="p1")
        second = build_schedule(sources, scenario, seed=19, protocol_id="p1")
        transformed = apply_schedule(sources, first)[("s1", "imu")]

        self.assertEqual(first["schedule_hash"], second["schedule_hash"])
        np.testing.assert_array_equal(transformed.event_end, source.event_end)
        np.testing.assert_array_equal(transformed.available_at, source.available_at + 1.0)

    def test_group_bootstrap_is_reproducible_and_seed_summary_is_not_best_seed(self):
        frame = pd.DataFrame(
            {
                "decision_id": ["d1", "d2", "d3", "d4"],
                "group_id": ["g1", "g1", "g2", "g2"],
                "target_class": [0, 1, 0, 1],
                "pred": [0, 0, 0, 1],
                "score_eligible": [True] * 4,
                "computable": [True, True, True, False],
            }
        )
        confusion = group_confusions(frame, 2)
        paired = paired_bootstrap(confusion, confusion, 2, B=25, seed=12)
        paired_again = paired_bootstrap(confusion, confusion, 2, B=25, seed=12)

        self.assertEqual(intersection_ids(frame, frame), {"d1", "d2", "d3"})
        self.assertEqual(paired["ci95"], [0.0, 0.0])
        self.assertEqual(paired["resampled_group_indices"], paired_again["resampled_group_indices"])
        self.assertAlmostEqual(seed_summary([0.2, 0.6, 0.4])["mean"], 0.4)
        self.assertAlmostEqual(seed_summary([0.2, 0.6, 0.4])["std"], 0.2)

    def test_acceptance_without_a_frozen_bound_is_not_qualified(self):
        metrics = {
            "counts": {"eligible": 4},
            "coverage": {"availability_coverage": 0.75},
            "quality_S": {"macro_f1": 0.5},
            "quality_A": {},
            "probability": {},
        }
        rules = [
            {
                "name": "quality",
                "metric": "macro_f1",
                "scenario": "clean",
                "direction": "min",
                "bound": None,
            }
        ]

        rows, status = evaluate_acceptance(rules, {"clean": metrics})

        self.assertEqual(status, "not_qualified")
        self.assertEqual(rows[0]["result"], "not_qualified")
        self.assertEqual(
            final_status(True, status),
            {
                "implementation_verified": True,
                "predictive_qualification": "not_qualified",
            },
        )

    def test_frozen_manifest_detects_changed_input(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "artifact.json"
            artifact.write_text('{"value":1}')
            manifest = freeze_manifest({"artifact": artifact})
            verify_frozen(manifest, {"artifact": artifact})
            artifact.write_text('{"value":2}')
            with self.assertRaises(RuntimeError):
                verify_frozen(manifest, {"artifact": artifact})

    def test_report_renders_saved_metrics_and_separate_qualification_statuses(self):
        metrics = {
            "counts": {"eligible": 2, "computable": 2},
            "quality_S": {"macro_f1": 1.0, "balanced_accuracy": 1.0},
            "probability": {"nll": 0.1, "brier": 0.02, "ece": 0.1},
            "coverage": {
                "availability_coverage": 1.0,
                "accepted_coverage": 1.0,
                "selective_risk": 0.0,
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "clean"
            run_dir.mkdir()
            (run_dir / "metrics.json").write_text(json.dumps(metrics))
            report_path = Path(directory) / "report.html"

            result = render_report(
                {"clean": run_dir},
                report_path,
                "Synthetic test",
                final_status(True, "not_qualified"),
                limitations=["synthetic fixture only"],
            )
            html = report_path.read_text()

        self.assertEqual(result, report_path)
        self.assertIn("implementation_verified = True", html)
        self.assertIn("predictive_qualification = not_qualified", html)
        self.assertIn("synthetic fixture only", html)


if __name__ == "__main__":
    unittest.main()
