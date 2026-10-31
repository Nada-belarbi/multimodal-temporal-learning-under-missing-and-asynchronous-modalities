import itertools
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from typer.testing import CliRunner

from mmt.cli import app
from mmt.config import resolve_config
from mmt.evaluation.perturbations import build_schedule, save_schedule
from mmt.services.evaluation import evaluate_run
from mmt.storage import load_sequence, write_prepared


def source(value: float, observation_base: int):
    values = np.asarray([[value], [value + 1.0]], dtype=np.float32)
    times = np.asarray([0.0, 1.0], dtype=np.float64)
    return {
        "values": values,
        "valid_mask": np.ones((2, 1), dtype=bool),
        "observation_ids": np.asarray([observation_base, observation_base + 1], dtype=np.int64),
        "event_start": times,
        "event_end": times,
        "available_at": times,
    }


class V0TrainingServiceTests(unittest.TestCase):
    def test_joint_models_train_and_export_the_best_checkpoint(self):
        for model_name, application in itertools.product(("v1", "v2"), ("pamap2", "mosi")):
            with (
                self.subTest(model=model_name, application=application),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                (root / "pyproject.toml").write_text("[project]\nname='fixture'\n")
                membership = pd.DataFrame(
                    {
                        "sequence_id": [f"s{i}" for i in range(5)],
                        "group_id": [f"g{i}" for i in range(5)],
                        "split": ["train", "train", "validation", "calibration", "test"],
                    }
                )
                decisions = pd.DataFrame(
                    [
                        {
                            "decision_id": f"d{i}",
                            "sequence_id": f"s{i}",
                            "decision_time": 1.0,
                            "support_start": 0.0,
                            "support_end": 1.0,
                        }
                        for i in range(5)
                    ]
                )
                targets = pd.DataFrame(
                    {
                        "decision_id": [f"d{i}" for i in range(5)],
                        "target_class": [0, 1, 0, 1, 0],
                        "score_eligible": [True] * 5,
                    }
                )
                prepared = write_prepared(
                    root / "prepared",
                    application,
                    "joint",
                    {
                        f"s{i}": {
                            "z_source": source(float(i), i * 10),
                            "a_source": source(float(i + 2), i * 10),
                        }
                        for i in range(5)
                    },
                    {"z_source": {}, "a_source": {}},
                    extra_manifest={"application": application, "class_map": {"-1": 0, "1": 1}},
                    tables={
                        "partitions.csv": membership,
                        "decisions.csv": decisions,
                        "targets.csv": targets,
                    },
                )
                config = {
                    "data": {
                        "application": application,
                        "manifest_path": str(prepared / "manifest.json"),
                        "partitions_path": str(prepared / "partitions.csv"),
                        "source_schema": {"z_source": 1, "a_source": 1},
                        "feature_order": ["z_source", "a_source"],
                        "decision_policy": "segment" if application == "mosi" else "window",
                        "availability_mode": "offline" if application == "mosi" else "simulated",
                    },
                    "run": {
                        "run_id": f"{model_name}-smoke",
                        "model_name": model_name,
                        "seed": 11,
                        "regime": "clean",
                        "partition_hashes": {"train": "fixture"},
                        "model_config": {"hidden_size": 8, "dropout": 0.0},
                        "training": {"batch_size": 2, "max_epochs": 1, "patience": 1},
                        "output_root": "runs",
                    },
                }
                path = root / "config.yaml"
                path.write_text(yaml.safe_dump(config))
                result = CliRunner().invoke(app, ["train", "--config", str(path)])
                self.assertEqual(result.exit_code, 0, result.output)
                run = Path(json.loads(result.output)["run_path"])
                model_config = json.loads((run / "model_config.json").read_text())
                self.assertEqual(model_config["model_name"], model_name)
                self.assertEqual(model_config["source_order"], ["z_source", "a_source"])
                self.assertEqual(model_config["score_kind"], "logits")
                weights = torch.load(run / "weights.pt", weights_only=True)
                best = torch.load(run / "checkpoints/best.pt", weights_only=True)
                for key in weights["state_dict"]:
                    torch.testing.assert_close(weights["state_dict"][key], best["state_dict"][key])
                verified = CliRunner().invoke(app, ["runs", "verify", "--run-dir", str(run)])
                self.assertEqual(verified.exit_code, 0, verified.output)
                repeated = CliRunner().invoke(
                    app,
                    [
                        "train",
                        "--config",
                        str(path),
                        "--seed",
                        "11",
                        "--run-id",
                        f"{model_name}-repeat",
                    ],
                )
                self.assertEqual(repeated.exit_code, 0, repeated.output)
                repeated_run = Path(json.loads(repeated.output)["run_path"])
                repeated_weights = torch.load(repeated_run / "weights.pt", weights_only=True)
                for key in weights["state_dict"]:
                    torch.testing.assert_close(
                        weights["state_dict"][key],
                        repeated_weights["state_dict"][key],
                        atol=0,
                        rtol=0,
                    )
                config["data"]["normalizer_ref"] = str(run / "normalizer.npz")
                path.write_text(yaml.safe_dump(config))
                for partition in ("validation", "calibration", "test"):
                    evaluation_dir = root / f"evaluation-{partition}"
                    result = CliRunner().invoke(
                        app,
                        [
                            "evaluate",
                            "--config",
                            str(path),
                            "--run-dir",
                            str(run),
                            "--partition",
                            partition,
                            "--output-dir",
                            str(evaluation_dir),
                        ],
                    )
                    self.assertEqual(result.exit_code, 0, result.output)
                    provenance = json.loads((evaluation_dir / "provenance.json").read_text())
                    self.assertEqual(provenance["score_kind"], "logits")
                calibration_path = root / "calibration.json"
                result = CliRunner().invoke(
                    app,
                    [
                        "calibrate",
                        "--run-dir",
                        str(run),
                        "--evaluation-dir",
                        str(root / "evaluation-calibration"),
                        "--output",
                        str(calibration_path),
                    ],
                )
                self.assertEqual(result.exit_code, 0, result.output)
                calibration = json.loads(calibration_path.read_text())
                self.assertEqual(calibration["score_kind"], "logits")
                policy_path = root / "policy.json"
                result = CliRunner().invoke(
                    app,
                    [
                        "select-policy",
                        "--run-dir",
                        str(run),
                        "--validation-dir",
                        str(root / "evaluation-validation"),
                        "--calibration",
                        str(calibration_path),
                        "--output",
                        str(policy_path),
                    ],
                )
                self.assertEqual(result.exit_code, 0, result.output)
                result = CliRunner().invoke(
                    app,
                    [
                        "report",
                        "--evaluation-dir",
                        str(root / "evaluation-test"),
                        "--output",
                        str(root / "report.html"),
                    ],
                )
                self.assertEqual(result.exit_code, 0, result.output)
                self.assertTrue((root / "report.html").is_file())
                calibrated_dir = root / "calibrated-test"
                result = CliRunner().invoke(
                    app,
                    [
                        "evaluate",
                        "--config",
                        str(path),
                        "--run-dir",
                        str(run),
                        "--partition",
                        "test",
                        "--output-dir",
                        str(calibrated_dir),
                        "--calibration",
                        str(calibration_path),
                        "--policy",
                        str(policy_path),
                    ],
                )
                self.assertEqual(result.exit_code, 0, result.output)
                replay_dir = root / "replay"
                result = CliRunner().invoke(
                    app,
                    [
                        "replay",
                        "--config",
                        str(path),
                        "--run-dir",
                        str(run),
                        "--sequence",
                        "s4",
                        "--output-dir",
                        str(replay_dir),
                    ],
                )
                self.assertEqual(result.exit_code, 0, result.output)
                prediction = json.loads((replay_dir / "predictions.jsonl").read_text())
                self.assertTrue(prediction["computable"])
                self.assertEqual(prediction["score_kind"], "logits")
                csv_path = root / "evaluation-test/predictions.csv"
                evaluated = (
                    pd.read_csv(csv_path)
                    if csv_path.exists()
                    else pd.read_parquet(root / "evaluation-test/predictions.parquet")
                )
                raw_scores = evaluated.iloc[0].scores
                scores = (
                    np.fromstring(raw_scores, sep=";")
                    if isinstance(raw_scores, str)
                    else np.asarray(raw_scores)
                )
                np.testing.assert_allclose(prediction["scores"], scores, atol=1e-6, rtol=1e-5)
                _, batched = evaluate_run(
                    resolve_config(path), run, "test", root / "batched", batch_size=2
                )
                np.testing.assert_allclose(batched.iloc[0].scores, scores, atol=1e-6, rtol=1e-5)
                result = CliRunner().invoke(
                    app,
                    [
                        "replay",
                        "--config",
                        str(path),
                        "--run-dir",
                        str(run),
                        "--sequence",
                        "s4",
                        "--output-dir",
                        str(root / "replay-calibrated"),
                        "--calibration",
                        str(calibration_path),
                        "--policy",
                        str(policy_path),
                    ],
                )
                self.assertEqual(result.exit_code, 0, result.output)
                calibrated_prediction = json.loads(
                    (root / "replay-calibrated/predictions.jsonl").read_text()
                )
                expected_probabilities = torch.softmax(
                    torch.tensor(scores, dtype=torch.float64) / float(calibration["T"]), dim=-1
                ).numpy()
                np.testing.assert_allclose(
                    calibrated_prediction["probabilities"],
                    expected_probabilities,
                    atol=1e-6,
                    rtol=1e-5,
                )
                schedule = build_schedule(
                    {
                        ("s4", source_id): arrays
                        for source_id, arrays in load_sequence(prepared, "s4").items()
                    },
                    {
                        "id": "no-sources",
                        "kind": "subset",
                        "sources": [],
                        "param": None,
                        "R": 1.0,
                        "profile": "segment" if application == "mosi" else "window",
                    },
                    seed=1701,
                    protocol_id="joint-fixture",
                )
                schedule_path = root / "schedule.json"
                save_schedule(schedule_path, schedule)
                result = CliRunner().invoke(
                    app,
                    [
                        "replay",
                        "--config",
                        str(path),
                        "--run-dir",
                        str(run),
                        "--sequence",
                        "s4",
                        "--scenario",
                        "no-sources",
                        "--schedule",
                        str(schedule_path),
                        "--output-dir",
                        str(root / "replay-empty"),
                    ],
                )
                self.assertEqual(result.exit_code, 0, result.output)
                empty = json.loads((root / "replay-empty/predictions.jsonl").read_text())
                self.assertFalse(empty["computable"])
                self.assertIsNone(empty["probabilities"])
                incompatible = {
                    **calibration,
                    "identity": {**calibration["identity"], "checkpoint_sha256": "0" * 64},
                }
                wrong_calibration = root / "wrong-calibration.json"
                wrong_calibration.write_text(json.dumps(incompatible))
                result = CliRunner().invoke(
                    app,
                    [
                        "evaluate",
                        "--config",
                        str(path),
                        "--run-dir",
                        str(run),
                        "--partition",
                        "test",
                        "--output-dir",
                        str(root / "rejected-calibration"),
                        "--calibration",
                        str(wrong_calibration),
                    ],
                )
                self.assertEqual(result.exit_code, 2, result.output)
                self.assertFalse((root / "rejected-calibration").exists())
                config["run"]["model_name"] = "v2" if model_name == "v1" else "v1"
                path.write_text(yaml.safe_dump(config))
                result = CliRunner().invoke(
                    app,
                    [
                        "replay",
                        "--config",
                        str(path),
                        "--run-dir",
                        str(run),
                        "--sequence",
                        "s4",
                        "--output-dir",
                        str(root / "rejected-model"),
                    ],
                )
                self.assertEqual(result.exit_code, 2, result.output)
                self.assertFalse((root / "rejected-model").exists())

    def test_prepared_fixture_trains_sources_assembles_v0_and_publishes_complete_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "pyproject.toml").write_text("[project]\nname='fixture'\n")
            sequences = {f"s{i}": {"imu": source(float(i), i * 10)} for i in range(1, 6)}
            membership = pd.DataFrame(
                {
                    "sequence_id": ["s1", "s2", "s3", "s4", "s5"],
                    "group_id": ["g1", "g2", "g3", "g4", "g5"],
                    "split": ["train", "train", "validation", "calibration", "test"],
                }
            )
            decisions = pd.DataFrame(
                [
                    {
                        "decision_id": f"d{i}",
                        "sequence_id": f"s{i}",
                        "decision_time": 1.0,
                        "support_start": 0.0,
                        "support_end": 1.0,
                    }
                    for i in range(1, 6)
                ]
            )
            targets = pd.DataFrame(
                {
                    "decision_id": [f"d{i}" for i in range(1, 6)],
                    "target_class": [0, 1, 0, 1, 1],
                    "score_eligible": [True] * 5,
                    "target_time": [1.0] * 5,
                    "exclusion_reason": [None] * 5,
                }
            )
            prepared = write_prepared(
                root / "data" / "prepared",
                "mosi",
                "prep-1",
                sequences,
                {"imu": {"feature_names": ["feature_x"]}},
                extra_manifest={
                    "application": "mosi",
                    "availability_mode": "offline",
                    "class_map": {"negative": 0, "positive": 1},
                },
                tables={
                    "partitions.csv": membership,
                    "decisions.csv": decisions,
                    "targets.csv": targets,
                },
            )
            config_path = root / "configs" / "mosi-v0.yaml"
            config_path.parent.mkdir()
            config = {
                "data": {
                    "application": "mosi",
                    "manifest_path": "../data/prepared/mosi/prep-1/manifest.json",
                    "partitions_path": "../data/prepared/mosi/prep-1/partitions.csv",
                    "source_schema": {"imu": 1},
                    "decision_policy": "segment",
                    "availability_mode": "offline",
                    "feature_order": ["imu"],
                },
                "run": {
                    "run_id": "v0-smoke",
                    "model_name": "v0",
                    "seed": 17,
                    "regime": "clean",
                    "partition_hashes": {"train": "a" * 64, "validation": "b" * 64},
                    "model_config": {"hidden_size": 8, "dropout": 0.0},
                    "training": {"batch_size": 2, "max_epochs": 1, "patience": 1},
                    "output_root": "runs",
                },
            }
            config_path.write_text(yaml.safe_dump(config))

            cli_result = CliRunner().invoke(app, ["train", "--config", str(config_path)])
            self.assertEqual(cli_result.exit_code, 0, cli_result.output)
            run_dir = Path(json.loads(cli_result.output)["run_path"])
            verify_result = CliRunner().invoke(app, ["runs", "verify", "--run-dir", str(run_dir)])
            list_result = CliRunner().invoke(app, ["runs", "list", "--root", str(run_dir.parent)])
            self.assertEqual(verify_result.exit_code, 0, verify_result.output)
            self.assertEqual(json.loads(verify_result.output)["integrity"], "valid")
            self.assertIn("v0-smoke", list_result.output)
            evaluation_dir = root / "runs" / "evaluation-test"
            evaluation_result = CliRunner().invoke(
                app,
                [
                    "evaluate",
                    "--config",
                    str(config_path),
                    "--run-dir",
                    str(run_dir),
                    "--partition",
                    "test",
                    "--output-dir",
                    str(evaluation_dir),
                ],
            )
            self.assertEqual(evaluation_result.exit_code, 0, evaluation_result.output)
            validation_dir = root / "runs" / "evaluation-validation"
            validation_result = CliRunner().invoke(
                app,
                [
                    "evaluate",
                    "--config",
                    str(config_path),
                    "--run-dir",
                    str(run_dir),
                    "--partition",
                    "validation",
                    "--output-dir",
                    str(validation_dir),
                ],
            )
            self.assertEqual(validation_result.exit_code, 0, validation_result.output)
            calibration_dir = root / "runs" / "evaluation-calibration"
            calibration_result = CliRunner().invoke(
                app,
                [
                    "evaluate",
                    "--config",
                    str(config_path),
                    "--run-dir",
                    str(run_dir),
                    "--partition",
                    "calibration",
                    "--output-dir",
                    str(calibration_dir),
                ],
            )
            self.assertEqual(calibration_result.exit_code, 0, calibration_result.output)
            calibration_path = root / "runs" / "calibration.json"
            calibrate_result = CliRunner().invoke(
                app,
                [
                    "calibrate",
                    "--run-dir",
                    str(run_dir),
                    "--evaluation-dir",
                    str(calibration_dir),
                    "--output",
                    str(calibration_path),
                ],
            )
            self.assertEqual(calibrate_result.exit_code, 0, calibrate_result.output)
            policy_path = root / "runs" / "policy.json"
            policy_result = CliRunner().invoke(
                app,
                [
                    "select-policy",
                    "--run-dir",
                    str(run_dir),
                    "--validation-dir",
                    str(validation_dir),
                    "--calibration",
                    str(calibration_path),
                    "--output",
                    str(policy_path),
                ],
            )
            self.assertEqual(policy_result.exit_code, 0, policy_result.output)
            replay_dir = root / "runs" / "replay-test"
            replay_result = CliRunner().invoke(
                app,
                [
                    "replay",
                    "--config",
                    str(config_path),
                    "--run-dir",
                    str(run_dir),
                    "--sequence",
                    "s5",
                    "--output-dir",
                    str(replay_dir),
                ],
            )
            self.assertEqual(replay_result.exit_code, 0, replay_result.output)
            sequence_sources = load_sequence(prepared, "s5")
            schedule = build_schedule(
                {("s5", source_id): arrays for source_id, arrays in sequence_sources.items()},
                {
                    "id": "no-sources",
                    "kind": "subset",
                    "sources": [],
                    "param": None,
                    "R": 1.0,
                    "profile": "segment",
                },
                seed=17,
                protocol_id="fixture-protocol",
            )
            schedule_path = root / "runs" / "no-sources.json"
            save_schedule(schedule_path, schedule)
            unavailable_replay_dir = root / "runs" / "replay-no-sources"
            unavailable_replay_result = CliRunner().invoke(
                app,
                [
                    "replay",
                    "--config",
                    str(config_path),
                    "--run-dir",
                    str(run_dir),
                    "--sequence",
                    "s5",
                    "--scenario",
                    "no-sources",
                    "--schedule",
                    str(schedule_path),
                    "--output-dir",
                    str(unavailable_replay_dir),
                ],
            )
            self.assertEqual(
                unavailable_replay_result.exit_code, 0, unavailable_replay_result.output
            )

            status = json.loads((run_dir / "status.json").read_text())
            artifact = torch.load(run_dir / "weights.pt", map_location="cpu", weights_only=True)
            metrics = json.loads((evaluation_dir / "metrics.json").read_text())
            calibration_artifact = json.loads(calibration_path.read_text())
            policy = json.loads(policy_path.read_text())
            replay_status = json.loads((replay_dir / "status.json").read_text())
            replay_rows = [
                json.loads(line)
                for line in (replay_dir / "predictions.jsonl").read_text().splitlines()
            ]
            unavailable_rows = [
                json.loads(line)
                for line in (unavailable_replay_dir / "predictions.jsonl").read_text().splitlines()
            ]
            self.assertEqual(status["status"], "completed")
            self.assertEqual(tuple(artifact["source_order"]), ("imu",))
            self.assertTrue((run_dir / "normalizer.npz").is_file())
            self.assertTrue((run_dir / "sha256sums.json").is_file())
            self.assertEqual(metrics["counts"]["scheduled"], 1)
            for reference in ("majority", "imu"):
                baseline_metrics, _ = evaluate_run(
                    resolve_config(config_path),
                    run_dir,
                    "test",
                    root / f"baseline-{reference}",
                    reference=reference,
                    batch_size=2,
                )
                self.assertEqual(baseline_metrics["counts"]["scheduled"], 1)
                if reference == "majority":
                    self.assertEqual(baseline_metrics["probability"]["status"], "not_available")
            self.assertTrue((evaluation_dir / "status.json").is_file())
            self.assertIn(calibration_artifact["status"], {"ok", "fit_failed"})
            self.assertIn(
                policy["status"], {"selected", "disabled_no_constraints", "no_feasible_policy"}
            )
            self.assertEqual(replay_status["status"], "completed")
            self.assertEqual(replay_status["rows_written"], 1)
            self.assertEqual(replay_rows[0]["decision_id"], "d5")
            self.assertFalse(unavailable_rows[0]["computable"])
            self.assertEqual(unavailable_rows[0]["status"], "unavailable_no_data")
            self.assertIsNone(unavailable_rows[0]["probabilities"])


if __name__ == "__main__":
    unittest.main()
