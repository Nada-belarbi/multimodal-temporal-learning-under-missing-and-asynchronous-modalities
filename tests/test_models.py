import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from mmt.models import (
    SourceBatch,
    SourceEncoder,
    V0LateFusion,
    V1GatedFusion,
    V2AttentionFusion,
    build_model,
)
from mmt.models.baselines import MajorityReference
from mmt.models.source_encoder import ModelOutput
from mmt.training import TrainingBatch, TrainingConfig, train
from mmt.training.checkpoint import load_fusion_model, load_state_dict, save_checkpoint
from mmt.training.source_models import assemble_v0, train_source_models


def source_batch(feature_dim, values, lengths, feature_mask=None):
    values = torch.tensor(values, dtype=torch.float32)
    batch_size, time_steps, _ = values.shape
    if feature_mask is None:
        feature_mask = torch.ones_like(values, dtype=torch.bool)
    else:
        feature_mask = torch.tensor(feature_mask, dtype=torch.bool)
    lengths = torch.tensor(lengths, dtype=torch.long)
    positions = torch.arange(time_steps).unsqueeze(0)
    padding_mask = positions >= lengths.unsqueeze(1)
    time_features = torch.zeros((batch_size, time_steps, 3), dtype=torch.float32)
    return SourceBatch(values, feature_mask, time_features, lengths, padding_mask, lengths > 0)


class LinearClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(1, 2)

    def forward(self, inputs):
        return ModelOutput(
            self.layer(inputs),
            torch.ones(inputs.shape[0], dtype=torch.bool, device=inputs.device),
            "logits",
        )


class ModelTests(unittest.TestCase):
    def test_joint_artifact_reloads_in_a_fresh_process_with_exact_source_order(self):
        for model_name in ("v1", "v2"):
            with self.subTest(model=model_name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                dims = {"z_source": 1, "a_source": 2}
                class_map = {"-1": 0, "1": 1}
                options = {"hidden_size": 8, "dropout": 0.0}
                if model_name == "v2":
                    options.update({"n_heads": 2, "feedforward_size": 24})
                torch.manual_seed(11)
                model = build_model(model_name, dims, class_map, options).eval()
                inputs = {
                    "z_source": source_batch(1, [[[1.0], [2.0]]], [2]),
                    "a_source": source_batch(2, [[[2.0, 3.0], [0.0, 0.0]]], [1]),
                }
                with torch.inference_mode():
                    expected = model(inputs).scores
                metadata = {
                    "model_name": model_name,
                    "source_order": list(dims),
                    "source_dimensions": dims,
                    "class_map": class_map,
                    "score_kind": "logits",
                    **options,
                }
                (root / "metadata.json").write_text(json.dumps(metadata, sort_keys=True))
                save_checkpoint(root / "weights.pt", {"state_dict": model.state_dict()})
                torch.save(
                    {name: vars(batch) for name, batch in inputs.items()}, root / "inputs.pt"
                )
                code = (
                    "import json,sys,torch; from pathlib import Path; "
                    "from mmt.training.checkpoint import load_fusion_model; "
                    "from mmt.models.source_encoder import SourceBatch; "
                    "root=Path(sys.argv[1]); metadata=json.loads((root/'metadata.json').read_text()); "
                    "model=load_fusion_model(metadata,metadata['class_map'],root/'weights.pt',"
                    "expected_model_name=metadata['model_name']); "
                    "inputs={key:SourceBatch(**value) for key,value in "
                    "torch.load(root/'inputs.pt',weights_only=True).items()}; "
                    "torch.save(model(inputs).scores.detach(),root/'scores.pt')"
                )
                result = subprocess.run(
                    [sys.executable, "-c", code, str(root)],
                    capture_output=True,
                    text=True,
                    cwd=root,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                actual = torch.load(root / "scores.pt", weights_only=True)
                torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
                with self.assertRaises(ValueError):
                    load_fusion_model(
                        metadata | {"score_kind": "log_probabilities"},
                        class_map,
                        root / "weights.pt",
                        expected_model_name=model_name,
                    )

    def test_source_encoder_handles_unequal_lengths_and_absent_rows(self):
        batch = source_batch(
            2,
            [
                [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                [[2.0, 1.0], [0.0, 0.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            ],
            [3, 1, 0],
            [
                [[True, True], [True, True], [True, True]],
                [[True, True], [False, False], [False, False]],
                [[False, False], [False, False], [False, False]],
            ],
        )
        encoder = SourceEncoder(2, hidden_size=8).eval()

        output = encoder(batch)

        self.assertEqual(tuple(output.states.shape), (3, 3, 8))
        self.assertTrue(torch.equal(output.last_state[2], torch.zeros(8)))
        self.assertTrue(torch.isfinite(output.states).all())

    def test_v0_and_fusions_are_finite_on_partial_and_empty_source_rows(self):
        torch.manual_seed(4)
        inputs = {
            "a": source_batch(
                2,
                [
                    [[1.0, 0.0], [2.0, 1.0]],
                    [[3.0, 1.0], [0.0, 0.0]],
                    [[0.0, 0.0], [0.0, 0.0]],
                ],
                [2, 1, 0],
                [
                    [[True, True], [True, True]],
                    [[True, True], [False, False]],
                    [[False, False], [False, False]],
                ],
            ),
            "b": source_batch(
                1,
                [
                    [[1.0], [2.0]],
                    [[0.0], [0.0]],
                    [[0.0], [0.0]],
                ],
                [2, 0, 0],
                [
                    [[True], [True]],
                    [[False], [False]],
                    [[False], [False]],
                ],
            ),
        }
        models = [
            V0LateFusion({"a": 2, "b": 1}, 3, hidden_size=8).eval(),
            V1GatedFusion({"a": 2, "b": 1}, 3, hidden_size=8).eval(),
            V2AttentionFusion(
                {"a": 2, "b": 1}, 3, hidden_size=8, n_heads=2, feedforward_size=16
            ).eval(),
        ]

        for model in models:
            with self.subTest(model=type(model).__name__):
                output = model(inputs)
                self.assertEqual(tuple(output.scores.shape), (3, 3))
                self.assertEqual(output.computable.tolist(), [True, True, False])
                self.assertTrue(torch.isfinite(output.scores).all())
                self.assertTrue(torch.equal(output.scores[2], torch.zeros(3)))

    def test_factory_validates_class_order_and_builds_models(self):
        model = build_model("v1", {"a": 2}, {"walk": 0, "run": 1}, {"hidden_size": 8})

        self.assertIsInstance(model, V1GatedFusion)
        with self.assertRaises(ValueError):
            build_model("v1", {"a": 2}, {"walk": 1, "run": 2})

    def test_majority_tie_breaks_to_lowest_class_and_has_no_confidence(self):
        baseline = MajorityReference(3).fit([2, 1, 0, 1, 0], [True] * 5)

        self.assertEqual(baseline.class_counts, (2, 2, 1))
        self.assertEqual(baseline.predict(True).predicted_class, 0)
        self.assertFalse(baseline.predict(False).computable)

    def test_training_writes_and_strictly_reloads_best_checkpoint(self):
        batches = [
            TrainingBatch(
                torch.tensor([[-1.0], [1.0]]),
                torch.tensor([0, 1]),
                torch.tensor([True, True]),
                ("a", "b"),
            ),
        ]
        model = LinearClassifier()
        with tempfile.TemporaryDirectory() as temp_dir:
            result = train(
                model,
                batches,
                batches,
                TrainingConfig(n_classes=2, max_epochs=2, patience=2),
                temp_dir,
            )
            reloaded = LinearClassifier()
            payload = load_state_dict(reloaded, Path(temp_dir) / "best.pt")
            self.assertTrue((Path(temp_dir) / "best.pt").is_file())
            self.assertTrue((Path(temp_dir) / "last.pt").is_file())

        self.assertGreaterEqual(result.best_epoch, 1)
        self.assertEqual(payload["epoch"], result.best_epoch)
        self.assertFalse(reloaded.training)

    def test_epoch_boundary_resume_matches_uninterrupted_training(self):
        batches = [
            TrainingBatch(
                torch.tensor([[-1.0], [1.0]]),
                torch.tensor([0, 1]),
                torch.tensor([True, True]),
                ("a", "b"),
            ),
        ]
        config = TrainingConfig(n_classes=2, max_epochs=2, patience=3, seed=7)
        identity = {"manifest_hash": "m" * 64, "partition_hash": "p" * 64}
        with tempfile.TemporaryDirectory() as directory:
            full_dir = Path(directory) / "full"
            resumed_dir = Path(directory) / "resumed"
            torch.manual_seed(123)
            uninterrupted = LinearClassifier()
            full_result = train(
                uninterrupted, batches, batches, config, full_dir, run_identity=identity
            )

            torch.manual_seed(123)
            interrupted = LinearClassifier()
            partial_result = train(
                interrupted,
                batches,
                batches,
                config,
                resumed_dir,
                run_identity=identity,
                stop_after_epochs=1,
            )
            self.assertFalse(partial_result.completed)

            torch.manual_seed(123)
            resumed = LinearClassifier()
            resumed_result = train(
                resumed,
                batches,
                batches,
                config,
                resumed_dir,
                resume_from=resumed_dir / "last.pt",
                run_identity=identity,
            )

        self.assertTrue(full_result.completed)
        self.assertTrue(resumed_result.completed)
        for name, expected in uninterrupted.state_dict().items():
            torch.testing.assert_close(resumed.state_dict()[name], expected, rtol=0.0, atol=0.0)

    def test_resume_rejects_changed_data_identity(self):
        batch = TrainingBatch(
            torch.tensor([[-1.0], [1.0]]), torch.tensor([0, 1]), torch.tensor([True, True])
        )
        config = TrainingConfig(n_classes=2, max_epochs=2, seed=5)
        with tempfile.TemporaryDirectory() as directory:
            model = LinearClassifier()
            train(model, [batch], [batch], config, directory, run_identity={"manifest_hash": "a"})
            other_model = LinearClassifier()
            with self.assertRaises(ValueError):
                train(
                    other_model,
                    [batch],
                    [batch],
                    config,
                    directory,
                    resume_from=Path(directory) / "last.pt",
                    run_identity={"manifest_hash": "b"},
                )

    def test_unimodal_training_and_v0_checkpoint_assembly(self):
        first = source_batch(
            1,
            [[[1.0], [2.0]], [[3.0], [0.0]]],
            [2, 0],
            [
                [[True], [True]],
                [[False], [False]],
            ],
        )
        second = source_batch(2, [[[2.0, 1.0], [3.0, 2.0]], [[1.0, 1.0], [2.0, 2.0]]], [2, 2])
        batch = TrainingBatch(
            {"a": first, "b": second},
            torch.tensor([0, 1]),
            torch.tensor([True, True]),
            ("d1", "d2"),
        )
        config = TrainingConfig(n_classes=2, max_epochs=1, patience=1, seed=17)
        with tempfile.TemporaryDirectory() as directory:
            results = train_source_models(
                {"a": 1, "b": 2},
                2,
                [batch],
                [batch],
                config,
                Path(directory) / "sources",
                hidden_size=8,
                dropout=0.0,
            )
            source_checkpoint = torch.load(
                results["a"].checkpoint_path, map_location="cpu", weights_only=True
            )
            artifact = assemble_v0(
                {"a": 1, "b": 2},
                2,
                {source_id: result.checkpoint_path for source_id, result in results.items()},
                Path(directory) / "v0.pt",
                hidden_size=8,
                dropout=0.0,
            )
            loaded = torch.load(artifact, map_location="cpu", weights_only=True)

        self.assertEqual(tuple(loaded["source_order"]), ("a", "b"))
        self.assertEqual(loaded["artifact_type"], "v0_late_fusion")
        self.assertEqual(set(results), {"a", "b"})
        self.assertEqual(source_checkpoint["config"]["selection_metric"], "validation_loss")

    def test_source_training_is_deterministic_for_fixed_seed_and_order(self):
        first = source_batch(
            1,
            [[[1.0], [2.0]], [[3.0], [0.0]]],
            [2, 0],
            [
                [[True], [True]],
                [[False], [False]],
            ],
        )
        second = source_batch(2, [[[2.0, 1.0], [3.0, 2.0]], [[1.0, 1.0], [2.0, 2.0]]], [2, 2])
        batch = TrainingBatch(
            {"a": first, "b": second},
            torch.tensor([0, 1]),
            torch.tensor([True, True]),
            ("d1", "d2"),
        )
        config = TrainingConfig(n_classes=2, max_epochs=1, patience=1, seed=31)
        with tempfile.TemporaryDirectory() as directory:
            first_results = train_source_models(
                {"a": 1, "b": 2},
                2,
                [batch],
                [batch],
                config,
                Path(directory) / "first",
                hidden_size=8,
                dropout=0.0,
            )
            second_results = train_source_models(
                {"a": 1, "b": 2},
                2,
                [batch],
                [batch],
                config,
                Path(directory) / "second",
                hidden_size=8,
                dropout=0.0,
            )
            first_state = {
                source_id: torch.load(
                    result.checkpoint_path, map_location="cpu", weights_only=True
                )["state_dict"]
                for source_id, result in first_results.items()
            }
            second_state = {
                source_id: torch.load(
                    result.checkpoint_path, map_location="cpu", weights_only=True
                )["state_dict"]
                for source_id, result in second_results.items()
            }

        for source_id in first_state:
            for name, expected in first_state[source_id].items():
                torch.testing.assert_close(
                    second_state[source_id][name], expected, rtol=0.0, atol=0.0
                )


if __name__ == "__main__":
    unittest.main()
