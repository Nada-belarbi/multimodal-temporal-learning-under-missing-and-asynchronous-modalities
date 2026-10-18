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
from mmt.training.checkpoint import load_state_dict


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
        return ModelOutput(self.layer(inputs), torch.ones(inputs.shape[0], dtype=torch.bool,
                                                          device=inputs.device), "logits")


class ModelTests(unittest.TestCase):
    def test_source_encoder_handles_unequal_lengths_and_absent_rows(self):
        batch = source_batch(2, [
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            [[2.0, 1.0], [0.0, 0.0], [0.0, 0.0]],
            [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        ], [3, 1, 0], [
            [[True, True], [True, True], [True, True]],
            [[True, True], [False, False], [False, False]],
            [[False, False], [False, False], [False, False]],
        ])
        encoder = SourceEncoder(2, hidden_size=8).eval()

        output = encoder(batch)

        self.assertEqual(tuple(output.states.shape), (3, 3, 8))
        self.assertTrue(torch.equal(output.last_state[2], torch.zeros(8)))
        self.assertTrue(torch.isfinite(output.states).all())

    def test_v0_and_fusions_are_finite_on_partial_and_empty_source_rows(self):
        torch.manual_seed(4)
        inputs = {
            "a": source_batch(2, [
                [[1.0, 0.0], [2.0, 1.0]], [[3.0, 1.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]],
            ], [2, 1, 0], [
                [[True, True], [True, True]], [[True, True], [False, False]],
                [[False, False], [False, False]],
            ]),
            "b": source_batch(1, [
                [[1.0], [2.0]], [[0.0], [0.0]], [[0.0], [0.0]],
            ], [2, 0, 0], [
                [[True], [True]], [[False], [False]], [[False], [False]],
            ]),
        }
        models = [
            V0LateFusion({"a": 2, "b": 1}, 3, hidden_size=8).eval(),
            V1GatedFusion({"a": 2, "b": 1}, 3, hidden_size=8).eval(),
            V2AttentionFusion({"a": 2, "b": 1}, 3, hidden_size=8, n_heads=2,
                              feedforward_size=16).eval(),
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
            TrainingBatch(torch.tensor([[-1.0], [1.0]]), torch.tensor([0, 1]),
                          torch.tensor([True, True]), ("a", "b")),
        ]
        model = LinearClassifier()
        with tempfile.TemporaryDirectory() as temp_dir:
            result = train(model, batches, batches, TrainingConfig(n_classes=2, max_epochs=2,
                                                                  patience=2), temp_dir)
            reloaded = LinearClassifier()
            payload = load_state_dict(reloaded, Path(temp_dir) / "best.pt")
            self.assertTrue((Path(temp_dir) / "best.pt").is_file())
            self.assertTrue((Path(temp_dir) / "last.pt").is_file())

        self.assertGreaterEqual(result.best_epoch, 1)
        self.assertEqual(payload["epoch"], result.best_epoch)
        self.assertFalse(reloaded.training)


if __name__ == "__main__":
    unittest.main()