import unittest

import torch

from mmt.data.grid import GridSequence
from mmt.models.baselines import BGridClassifier, collate_grid_sequences


class GridModelTests(unittest.TestCase):
    def test_variable_length_grid_collation_and_model_forward(self):
        feature_order = (("imu", 0), ("audio", 0))
        grids = [
            GridSequence(
                (0.5, 1.0),
                ((1.0, 0.0), (2.0, 1.0)),
                ((True, False), (True, True)),
                ((0.0, 0.5), (0.0, 0.0)),
                1.0,
                feature_order,
            ),
            GridSequence(
                (1.0,), ((0.0, 3.0),), ((False, True),), ((0.5, 0.0),), 1.0, feature_order
            ),
        ]

        batch = collate_grid_sequences(grids)
        model = BGridClassifier(feature_count=2, n_classes=2, hidden_size=8).eval()
        output = model(batch)

        self.assertEqual(batch.lengths.tolist(), [2, 1])
        self.assertEqual(batch.padding_mask.tolist(), [[False, False], [False, True]])
        self.assertEqual(tuple(output.scores.shape), (2, 2))
        self.assertEqual(output.computable.tolist(), [True, True])
        self.assertTrue(torch.isfinite(output.scores).all())


if __name__ == "__main__":
    unittest.main()
