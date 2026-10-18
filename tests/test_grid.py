import unittest

from mmt.data.grid import build_common_grid
from mmt.data.selection import SelectionResult


def selection(ids, values, starts, ends, masks):
    return SelectionResult(
        observation_ids=tuple(ids), values=tuple(tuple(row) for row in values),
        valid_mask=tuple(tuple(row) for row in masks), event_start=tuple(starts),
        event_end=tuple(ends), available_at=tuple(ends), time_features=(),
        candidate_count=len(ids), selected_count_before_freshness=len(ids),
        selected_count=len(ids), latest_event_end=max(ends, default=None),
        latest_age=None, valid_feature_fraction=None, source_status="present" if ids else "empty",
    )


class GridTests(unittest.TestCase):
    def test_grid_holds_latest_valid_value_without_future_interpolation(self):
        chosen = selection([1, 2], [[2.0], [8.0]], [1.0, 3.0], [1.0, 3.0], [[True], [True]])

        grid = build_common_grid({"s": chosen}, {"s": 1}, 0.0, 4.0, 4.0,
                                 grid_step_seconds=2.0, hold_limit_seconds=2.0)

        self.assertEqual(grid.grid_times, (2.0, 4.0))
        self.assertEqual(grid.values, ((2.0,), (8.0,)))
        self.assertEqual(grid.feature_mask, ((True,), (True,)))
        self.assertEqual(grid.ages, ((1.0,), (1.0,)))

    def test_expired_value_becomes_masked_zero(self):
        chosen = selection([1], [[2.0]], [0.0], [0.0], [[True]])

        grid = build_common_grid({"s": chosen}, {"s": 1}, 0.0, 1.0, 1.0,
                                 grid_step_seconds=1.0, hold_limit_seconds=0.5)

        self.assertEqual(grid.values, ((0.0,),))
        self.assertEqual(grid.feature_mask, ((False,),))
        self.assertEqual(grid.ages, ((0.5,),))

    def test_rejects_zero_duration_support(self):
        with self.assertRaises(ValueError):
            build_common_grid({"s": selection([], [], [], [], [])}, {"s": 1}, 1.0, 1.0, 1.0)


if __name__ == "__main__":
    unittest.main()