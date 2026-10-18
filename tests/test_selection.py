import unittest

from mmt.data.selection import select_observations


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.decision = {"decision_time": 10.0, "support_start": 5.0, "support_end": 9.0}
        self.source = {
            "observation_ids": [4, 2, 3, 1, 5],
            "values": [[4.0], [2.0], [3.0], [1.0], [5.0]],
            "valid_mask": [[True], [True], [True], [True], [False]],
            "event_start": [8.0, 6.0, 7.0, 4.9, 8.0],
            "event_end": [8.0, 6.0, 7.0, 5.5, 8.0],
            "available_at": [8.0, 10.0, 10.0001, 5.5, 8.0],
        }

    def test_applies_closed_support_and_availability_cutoff_then_sorts(self):
        selected = select_observations(self.source, self.decision)

        self.assertEqual(selected.observation_ids, (2, 4))
        self.assertEqual(selected.candidate_count, 3)
        self.assertEqual(selected.source_status, "present")
        self.assertEqual(selected.time_features[0][0], 0.6931471805599453)

    def test_invalid_rows_do_not_make_a_source_present(self):
        source = {key: value[4:] for key, value in self.source.items()}

        selected = select_observations(source, self.decision)

        self.assertEqual(selected.observation_ids, ())
        self.assertEqual(selected.source_status, "no_eligible_rows")
        self.assertIsNone(selected.valid_feature_fraction)

    def test_freshness_equality_is_usable_and_older_source_is_stale(self):
        at_limit = select_observations(self.source, self.decision, max_source_age=2.0)
        stale = select_observations(self.source, self.decision, max_source_age=1.99)

        self.assertEqual(at_limit.source_status, "present")
        self.assertEqual(stale.source_status, "stale")
        self.assertEqual(stale.selected_count_before_freshness, 2)
        self.assertEqual(stale.selected_count, 0)

    def test_rejects_invalid_decision_bounds(self):
        with self.assertRaises(ValueError):
            select_observations(self.source, {"decision_time": 8.0, "support_start": 5.0, "support_end": 9.0})


if __name__ == "__main__":
    unittest.main()