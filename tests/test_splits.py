import unittest

import numpy as np
import pandas as pd

from mmt.data.splits import SplitError, build_pamap2_window_decisions, validate_membership


class SplitAndDecisionTests(unittest.TestCase):
    def test_membership_requires_all_groups_to_be_split_disjointly(self):
        membership = pd.DataFrame(
            {
                "sequence_id": ["s1", "s2", "s3", "s4"],
                "group_id": ["g1", "g2", "g3", "g4"],
                "split": ["train", "validation", "calibration", "test"],
            }
        )
        report = validate_membership(["s1", "s2", "s3", "s4"], membership)

        self.assertEqual(report["counts"]["train"], {"sequences": 1, "groups": 1})
        leaking = membership.copy()
        leaking.loc[1, "group_id"] = "g1"
        with self.assertRaises(SplitError):
            validate_membership(["s1", "s2", "s3", "s4"], leaking)

    def test_pamap2_decisions_use_integer_stride_and_keep_targets_separate(self):
        membership = pd.DataFrame([{"sequence_id": "s1", "group_id": "p1", "split": "train"}])
        times = np.arange(0.0, 8.0, 1.0)
        labels = np.asarray([1, 1, 1, 1, 1, 0, np.nan, 24.0])

        decisions, targets = build_pamap2_window_decisions(
            "s1",
            times,
            labels,
            membership,
            {1: 0, 24: 1},
            window_seconds=5,
            stride_seconds=1,
            max_label_age_seconds=0.02,
        )

        self.assertEqual(decisions.decision_time.tolist(), [5.0, 6.0, 7.0])
        self.assertNotIn("target_class", decisions.columns)
        self.assertTrue(pd.isna(targets.target_class.iloc[0]))
        self.assertTrue(pd.isna(targets.target_class.iloc[1]))
        self.assertEqual(targets.target_class.iloc[2], 1)
        self.assertEqual(
            targets.exclusion_reason.iloc[:2].tolist(), ["activity_zero", "missing_label"]
        )
        self.assertTrue(pd.isna(targets.exclusion_reason.iloc[2]))
        self.assertEqual(decisions.support_start.tolist(), [0.0, 1.0, 2.0])

    def test_label_age_boundary_and_duplicate_timestamps_are_supported(self):
        membership = pd.DataFrame([{"sequence_id": "s1", "group_id": "p1", "split": "train"}])
        decisions, targets = build_pamap2_window_decisions(
            "s1",
            [0, 1, 2, 3, 4, 5, 5, 6],
            [1, 1, 1, 1, 1, 1, 1, 1],
            membership,
            {1: 0},
            window_seconds=5,
            stride_seconds=1,
        )

        self.assertEqual(len(decisions), 2)
        self.assertEqual(targets.score_eligible.tolist(), [True, True])
        self.assertEqual(decisions.decision_id.iloc[0], "pamap2|s1|00000000")


if __name__ == "__main__":
    unittest.main()
