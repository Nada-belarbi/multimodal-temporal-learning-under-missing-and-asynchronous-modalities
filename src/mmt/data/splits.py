"""Group-disjoint membership checks and target-separated PAMAP2 window decisions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

SPLITS = ("train", "validation", "calibration", "test")


class SplitError(ValueError):
    """Raised when group membership or decision construction violates the protocol."""


def validate_membership(
    sequence_ids: Sequence[str], membership: pd.DataFrame, require_all_splits: bool = True
) -> dict[str, Any]:
    required = {"sequence_id", "group_id", "split"}
    if not required <= set(membership.columns):
        raise SplitError(f"membership requires columns {sorted(required)}")
    if membership.sequence_id.isna().any() or membership.group_id.isna().any():
        raise SplitError("sequence_id and group_id cannot be null")
    if membership.sequence_id.duplicated().any():
        raise SplitError("each sequence must appear exactly once in membership")
    unknown_splits = set(membership.split) - set(SPLITS)
    if unknown_splits:
        raise SplitError(f"unknown split values: {sorted(unknown_splits)}")
    expected = set(sequence_ids)
    actual = set(membership.sequence_id)
    if expected != actual:
        raise SplitError(
            f"membership sequence mismatch: missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}"
        )
    overlapping_groups = membership.groupby("group_id").split.nunique()
    leaking = sorted(str(group) for group in overlapping_groups[overlapping_groups > 1].index)
    if leaking:
        raise SplitError(f"groups occur in more than one split: {leaking}")
    counts = {
        split: {
            "sequences": int((membership.split == split).sum()),
            "groups": int(membership.loc[membership.split == split, "group_id"].nunique()),
        }
        for split in SPLITS
    }
    if require_all_splits:
        empty = [split for split, count in counts.items() if count["sequences"] == 0]
        if empty:
            raise SplitError(f"required partitions are empty: {empty}")
    return {
        "counts": counts,
        "membership_hash_input": membership.sort_values("sequence_id").to_dict(orient="records"),
    }


def build_pamap2_window_decisions(
    sequence_id: str,
    timestamps: Sequence[float],
    raw_labels: Sequence[float],
    membership: pd.DataFrame,
    class_map: Mapping[int, int],
    window_seconds: float = 5.0,
    stride_seconds: float = 1.0,
    max_label_age_seconds: float = 0.02,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build stable decisions and a separate targets table for one PAMAP2 sequence."""
    if not sequence_id:
        raise SplitError("sequence_id must be nonempty")
    if window_seconds <= 0 or stride_seconds <= 0 or max_label_age_seconds < 0:
        raise SplitError("window and stride must be positive; label age must be nonnegative")
    times = np.asarray(timestamps, dtype=np.float64)
    labels = np.asarray(raw_labels, dtype=np.float64)
    if times.ndim != 1 or labels.shape != times.shape or len(times) == 0:
        raise SplitError("timestamps and raw_labels must be nonempty matching vectors")
    if not np.isfinite(times).all() or (np.diff(times) < 0).any():
        raise SplitError("timestamps must be finite and nondecreasing")
    membership_rows = membership[membership.sequence_id == sequence_id]
    if len(membership_rows) != 1:
        raise SplitError(f"sequence {sequence_id!r} must have exactly one membership row")

    start = float(times[0])
    end = float(times[-1])
    decisions: list[dict[str, Any]] = []
    targets: list[dict[str, Any]] = []
    decision_index = 0
    decision_time = start + window_seconds
    while decision_time <= end:
        decision_id = f"pamap2|{sequence_id}|{decision_index:08d}"
        target_row = int(np.searchsorted(times, decision_time, side="right") - 1)
        raw_label: float | None = None
        target_time: float | None = None
        target_class: int | None = None
        exclusion_reason: str | None = None
        if target_row < 0 or not np.isfinite(labels[target_row]):
            exclusion_reason = "missing_label"
        else:
            raw_value = float(labels[target_row])
            raw_label = raw_value
            target_time = float(times[target_row])
            label_age = decision_time - target_time
            if label_age > max_label_age_seconds:
                exclusion_reason = "expired_label"
            elif raw_value == 0:
                exclusion_reason = "activity_zero"
            elif raw_value != int(raw_value) or int(raw_value) not in class_map:
                raise SplitError(f"unknown nonzero PAMAP2 class {raw_value} in {sequence_id!r}")
            else:
                target_class = int(class_map[int(raw_value)])

        decisions.append(
            {
                "decision_id": decision_id,
                "sequence_id": sequence_id,
                "decision_time": float(decision_time),
                "support_start": float(decision_time - window_seconds),
                "support_end": float(decision_time),
            }
        )
        targets.append(
            {
                "decision_id": decision_id,
                "target_class": target_class,
                "target_time": target_time,
                "score_eligible": target_class is not None,
                "exclusion_reason": exclusion_reason,
                "raw_label": raw_label,
            }
        )
        decision_index += 1
        decision_time = start + window_seconds + decision_index * stride_seconds
    return pd.DataFrame(decisions), pd.DataFrame(targets)
