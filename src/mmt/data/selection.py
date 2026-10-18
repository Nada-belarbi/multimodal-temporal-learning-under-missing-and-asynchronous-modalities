"""Pure temporal selection shared by offline batches and streaming inference."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class SelectionResult:
    observation_ids: tuple[int, ...]
    values: tuple[tuple[float, ...], ...]
    valid_mask: tuple[tuple[bool, ...], ...]
    event_start: tuple[float, ...]
    event_end: tuple[float, ...]
    available_at: tuple[float, ...]
    time_features: tuple[tuple[float, float, float], ...]
    candidate_count: int
    selected_count_before_freshness: int
    selected_count: int
    latest_event_end: float | None
    latest_age: float | None
    valid_feature_fraction: float | None
    source_status: str


def _field(record: Any, name: str) -> Any:
    if isinstance(record, Mapping):
        return record[name]
    return getattr(record, name)


def _finite(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def select_observations(
    source: Any,
    decision: Any,
    max_source_age: float | None = None,
) -> SelectionResult:
    """Select fully contained, available and feature-valid observations for a decision."""
    decision_time = _finite(_field(decision, "decision_time"), "decision_time")
    support_start = _finite(_field(decision, "support_start"), "support_start")
    support_end = _finite(_field(decision, "support_end"), "support_end")
    if support_start > support_end or support_end > decision_time:
        raise ValueError("decision support must satisfy start <= end <= decision_time")
    if max_source_age is not None:
        max_source_age = _finite(max_source_age, "max_source_age")
        if max_source_age < 0:
            raise ValueError("max_source_age must be nonnegative")

    ids = _field(source, "observation_ids")
    values = _field(source, "values")
    masks = _field(source, "valid_mask")
    starts = _field(source, "event_start")
    ends = _field(source, "event_end")
    arrivals = _field(source, "available_at")
    lengths = {len(ids), len(values), len(masks), len(starts), len(ends), len(arrivals)}
    if len(lengths) != 1:
        raise ValueError("source arrays must have equal row counts")

    rows: list[tuple[int, float, float, float, tuple[float, ...], tuple[bool, ...]]] = []
    candidates = 0
    feature_count: int | None = None
    for index in range(len(ids)):
        observation_id = int(ids[index])
        event_start = _finite(starts[index], "event_start")
        event_end = _finite(ends[index], "event_end")
        available_at = _finite(arrivals[index], "available_at")
        if event_start > event_end or event_end > available_at:
            raise ValueError(f"invalid event/availability order for observation {observation_id}")
        row_values = tuple(float(value) for value in values[index])
        row_mask = tuple(bool(value) for value in masks[index])
        if len(row_values) != len(row_mask):
            raise ValueError(f"value and validity dimensions differ for observation {observation_id}")
        if feature_count is None:
            feature_count = len(row_values)
        elif len(row_values) != feature_count:
            raise ValueError("source feature dimensions must be consistent")
        if any(valid and not math.isfinite(value) for value, valid in zip(row_values, row_mask)):
            raise ValueError(f"valid features must be finite for observation {observation_id}")

        if available_at <= decision_time and event_start >= support_start and event_end <= support_end:
            candidates += 1
            if any(row_mask):
                rows.append((observation_id, event_start, event_end, available_at, row_values, row_mask))

    rows.sort(key=lambda row: (row[2], row[1], row[0]))
    before_freshness = len(rows)
    latest_event_end = max((row[2] for row in rows), default=None)
    latest_age = None if latest_event_end is None else decision_time - latest_event_end
    stale = (max_source_age is not None and latest_age is not None and latest_age > max_source_age)
    if stale:
        rows = []

    n_features = feature_count or 0
    valid_count = sum(sum(row[5]) for row in rows)
    valid_fraction = valid_count / (len(rows) * n_features) if rows and n_features else None
    if stale:
        source_status = "stale"
    elif rows:
        source_status = "present"
    elif len(ids) == 0:
        source_status = "empty"
    else:
        source_status = "no_eligible_rows"

    time_features: list[tuple[float, float, float]] = []
    previous_end = support_start
    for row in rows:
        _, event_start, event_end, _, _, _ = row
        delta = event_end - previous_end
        age = decision_time - event_end
        duration = event_end - event_start
        if min(delta, age, duration) < 0:
            raise ValueError("selected event times must produce nonnegative temporal features")
        time_features.append((math.log1p(delta), math.log1p(age), math.log1p(duration)))
        previous_end = event_end

    return SelectionResult(
        observation_ids=tuple(row[0] for row in rows),
        values=tuple(row[4] for row in rows),
        valid_mask=tuple(row[5] for row in rows),
        event_start=tuple(row[1] for row in rows),
        event_end=tuple(row[2] for row in rows),
        available_at=tuple(row[3] for row in rows),
        time_features=tuple(time_features),
        candidate_count=candidates,
        selected_count_before_freshness=before_freshness,
        selected_count=len(rows),
        latest_event_end=latest_event_end,
        latest_age=latest_age,
        valid_feature_fraction=valid_fraction,
        source_status=source_status,
    )