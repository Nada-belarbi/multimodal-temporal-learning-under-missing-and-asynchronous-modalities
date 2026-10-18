"""Deterministic hold-based common-grid representation for B-GRID."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

from .selection import SelectionResult


@dataclass(frozen=True)
class GridSequence:
    grid_times: tuple[float, ...]
    values: tuple[tuple[float, ...], ...]
    feature_mask: tuple[tuple[bool, ...], ...]
    ages: tuple[tuple[float, ...], ...]
    decision_time: float
    feature_order: tuple[tuple[str, int], ...]


def build_common_grid(
    selections: Mapping[str, SelectionResult],
    source_dimensions: Mapping[str, int],
    support_start: float,
    support_end: float,
    decision_time: float,
    grid_step_seconds: float = 0.05,
    hold_limit_seconds: float = 0.5,
) -> GridSequence:
    """Hold the latest valid event-time value; never interpolate toward future samples."""
    values_to_validate = (support_start, support_end, decision_time,
                          grid_step_seconds, hold_limit_seconds)
    if not all(math.isfinite(float(value)) for value in values_to_validate):
        raise ValueError("grid times and settings must be finite")
    if support_start >= support_end or support_end > decision_time:
        raise ValueError("grid support must have positive duration and end by decision_time")
    if grid_step_seconds <= 0 or hold_limit_seconds < 0:
        raise ValueError("grid step must be positive and hold limit nonnegative")
    if set(selections) != set(source_dimensions):
        raise ValueError("selection keys must match source dimensions")
    feature_order = tuple((source_id, feature_index)
                         for source_id, dimension in source_dimensions.items()
                         for feature_index in range(dimension))
    if any(dimension <= 0 for dimension in source_dimensions.values()):
        raise ValueError("source dimensions must be positive")

    duration = support_end - support_start
    regular_count = int(duration // grid_step_seconds)
    grid = [support_start + index * grid_step_seconds for index in range(1, regular_count + 1)]
    if not grid or grid[-1] != support_end:
        grid.append(support_end)

    grid_values: list[tuple[float, ...]] = []
    grid_masks: list[tuple[bool, ...]] = []
    grid_ages: list[tuple[float, ...]] = []
    for grid_time in grid:
        row_values: list[float] = []
        row_masks: list[bool] = []
        row_ages: list[float] = []
        for source_id, dimension in source_dimensions.items():
            selected = selections[source_id]
            if any(len(row) != dimension for row in selected.values):
                raise ValueError(f"selected feature dimension differs for source {source_id!r}")
            for feature_index in range(dimension):
                candidates = [index for index, end in enumerate(selected.event_end)
                              if end <= grid_time and selected.valid_mask[index][feature_index]]
                if candidates:
                    chosen = max(candidates, key=lambda index: (
                        selected.event_end[index], selected.event_start[index],
                        selected.observation_ids[index],
                    ))
                    age = grid_time - selected.event_end[chosen]
                    if age <= hold_limit_seconds:
                        row_values.append(selected.values[chosen][feature_index])
                        row_masks.append(True)
                        row_ages.append(age)
                        continue
                row_values.append(0.0)
                row_masks.append(False)
                row_ages.append(hold_limit_seconds)
        grid_values.append(tuple(row_values))
        grid_masks.append(tuple(row_masks))
        grid_ages.append(tuple(row_ages))

    return GridSequence(tuple(grid), tuple(grid_values), tuple(grid_masks), tuple(grid_ages),
                        float(decision_time), feature_order)