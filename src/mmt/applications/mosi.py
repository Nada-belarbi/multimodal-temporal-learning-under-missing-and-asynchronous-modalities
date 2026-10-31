"""MOSI segment adapter consuming explicitly audited feature bindings."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import h5py
import numpy as np


class MosiDataError(ValueError):
    """Raised when audited MOSI segment/feature bindings are ambiguous or invalid."""


@dataclass(frozen=True)
class MosiSegment:
    sequence_id: str
    video_id: str
    decision: dict[str, Any]
    sources: dict[str, dict[str, np.ndarray]]
    target: dict[str, Any]
    audit: dict[str, Any]


class MosiAdapter:
    """Convert one segment using feature keys and timing already resolved by an audit step."""

    def read_hdf5_feature_bindings(
        self,
        path: str,
        bindings: Mapping[str, Mapping[str, str]],
        *,
        availability_mode: str,
    ) -> dict[str, dict[str, np.ndarray]]:
        """Read HDF5 feature sequences through paths supplied by the dataset audit.

        Each binding requires `values` and `event_intervals` dataset paths. Optional
        keys are `valid_mask`, `observation_ids`, and `available_at`. Offline access
        may use event-end availability; causal access must provide audited arrival data.
        """
        if availability_mode not in {"offline", "causal"}:
            raise MosiDataError("availability_mode must be 'offline' or 'causal'")
        if not bindings:
            raise MosiDataError("at least one audited HDF5 feature binding is required")
        loaded: dict[str, dict[str, np.ndarray]] = {}
        try:
            with h5py.File(path, "r") as hdf:
                for source_id, binding in bindings.items():
                    if not source_id or not isinstance(binding, Mapping):
                        raise MosiDataError("each feature binding needs a source ID and mapping")
                    required = {"values", "event_intervals"}
                    if not required <= set(binding):
                        raise MosiDataError(
                            f"{source_id}: bindings require {sorted(required)} dataset paths"
                        )
                    values = np.asarray(hdf[binding["values"]], dtype=np.float32)
                    intervals = np.asarray(hdf[binding["event_intervals"]], dtype=np.float64)
                    if values.ndim != 2 or intervals.shape != (len(values), 2):
                        raise MosiDataError(
                            f"{source_id}: values must be [N,D] and intervals must be [N,2]"
                        )
                    count = len(values)
                    mask = (
                        np.asarray(hdf[binding["valid_mask"]], dtype=bool)
                        if "valid_mask" in binding
                        else np.isfinite(values)
                    )
                    observation_ids = (
                        np.asarray(hdf[binding["observation_ids"]], dtype=np.int64)
                        if "observation_ids" in binding
                        else np.arange(count, dtype=np.int64)
                    )
                    if "available_at" in binding:
                        available_at = np.asarray(hdf[binding["available_at"]], dtype=np.float64)
                    elif availability_mode == "offline":
                        available_at = intervals[:, 1].copy()
                    else:
                        raise MosiDataError(
                            f"{source_id}: causal mode requires an audited available_at dataset"
                        )
                    if mask.shape != values.shape or observation_ids.shape != (count,):
                        raise MosiDataError(f"{source_id}: mask or observation ID shape mismatch")
                    if available_at.shape != (count,):
                        raise MosiDataError(f"{source_id}: available_at must have shape ({count},)")
                    loaded[source_id] = {
                        "values": np.where(mask, values, 0.0).astype(np.float32),
                        "valid_mask": mask,
                        "observation_ids": observation_ids,
                        "event_start": intervals[:, 0],
                        "event_end": intervals[:, 1],
                        "available_at": available_at,
                    }
        except MosiDataError:
            raise
        except (OSError, KeyError, TypeError, ValueError) as error:
            raise MosiDataError(f"cannot read audited HDF5 feature bindings: {error}") from error
        return loaded

    def prepare_segment(
        self,
        segment: Mapping[str, Any],
        features_by_source: Mapping[str, Mapping[str, Any]],
        source_dimensions: Mapping[str, int],
        *,
        timestamps_relative: bool,
        availability_mode: str = "offline",
        extraction_context: Mapping[str, str] | None = None,
    ) -> MosiSegment:
        sequence_id = str(segment.get("sequence_id", ""))
        video_id = str(segment.get("video_id", ""))
        if not sequence_id or not video_id or any(char in sequence_id + video_id for char in "/\\"):
            raise MosiDataError("sequence_id and video_id must be declared path-safe identities")
        start = _finite(segment.get("segment_start"), "segment_start")
        end = _finite(segment.get("segment_end"), "segment_end")
        if end <= start:
            raise MosiDataError("segment_end must be greater than segment_start")
        duration = end - start
        if availability_mode not in {"offline", "causal"}:
            raise MosiDataError("availability_mode must be 'offline' or 'causal'")
        if not source_dimensions or any(dim <= 0 for dim in source_dimensions.values()):
            raise MosiDataError("source_dimensions must be explicit positive audited dimensions")
        unknown_sources = set(features_by_source) - set(source_dimensions)
        if unknown_sources:
            raise MosiDataError(
                f"feature bindings include unknown sources {sorted(unknown_sources)}"
            )

        causal_contexts = dict(extraction_context or {})
        sources: dict[str, dict[str, np.ndarray]] = {}
        audit_sources: dict[str, dict[str, int]] = {}
        for source_id, dimension in source_dimensions.items():
            raw = features_by_source.get(source_id)
            if raw is None:
                sources[source_id] = _empty_source(dimension)
                audit_sources[source_id] = {"rows_in": 0, "rows_used": 0, "crossing_intervals": 0}
                continue
            if availability_mode == "causal" and not causal_contexts.get(source_id, "").strip():
                raise MosiDataError(
                    f"causal availability requires audited extraction context for {source_id!r}"
                )

            values = np.asarray(raw.get("values"), dtype=np.float32)
            mask = np.asarray(raw.get("valid_mask"), dtype=bool)
            observation_ids = np.asarray(raw.get("observation_ids"), dtype=np.int64)
            event_start = np.asarray(raw.get("event_start"), dtype=np.float64)
            event_end = np.asarray(raw.get("event_end"), dtype=np.float64)
            available_at = np.asarray(raw.get("available_at"), dtype=np.float64)
            if values.ndim != 2 or values.shape[1] != dimension:
                raise MosiDataError(f"{source_id}: values must use audited dimension {dimension}")
            count = values.shape[0]
            if mask.shape != values.shape or observation_ids.shape != (count,):
                raise MosiDataError(f"{source_id}: values, masks and IDs have incompatible shapes")
            if any(array.shape != (count,) for array in (event_start, event_end, available_at)):
                raise MosiDataError(f"{source_id}: interval arrays have incompatible shapes")
            if not all(
                np.isfinite(array).all() for array in (event_start, event_end, available_at)
            ):
                raise MosiDataError(f"{source_id}: event and availability times must be finite")
            if len(np.unique(observation_ids)) != count:
                raise MosiDataError(
                    f"{source_id}: observation IDs must be unique within the segment"
                )
            if not np.isfinite(values[mask]).all():
                raise MosiDataError(f"{source_id}: valid feature values must be finite")
            if (event_end < event_start).any() or (available_at < event_end).any():
                raise MosiDataError(f"{source_id}: invalid event/availability order")

            if not timestamps_relative:
                event_start = event_start - start
                event_end = event_end - start
                available_at = available_at - start
            inside = (event_start >= 0.0) & (event_end <= duration)
            crossing_count = int((~inside).sum())
            indices = np.flatnonzero(inside)
            indices = np.asarray(
                sorted(
                    indices, key=lambda i: (event_end[i], event_start[i], int(observation_ids[i]))
                ),
                dtype=np.int64,
            )
            event_start = event_start[indices]
            event_end = event_end[indices]
            available_at = available_at[indices]
            observation_ids = observation_ids[indices]
            values = values[indices]
            mask = mask[indices]
            row_usable = mask.any(axis=1)
            event_start = event_start[row_usable]
            event_end = event_end[row_usable]
            available_at = available_at[row_usable]
            observation_ids = observation_ids[row_usable]
            values = values[row_usable]
            mask = mask[row_usable]
            if availability_mode == "offline":
                available_at = np.full(len(indices[row_usable]), duration, dtype=np.float64)
            if (available_at < event_end).any():
                raise MosiDataError(f"{source_id}: converted availability precedes event end")
            sources[source_id] = {
                "values": np.where(mask, values, 0.0).astype(np.float32),
                "valid_mask": mask.astype(bool, copy=False),
                "observation_ids": observation_ids.astype(np.int64, copy=False),
                "event_start": event_start.astype(np.float64, copy=False),
                "event_end": event_end.astype(np.float64, copy=False),
                "available_at": available_at.astype(np.float64, copy=False),
            }
            audit_sources[source_id] = {
                "rows_in": count,
                "rows_used": int(len(observation_ids)),
                "crossing_intervals": crossing_count,
                "all_invalid_rows": int(count - int(mask.shape[0]) - crossing_count),
            }

        raw_sentiment = segment.get("sentiment")
        target_class: int | None = None
        exclusion_reason: str | None = None
        sentiment_value: float | None
        if raw_sentiment is None or (
            isinstance(raw_sentiment, float) and math.isnan(raw_sentiment)
        ):
            sentiment_value = None
            exclusion_reason = "missing_label"
        else:
            sentiment_value = _finite(raw_sentiment, "sentiment")
            if sentiment_value < 0:
                target_class = 0
            elif sentiment_value > 0:
                target_class = 1
            else:
                exclusion_reason = "zero_sentiment"

        decision = {
            "decision_id": f"mosi|{sequence_id}|closure",
            "sequence_id": sequence_id,
            "decision_time": duration,
            "support_start": 0.0,
            "support_end": duration,
        }
        target = {
            "decision_id": decision["decision_id"],
            "target_class": target_class,
            "score_eligible": target_class is not None,
            "exclusion_reason": exclusion_reason,
            "raw_sentiment": sentiment_value,
        }
        audit = {
            "sequence_id": sequence_id,
            "original_video_id": video_id,
            "segment_start": start,
            "segment_end": end,
            "duration": duration,
            "timestamps_relative": timestamps_relative,
            "availability_mode": availability_mode,
            "extraction_context": causal_contexts if availability_mode == "causal" else {},
            "sources": audit_sources,
        }
        return MosiSegment(sequence_id, video_id, decision, sources, target, audit)


def _empty_source(dimension: int) -> dict[str, np.ndarray]:
    return {
        "values": np.zeros((0, dimension), dtype=np.float32),
        "valid_mask": np.zeros((0, dimension), dtype=bool),
        "observation_ids": np.zeros((0,), dtype=np.int64),
        "event_start": np.zeros((0,), dtype=np.float64),
        "event_end": np.zeros((0,), dtype=np.float64),
        "available_at": np.zeros((0,), dtype=np.float64),
    }


def _finite(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise MosiDataError(f"{name} must be finite numeric data") from error
    if not math.isfinite(result):
        raise MosiDataError(f"{name} must be finite")
    return result
