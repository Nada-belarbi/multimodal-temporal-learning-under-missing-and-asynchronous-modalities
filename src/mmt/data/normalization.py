"""Training-only, unique-observation normalization for canonical source streams."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .selection import select_observations

NORMALIZER_SCHEMA_VERSION = "1.0"


class NormalizerError(ValueError):
    """Raised when fit data or a persisted normalizer identity is invalid."""


@dataclass(frozen=True)
class Normalizer:
    source_order: tuple[str, ...]
    counts: Mapping[str, np.ndarray]
    means: Mapping[str, np.ndarray]
    scales: Mapping[str, np.ndarray]
    identity: Mapping[str, str]

    def transform(
        self, source_id: str, values: np.ndarray, valid_mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if source_id not in self.means:
            raise NormalizerError(f"unknown source {source_id!r}")
        values = np.asarray(values, dtype=np.float32)
        valid_mask = np.asarray(valid_mask, dtype=bool)
        if values.shape != valid_mask.shape or values.ndim < 1:
            raise NormalizerError("values and valid_mask must have matching shapes")
        if values.shape[-1] != len(self.means[source_id]):
            raise NormalizerError("feature dimension does not match normalizer")
        if not np.isfinite(values[valid_mask]).all():
            raise NormalizerError("valid feature values must be finite")
        normalized = np.zeros(values.shape, dtype=np.float32)
        normalized[valid_mask] = ((values - self.means[source_id]) / self.scales[source_id])[
            valid_mask
        ]
        return normalized, valid_mask.copy()


def fit_normalizer(
    sources: Mapping[tuple[str, str], Any],
    train_decisions: Sequence[Any],
    source_order: Sequence[str],
    identity: Mapping[str, str],
) -> Normalizer:
    """Fit float64 population statistics once per eligible observation ID."""
    if not source_order or len(set(source_order)) != len(source_order):
        raise NormalizerError("source_order must be nonempty and unique")
    required_identity = {"manifest_hash", "training_membership_hash"}
    if not required_identity <= set(identity) or any(
        not identity[key] for key in required_identity
    ):
        raise NormalizerError("identity requires manifest_hash and training_membership_hash")
    if not train_decisions:
        raise NormalizerError("at least one training decision is required")

    decisions_by_sequence: dict[str, list[Any]] = {}
    for decision in train_decisions:
        split = _field(decision, "split", default="train")
        if split != "train":
            raise NormalizerError(f"normalizer may only fit train decisions, got split={split!r}")
        decision_time = float(_field(decision, "decision_time"))
        support_start = float(_field(decision, "support_start"))
        support_end = float(_field(decision, "support_end"))
        if not np.isfinite([decision_time, support_start, support_end]).all():
            raise NormalizerError("decision times must be finite")
        if not support_start <= support_end <= decision_time:
            raise NormalizerError("decision support must satisfy start <= end <= decision_time")
        sequence_id = str(_field(decision, "sequence_id"))
        decisions_by_sequence.setdefault(sequence_id, []).append(decision)

    dimensions: dict[str, int] = {}
    counts_by_source: dict[str, np.ndarray] = {}
    means_by_source: dict[str, np.ndarray] = {}
    m2_by_source: dict[str, np.ndarray] = {}
    for source_id in source_order:
        for sequence_id, sequence_decisions in decisions_by_sequence.items():
            source = sources.get((sequence_id, source_id))
            if source is None:
                continue
            values = np.asarray(source["values"], dtype=np.float32)
            masks = np.asarray(source["valid_mask"], dtype=bool)
            observation_ids = np.asarray(source["observation_ids"], dtype=np.int64)
            if values.ndim != 2 or masks.shape != values.shape:
                raise NormalizerError(f"invalid values or masks for {sequence_id}/{source_id}")
            if observation_ids.shape != (len(values),):
                raise NormalizerError(f"invalid observation IDs for {sequence_id}/{source_id}")
            dimension = values.shape[1]
            if source_id not in dimensions:
                dimensions[source_id] = dimension
                counts_by_source[source_id] = np.zeros(dimension, dtype=np.int64)
                means_by_source[source_id] = np.zeros(dimension, dtype=np.float64)
                m2_by_source[source_id] = np.zeros(dimension, dtype=np.float64)
            elif dimensions[source_id] != dimension:
                raise NormalizerError(f"inconsistent feature dimension for {source_id!r}")

            selected_rows = _training_observation_mask(source, sequence_decisions)
            if not selected_rows.any():
                continue

            selected_values = values[selected_rows]
            selected_masks = masks[selected_rows]
            selected_ids = observation_ids[selected_rows]
            if len(np.unique(selected_ids)) != len(selected_ids):
                raise NormalizerError(
                    f"duplicate observation IDs selected for {sequence_id}/{source_id}"
                )
            if not np.isfinite(selected_values[selected_masks]).all():
                raise NormalizerError(f"non-finite valid values for {sequence_id}/{source_id}")

            batch_counts = selected_masks.sum(axis=0, dtype=np.int64)
            for feature_index, batch_count in enumerate(batch_counts):
                if batch_count == 0:
                    continue
                batch_values = selected_values[selected_masks[:, feature_index], feature_index]
                batch_mean = float(np.mean(batch_values, dtype=np.float64))
                batch_delta = batch_values.astype(np.float64) - batch_mean
                batch_m2 = float(np.dot(batch_delta, batch_delta))
                previous_count = int(counts_by_source[source_id][feature_index])
                total_count = previous_count + int(batch_count)
                delta = batch_mean - means_by_source[source_id][feature_index]
                m2_by_source[source_id][feature_index] += (
                    batch_m2 + delta * delta * previous_count * int(batch_count) / total_count
                )
                means_by_source[source_id][feature_index] += delta * int(batch_count) / total_count
                counts_by_source[source_id][feature_index] = total_count

    missing_sources = set(source_order) - set(dimensions)
    if missing_sources:
        raise NormalizerError(
            f"no eligible train observations for sources {sorted(missing_sources)}"
        )
    for source_id, counts in counts_by_source.items():
        if (counts == 0).any():
            raise NormalizerError(f"features without valid training values in {source_id!r}")

    scales: dict[str, np.ndarray] = {}
    for source_id, counts in counts_by_source.items():
        variance = m2_by_source[source_id] / counts
        scale = np.sqrt(np.maximum(variance, 0.0))
        scale[scale == 0] = 1.0
        scales[source_id] = scale
    return Normalizer(
        tuple(source_order), counts_by_source, means_by_source, scales, dict(identity)
    )


def _training_observation_mask(source: Any, decisions: Sequence[Any]) -> np.ndarray:
    """Union selected rows once, using index ranges for monotonic feature timelines."""
    observation_ids = np.asarray(source["observation_ids"], dtype=np.int64)
    event_start = np.asarray(source["event_start"], dtype=np.float64)
    event_end = np.asarray(source["event_end"], dtype=np.float64)
    available_at = np.asarray(source["available_at"], dtype=np.float64)
    values = np.asarray(source["values"])
    valid_mask = np.asarray(source["valid_mask"], dtype=bool)
    row_count = len(observation_ids)
    if any(array.shape != (row_count,) for array in (event_start, event_end, available_at)):
        raise NormalizerError("source times and observation IDs must have matching lengths")
    if valid_mask.shape != values.shape or values.ndim != 2 or len(values) != row_count:
        raise NormalizerError("source values, masks and IDs must have matching row counts")
    if not all(np.isfinite(array).all() for array in (event_start, event_end, available_at)):
        raise NormalizerError("source event and availability times must be finite")
    if (event_start > event_end).any() or (event_end > available_at).any():
        raise NormalizerError("source event/availability order is invalid")

    monotonic = (np.diff(event_start) >= 0).all() and (np.diff(event_end) >= 0).all()
    selected_rows = np.zeros(row_count, dtype=bool)
    if monotonic and len(np.unique(observation_ids)) == row_count:
        valid_rows = valid_mask.any(axis=1)
        for decision in decisions:
            decision_time = float(_field(decision, "decision_time"))
            support_start = float(_field(decision, "support_start"))
            support_end = float(_field(decision, "support_end"))
            left = int(np.searchsorted(event_start, support_start, side="left"))
            right = int(np.searchsorted(event_end, support_end, side="right"))
            candidates = np.arange(left, right, dtype=np.intp)
            if candidates.size:
                eligible = (available_at[candidates] <= decision_time) & valid_rows[candidates]
                selected_rows[candidates[eligible]] = True
        return selected_rows

    selected_ids: set[int] = set()
    for decision in decisions:
        selected_ids.update(select_observations(source, decision).observation_ids)
    if not selected_ids:
        return selected_rows
    return np.isin(observation_ids, np.fromiter(selected_ids, dtype=np.int64))


def save_normalizer(path: str | Path, normalizer: Normalizer) -> Path:
    """Atomically save arrays and metadata in a pickle-free NPZ archive."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        prefix=destination.name + ".tmp-", suffix=".npz", dir=destination.parent
    )
    os.close(descriptor)
    metadata = json.dumps(
        {
            "schema_version": NORMALIZER_SCHEMA_VERSION,
            "source_order": normalizer.source_order,
            "identity": normalizer.identity,
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    arrays: dict[str, np.ndarray] = {"metadata_json": np.asarray(metadata)}
    for source_id in normalizer.source_order:
        arrays[f"{source_id}__counts"] = normalizer.counts[source_id]
        arrays[f"{source_id}__mean"] = normalizer.means[source_id]
        arrays[f"{source_id}__scale"] = normalizer.scales[source_id]
    try:
        with open(temp_name, "wb") as stream:
            np.savez_compressed(stream, **arrays)
        os.replace(temp_name, destination)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise
    return destination


def load_normalizer(
    path: str | Path, expected_identity: Mapping[str, str] | None = None
) -> Normalizer:
    try:
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata_json"].item()))
            if metadata.get("schema_version") != NORMALIZER_SCHEMA_VERSION:
                raise NormalizerError("incompatible normalizer schema version")
            source_order = tuple(metadata["source_order"])
            identity = metadata["identity"]
            if expected_identity is not None and dict(expected_identity) != identity:
                raise NormalizerError("normalizer identity does not match prepared training data")
            counts = {
                source_id: np.array(archive[f"{source_id}__counts"], copy=True)
                for source_id in source_order
            }
            means = {
                source_id: np.array(archive[f"{source_id}__mean"], copy=True)
                for source_id in source_order
            }
            scales = {
                source_id: np.array(archive[f"{source_id}__scale"], copy=True)
                for source_id in source_order
            }
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as error:
        if isinstance(error, NormalizerError):
            raise
        raise NormalizerError(f"cannot load normalizer: {error}") from error
    return Normalizer(source_order, counts, means, scales, identity)


def _field(record: Any, name: str, default: Any = None) -> Any:
    if isinstance(record, Mapping):
        return record.get(name, default)
    return getattr(record, name, default)
