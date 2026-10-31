"""Audited PAMAP2 Protocol-file reader and canonical observation adapter."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PAMAP2_COLUMN_COUNT = 54
PAMAP2_CLASS_IDS = (1, 2, 3, 4, 5, 6, 7, 12, 13, 16, 17, 24)
PAMAP2_CLASS_MAP = {raw_id: index for index, raw_id in enumerate(PAMAP2_CLASS_IDS)}
PAMAP2_SOURCE_COLUMNS = {
    "hand": (4, 5, 6, 10, 11, 12),
    "chest": (21, 22, 23, 27, 28, 29),
    "ankle": (38, 39, 40, 44, 45, 46),
    "heart_rate": (2,),
}


class Pamap2DataError(ValueError):
    """Raised when a PAMAP2 Protocol file does not match its audited schema."""


@dataclass(frozen=True)
class Pamap2Recording:
    sequence_id: str
    sources: dict[str, dict[str, np.ndarray]]
    timestamps: np.ndarray
    raw_labels: np.ndarray
    target_class: np.ndarray
    target_valid: np.ndarray
    audit: dict[str, Any]


class Pamap2Adapter:
    def __init__(self, chunksize: int = 100_000) -> None:
        if chunksize <= 0:
            raise ValueError("chunksize must be positive")
        self.chunksize = chunksize

    def read_recording(self, path: str | Path, sequence_id: str) -> Pamap2Recording:
        """Read one inventoried Protocol recording without guessing participant IDs."""
        path = Path(path)
        if (
            not sequence_id
            or sequence_id in (".", "..")
            or "/" in sequence_id
            or "\\" in sequence_id
        ):
            raise Pamap2DataError("sequence_id must be a nonempty opaque path-safe ID")
        if not path.is_file():
            raise Pamap2DataError(f"recording does not exist: {path}")

        frames: list[np.ndarray] = []
        try:
            with pd.read_csv(path, sep=r"\s+", header=None, chunksize=self.chunksize) as chunks:
                for chunk in chunks:
                    if chunk.shape[1] != PAMAP2_COLUMN_COUNT:
                        raise Pamap2DataError(
                            f"expected {PAMAP2_COLUMN_COUNT} columns, got {chunk.shape[1]}"
                        )
                    numeric = chunk.apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float64)
                    if numeric.ndim != 2:
                        raise Pamap2DataError(
                            "recording rows must be a two-dimensional numeric table"
                        )
                    frames.append(numeric)
        except Pamap2DataError:
            raise
        except (OSError, ValueError, pd.errors.ParserError) as error:
            raise Pamap2DataError(f"cannot parse PAMAP2 recording {path}: {error}") from error
        if not frames:
            raise Pamap2DataError("recording is empty")

        matrix = np.concatenate(frames, axis=0)
        timestamps = matrix[:, 0]
        if not np.isfinite(timestamps).all():
            raise Pamap2DataError("timestamp column must be finite")
        if (np.diff(timestamps) < 0).any():
            raise Pamap2DataError("timestamp column must be nondecreasing")

        raw_labels = matrix[:, 1]
        label_present = np.isfinite(raw_labels)
        rounded_labels = np.zeros(len(raw_labels), dtype=np.int64)
        rounded_labels[label_present] = np.rint(raw_labels[label_present]).astype(np.int64)
        if (label_present & (rounded_labels != raw_labels)).any():
            raise Pamap2DataError("activity labels must be integer IDs")
        unknown = sorted(
            {
                int(label)
                for label in rounded_labels[label_present]
                if label != 0 and label not in PAMAP2_CLASS_MAP
            }
        )
        if unknown:
            raise Pamap2DataError(f"unknown nonzero activity labels: {unknown}")
        target_class = np.zeros(len(raw_labels), dtype=np.int64)
        target_valid = label_present & (rounded_labels != 0)
        for raw_id, class_index in PAMAP2_CLASS_MAP.items():
            target_class[target_valid & (rounded_labels == raw_id)] = class_index

        row_ids = np.arange(len(matrix), dtype=np.int64)
        sources: dict[str, dict[str, np.ndarray]] = {}
        source_audit: dict[str, dict[str, int]] = {}
        for source_id, columns in PAMAP2_SOURCE_COLUMNS.items():
            values = matrix[:, columns]
            valid_mask = np.isfinite(values)
            sources[source_id] = {
                "values": np.where(valid_mask, values, 0.0).astype(np.float32),
                "valid_mask": valid_mask,
                "observation_ids": row_ids.copy(),
                "event_start": timestamps.astype(np.float64, copy=True),
                "event_end": timestamps.astype(np.float64, copy=True),
                "available_at": timestamps.astype(np.float64, copy=True),
            }
            source_audit[source_id] = {
                "feature_count": len(columns),
                "missing_entries": int((~valid_mask).sum()),
                "all_invalid_rows": int((~valid_mask.any(axis=1)).sum()),
            }

        positive_intervals = np.diff(timestamps)
        positive_intervals = positive_intervals[positive_intervals > 0]
        audit = {
            "relative_path": path.name,
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
            "row_count": int(len(matrix)),
            "column_count": int(matrix.shape[1]),
            "timestamp_start": float(timestamps[0]),
            "timestamp_end": float(timestamps[-1]),
            "repeated_timestamp_intervals": int((np.diff(timestamps) == 0).sum()),
            "positive_interval_min": float(positive_intervals.min())
            if len(positive_intervals)
            else None,
            "positive_interval_median": float(np.median(positive_intervals))
            if len(positive_intervals)
            else None,
            "positive_interval_max": float(positive_intervals.max())
            if len(positive_intervals)
            else None,
            "missing_labels": int((~label_present).sum()),
            "excluded_activity_zero": int((label_present & (rounded_labels == 0)).sum()),
            "class_support": {
                str(raw_id): int((label_present & (rounded_labels == raw_id)).sum())
                for raw_id in PAMAP2_CLASS_IDS
            },
            "sources": source_audit,
            "availability_mode": "simulated",
        }
        return Pamap2Recording(
            sequence_id,
            sources,
            timestamps.astype(np.float64),
            raw_labels,
            target_class,
            target_valid,
            audit,
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
