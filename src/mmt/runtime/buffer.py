"""Bounded observation storage with explicit temporal and duplicate semantics."""

from __future__ import annotations

from bisect import insort
from dataclasses import dataclass
from enum import Enum
import math
from typing import Any, Mapping


BYTE_ESTIMATOR_VERSION = "mmt-runtime-bytes-v1"
_RECORD_OVERHEAD_BYTES = 64


class BufferError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


class InsertResult(str, Enum):
    ACCEPTED = "accepted"
    DUPLICATE_IGNORED = "duplicate_ignored"


@dataclass(frozen=True)
class Observation:
    sequence_id: str
    source_id: str
    observation_id: int
    values: tuple[float, ...]
    valid_mask: tuple[bool, ...]
    event_start: float
    event_end: float
    available_at: float

    @classmethod
    def from_record(cls, record: Any) -> Observation:
        def field(name: str) -> Any:
            return record[name] if isinstance(record, Mapping) else getattr(record, name)

        values = tuple(float(value) for value in field("values"))
        valid_mask = tuple(bool(value) for value in field("valid_mask"))
        if len(values) != len(valid_mask):
            raise BufferError("invalid_data", "values and valid_mask dimensions differ")
        times = tuple(float(field(name)) for name in ("event_start", "event_end", "available_at"))
        if not all(math.isfinite(value) for value in times):
            raise BufferError("invalid_data", "observation timestamps must be finite")
        event_start, event_end, available_at = times
        if event_start > event_end or event_end > available_at:
            raise BufferError("invalid_data", "timestamps must satisfy event_start <= event_end <= available_at")
        if any(valid and not math.isfinite(value) for value, valid in zip(values, valid_mask)):
            raise BufferError("invalid_data", "valid feature values must be finite")
        if any(not valid and value != 0.0 for value, valid in zip(values, valid_mask)):
            raise BufferError("invalid_data", "invalid feature entries must be stored as zero")
        sequence_id = str(field("sequence_id"))
        source_id = str(field("source_id"))
        if not sequence_id or not source_id:
            raise BufferError("invalid_data", "sequence_id and source_id must be nonempty")
        return cls(sequence_id, source_id, int(field("observation_id")), values, valid_mask,
                   event_start, event_end, available_at)

    @property
    def estimated_bytes(self) -> int:
        return _RECORD_OVERHEAD_BYTES + 5 * len(self.values) + 32


@dataclass(frozen=True)
class SourceSnapshot:
    observation_ids: tuple[int, ...]
    values: tuple[tuple[float, ...], ...]
    valid_mask: tuple[tuple[bool, ...], ...]
    event_start: tuple[float, ...]
    event_end: tuple[float, ...]
    available_at: tuple[float, ...]


class ObservationBuffer:
    """Single-owner buffer; sequence clocks and capacity are shared across sources."""

    def __init__(self, max_active_sequences: int = 1, max_observations_per_source: int = 100_000,
                 max_buffer_bytes: int = 268_435_456) -> None:
        if max_active_sequences <= 0 or max_observations_per_source <= 0 or max_buffer_bytes <= 0:
            raise ValueError("buffer limits must be positive")
        self.max_active_sequences = max_active_sequences
        self.max_observations_per_source = max_observations_per_source
        self.max_buffer_bytes = max_buffer_bytes
        self._records: dict[tuple[str, str, int], Observation] = {}
        self._indexes: dict[tuple[str, str], list[tuple[float, float, int]]] = {}
        self._last_arrival: dict[str, float] = {}
        self._retention_floor: dict[str, float] = {}
        self._dimensions: dict[str, int] = {}
        self._estimated_bytes = 0
        self._high_water_bytes = 0

    @property
    def estimated_bytes(self) -> int:
        return self._estimated_bytes

    @property
    def high_water_bytes(self) -> int:
        return self._high_water_bytes

    @property
    def row_count(self) -> int:
        return len(self._records)

    def advance_retention_floor(self, sequence_id: str, support_start: float) -> None:
        floor = float(support_start)
        if not math.isfinite(floor):
            raise BufferError("invalid_data", "retention floor must be finite")
        previous = self._retention_floor.get(sequence_id)
        if previous is not None and floor < previous:
            raise BufferError("invalid_decision_order", "retention floor cannot move backward")
        self._retention_floor[sequence_id] = floor
        expired = [key for key, record in self._records.items()
                   if record.sequence_id == sequence_id and record.event_start < floor]
        for key in expired:
            self._remove(key)

    def insert(self, record: Any) -> InsertResult:
        observation = Observation.from_record(record)
        key = (observation.sequence_id, observation.source_id, observation.observation_id)
        existing = self._records.get(key)
        if existing is not None:
            if existing == observation:
                return InsertResult.DUPLICATE_IGNORED
            raise BufferError("duplicate_conflict", "retained observation identity has a different payload")

        floor = self._retention_floor.get(observation.sequence_id)
        if floor is not None and observation.event_start < floor:
            raise BufferError("expired_observation", "observation starts before the retained support floor")
        last_arrival = self._last_arrival.get(observation.sequence_id)
        if last_arrival is not None and observation.available_at < last_arrival:
            raise BufferError("invalid_arrival_order", "arrival timestamps must be nondecreasing per sequence")

        active_sequences = set(self._last_arrival)
        if observation.sequence_id not in active_sequences and len(active_sequences) >= self.max_active_sequences:
            raise BufferError("buffer_overflow", "maximum active sequence count reached")
        source_key = (observation.sequence_id, observation.source_id)
        source_index = self._indexes.get(source_key, [])
        if len(source_index) >= self.max_observations_per_source:
            raise BufferError("buffer_overflow", "maximum observations per source reached")
        expected_dim = self._dimensions.get(observation.source_id)
        if expected_dim is not None and len(observation.values) != expected_dim:
            raise BufferError("invalid_data", "source feature dimension changed")
        if self._estimated_bytes + observation.estimated_bytes > self.max_buffer_bytes:
            raise BufferError("buffer_overflow", "maximum accounted buffer bytes reached")

        self._records[key] = observation
        insort(self._indexes.setdefault(source_key, []),
               (observation.event_end, observation.event_start, observation.observation_id))
        self._dimensions.setdefault(observation.source_id, len(observation.values))
        self._last_arrival[observation.sequence_id] = observation.available_at
        self._estimated_bytes += observation.estimated_bytes
        self._high_water_bytes = max(self._high_water_bytes, self._estimated_bytes)
        return InsertResult.ACCEPTED

    def duplicate_result(self, record: Any) -> InsertResult | None:
        """Check a retained identity before enforcing a sealed-decision boundary."""
        observation = Observation.from_record(record)
        key = (observation.sequence_id, observation.source_id, observation.observation_id)
        existing = self._records.get(key)
        if existing is None:
            return None
        if existing == observation:
            return InsertResult.DUPLICATE_IGNORED
        raise BufferError("duplicate_conflict", "retained observation identity has a different payload")

    def snapshot(self, sequence_id: str, source_id: str) -> SourceSnapshot:
        keys = self._indexes.get((sequence_id, source_id), ())
        records = [self._records[(sequence_id, source_id, observation_id)]
                   for _, _, observation_id in keys]
        return SourceSnapshot(
            observation_ids=tuple(record.observation_id for record in records),
            values=tuple(record.values for record in records),
            valid_mask=tuple(record.valid_mask for record in records),
            event_start=tuple(record.event_start for record in records),
            event_end=tuple(record.event_end for record in records),
            available_at=tuple(record.available_at for record in records),
        )

    def close_sequence(self, sequence_id: str) -> None:
        for key in [key for key, record in self._records.items() if record.sequence_id == sequence_id]:
            self._remove(key)
        self._last_arrival.pop(sequence_id, None)
        self._retention_floor.pop(sequence_id, None)

    def _remove(self, key: tuple[str, str, int]) -> None:
        record = self._records.pop(key)
        source_key = (record.sequence_id, record.source_id)
        index = self._indexes[source_key]
        index.remove((record.event_end, record.event_start, record.observation_id))
        if not index:
            del self._indexes[source_key]
        self._estimated_bytes -= record.estimated_bytes