"""Deterministic headless replay over pre-sorted arrival and decision iterators."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from typing import Any

from .buffer import BufferError
from .engine import InferenceEngine, PredictionRecord


def replay(engine: InferenceEngine, arrivals: Iterable[Any],
           decisions: Iterable[Any]) -> Iterator[PredictionRecord]:
    """Merge one sequence with one-item arrival lookahead; both inputs must be sorted."""
    arrival_iter = iter(arrivals)
    next_arrival = next(arrival_iter, None)
    prior_arrival_key: tuple[float, str, int] | None = None
    active_sequence: str | None = None
    prior_decision_time: float | None = None

    for decision in decisions:
        sequence_id = str(_field(decision, "sequence_id"))
        decision_time = float(_field(decision, "decision_time"))
        if active_sequence is None:
            active_sequence = sequence_id
        elif sequence_id != active_sequence:
            raise BufferError("unsupported_profile", "replay handles one sequence per invocation")
        if prior_decision_time is not None and decision_time <= prior_decision_time:
            raise BufferError("invalid_decision_order", "replay decisions must be strictly chronological")
        prior_decision_time = decision_time

        while next_arrival is not None and float(_field(next_arrival, "available_at")) <= decision_time:
            if str(_field(next_arrival, "sequence_id")) != active_sequence:
                raise BufferError("unsupported_profile", "replay arrivals must belong to the decision sequence")
            key = (float(_field(next_arrival, "available_at")),
                   str(_field(next_arrival, "source_id")), int(_field(next_arrival, "observation_id")))
            if prior_arrival_key is not None and key < prior_arrival_key:
                raise BufferError("invalid_arrival_order", "replay arrivals must be sorted by availability and identity")
            prior_arrival_key = key
            engine.update(next_arrival)
            next_arrival = next(arrival_iter, None)
        yield engine.predict(decision)


def _field(record: Any, name: str) -> Any:
    return record[name] if isinstance(record, Mapping) else getattr(record, name)