"""Stateless prediction and stateful inference over the shared temporal selector."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
import math
from typing import Any

from mmt.data.selection import SelectionResult, select_observations
from .buffer import BufferError, InsertResult, ObservationBuffer


def _field(record: Any, name: str) -> Any:
    return record[name] if isinstance(record, Mapping) else getattr(record, name)


def _finite_time(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise BufferError("invalid_data", f"{name} must be finite")
    return result


def _python(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    return value


def _as_bool(value: Any) -> bool:
    value = _python(value)
    if isinstance(value, list):
        if len(value) != 1:
            raise ValueError("expected a single computable flag")
        value = value[0]
    return bool(value)


def _as_scores(value: Any) -> list[float]:
    values = _python(value)
    if values and isinstance(values[0], list):
        if len(values) != 1:
            raise ValueError("expected scores for one decision")
        values = values[0]
    return [float(score) for score in values]


def _softmax(scores: Sequence[float], temperature: float) -> list[float]:
    scaled = [score / temperature for score in scores]
    maximum = max(scaled)
    exponents = [math.exp(score - maximum) for score in scaled]
    denominator = sum(exponents)
    return [value / denominator for value in exponents]


@dataclass(frozen=True)
class PredictionRecord:
    sequence_id: str
    decision_id: str
    decision_time: float
    support_start: float
    support_end: float
    computable: bool
    score_kind: str | None
    scores: tuple[float, ...] | None
    probabilities: tuple[float, ...] | None
    predicted_class: int | None
    confidence: float | None
    accepted: bool
    status: str
    reason: str | None
    source_evidence: Mapping[str, Mapping[str, Any]]
    selected_ids: Mapping[str, tuple[int, ...]]
    policy_applicable: bool
    session_id: str | None = None
    generation: int = 0
    run_id: str | None = None
    scenario_id: str | None = None
    artifact_hashes: Mapping[str, str] | None = None


BatchBuilder = Callable[[Mapping[str, SelectionResult]], Any]
PredictFunction = Callable[[Any], Any]


class InferenceEngine:
    """Callbacks adapt Member 1 batches and Member 2 models without duplicating either."""

    def __init__(self, source_ids: Sequence[str], batch_builder: BatchBuilder,
                 predict_fn: PredictFunction, n_classes: int, max_source_age: float | None = None,
                 temperature: float = 1.0, threshold: float | None = None,
                 max_active_sequences: int = 1, max_observations_per_source: int = 100_000,
                 max_buffer_bytes: int = 268_435_456, session_id: str | None = None,
                 run_id: str | None = None, scenario_id: str | None = None,
                 artifact_hashes: Mapping[str, str] | None = None) -> None:
        if not source_ids or len(set(source_ids)) != len(source_ids):
            raise ValueError("source_ids must be nonempty and unique")
        if n_classes <= 1:
            raise ValueError("n_classes must be greater than one")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        if threshold is not None and (not math.isfinite(threshold) or not 0 <= threshold <= 1):
            raise ValueError("threshold must be null or between zero and one")
        self.source_ids = tuple(source_ids)
        self.batch_builder = batch_builder
        self.predict_fn = predict_fn
        self.n_classes = n_classes
        self.max_source_age = max_source_age
        self.temperature = temperature
        self.threshold = threshold
        self.session_id = session_id
        self.run_id = run_id
        self.scenario_id = scenario_id
        self.artifact_hashes = dict(artifact_hashes or {})
        self.buffer = ObservationBuffer(max_active_sequences, max_observations_per_source, max_buffer_bytes)
        self._last_decision: dict[str, tuple[float, float]] = {}
        self._generation: dict[str, int] = {}

    def update(self, observation: Any) -> InsertResult:
        sequence_id = str(_field(observation, "sequence_id"))
        available_at = float(_field(observation, "available_at"))
        sealed = self._last_decision.get(sequence_id)
        duplicate = self.buffer.duplicate_result(observation)
        if duplicate is not None:
            return duplicate
        if sealed is not None and available_at <= sealed[0]:
            raise BufferError("invalid_arrival_order", "a new arrival cannot alter a sealed decision time")
        return self.buffer.insert(observation)

    def predict(self, decision: Any) -> PredictionRecord:
        sequence_id = str(_field(decision, "sequence_id"))
        decision_time = _finite_time(_field(decision, "decision_time"), "decision_time")
        support_start = _finite_time(_field(decision, "support_start"), "support_start")
        support_end = _finite_time(_field(decision, "support_end"), "support_end")
        decision_id = str(_field(decision, "decision_id"))
        if support_start > support_end or support_end > decision_time:
            raise BufferError("invalid_data", "decision support must satisfy start <= end <= decision_time")
        previous = self._last_decision.get(sequence_id)
        if previous is not None and decision_time <= previous[0]:
            raise BufferError("invalid_decision_order", "stateful decision times must increase strictly")
        if previous is not None and support_start < previous[1]:
            raise BufferError("invalid_decision_order", "stateful support start cannot move backward")
        self.buffer.advance_retention_floor(sequence_id, support_start)
        selections = self._select(sequence_id, decision)
        self._last_decision[sequence_id] = (decision_time, support_start)
        return self._decorate(sequence_id, self._predict_selected(decision, selections))

    def predict_batch(self, decisions: Sequence[Any],
                      sources: Mapping[tuple[str, str], Any]) -> list[PredictionRecord]:
        """Predict independently from complete source sequences; this method is stateless."""
        predictions = []
        for decision in decisions:
            sequence_id = str(_field(decision, "sequence_id"))
            record = self._predict_selected(decision, self._select_from_sources(decision, sources))
            predictions.append(self._decorate(sequence_id, record))
        return predictions

    def reset(self, sequence_id: str) -> int:
        self.buffer.close_sequence(sequence_id)
        self._last_decision.pop(sequence_id, None)
        generation = self._generation.get(sequence_id, 0) + 1
        self._generation[sequence_id] = generation
        return generation

    def close(self, sequence_id: str) -> None:
        self.buffer.close_sequence(sequence_id)
        self._last_decision.pop(sequence_id, None)

    def _select(self, sequence_id: str, decision: Any) -> dict[str, SelectionResult]:
        return {
            source_id: select_observations(
                self.buffer.snapshot(sequence_id, source_id), decision, self.max_source_age
            )
            for source_id in self.source_ids
        }

    def _select_from_sources(self, decision: Any,
                             sources: Mapping[tuple[str, str], Any]) -> dict[str, SelectionResult]:
        sequence_id = str(_field(decision, "sequence_id"))
        return {
            source_id: select_observations(
                sources.get((sequence_id, source_id), _empty_source()), decision, self.max_source_age
            )
            for source_id in self.source_ids
        }

    def _decorate(self, sequence_id: str, record: PredictionRecord) -> PredictionRecord:
        return replace(
            record,
            session_id=self.session_id,
            generation=self._generation.get(sequence_id, 0),
            run_id=self.run_id,
            scenario_id=self.scenario_id,
            artifact_hashes=self.artifact_hashes,
        )

    def _predict_selected(self, decision: Any,
                          selections: Mapping[str, SelectionResult]) -> PredictionRecord:
        sequence_id = str(_field(decision, "sequence_id"))
        decision_id = str(_field(decision, "decision_id"))
        decision_time = float(_field(decision, "decision_time"))
        support_start = float(_field(decision, "support_start"))
        support_end = float(_field(decision, "support_end"))
        evidence = {
            source_id: {
                "selected_count": result.selected_count,
                "latest_age": result.latest_age,
                "valid_feature_fraction": result.valid_feature_fraction,
                "source_status": result.source_status,
            }
            for source_id, result in selections.items()
        }
        selected_ids = {source_id: result.observation_ids for source_id, result in selections.items()}
        if not any(result.selected_count for result in selections.values()):
            return PredictionRecord(
                sequence_id, decision_id, decision_time, support_start, support_end,
                False, None, None, None, None, None, False, "unavailable_no_data",
                "all_sources_unusable", evidence, selected_ids, False,
            )

        try:
            output = self.predict_fn(self.batch_builder(selections))
        except Exception as error:
            raise RuntimeError(
                f"model_execution_failed: decision {decision_id}: {type(error).__name__}: {error}"
            ) from error
        computable = _as_bool(_get_output(output, "computable"))
        if not computable:
            return PredictionRecord(
                sequence_id, decision_id, decision_time, support_start, support_end,
                False, None, None, None, None, None, False, "unavailable_no_data",
                "model_input_unusable", evidence, selected_ids, False,
            )
        score_kind = str(_get_output(output, "score_kind"))
        if score_kind == "label_only":
            predicted_class = int(_get_output(output, "predicted_class"))
            if not 0 <= predicted_class < self.n_classes:
                raise RuntimeError("model_execution_failed: predicted class is outside the class map")
            return PredictionRecord(
                sequence_id, decision_id, decision_time, support_start, support_end,
                True, score_kind, None, None, predicted_class, None, True, "accepted",
                None, evidence, selected_ids, False,
            )
        if score_kind not in ("logits", "log_probabilities"):
            raise RuntimeError(f"model_execution_failed: unsupported score_kind {score_kind!r}")
        scores = _as_scores(_get_output(output, "scores"))
        if len(scores) != self.n_classes or not all(math.isfinite(score) for score in scores):
            raise RuntimeError("model_execution_failed: invalid computable model scores")
        probabilities = _softmax(scores, self.temperature)
        predicted_class = max(range(self.n_classes), key=probabilities.__getitem__)
        confidence = probabilities[predicted_class]
        accepted = self.threshold is None or confidence >= self.threshold
        status = "accepted" if accepted else "abstained_low_confidence"
        return PredictionRecord(
            sequence_id, decision_id, decision_time, support_start, support_end,
            True, score_kind, tuple(scores), tuple(probabilities), predicted_class,
            confidence, accepted, status, None, evidence, selected_ids, True,
        )


def _get_output(output: Any, name: str) -> Any:
    return output[name] if isinstance(output, Mapping) else getattr(output, name)


def _empty_source() -> Mapping[str, tuple[Any, ...]]:
    return {
        "observation_ids": (), "values": (), "valid_mask": (),
        "event_start": (), "event_end": (), "available_at": (),
    }