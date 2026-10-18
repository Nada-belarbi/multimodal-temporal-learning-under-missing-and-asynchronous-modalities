"""Required label-only majority and common-grid GRU references."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from .source_encoder import ModelOutput


@dataclass(frozen=True)
class MajorityOutput:
    predicted_class: int | None
    computable: bool
    score_kind: str = "label_only"


class MajorityReference:
    def __init__(self, n_classes: int) -> None:
        if n_classes <= 1:
            raise ValueError("n_classes must be greater than one")
        self.n_classes = n_classes
        self.class_counts: tuple[int, ...] | None = None
        self.selected_class: int | None = None

    def fit(self, targets: Iterable[int], target_valid: Iterable[bool]) -> MajorityReference:
        counts = [0] * self.n_classes
        for target, valid in zip(targets, target_valid):
            if not valid:
                continue
            target = int(target)
            if not 0 <= target < self.n_classes:
                raise ValueError("training target outside the class map")
            counts[target] += 1
        if not any(counts):
            raise ValueError("majority reference needs at least one eligible training target")
        self.class_counts = tuple(counts)
        self.selected_class = max(range(self.n_classes), key=counts.__getitem__)
        return self

    def predict(self, input_available: bool) -> MajorityOutput:
        if self.selected_class is None:
            raise RuntimeError("majority reference must be fitted before prediction")
        return MajorityOutput(self.selected_class if input_available else None,
                              bool(input_available))


@dataclass(frozen=True)
class GridBatch:
    values: torch.Tensor
    feature_mask: torch.Tensor
    ages: torch.Tensor
    grid_times: torch.Tensor
    decision_times: torch.Tensor
    lengths: torch.Tensor
    padding_mask: torch.Tensor


class BGridClassifier(nn.Module):
    def __init__(self, feature_count: int, n_classes: int, hidden_size: int = 64,
                 dropout: float = 0.1) -> None:
        super().__init__()
        if feature_count <= 0 or n_classes <= 1:
            raise ValueError("feature_count and n_classes must be valid")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.feature_count = feature_count
        self.projection = nn.Linear(3 * feature_count + 1, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=1, batch_first=True,
                           bidirectional=False, dropout=0.0)
        self.classifier = nn.Linear(hidden_size, n_classes)
        self.n_classes = n_classes

    def forward(self, batch: GridBatch) -> ModelOutput:
        if batch.values.ndim != 3 or batch.values.shape[-1] != self.feature_count:
            raise ValueError("grid values must have shape [batch, time, feature_count]")
        batch_size, time_steps, _ = batch.values.shape
        if batch.feature_mask.shape != batch.values.shape or batch.ages.shape != batch.values.shape:
            raise ValueError("grid masks and ages must match values")
        if batch.grid_times.shape != (batch_size, time_steps):
            raise ValueError("grid_times must have shape [batch, time]")
        if batch.decision_times.shape != (batch_size,) or batch.lengths.shape != (batch_size,):
            raise ValueError("decision_times and lengths must have shape [batch]")
        if batch.padding_mask.shape != (batch_size, time_steps):
            raise ValueError("padding_mask must have shape [batch, time]")
        if bool(((batch.lengths <= 0) | (batch.lengths > time_steps)).any()):
            raise ValueError("every common grid must contain at least one point")
        expected_padding = torch.arange(time_steps, device=batch.lengths.device).unsqueeze(0) >= batch.lengths.unsqueeze(1)
        if not torch.equal(batch.padding_mask, expected_padding):
            raise ValueError("padding_mask must agree with lengths")
        if bool((batch.ages < 0).any()):
            raise ValueError("grid ages must be nonnegative")
        valid_values = batch.values.masked_select(batch.feature_mask)
        if not bool(torch.isfinite(valid_values).all()):
            raise ValueError("valid grid values must be finite")

        clock_age = batch.decision_times.unsqueeze(1) - batch.grid_times
        if bool((clock_age.masked_select(~batch.padding_mask) < 0).any()):
            raise ValueError("grid times cannot be after their decision time")
        clock_feature = torch.log1p(clock_age.clamp_min(0)).unsqueeze(-1)
        encoded_input = torch.cat((
            batch.values.masked_fill(~batch.feature_mask, 0.0),
            batch.feature_mask.to(batch.values.dtype),
            torch.log1p(batch.ages), clock_feature,
        ), dim=-1)
        projected = self.dropout(torch.relu(self.projection(encoded_input)))
        computable = batch.feature_mask.any(dim=(1, 2))
        scores = projected.new_zeros((batch_size, self.n_classes))
        active = torch.nonzero(computable, as_tuple=False).flatten()
        if active.numel():
            active_lengths = batch.lengths.index_select(0, active)
            packed = pack_padded_sequence(
                projected.index_select(0, active), active_lengths.cpu(), batch_first=True,
                enforce_sorted=False,
            )
            _, hidden = self.gru(packed)
            active_scores = self.classifier(hidden[-1])
            scores = scores.index_copy(0, active, active_scores)
        return ModelOutput(scores=scores, computable=computable, score_kind="logits")