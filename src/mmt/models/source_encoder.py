"""Shared source encoder and output contracts for temporal classifiers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


@dataclass(frozen=True)
class SourceBatch:
    values: torch.Tensor
    feature_mask: torch.Tensor
    time_features: torch.Tensor
    lengths: torch.Tensor
    padding_mask: torch.Tensor
    present: torch.Tensor

    def validate(self, feature_dim: int) -> None:
        if self.values.ndim != 3:
            raise ValueError("values must have shape [batch, time, features]")
        batch_size, time_steps, actual_dim = self.values.shape
        if actual_dim != feature_dim:
            raise ValueError(f"feature dimension {actual_dim} != configured {feature_dim}")
        if self.feature_mask.shape != self.values.shape or self.feature_mask.dtype != torch.bool:
            raise ValueError("feature_mask must be boolean and match values")
        if self.time_features.shape != (batch_size, time_steps, 3):
            raise ValueError("time_features must have shape [batch, time, 3]")
        if self.lengths.shape != (batch_size,) or self.lengths.dtype != torch.long:
            raise ValueError("lengths must be int64 with shape [batch]")
        if self.present.shape != (batch_size,) or self.present.dtype != torch.bool:
            raise ValueError("present must be boolean with shape [batch]")
        if self.padding_mask.shape != (batch_size, time_steps) or self.padding_mask.dtype != torch.bool:
            raise ValueError("padding_mask must be boolean with shape [batch, time]")
        if bool(((self.lengths < 0) | (self.lengths > time_steps)).any()):
            raise ValueError("lengths must be between zero and the padded time dimension")
        expected_present = self.lengths > 0
        if not torch.equal(self.present, expected_present):
            raise ValueError("present must equal lengths > 0")
        positions = torch.arange(time_steps, device=self.lengths.device).unsqueeze(0)
        expected_padding = positions >= self.lengths.unsqueeze(1)
        if not torch.equal(self.padding_mask, expected_padding):
            raise ValueError("padding_mask must mark positions at or beyond each length")
        if bool((~self.feature_mask.any(dim=-1) & ~self.padding_mask).any()):
            raise ValueError("all-invalid rows must be removed before batching")
        valid_values = self.values.masked_select(self.feature_mask)
        if not bool(torch.isfinite(valid_values).all()):
            raise ValueError("valid feature values must be finite")
        valid_times = self.time_features.masked_select(~self.padding_mask.unsqueeze(-1))
        if not bool(torch.isfinite(valid_times).all()):
            raise ValueError("non-padding time features must be finite")


@dataclass(frozen=True)
class EncoderOutput:
    states: torch.Tensor
    last_state: torch.Tensor
    present: torch.Tensor


@dataclass(frozen=True)
class ModelOutput:
    scores: torch.Tensor
    computable: torch.Tensor
    score_kind: str
    diagnostics: Mapping[str, torch.Tensor] | None = None


class SourceEncoder(nn.Module):
    def __init__(self, feature_dim: int, hidden_size: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        if feature_dim <= 0 or hidden_size <= 0:
            raise ValueError("feature_dim and hidden_size must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.feature_dim = feature_dim
        self.hidden_size = hidden_size
        self.projection = nn.Linear(2 * feature_dim + 3, hidden_size)
        self.input_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=1, batch_first=True,
                           bidirectional=False, dropout=0.0)

    def forward(self, batch: SourceBatch) -> EncoderOutput:
        batch.validate(self.feature_dim)
        clean_values = batch.values.masked_fill(~batch.feature_mask, 0.0)
        encoder_input = torch.cat(
            (clean_values, batch.feature_mask.to(batch.values.dtype), batch.time_features), dim=-1
        )
        projected = self.input_dropout(torch.relu(self.projection(encoder_input)))

        batch_size, time_steps, _ = projected.shape
        states = projected.new_zeros((batch_size, time_steps, self.hidden_size))
        last_state = projected.new_zeros((batch_size, self.hidden_size))
        active = torch.nonzero(batch.lengths > 0, as_tuple=False).flatten()
        if active.numel() == 0:
            return EncoderOutput(states, last_state, batch.present)

        active_lengths = batch.lengths.index_select(0, active)
        packed = pack_padded_sequence(
            projected.index_select(0, active), active_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_states, hidden = self.gru(packed)
        active_states, _ = pad_packed_sequence(
            packed_states, batch_first=True, total_length=time_steps
        )
        states = states.index_copy(0, active, active_states)
        last_state = last_state.index_copy(0, active, hidden[-1])
        return EncoderOutput(states, last_state, batch.present)


class SourceClassifier(nn.Module):
    def __init__(self, feature_dim: int, n_classes: int, hidden_size: int = 64,
                 dropout: float = 0.1) -> None:
        super().__init__()
        if n_classes <= 1:
            raise ValueError("n_classes must be greater than one")
        self.encoder = SourceEncoder(feature_dim, hidden_size, dropout)
        self.classifier = nn.Linear(hidden_size, n_classes)
        self.n_classes = n_classes

    def forward(self, batch: SourceBatch) -> ModelOutput:
        encoded = self.encoder(batch)
        scores = self.classifier(encoded.last_state)
        scores = scores.masked_fill(~encoded.present.unsqueeze(-1), 0.0)
        return ModelOutput(scores=scores, computable=encoded.present, score_kind="logits")