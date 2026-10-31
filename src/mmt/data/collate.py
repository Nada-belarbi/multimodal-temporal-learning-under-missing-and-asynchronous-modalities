"""Per-source temporal padding and tensor batch construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from mmt.models.source_encoder import SourceBatch

from .dataset import DecisionSample


@dataclass(frozen=True)
class DecisionBatch:
    inputs: dict[str, SourceBatch]
    targets: torch.Tensor
    target_valid: torch.Tensor
    decision_ids: tuple[str, ...]
    metadata: tuple[dict[str, str], ...]


def collate_decisions(samples: Sequence[DecisionSample]) -> DecisionBatch:
    if not samples:
        raise ValueError("cannot collate an empty sample list")
    source_order = tuple(samples[0].sources)
    if not source_order:
        raise ValueError("samples must declare at least one source")
    if any(tuple(sample.sources) != source_order for sample in samples):
        raise ValueError("all samples must have identical source order")

    batch_size = len(samples)
    inputs: dict[str, SourceBatch] = {}
    for source_id in source_order:
        feature_dim = samples[0].sources[source_id].values.shape[-1]
        if feature_dim <= 0:
            raise ValueError(f"source {source_id!r} has no declared features")
        lengths = torch.tensor(
            [sample.sources[source_id].values.shape[0] for sample in samples], dtype=torch.long
        )
        max_steps = max(1, int(lengths.max().item()))
        values = torch.zeros((batch_size, max_steps, feature_dim), dtype=torch.float32)
        feature_mask = torch.zeros((batch_size, max_steps, feature_dim), dtype=torch.bool)
        time_features = torch.zeros((batch_size, max_steps, 3), dtype=torch.float32)
        for index, sample in enumerate(samples):
            source = sample.sources[source_id]
            row_count = source.values.shape[0]
            if source.values.shape != (row_count, feature_dim):
                raise ValueError(f"inconsistent feature dimensions for source {source_id!r}")
            if source.feature_mask.shape != source.values.shape:
                raise ValueError(f"feature mask shape differs for source {source_id!r}")
            if source.time_features.shape != (row_count, 3):
                raise ValueError(f"time feature shape differs for source {source_id!r}")
            if row_count:
                values[index, :row_count] = source.values
                feature_mask[index, :row_count] = source.feature_mask
                time_features[index, :row_count] = source.time_features
        positions = torch.arange(max_steps).unsqueeze(0)
        padding_mask = positions >= lengths.unsqueeze(1)
        inputs[source_id] = SourceBatch(
            values=values,
            feature_mask=feature_mask,
            time_features=time_features,
            lengths=lengths,
            padding_mask=padding_mask,
            present=lengths > 0,
        )
    return DecisionBatch(
        inputs=inputs,
        targets=torch.tensor([sample.target_class for sample in samples], dtype=torch.long),
        target_valid=torch.tensor([sample.target_valid for sample in samples], dtype=torch.bool),
        decision_ids=tuple(sample.decision_id for sample in samples),
        metadata=tuple(dict(sample.metadata) for sample in samples),
    )
