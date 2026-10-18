"""Independent source classifiers combined by equal probability averaging."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import nn

from .source_encoder import ModelOutput, SourceBatch, SourceClassifier


class V0LateFusion(nn.Module):
    def __init__(self, source_dimensions: Mapping[str, int], n_classes: int,
                 hidden_size: int = 64, dropout: float = 0.1,
                 epsilon: float = 1e-8) -> None:
        super().__init__()
        if not source_dimensions:
            raise ValueError("at least one source is required")
        if not 0.0 < epsilon < 1.0:
            raise ValueError("epsilon must be between zero and one")
        self.source_order: tuple[str, ...] = tuple(source_dimensions)
        self.n_classes = n_classes
        self.epsilon = epsilon
        self.classifiers = nn.ModuleList(
            SourceClassifier(dimension, n_classes, hidden_size, dropout)
            for dimension in source_dimensions.values()
        )

    def forward(self, batches: Mapping[str, SourceBatch],
                diagnostics: bool = False) -> ModelOutput:
        if set(batches) != set(self.source_order):
            raise ValueError("input sources must exactly match the configured source order")
        first_batch = batches[self.source_order[0]]
        batch_size = first_batch.values.shape[0]
        probability_sum = first_batch.values.new_zeros((batch_size, self.n_classes))
        usable_count = first_batch.values.new_zeros((batch_size, 1))
        present_parts: list[torch.Tensor] = []

        for source_id, classifier in zip(self.source_order, self.classifiers):
            output = classifier(batches[source_id])
            if output.scores.shape != (batch_size, self.n_classes):
                raise ValueError("source batch sizes and classifier outputs must agree")
            present_parts.append(output.computable)
            active = torch.nonzero(output.computable, as_tuple=False).flatten()
            if active.numel() == 0:
                continue
            source_probabilities = torch.softmax(output.scores.index_select(0, active), dim=-1)
            probability_sum = probability_sum.index_add(0, active, source_probabilities)
            usable_count = usable_count.index_add(
                0, active, usable_count.new_ones((active.numel(), 1))
            )

        computable = usable_count.squeeze(-1) > 0
        denominator = usable_count.clamp_min(1.0)
        mean_probability = probability_sum / denominator
        scores = mean_probability.clamp_min(self.epsilon).log()
        scores = scores.masked_fill(~computable.unsqueeze(-1), 0.0)
        details = None
        if diagnostics:
            details = {"source_present": torch.stack(present_parts, dim=1)}
        return ModelOutput(scores=scores, computable=computable,
                           score_kind="log_probabilities", diagnostics=details)