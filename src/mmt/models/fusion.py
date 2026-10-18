"""Learned gated and hierarchical-attention multimodal fusion models."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn

from .source_encoder import EncoderOutput, ModelOutput, SourceBatch, SourceEncoder


class _FusionBase(nn.Module):
    def __init__(self, source_dimensions: Mapping[str, int], n_classes: int,
                 hidden_size: int, dropout: float) -> None:
        super().__init__()
        if not source_dimensions:
            raise ValueError("at least one source is required")
        if n_classes <= 1:
            raise ValueError("n_classes must be greater than one")
        self.source_order = tuple(source_dimensions)
        self.hidden_size = hidden_size
        self.encoders = nn.ModuleList(
            SourceEncoder(dimension, hidden_size, dropout)
            for dimension in source_dimensions.values()
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, n_classes)
        )
        self.n_classes = n_classes

    def _encode(self, batches: Mapping[str, SourceBatch]) -> tuple[list[EncoderOutput], int]:
        if set(batches) != set(self.source_order):
            raise ValueError("input sources must exactly match the configured source order")
        outputs = [encoder(batches[source_id])
                   for source_id, encoder in zip(self.source_order, self.encoders)]
        batch_sizes = {output.last_state.shape[0] for output in outputs}
        if len(batch_sizes) != 1:
            raise ValueError("all source batches must have the same batch size")
        return outputs, batch_sizes.pop()


class V1GatedFusion(_FusionBase):
    def __init__(self, source_dimensions: Mapping[str, int], n_classes: int,
                 hidden_size: int = 64, dropout: float = 0.1) -> None:
        super().__init__(source_dimensions, n_classes, hidden_size, dropout)
        self.gates = nn.ModuleList(
            nn.Sequential(nn.Linear(hidden_size + 2, 32), nn.ReLU(), nn.Linear(32, 1))
            for _ in self.source_order
        )

    def forward(self, batches: Mapping[str, SourceBatch],
                diagnostics: bool = False) -> ModelOutput:
        encoded, batch_size = self._encode(batches)
        gate_parts = []
        for source_id, output, gate in zip(self.source_order, encoded, self.gates):
            batch = batches[source_id]
            last_index = (batch.lengths - 1).clamp_min(0)
            row_index = torch.arange(batch_size, device=batch.lengths.device)
            latest_age = batch.time_features[row_index, last_index, 1].unsqueeze(-1)
            valid_entries = batch.feature_mask.sum(dim=(1, 2)).to(output.last_state.dtype)
            valid_denominator = (batch.lengths.clamp_min(1) * batch.feature_mask.shape[-1]).to(
                output.last_state.dtype
            )
            valid_fraction = (valid_entries / valid_denominator).unsqueeze(-1)
            gate_parts.append(gate(torch.cat((output.last_state, latest_age, valid_fraction), dim=-1)))

        present = torch.stack([output.present for output in encoded], dim=1)
        gate_scores = torch.cat(gate_parts, dim=1)
        fused = encoded[0].last_state.new_zeros((batch_size, self.hidden_size))
        alpha = gate_scores.new_zeros(gate_scores.shape)
        active_rows = torch.nonzero(present.any(dim=1), as_tuple=False).flatten()
        if active_rows.numel():
            active_scores = gate_scores.index_select(0, active_rows)
            active_present = present.index_select(0, active_rows)
            active_alpha = torch.softmax(active_scores.masked_fill(~active_present, -torch.inf), dim=1)
            alpha = alpha.index_copy(0, active_rows, active_alpha)
            states = torch.stack([output.last_state for output in encoded], dim=1)
            active_states = states.index_select(0, active_rows)
            active_fused = (active_states * active_alpha.unsqueeze(-1)).sum(dim=1)
            fused = fused.index_copy(0, active_rows, active_fused)

        scores = self.classifier(fused).masked_fill(~present.any(dim=1, keepdim=True), 0.0)
        details = {"source_weights": alpha, "source_present": present} if diagnostics else None
        return ModelOutput(scores=scores, computable=present.any(dim=1),
                           score_kind="logits", diagnostics=details)


class V2AttentionFusion(_FusionBase):
    def __init__(self, source_dimensions: Mapping[str, int], n_classes: int,
                 hidden_size: int = 64, dropout: float = 0.1, n_heads: int = 4,
                 feedforward_size: int = 128) -> None:
        super().__init__(source_dimensions, n_classes, hidden_size, dropout)
        if hidden_size % n_heads:
            raise ValueError("hidden_size must be divisible by n_heads")
        if feedforward_size <= 0:
            raise ValueError("feedforward_size must be positive")
        self.queries = nn.ParameterList(
            nn.Parameter(torch.empty(1, 1, hidden_size)) for _ in self.source_order
        )
        for query in self.queries:
            nn.init.normal_(query, mean=0.0, std=0.02)
        self.temporal_attention = nn.ModuleList(
            nn.MultiheadAttention(hidden_size, n_heads, dropout=dropout, batch_first=True)
            for _ in self.source_order
        )
        self.source_embeddings = nn.Parameter(torch.empty(len(self.source_order), hidden_size))
        nn.init.normal_(self.source_embeddings, mean=0.0, std=0.02)
        self.source_attention = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=n_heads, dim_feedforward=feedforward_size,
            dropout=dropout, activation="relu", batch_first=True, norm_first=False,
            layer_norm_eps=1e-5, bias=True,
        )

    def forward(self, batches: Mapping[str, SourceBatch],
                diagnostics: bool = False) -> ModelOutput:
        encoded, batch_size = self._encode(batches)
        present = torch.stack([output.present for output in encoded], dim=1)
        summaries = encoded[0].last_state.new_zeros(
            (batch_size, len(self.source_order), self.hidden_size)
        )
        temporal_weights: dict[str, torch.Tensor] = {}

        for source_index, (source_id, output, query, attention) in enumerate(
            zip(self.source_order, encoded, self.queries, self.temporal_attention)
        ):
            active_rows = torch.nonzero(output.present, as_tuple=False).flatten()
            if active_rows.numel() == 0:
                continue
            states = output.states.index_select(0, active_rows)
            padding = batches[source_id].padding_mask.index_select(0, active_rows)
            query_rows = query.expand(active_rows.numel(), -1, -1)
            attended, weights = attention(
                query_rows, states, states, key_padding_mask=padding,
                need_weights=diagnostics, average_attn_weights=False,
            )
            source_summaries = attended[:, 0, :] + self.source_embeddings[source_index]
            summaries = summaries.index_copy(0, active_rows,
                                             summaries.index_select(0, active_rows).index_copy(
                                                 1, torch.tensor([source_index], device=active_rows.device),
                                                 source_summaries.unsqueeze(1)))
            if diagnostics and weights is not None:
                temporal_weights[source_id] = weights

        active_rows = torch.nonzero(present.any(dim=1), as_tuple=False).flatten()
        transformed = summaries.new_zeros(summaries.shape)
        if active_rows.numel():
            active_summaries = summaries.index_select(0, active_rows)
            active_present = present.index_select(0, active_rows)
            active_transformed = self.source_attention(
                active_summaries, src_key_padding_mask=~active_present
            )
            active_transformed = active_transformed.masked_fill(
                ~active_present.unsqueeze(-1), 0.0
            )
            transformed = transformed.index_copy(0, active_rows, active_transformed)
        source_count = present.sum(dim=1, keepdim=True).clamp_min(1).to(transformed.dtype)
        fused = transformed.sum(dim=1) / source_count
        scores = self.classifier(fused).masked_fill(~present.any(dim=1, keepdim=True), 0.0)
        details = {"source_present": present, "temporal_attention": temporal_weights} if diagnostics else None
        return ModelOutput(scores=scores, computable=present.any(dim=1),
                           score_kind="logits", diagnostics=details)