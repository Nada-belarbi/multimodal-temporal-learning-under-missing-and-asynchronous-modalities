"""Stable model-name registry with schema-driven construction."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from torch import nn

from .baselines import BGridClassifier, MajorityReference
from .fusion import V1GatedFusion, V2AttentionFusion
from .source_encoder import SourceClassifier
from .v0_late_fusion import V0LateFusion


MODEL_NAMES = ("majority", "unimodal", "b_grid", "v0", "v1", "v2")


def _class_count(class_map: Mapping[Any, int]) -> int:
    indices = sorted(int(index) for index in class_map.values())
    if not indices or indices != list(range(len(indices))):
        raise ValueError("class map indices must be contiguous from zero")
    return len(indices)


def build_model(model_name: str, source_dimensions: Mapping[str, int],
                class_map: Mapping[Any, int], config: Mapping[str, Any] | None = None) -> nn.Module:
    """Build a torch model from the audited source dimensions and fixed class order."""
    if model_name not in MODEL_NAMES or model_name == "majority":
        raise ValueError(f"model name must be one of {MODEL_NAMES[1:]}")
    if not source_dimensions or any(not source_id or dimension <= 0
                                   for source_id, dimension in source_dimensions.items()):
        raise ValueError("source dimensions must contain named sources with positive dimensions")
    options = dict(config or {})
    n_classes = _class_count(class_map)
    hidden_size = int(options.pop("hidden_size", 64))
    dropout = float(options.pop("dropout", 0.1))
    if hidden_size <= 0 or not 0.0 <= dropout < 1.0:
        raise ValueError("hidden_size must be positive and dropout must be in [0, 1)")

    if model_name == "unimodal":
        source_id = options.pop("source_id", None)
        if source_id not in source_dimensions:
            raise ValueError("unimodal models require a configured source_id")
        if options:
            raise ValueError(f"unknown unimodal options: {sorted(options)}")
        return SourceClassifier(source_dimensions[source_id], n_classes, hidden_size, dropout)
    if model_name == "v0":
        epsilon = float(options.pop("epsilon", 1e-8))
        if options:
            raise ValueError(f"unknown V0 options: {sorted(options)}")
        return V0LateFusion(source_dimensions, n_classes, hidden_size, dropout, epsilon)
    if model_name == "v1":
        if options:
            raise ValueError(f"unknown V1 options: {sorted(options)}")
        return V1GatedFusion(source_dimensions, n_classes, hidden_size, dropout)
    if model_name == "v2":
        n_heads = int(options.pop("n_heads", 4))
        feedforward_size = int(options.pop("feedforward_size", 128))
        if options:
            raise ValueError(f"unknown V2 options: {sorted(options)}")
        return V2AttentionFusion(source_dimensions, n_classes, hidden_size, dropout,
                                 n_heads, feedforward_size)
    if model_name == "b_grid":
        feature_count = sum(source_dimensions.values())
        if options:
            raise ValueError(f"unknown B-GRID options: {sorted(options)}")
        return BGridClassifier(feature_count, n_classes, hidden_size, dropout)
    raise AssertionError("model registry branch is incomplete")


def build_majority_reference(class_map: Mapping[Any, int]) -> MajorityReference:
    return MajorityReference(_class_count(class_map))