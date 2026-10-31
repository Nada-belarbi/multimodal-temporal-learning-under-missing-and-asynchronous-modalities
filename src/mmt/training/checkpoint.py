"""Atomic state-dict checkpoint persistence."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from mmt.models.registry import build_model


def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=destination.name + ".tmp-", dir=destination.parent
    )
    os.close(descriptor)
    try:
        torch.save(payload, temporary_name)
        os.replace(temporary_name, destination)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def load_state_dict(
    model: nn.Module, path: str | Path, device: str | torch.device = "cpu"
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("state_dict"), dict):
        raise ValueError("checkpoint must contain a state_dict")
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(device)
    model.eval()
    return payload


def load_fusion_model(
    metadata: Mapping[str, Any],
    class_map: Mapping[str, int],
    checkpoint_path: str | Path,
    *,
    expected_model_name: str,
    device: str | torch.device = "cpu",
) -> nn.Module:
    """Reconstruct the exact saved fusion architecture and load its weights strictly."""
    name = metadata.get("model_name")
    if name not in {"v0", "v1", "v2"} or name != expected_model_name:
        raise ValueError("saved model architecture differs from the requested model")
    if metadata.get("class_map") != dict(class_map):
        raise ValueError("saved model class map differs from class_map.json")
    score_kind = "log_probabilities" if name == "v0" else "logits"
    if metadata.get("score_kind", score_kind) != score_kind:
        raise ValueError("saved score_kind is incompatible with the model architecture")
    order = tuple(metadata["source_order"])
    dimensions = metadata["source_dimensions"]
    if not order or len(order) != len(set(order)) or set(order) != set(dimensions):
        raise ValueError("saved source order and dimensions are incompatible")
    options = {"hidden_size": metadata["hidden_size"], "dropout": metadata["dropout"]}
    if name == "v0":
        options["epsilon"] = metadata["epsilon"]
    elif name == "v2":
        options["n_heads"] = metadata["n_heads"]
        options["feedforward_size"] = metadata["feedforward_size"]
    model = build_model(
        name, {source: int(dimensions[source]) for source in order}, class_map, options
    )
    load_state_dict(model, checkpoint_path, device)
    return model
