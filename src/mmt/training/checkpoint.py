"""Atomic state-dict checkpoint persistence."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
from typing import Any

import torch
from torch import nn


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


def load_state_dict(model: nn.Module, path: str | Path, device: str | torch.device = "cpu") -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("state_dict"), dict):
        raise ValueError("checkpoint must contain a state_dict")
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(device)
    model.eval()
    return payload