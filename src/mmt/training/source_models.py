"""Independent source-model training and strict V0 checkpoint assembly."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch

from mmt.models.source_encoder import SourceBatch, SourceClassifier
from mmt.models.v0_late_fusion import V0LateFusion

from .checkpoint import load_state_dict, save_checkpoint
from .trainer import TrainingBatch, TrainingConfig, TrainingResult, train


@dataclass(frozen=True)
class SourceTrainingResult:
    source_id: str
    checkpoint_path: Path
    training: TrainingResult


class _SourceBatchView:
    def __init__(self, batches: Iterable[Any], source_id: str) -> None:
        self.batches = batches
        self.source_id = source_id
        self.dataset = getattr(batches, "dataset", None)
        self.generator = getattr(batches, "generator", None)

    def __iter__(self) -> Iterator[TrainingBatch]:
        for batch in self.batches:
            inputs = batch.inputs
            if not isinstance(inputs, Mapping) or self.source_id not in inputs:
                raise ValueError(f"batch is missing source {self.source_id!r}")
            source_batch = inputs[self.source_id]
            if not isinstance(source_batch, SourceBatch):
                raise TypeError(f"source {self.source_id!r} must use SourceBatch")
            target_valid = batch.target_valid.to(dtype=torch.bool) & source_batch.present.cpu()
            yield TrainingBatch(
                source_batch, batch.targets, target_valid, getattr(batch, "decision_ids", ())
            )

    def __len__(self) -> int:
        return len(self.batches)  # type: ignore[arg-type]


def train_source_models(
    source_dimensions: Mapping[str, int],
    n_classes: int,
    train_batches: Iterable[Any],
    validation_batches: Iterable[Any],
    config: TrainingConfig,
    output_root: str | Path,
    hidden_size: int = 64,
    dropout: float = 0.1,
    device: str | torch.device = "cpu",
    run_identity: Mapping[str, str] | None = None,
) -> dict[str, SourceTrainingResult]:
    """Fit one checkpoint per source using only that source's present, valid targets."""
    output_root = Path(output_root)
    results = {}
    for source_index, (source_id, feature_dim) in enumerate(source_dimensions.items()):
        source_config = replace(
            config, seed=config.seed + source_index, selection_metric="validation_loss"
        )
        torch.manual_seed(source_config.seed)
        model = SourceClassifier(feature_dim, n_classes, hidden_size, dropout)
        result = train(
            model,
            _SourceBatchView(train_batches, source_id),
            _SourceBatchView(validation_batches, source_id),
            source_config,
            output_root / source_id,
            device=device,
            run_identity={**(run_identity or {}), "source_id": source_id},
        )
        checkpoint_path = output_root / source_id / "best.pt"
        results[source_id] = SourceTrainingResult(source_id, checkpoint_path, result)
    return results


def assemble_v0(
    source_dimensions: Mapping[str, int],
    n_classes: int,
    source_checkpoints: Mapping[str, str | Path],
    output_path: str | Path,
    hidden_size: int = 64,
    dropout: float = 0.1,
    device: str | torch.device = "cpu",
) -> Path:
    """Assemble source checkpoints in declared order and persist one strict V0 artifact."""
    if tuple(source_checkpoints) != tuple(source_dimensions):
        raise ValueError("checkpoint keys and order must match source_dimensions")
    model = V0LateFusion(source_dimensions, n_classes, hidden_size, dropout)
    identities = {}
    for index, source_id in enumerate(source_dimensions):
        source_model = SourceClassifier(
            source_dimensions[source_id], n_classes, hidden_size, dropout
        )
        payload = load_state_dict(source_model, source_checkpoints[source_id], device="cpu")
        model.classifiers[index].load_state_dict(source_model.state_dict(), strict=True)
        identities[source_id] = {
            "checkpoint": str(Path(source_checkpoints[source_id]).resolve()),
            "epoch": int(payload["epoch"]),
        }
    model.to(device)
    save_checkpoint(
        output_path,
        {
            "state_dict": model.state_dict(),
            "artifact_type": "v0_late_fusion",
            "source_order": tuple(source_dimensions),
            "source_dimensions": dict(source_dimensions),
            "n_classes": n_classes,
            "epsilon": model.epsilon,
            "source_checkpoints": identities,
        },
    )
    return Path(output_path)
