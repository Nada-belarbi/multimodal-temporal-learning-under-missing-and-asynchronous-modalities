"""Generic supervised trainer for source-only and joint fusion models."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, fields, is_dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from .checkpoint import save_checkpoint


@dataclass(frozen=True)
class TrainingBatch:
    inputs: Any
    targets: torch.Tensor
    target_valid: torch.Tensor
    decision_ids: Sequence[str] = ()


@dataclass(frozen=True)
class TrainingConfig:
    n_classes: int
    max_epochs: int = 100
    patience: int = 10
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    gradient_clip_norm: float = 1.0
    seed: int = 11
    selection_metric: Literal["macro_f1", "validation_loss"] = "macro_f1"

    def validate(self) -> None:
        if self.n_classes <= 1 or self.max_epochs <= 0 or self.patience <= 0:
            raise ValueError("class count, epoch count and patience must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0 or self.gradient_clip_norm <= 0:
            raise ValueError("optimizer and clipping settings are outside valid ranges")
        if self.selection_metric not in {"macro_f1", "validation_loss"}:
            raise ValueError("selection_metric must be macro_f1 or validation_loss")


@dataclass(frozen=True)
class TrainingResult:
    best_epoch: int
    best_macro_f1: float
    best_validation_loss: float
    epochs_completed: int
    optimizer_updates: int
    history: tuple[dict[str, float | int], ...]
    completed: bool


def _unpack_batch(batch: TrainingBatch | Any) -> TrainingBatch:
    if isinstance(batch, TrainingBatch):
        return batch
    return TrainingBatch(
        batch.inputs, batch.targets, batch.target_valid, getattr(batch, "decision_ids", ())
    )


def _forward(model: nn.Module, inputs: Any) -> Any:
    return model(inputs)


def _to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if is_dataclass(value) and not isinstance(value, type):
        return replace(
            value,
            **{
                field.name: _to_device(getattr(value, field.name), device)
                for field in fields(value)
            },
        )
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_to_device(item, device) for item in value]
    return value


def _validate_output(
    output: Any, targets: torch.Tensor, target_valid: torch.Tensor, n_classes: int
) -> None:
    scores = output.scores
    computable = output.computable
    if scores.ndim != 2 or scores.shape[1] != n_classes or scores.shape[0] != targets.shape[0]:
        raise ValueError("model output shape does not match targets or class map")
    if computable.shape != targets.shape or computable.dtype != torch.bool:
        raise ValueError("model computable mask must be boolean with one value per target")
    if target_valid.shape != targets.shape or target_valid.dtype != torch.bool:
        raise ValueError("target_valid must be boolean with one value per target")
    if bool(((targets[target_valid] < 0) | (targets[target_valid] >= n_classes)).any()):
        raise ValueError("target class is outside the configured class map")
    if bool(computable.any()) and not bool(torch.isfinite(scores[computable]).all()):
        raise FloatingPointError("non-finite scores on computable decisions")


def _macro_f1(confusion: torch.Tensor) -> float:
    true_positive = confusion.diag().to(torch.float64)
    false_positive = confusion.sum(dim=0).to(torch.float64) - true_positive
    false_negative = confusion.sum(dim=1).to(torch.float64) - true_positive
    denominator = 2 * true_positive + false_positive + false_negative
    per_class = torch.where(denominator > 0, 2 * true_positive / denominator.clamp_min(1), 0.0)
    return float(per_class.mean().item())


def _capture_rng_state(
    train_batches: Iterable[Any], validation_batches: Iterable[Any]
) -> dict[str, Any]:
    numpy_state = np.random.get_state()
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": {
            "algorithm": numpy_state[0],
            "keys": torch.from_numpy(numpy_state[1].copy()),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    for name, batches in (("train", train_batches), ("validation", validation_batches)):
        generator = getattr(batches, "generator", None)
        if generator is not None:
            state[f"{name}_loader_generator"] = generator.get_state()
    return state


def _restore_rng_state(
    state: Mapping[str, Any], train_batches: Iterable[Any], validation_batches: Iterable[Any]
) -> None:
    random.setstate(_nested_tuple(state["python"]))
    numpy_state = state["numpy"]
    np.random.set_state(
        (
            numpy_state["algorithm"],
            numpy_state["keys"].cpu().numpy().astype(np.uint32),
            int(numpy_state["position"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"]),
        )
    )
    torch.set_rng_state(state["torch_cpu"].cpu())
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    for name, batches in (("train", train_batches), ("validation", validation_batches)):
        generator = getattr(batches, "generator", None)
        state_key = f"{name}_loader_generator"
        if state_key in state:
            if generator is None:
                raise ValueError(f"resume checkpoint requires a {name} DataLoader generator")
            generator.set_state(state[state_key].cpu())


def _nested_tuple(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_nested_tuple(item) for item in value)
    return value


def _evaluate(
    model: nn.Module, batches: Iterable[Any], n_classes: int, device: torch.device
) -> dict[str, float | int]:
    confusion = torch.zeros((n_classes, n_classes), dtype=torch.int64)
    loss_sum = 0.0
    scored_count = 0
    eligible_count = 0
    prior_training = model.training
    model.eval()
    with torch.no_grad():
        for raw_batch in batches:
            batch = _unpack_batch(raw_batch)
            targets = batch.targets.to(device=device, dtype=torch.long)
            target_valid = batch.target_valid.to(device=device, dtype=torch.bool)
            output = _forward(model, _to_device(batch.inputs, device))
            _validate_output(output, targets, target_valid, n_classes)
            eligible = target_valid
            use = output.computable & eligible
            eligible_count += int(eligible.sum().item())
            if not bool(use.any()):
                continue
            selected_targets = targets[use]
            selected_scores = output.scores[use]
            loss_sum += float(
                nn.functional.cross_entropy(selected_scores, selected_targets).item()
            ) * int(use.sum().item())
            predicted = selected_scores.argmax(dim=1)
            flat = selected_targets * n_classes + predicted
            confusion += torch.bincount(flat.cpu(), minlength=n_classes * n_classes).reshape(
                n_classes, n_classes
            )
            scored_count += int(use.sum().item())
    model.train(prior_training)
    if scored_count == 0:
        raise RuntimeError("validation has no computable, label-eligible decisions")
    return {
        "loss": loss_sum / scored_count,
        "macro_f1": _macro_f1(confusion),
        "coverage": scored_count / eligible_count if eligible_count else 0.0,
        "scored_count": scored_count,
        "eligible_count": eligible_count,
    }


def train(
    model: nn.Module,
    train_batches: Iterable[Any],
    validation_batches: Iterable[Any],
    config: TrainingConfig,
    checkpoint_dir: str | Path,
    device: str | torch.device = "cpu",
    *,
    resume_from: str | Path | None = None,
    run_identity: Mapping[str, str] | None = None,
    stop_after_epochs: int | None = None,
) -> TrainingResult:
    """Train one model; callers provide re-iterable loaders and prepared batch contracts."""
    config.validate()
    device = torch.device(device)
    model.to(device)
    torch.manual_seed(config.seed)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    criterion = nn.CrossEntropyLoss(reduction="mean", label_smoothing=0.0)
    destination = Path(checkpoint_dir)
    if stop_after_epochs is not None and stop_after_epochs <= 0:
        raise ValueError("stop_after_epochs must be positive")
    expected_config = asdict(config)
    expected_identity = dict(run_identity or {})
    best_f1: float | None = None
    best_loss: float | None = None
    best_epoch = 0
    stale_epochs = 0
    update_count = 0
    completed_epoch = 0
    history: list[dict[str, float | int]] = []

    if resume_from is None:
        torch.manual_seed(config.seed)
        random.seed(config.seed)
        np.random.seed(config.seed)
        start_epoch = 1
    else:
        checkpoint = torch.load(resume_from, map_location="cpu", weights_only=True)
        if checkpoint.get("config") != expected_config:
            raise ValueError("resume checkpoint training configuration does not match")
        if checkpoint.get("run_identity", {}) != expected_identity:
            raise ValueError("resume checkpoint data/run identity does not match")
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        completed_epoch = int(checkpoint["epoch"])
        best_epoch = int(checkpoint["best_epoch"])
        best_f1 = checkpoint["best_macro_f1"]
        best_loss = checkpoint["best_validation_loss"]
        stale_epochs = int(checkpoint["stale_epochs"])
        update_count = int(checkpoint["optimizer_updates"])
        history = list(checkpoint.get("history", []))
        start_epoch = completed_epoch + 1
        _restore_rng_state(checkpoint["rng_state"], train_batches, validation_batches)
        if start_epoch > config.max_epochs:
            raise ValueError("resume checkpoint is already at or beyond max_epochs")

    completed = True

    for epoch in range(start_epoch, config.max_epochs + 1):
        model.train()
        epoch_loss_sum = 0.0
        scored_count = 0
        skipped_count = 0
        for raw_batch in train_batches:
            batch = _unpack_batch(raw_batch)
            targets = batch.targets.to(device=device, dtype=torch.long)
            target_valid = batch.target_valid.to(device=device, dtype=torch.bool)
            optimizer.zero_grad(set_to_none=True)
            output = _forward(model, _to_device(batch.inputs, device))
            _validate_output(output, targets, target_valid, config.n_classes)
            use = output.computable & target_valid
            if not bool(use.any()):
                skipped_count += int(targets.shape[0])
                continue
            loss = criterion(output.scores[use], targets[use])
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(
                    f"non-finite loss at epoch {epoch}; decisions={list(batch.decision_ids)}"
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.gradient_clip_norm, error_if_nonfinite=True
            )
            optimizer.step()
            count = int(use.sum().item())
            epoch_loss_sum += float(loss.detach().item()) * count
            scored_count += count
            update_count += 1
        if scored_count == 0:
            raise RuntimeError(f"epoch {epoch} contained no computable training decisions")

        validation = _evaluate(model, validation_batches, config.n_classes, device)
        train_loss = epoch_loss_sum / scored_count
        row: dict[str, float | int] = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation["loss"],
            "validation_macro_f1": validation["macro_f1"],
            "validation_coverage": validation["coverage"],
            "train_scored": scored_count,
            "train_skipped": skipped_count,
            "optimizer_updates": update_count,
        }
        history.append(row)
        destination.mkdir(parents=True, exist_ok=True)
        with (destination / "history.jsonl").open("a", encoding="utf-8") as progress:
            progress.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        completed_epoch = epoch
        f1 = float(validation["macro_f1"])
        val_loss = float(validation["loss"])
        if config.selection_metric == "validation_loss":
            improved = best_loss is None or val_loss < best_loss
        else:
            improved = (
                best_f1 is None
                or f1 > best_f1
                or (f1 == best_f1 and best_loss is not None and val_loss < best_loss)
            )
        if improved:
            best_f1, best_loss, best_epoch = f1, val_loss, epoch
            stale_epochs = 0
            save_checkpoint(
                destination / "best.pt",
                {
                    "state_dict": model.state_dict(),
                    "epoch": epoch,
                    "config": asdict(config),
                    "validation_macro_f1": f1,
                    "validation_loss": val_loss,
                },
            )
        else:
            stale_epochs += 1
        rng_state = _capture_rng_state(train_batches, validation_batches)
        save_checkpoint(
            destination / "last.pt",
            {
                "state_dict": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": epoch,
                "best_epoch": best_epoch,
                "best_macro_f1": best_f1,
                "best_validation_loss": best_loss,
                "stale_epochs": stale_epochs,
                "optimizer_updates": update_count,
                "config": expected_config,
                "run_identity": expected_identity,
                "history": history,
                "rng_state": rng_state,
            },
        )
        if stale_epochs >= config.patience:
            break
        if stop_after_epochs is not None and epoch - start_epoch + 1 >= stop_after_epochs:
            completed = False
            break

    if best_f1 is None or best_loss is None:
        raise RuntimeError("training completed without selecting a validation checkpoint")
    return TrainingResult(
        best_epoch, best_f1, best_loss, completed_epoch, update_count, tuple(history), completed
    )
