"""Prepared-data training for independent-source V0 and joint V1/V2 fusion."""

from __future__ import annotations

import json
import platform
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from mmt.config import ResolvedConfig, save_resolved_config
from mmt.data.collate import collate_decisions
from mmt.data.dataset import DecisionDataset
from mmt.data.normalization import fit_normalizer, save_normalizer
from mmt.models.registry import build_model
from mmt.storage import load_manifest, load_sequence
from mmt.training import TrainingConfig, assemble_v0, train, train_source_models

from .runs import RunDirectory, write_sha256sums


class TrainingServiceError(ValueError):
    """Raised when a prepared fusion run cannot satisfy its declared contract."""


def train_prepared(resolved: ResolvedConfig, device: str | torch.device = "cpu") -> Path:
    """Train V0/V1/V2 on train data and select checkpoints on clean validation only."""
    config = resolved.config
    run_config = config.run
    model_name = run_config.model_name
    if model_name not in {"v0", "v1", "v2"}:
        raise TrainingServiceError("prepared training supports v0, v1 and v2")
    if run_config.regime != "clean":
        raise TrainingServiceError(
            "robust training schedules are not integrated; use regime='clean'"
        )
    manifest_path = config.data.manifest_path
    prepared_root = manifest_path.parent
    manifest = load_manifest(prepared_root, verify=True)
    if manifest.get("application") != config.data.application:
        raise TrainingServiceError(
            "prepared manifest application does not match the resolved config"
        )
    if manifest.get("source_dims") != config.data.source_schema:
        raise TrainingServiceError("resolved source_schema differs from the prepared manifest")
    if tuple(manifest.get("source_order", ())) != tuple(config.data.feature_order):
        raise TrainingServiceError("feature_order must match the manifest's declared source_order")
    class_map_raw = manifest.get("class_map")
    if not isinstance(class_map_raw, dict) or not class_map_raw:
        raise TrainingServiceError("prepared manifest must include an audited class_map")
    class_map = {str(key): int(value) for key, value in class_map_raw.items()}
    n_classes = len(class_map)
    if sorted(class_map.values()) != list(range(n_classes)):
        raise TrainingServiceError("prepared class_map indices must be contiguous from zero")

    partitions = pd.read_csv(config.data.partitions_path)
    decisions_path = prepared_root / "decisions.csv"
    targets_path = prepared_root / "targets.csv"
    decisions = pd.read_csv(decisions_path)
    targets = pd.read_csv(targets_path)
    membership_hash = _canonical_hash(
        partitions.sort_values("sequence_id").to_dict(orient="records")
    )
    manifest_hash = resolved.input_hashes.get("data.manifest_path")
    if manifest_hash is None:
        raise TrainingServiceError("resolved config must hash the prepared manifest")

    decision_splits = decisions.merge(
        partitions[["sequence_id", "split"]], on="sequence_id", how="left", validate="many_to_one"
    )
    train_decisions = decision_splits[decision_splits.split == "train"].to_dict(orient="records")
    if not train_decisions:
        raise TrainingServiceError("training partition contains no decisions")
    training_sequences = sorted({str(row["sequence_id"]) for row in train_decisions})
    training_sources = {
        (sequence_id, source_id): arrays
        for sequence_id in training_sequences
        for source_id, arrays in load_sequence(prepared_root, sequence_id).items()
        if source_id in config.data.source_schema
    }
    normalizer_identity = {
        "manifest_hash": manifest_hash,
        "training_membership_hash": membership_hash,
    }
    normalizer = fit_normalizer(
        training_sources, train_decisions, config.data.feature_order, normalizer_identity
    )
    train_dataset = DecisionDataset(
        prepared_root,
        decisions,
        partitions,
        normalizer,
        config.data.feature_order,
        targets,
        split="train",
    )
    validation_dataset = DecisionDataset(
        prepared_root,
        decisions,
        partitions,
        normalizer,
        config.data.feature_order,
        targets,
        split="validation",
    )
    if not len(validation_dataset):
        raise TrainingServiceError("validation partition contains no decisions")

    model_options = dict(run_config.model_options)
    hidden_size = int(model_options.pop("hidden_size", 64))
    dropout = float(model_options.pop("dropout", 0.1))
    architecture_options = {"hidden_size": hidden_size, "dropout": dropout}
    if model_name == "v0":
        architecture_options["epsilon"] = float(model_options.pop("epsilon", 1e-8))
    elif model_name == "v2":
        architecture_options["n_heads"] = int(model_options.pop("n_heads", 4))
        architecture_options["feedforward_size"] = int(model_options.pop("feedforward_size", 128))
    if model_options:
        raise TrainingServiceError(f"unknown {model_name} model options: {sorted(model_options)}")
    source_dimensions = {
        source: config.data.source_schema[source] for source in config.data.feature_order
    }
    torch.manual_seed(run_config.seed)
    model = build_model(model_name, source_dimensions, class_map, architecture_options)

    training_options = dict(run_config.training)
    batch_size = int(training_options.pop("batch_size", 32))
    allowed_training = {
        "max_epochs",
        "patience",
        "learning_rate",
        "weight_decay",
        "gradient_clip_norm",
    }
    unknown_training = set(training_options) - allowed_training
    if unknown_training:
        raise TrainingServiceError(f"unknown training options: {sorted(unknown_training)}")
    if batch_size <= 0:
        raise TrainingServiceError("batch_size must be positive")
    training_config = TrainingConfig(n_classes=n_classes, seed=run_config.seed, **training_options)
    generator = torch.Generator().manual_seed(run_config.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        collate_fn=collate_decisions,
        drop_last=False,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_decisions,
        drop_last=False,
    )

    run = RunDirectory(run_config.output_root, run_config.run_id)
    run_path = run.start()
    try:
        save_resolved_config(resolved, run_path)
        save_normalizer(run_path / "normalizer.npz", normalizer)
        (run_path / "environment.json").write_text(
            json.dumps(
                {
                    "python": platform.python_version(),
                    "numpy": np.__version__,
                    "torch": torch.__version__,
                    "device": str(device),
                    "mps_available": bool(torch.backends.mps.is_available()),
                },
                sort_keys=True,
                indent=2,
            ),
            encoding="utf-8",
        )
        (run_path / "source_schema.json").write_text(
            json.dumps(
                {
                    "source_order": list(config.data.feature_order),
                    "source_dimensions": config.data.source_schema,
                },
                sort_keys=True,
                indent=2,
            ),
            encoding="utf-8",
        )
        (run_path / "class_map.json").write_text(
            json.dumps(class_map, sort_keys=True, indent=2), encoding="utf-8"
        )
        run_identity = {
            "manifest_hash": manifest_hash,
            "training_membership_hash": membership_hash,
            "partition_hashes": _canonical_hash(run_config.partition_hashes),
        }
        checkpoint_files = []
        if model_name == "v0":
            source_results = train_source_models(
                source_dimensions,
                n_classes,
                train_loader,
                validation_loader,
                training_config,
                run_path / "source_models",
                hidden_size=hidden_size,
                dropout=dropout,
                device=device,
                run_identity=run_identity,
            )
            assemble_v0(
                source_dimensions,
                n_classes,
                {source_id: result.checkpoint_path for source_id, result in source_results.items()},
                run_path / "weights.pt",
                hidden_size=hidden_size,
                dropout=dropout,
                device=device,
            )
            history = {source: asdict(result.training) for source, result in source_results.items()}
            for source in config.data.feature_order:
                checkpoint_files.extend(
                    [
                        Path("source_models") / source / "best.pt",
                        Path("source_models") / source / "last.pt",
                        Path("source_models") / source / "history.jsonl",
                    ]
                )
        else:
            result = train(
                model,
                train_loader,
                validation_loader,
                training_config,
                run_path / "checkpoints",
                device=device,
                run_identity=run_identity,
            )
            shutil.copyfile(run_path / "checkpoints/best.pt", run_path / "weights.pt")
            history = asdict(result)
            checkpoint_files = [
                Path("checkpoints/best.pt"),
                Path("checkpoints/last.pt"),
                Path("checkpoints/history.jsonl"),
            ]
        (run_path / "model_config.json").write_text(
            json.dumps(
                {
                    "model_name": model_name,
                    "source_order": list(config.data.feature_order),
                    "source_dimensions": config.data.source_schema,
                    "class_map": class_map,
                    **architecture_options,
                    "score_kind": "log_probabilities" if model_name == "v0" else "logits",
                    "normalizer_identity": normalizer_identity,
                },
                sort_keys=True,
                indent=2,
            ),
            encoding="utf-8",
        )
        (run_path / "training_history.json").write_text(
            json.dumps(
                history,
                sort_keys=True,
                indent=2,
            ),
            encoding="utf-8",
        )
        artifact_files = [
            Path("resolved_config.yaml"),
            Path("config.sha256"),
            Path("environment.json"),
            Path("source_schema.json"),
            Path("class_map.json"),
            Path("normalizer.npz"),
            Path("model_config.json"),
            Path("weights.pt"),
            Path("training_history.json"),
        ]
        artifact_files.extend(checkpoint_files)
        write_sha256sums(run_path, [path.as_posix() for path in artifact_files])
        return run.complete([path.as_posix() for path in artifact_files] + ["sha256sums.json"])
    except BaseException as error:
        run.fail(error, incomplete=True)
        raise


def _canonical_hash(value: Any) -> str:
    import hashlib

    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


train_prepared_v0 = train_prepared
