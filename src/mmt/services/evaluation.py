"""Evaluate saved fusion artifacts through the shared runner and temporal selector."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
import yaml

from mmt.config import ResolvedConfig
from mmt.data.collate import collate_decisions
from mmt.data.dataset import DecisionSample, SourceSample
from mmt.data.normalization import load_normalizer
from mmt.data.selection import SelectionResult, select_observations
from mmt.evaluation.io_utils import canonical_hash
from mmt.evaluation.perturbations import _observation_digest, apply_schedule
from mmt.evaluation.protocol import attach_groups
from mmt.evaluation.runner import evaluate
from mmt.models.baselines import MajorityReference
from mmt.models.source_encoder import SourceBatch
from mmt.storage import load_manifest, load_sequence
from mmt.training.checkpoint import load_fusion_model
from mmt.training.trainer import _to_device

from .evaluation_workflows import load_saved_postprocessing


class EvaluationServiceError(ValueError):
    """Raised when saved run/data identities cannot be reconciled for evaluation."""


def evaluate_run(
    resolved: ResolvedConfig,
    run_dir: str | Path,
    partition: str,
    output_dir: str | Path,
    *,
    scenario_id: str = "clean",
    schedule: Mapping[str, Any] | None = None,
    device: str | torch.device = "cpu",
    calibration_path: str | Path | None = None,
    policy_path: str | Path | None = None,
    reference: str | None = None,
    batch_size: int = 1,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Score one prepared partition without passing target labels into the model adapter."""
    run_dir = Path(run_dir).expanduser().resolve()
    if partition not in {"train", "validation", "calibration", "test"}:
        raise EvaluationServiceError(f"unknown partition {partition!r}")
    model_name = resolved.config.run.model_name
    if model_name not in {"v0", "v1", "v2"}:
        raise EvaluationServiceError("evaluation supports V0, V1 and V2 artifacts")
    prepared_root = resolved.config.data.manifest_path.parent
    manifest = load_manifest(prepared_root, verify=True)
    if manifest.get("application") != resolved.config.data.application:
        raise EvaluationServiceError("prepared manifest application does not match config")

    required_artifacts = ("weights.pt", "normalizer.npz", "model_config.json", "class_map.json")
    missing = [
        name for name in (*required_artifacts, "sha256sums.json") if not (run_dir / name).is_file()
    ]
    if missing:
        raise EvaluationServiceError(f"fusion run is missing artifacts: {missing}")
    hashes = json.loads((run_dir / "sha256sums.json").read_text(encoding="utf-8"))
    actual_artifact_hashes = {
        name: _sha256(run_dir / name) for name in required_artifacts if name in hashes
    }
    mismatched = [name for name, digest in actual_artifact_hashes.items() if hashes[name] != digest]
    absent_hashes = [name for name in required_artifacts if name not in hashes]
    if absent_hashes or mismatched:
        raise EvaluationServiceError(
            f"run artifact hashes incomplete or mismatched: {absent_hashes + mismatched}"
        )

    model_config = json.loads((run_dir / "model_config.json").read_text(encoding="utf-8"))
    class_map = json.loads((run_dir / "class_map.json").read_text(encoding="utf-8"))
    source_order = tuple(model_config["source_order"])
    source_dimensions = {
        key: int(value) for key, value in model_config["source_dimensions"].items()
    }
    if source_order != tuple(resolved.config.data.feature_order):
        raise EvaluationServiceError("run source order differs from current data config")
    if source_dimensions != resolved.config.data.source_schema:
        raise EvaluationServiceError("run source dimensions differ from current data config")
    normalizer_identity = model_config["normalizer_identity"]
    if normalizer_identity.get("manifest_hash") != _sha256(resolved.config.data.manifest_path):
        raise EvaluationServiceError("model normalizer belongs to a different dataset manifest")
    if class_map != manifest.get("class_map"):
        raise EvaluationServiceError("run class map differs from the prepared dataset")
    normalizer = load_normalizer(run_dir / "normalizer.npz", normalizer_identity)
    model = load_fusion_model(
        model_config,
        class_map,
        run_dir / "weights.pt",
        expected_model_name=model_name,
        device=device,
    )
    model.eval()
    if batch_size <= 0:
        raise EvaluationServiceError("batch_size must be positive")
    reference_source = None
    if reference is not None and reference != "majority":
        if model_name != "v0" or reference not in source_order:
            raise EvaluationServiceError("source-only references require a V0 run and known source")
        reference_source = reference
    if reference is not None and (calibration_path is not None or policy_path is not None):
        raise EvaluationServiceError("fusion calibration cannot be reused for a baseline")

    membership = pd.read_csv(resolved.config.data.partitions_path)
    decisions_path = prepared_root / "decisions.csv"
    targets_path = prepared_root / "targets.csv"
    if not decisions_path.is_file() or not targets_path.is_file():
        raise EvaluationServiceError("prepared run needs decisions.csv and targets.csv")
    registry = pd.read_csv(decisions_path).merge(
        pd.read_csv(targets_path), on="decision_id", how="left", validate="one_to_one"
    )
    if registry.score_eligible.isna().any():
        raise EvaluationServiceError("some prepared decisions have no target/eligibility record")
    registry = attach_groups(registry, membership)
    if registry.group_id.isna().any() or registry.split.isna().any():
        raise EvaluationServiceError("prepared decisions have incomplete group/split membership")
    majority = None
    if reference == "majority":
        train_rows = registry[(registry.split == "train") & registry.score_eligible]
        majority = MajorityReference(len(class_map)).fit(
            train_rows.target_class.astype(int), [True] * len(train_rows)
        )
    sources = {
        (sequence_id, source_id): data
        for sequence_id in registry.loc[registry.split == partition, "sequence_id"]
        .astype(str)
        .unique()
        for source_id, data in load_sequence(prepared_root, sequence_id).items()
    }
    sorted_sources = {
        key: bool(
            np.all(arrays["event_start"][1:] >= arrays["event_start"][:-1])
            and np.all(arrays["event_end"][1:] >= arrays["event_end"][:-1])
        )
        for key, arrays in sources.items()
    }
    if schedule is not None:
        expected_schedule_hash = canonical_hash(
            {
                "entries": schedule["entries"],
                "scenario": schedule["scenario"],
                "seed": schedule["seed"],
            }
        )
        if schedule.get("schedule_hash") != expected_schedule_hash or schedule.get(
            "input_hash"
        ) != _observation_digest(sources):
            raise EvaluationServiceError("schedule hash or observation identity mismatch")
        if schedule["scenario"]["id"] != scenario_id:
            raise EvaluationServiceError("schedule scenario ID differs from requested scenario")

    prediction_cache = {}
    if batch_size > 1:
        transformed = apply_schedule(sources, schedule) if schedule is not None else sources
        rows = registry[registry.split == partition].to_dict(orient="records")
        for offset in range(0, len(rows), batch_size):
            chunk = rows[offset : offset + batch_size]
            samples = []
            for row in chunk:
                sample_sources = {}
                for source_id in source_order:
                    raw = transformed.get((str(row["sequence_id"]), source_id))
                    selected = (
                        _empty_selection(0)
                        if raw is None
                        else select_observations(
                            raw,
                            row,
                            resolved.config.runtime.max_source_age,
                            assume_sorted=sorted_sources[(str(row["sequence_id"]), source_id)],
                        )
                    )
                    dimension = source_dimensions[source_id]
                    values, mask = normalizer.transform(
                        source_id,
                        np.asarray(selected.values, dtype=np.float32).reshape(-1, dimension),
                        np.asarray(selected.valid_mask, dtype=bool).reshape(-1, dimension),
                    )
                    sample_sources[source_id] = SourceSample(
                        torch.from_numpy(values),
                        torch.from_numpy(mask),
                        torch.tensor(selected.time_features, dtype=torch.float32).reshape(-1, 3),
                    )
                samples.append(
                    DecisionSample(
                        str(row["decision_id"]),
                        str(row["sequence_id"]),
                        float(row["decision_time"]),
                        float(row["support_start"]),
                        float(row["support_end"]),
                        sample_sources,
                        0,
                        False,
                        {},
                    )
                )
            batch = collate_decisions(samples)
            inputs = _to_device(batch.inputs, torch.device(device))
            with torch.inference_mode():
                output = (
                    None
                    if majority is not None
                    else (
                        model(inputs)
                        if reference_source is None
                        else model.classifiers[source_order.index(reference_source)](
                            inputs[reference_source]
                        )
                    )
                )
            for index, sample in enumerate(samples):
                presence = {source: bool(inputs[source].present[index]) for source in source_order}
                if majority is not None:
                    prediction = majority.predict(any(presence.values()))
                    result = {
                        "computable": prediction.computable,
                        "scores": None,
                        "predicted_class": prediction.predicted_class,
                        "presence": presence,
                    }
                else:
                    computable = bool(output.computable[index])
                    scores = output.scores[index].detach().cpu().numpy() if computable else None
                    result = {
                        "computable": computable,
                        "scores": scores,
                        "predicted_class": int(scores.argmax()) if computable else None,
                        "presence": presence,
                    }
                prediction_cache[sample.decision_id] = result

    def predict_fn(row: Mapping[str, Any], sequence_sources: Mapping[str, Any]) -> dict[str, Any]:
        if prediction_cache:
            return prediction_cache[str(row["decision_id"])]
        batches = {}
        for source_id in source_order:
            raw = sequence_sources.get(source_id)
            selection = (
                _empty_selection(source_dimensions[source_id])
                if raw is None
                else select_observations(
                    raw,
                    row,
                    resolved.config.runtime.max_source_age,
                    assume_sorted=sorted_sources[(str(row["sequence_id"]), source_id)],
                )
            )
            values, mask = normalizer.transform(
                source_id,
                np.asarray(selection.values, dtype=np.float32).reshape(
                    -1, source_dimensions[source_id]
                ),
                np.asarray(selection.valid_mask, dtype=bool).reshape(
                    -1, source_dimensions[source_id]
                ),
            )
            steps = max(1, len(values))
            padded_values = torch.zeros(
                (1, steps, source_dimensions[source_id]), dtype=torch.float32
            )
            padded_mask = torch.zeros_like(padded_values, dtype=torch.bool)
            time_features = torch.zeros((1, steps, 3), dtype=torch.float32)
            if len(values):
                padded_values[0, : len(values)] = torch.from_numpy(values)
                padded_mask[0, : len(values)] = torch.from_numpy(mask)
                time_features[0, : len(values)] = torch.tensor(
                    selection.time_features, dtype=torch.float32
                )
            lengths = torch.tensor([len(values)], dtype=torch.long)
            padding = torch.arange(steps).unsqueeze(0) >= lengths.unsqueeze(1)
            batches[source_id] = SourceBatch(
                values=padded_values.to(device),
                feature_mask=padded_mask.to(device),
                time_features=time_features.to(device),
                lengths=lengths.to(device),
                padding_mask=padding.to(device),
                present=(lengths > 0).to(device),
            )
        with torch.inference_mode():
            if majority is not None:
                output = majority.predict(any(bool(batch.present[0]) for batch in batches.values()))
                return {
                    "computable": output.computable,
                    "scores": None,
                    "predicted_class": output.predicted_class,
                    "presence": {
                        source: bool(batch.present[0]) for source, batch in batches.items()
                    },
                }
            output = (
                model(batches)
                if reference_source is None
                else model.classifiers[source_order.index(reference_source)](
                    batches[reference_source]
                )
            )
        scores = output.scores[0].detach().cpu().numpy()
        presence = {source_id: bool(batches[source_id].present[0]) for source_id in source_order}
        return {
            "computable": bool(output.computable[0].item()),
            "scores": scores,
            "predicted_class": int(scores.argmax()) if bool(output.computable[0].item()) else None,
            "presence": presence,
        }

    expected_inputs = resolved.input_hashes
    actual_inputs = {
        "data.manifest_path": _sha256(resolved.config.data.manifest_path),
        "data.partitions_path": _sha256(resolved.config.data.partitions_path),
    }
    if resolved.config.data.normalizer_ref is not None:
        actual_inputs["data.normalizer_ref"] = _sha256(resolved.config.data.normalizer_ref)
    actual_inputs.update({f"run.{key}": value for key, value in actual_artifact_hashes.items()})
    score_kind = (
        "label_only"
        if reference == "majority"
        else (
            "logits" if reference_source is not None or model_name != "v0" else "log_probabilities"
        )
    )
    calibration, policy = load_saved_postprocessing(
        run_dir, actual_inputs, score_kind, calibration_path, policy_path
    )
    if calibration_path is not None:
        actual_inputs["calibration"] = _sha256(Path(calibration_path))
    if policy_path is not None:
        actual_inputs["policy"] = _sha256(Path(policy_path))
    return evaluate(
        registry,
        sources,
        schedule,
        predict_fn,
        len(class_map),
        output_dir,
        run_config_id(run_dir) + (f"::{reference}" if reference is not None else ""),
        scenario_id,
        score_kind,
        calibration=calibration,
        policy=policy,
        expected_ids=expected_inputs,
        actual_ids=actual_inputs,
        partition=partition,
    )


def run_config_id(run_dir: Path) -> str:
    try:
        run_yaml = yaml.safe_load((run_dir / "resolved_config.yaml").read_text(encoding="utf-8"))
        return str(run_yaml["run"]["run_id"])
    except (OSError, KeyError, TypeError, yaml.YAMLError) as error:
        raise EvaluationServiceError(f"run configuration is missing or invalid: {error}") from error


def _empty_selection(feature_count: int) -> SelectionResult:
    return SelectionResult((), (), (), (), (), (), (), 0, 0, 0, None, None, None, "empty")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


evaluate_v0_run = evaluate_run
