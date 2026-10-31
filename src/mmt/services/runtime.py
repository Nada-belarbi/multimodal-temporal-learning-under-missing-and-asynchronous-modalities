"""Load a saved fusion predictor and replay one sequence without target access."""

from __future__ import annotations

import hashlib
import heapq
import json
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from mmt.config import ResolvedConfig
from mmt.data.normalization import load_normalizer
from mmt.data.selection import SelectionResult
from mmt.evaluation.io_utils import canonical_hash
from mmt.evaluation.perturbations import (
    _observation_digest,
    apply_schedule,
    load_schedule,
)
from mmt.models.source_encoder import SourceBatch
from mmt.runtime.engine import InferenceEngine
from mmt.runtime.replay import replay
from mmt.runtime.sink import ReplayOutputSink
from mmt.storage import load_manifest, load_sequence
from mmt.training.checkpoint import load_fusion_model

from .evaluation import run_config_id
from .evaluation_workflows import load_saved_postprocessing


class RuntimeServiceError(ValueError):
    """Raised when a saved predictor cannot replay a prepared sequence."""


def replay_sequence(
    resolved: ResolvedConfig,
    run_dir: str | Path,
    sequence_id: str,
    output_dir: str | Path,
    *,
    scenario_id: str = "clean",
    schedule_path: str | Path | None = None,
    temperature: float = 1.0,
    threshold: float | None = None,
    device: str | torch.device = "cpu",
    calibration_path: str | Path | None = None,
    policy_path: str | Path | None = None,
) -> Path:
    run_dir = Path(run_dir).expanduser().resolve()
    run_config = resolved.config.run
    if run_config.model_name not in {"v0", "v1", "v2"}:
        raise RuntimeServiceError("replay supports V0, V1 and V2 artifacts")
    prepared_root = resolved.config.data.manifest_path.parent
    manifest = load_manifest(prepared_root, verify=True)
    if manifest.get("application") != resolved.config.data.application:
        raise RuntimeServiceError("prepared manifest application does not match config")
    required = (
        "weights.pt",
        "normalizer.npz",
        "model_config.json",
        "class_map.json",
        "sha256sums.json",
    )
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        raise RuntimeServiceError(f"fusion run is missing artifacts: {missing}")
    hashes = json.loads((run_dir / "sha256sums.json").read_text(encoding="utf-8"))
    for name in required[:-1]:
        if hashes.get(name) != _sha256(run_dir / name):
            raise RuntimeServiceError(f"run artifact checksum mismatch for {name}")

    model_config = json.loads((run_dir / "model_config.json").read_text(encoding="utf-8"))
    class_map = json.loads((run_dir / "class_map.json").read_text(encoding="utf-8"))
    source_order = tuple(model_config["source_order"])
    source_dimensions = {
        name: int(value) for name, value in model_config["source_dimensions"].items()
    }
    if source_order != tuple(resolved.config.data.feature_order):
        raise RuntimeServiceError("run source order differs from data config")
    if source_dimensions != resolved.config.data.source_schema:
        raise RuntimeServiceError("run source dimensions differ from data config")
    normalizer_identity = model_config["normalizer_identity"]
    if normalizer_identity.get("manifest_hash") != _sha256(resolved.config.data.manifest_path):
        raise RuntimeServiceError("model normalizer belongs to a different dataset manifest")
    if class_map != manifest.get("class_map"):
        raise RuntimeServiceError("run class map differs from the prepared dataset")
    normalizer = load_normalizer(run_dir / "normalizer.npz", normalizer_identity)
    model = load_fusion_model(
        model_config,
        class_map,
        run_dir / "weights.pt",
        expected_model_name=run_config.model_name,
        device=device,
    )
    calibration, policy = load_saved_postprocessing(
        run_dir,
        {
            "data.manifest_path": _sha256(resolved.config.data.manifest_path),
            "data.partitions_path": _sha256(resolved.config.data.partitions_path),
        },
        "log_probabilities" if run_config.model_name == "v0" else "logits",
        calibration_path,
        policy_path,
    )
    if calibration is not None:
        if temperature != 1.0 or threshold is not None:
            raise RuntimeServiceError(
                "saved postprocessing cannot be combined with scalar overrides"
            )
        temperature = float(calibration["T"])
    if policy is not None:
        threshold = policy.get("tau")

    decisions_path = prepared_root / "decisions.csv"
    if not decisions_path.is_file():
        raise RuntimeServiceError("prepared artifact has no decisions.csv")
    decisions = pd.read_csv(
        decisions_path,
        usecols=[
            "decision_id",
            "sequence_id",
            "decision_time",
            "support_start",
            "support_end",
        ],
    )
    sequence_decisions = decisions[decisions.sequence_id.astype(str) == sequence_id]
    if sequence_decisions.empty:
        raise RuntimeServiceError(f"unknown sequence or no decisions: {sequence_id!r}")
    sequence_sources = load_sequence(prepared_root, sequence_id)
    schedule = None
    if schedule_path is not None:
        schedule = load_schedule(schedule_path)
        schedule_scenario = schedule.get("scenario", {})
        if schedule_scenario.get("id") != scenario_id:
            raise RuntimeServiceError("saved schedule ID does not match --scenario")
        expected_schedule_hash = canonical_hash(
            {
                "entries": schedule.get("entries"),
                "scenario": schedule_scenario,
                "seed": schedule.get("seed"),
            }
        )
        if schedule.get("schedule_hash") != expected_schedule_hash:
            raise RuntimeServiceError("saved scenario schedule hash is invalid")
        input_sources = {
            (sequence_id, source_id): values for source_id, values in sequence_sources.items()
        }
        if schedule.get("input_hash") != _observation_digest(input_sources):
            raise RuntimeServiceError("saved scenario schedule belongs to different observations")
        sequence_sources = {
            source_id: values
            for (_, source_id), values in apply_schedule(input_sources, schedule).items()
        }
    elif scenario_id != "clean":
        raise RuntimeServiceError("non-clean replay requires an explicit saved --schedule")

    def batch_builder(selections: Mapping[str, SelectionResult]) -> dict[str, SourceBatch]:
        result = {}
        for source_id in source_order:
            selection = selections[source_id]
            dimension = source_dimensions[source_id]
            values, mask = normalizer.transform(
                source_id,
                np.asarray(selection.values, dtype=np.float32).reshape(-1, dimension),
                np.asarray(selection.valid_mask, dtype=bool).reshape(-1, dimension),
            )
            length = len(values)
            steps = max(1, length)
            padded_values = torch.zeros((1, steps, dimension), dtype=torch.float32)
            padded_mask = torch.zeros((1, steps, dimension), dtype=torch.bool)
            time_features = torch.zeros((1, steps, 3), dtype=torch.float32)
            if length:
                padded_values[0, :length] = torch.from_numpy(values)
                padded_mask[0, :length] = torch.from_numpy(mask)
                time_features[0, :length] = torch.tensor(
                    selection.time_features, dtype=torch.float32
                )
            lengths = torch.tensor([length], dtype=torch.long)
            padding = torch.arange(steps).unsqueeze(0) >= lengths.unsqueeze(1)
            result[source_id] = SourceBatch(
                padded_values, padded_mask, time_features, lengths, padding, lengths > 0
            )
        return result

    def predict_fn(batches: Mapping[str, SourceBatch]) -> Any:
        with torch.inference_mode():
            return model(
                {
                    source_id: _to_device(batch, torch.device(device))
                    for source_id, batch in batches.items()
                }
            )

    identity = {
        "dataset_manifest": _sha256(resolved.config.data.manifest_path),
        "partition_manifest": _sha256(resolved.config.data.partitions_path),
        "checkpoint": _sha256(run_dir / "weights.pt"),
        "normalizer": _sha256(run_dir / "normalizer.npz"),
        "policy": "none" if threshold is None else f"threshold:{threshold:.17g}",
        "schedule": "none" if schedule is None else str(schedule["schedule_hash"]),
    }
    if calibration_path is not None:
        identity["calibration"] = _sha256(Path(calibration_path))
    if policy_path is not None:
        identity["policy"] = _sha256(Path(policy_path))
    saved_run_id = run_config_id(run_dir)
    engine = InferenceEngine(
        source_order,
        batch_builder,
        predict_fn,
        len(class_map),
        max_source_age=resolved.config.runtime.max_source_age,
        temperature=temperature,
        threshold=threshold,
        max_active_sequences=resolved.config.runtime.max_active_sequences,
        max_observations_per_source=resolved.config.runtime.max_observations_per_source,
        max_buffer_bytes=resolved.config.runtime.max_buffer_bytes,
        session_id=f"{saved_run_id}:{scenario_id}",
        run_id=saved_run_id,
        scenario_id=scenario_id,
        artifact_hashes=identity,
    )
    arrivals = _merge_arrivals(sequence_id, sequence_sources)
    decision_rows = sequence_decisions.to_dict(orient="records")
    sink = ReplayOutputSink(output_dir)
    try:
        for prediction in replay(engine, arrivals, decision_rows):
            sink.append(prediction)
        return sink.finalize(expected_count=len(decision_rows))
    except BaseException as error:
        sink.abort(error)
        raise


def _merge_arrivals(
    sequence_id: str, sources: Mapping[str, Mapping[str, np.ndarray]]
) -> Iterator[dict[str, Any]]:
    streams = []
    for source_id, arrays in sources.items():
        order = np.lexsort((arrays["observation_ids"], arrays["available_at"]))

        def stream(current_source: str = source_id, current_arrays=arrays, current_order=order):
            for index in current_order:
                yield {
                    "sequence_id": sequence_id,
                    "source_id": current_source,
                    "observation_id": int(current_arrays["observation_ids"][index]),
                    "values": current_arrays["values"][index],
                    "valid_mask": current_arrays["valid_mask"][index],
                    "event_start": float(current_arrays["event_start"][index]),
                    "event_end": float(current_arrays["event_end"][index]),
                    "available_at": float(current_arrays["available_at"][index]),
                }

        streams.append(stream())
    return heapq.merge(
        *streams, key=lambda row: (row["available_at"], row["source_id"], row["observation_id"])
    )


def _to_device(batch: SourceBatch, device: torch.device) -> SourceBatch:
    return SourceBatch(
        batch.values.to(device),
        batch.feature_mask.to(device),
        batch.time_features.to(device),
        batch.lengths.to(device),
        batch.padding_mask.to(device),
        batch.present.to(device),
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


replay_v0_sequence = replay_sequence
