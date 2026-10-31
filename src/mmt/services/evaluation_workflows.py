"""Calibration and validation-only policy selection over saved evaluation exports."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from mmt.evaluation.abstention import save_policy, select_policy
from mmt.evaluation.calibration import fit_temperature, save_calibration
from mmt.evaluation.metrics import log_probs


class EvaluationWorkflowError(ValueError):
    """Raised when an evaluation export has the wrong partition or artifact identity."""


def calibrate_saved_scores(
    run_dir: str | Path, evaluation_dir: str | Path, output_path: str | Path
) -> dict[str, Any]:
    evaluation_dir = Path(evaluation_dir).resolve()
    run_dir = Path(run_dir).resolve()
    provenance, predictions = _load_evaluation(evaluation_dir, "calibration")
    run_id = _load_run_id(run_dir)
    if provenance.get("run_id") != run_id:
        raise EvaluationWorkflowError("calibration export belongs to a different run")
    eligible = predictions.score_eligible.astype(bool) & predictions.computable.astype(bool)
    rows = predictions[eligible]
    if rows.empty or rows.target_class.isna().any() or rows.scores.isna().any():
        raise EvaluationWorkflowError(
            "calibration partition needs eligible computable scores and labels"
        )
    scores = np.vstack([_parse_scores(value) for value in rows.scores])
    labels = rows.target_class.to_numpy(dtype=np.int64)
    identity = _artifact_identity(run_dir, provenance)
    score_kind = str(provenance.get("score_kind") or "logits")
    artifact = fit_temperature(scores, labels, scores.shape[1], identity, score_kind)
    save_calibration(output_path, artifact)
    return artifact


def select_saved_policy(
    run_dir: str | Path,
    validation_dir: str | Path,
    calibration_path: str | Path,
    output_path: str | Path,
    *,
    risk_limit: float | None = None,
    minimum_accepted_count: int | None = None,
    minimum_accepted_coverage: float | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    run_dir = Path(run_dir).resolve()
    provenance, predictions = _load_evaluation(Path(validation_dir).resolve(), "validation")
    if provenance.get("run_id") != _load_run_id(run_dir):
        raise EvaluationWorkflowError("validation export belongs to a different run")
    calibration = json.loads(Path(calibration_path).read_text(encoding="utf-8"))
    identity = _artifact_identity(run_dir, provenance)
    for key in ("run_id", "checkpoint_sha256", "class_map_sha256"):
        if calibration.get("identity", {}).get(key) != identity.get(key):
            raise EvaluationWorkflowError(f"calibration artifact identity differs on {key}")
    if calibration.get("status") not in {"ok", "fit_failed"}:
        raise EvaluationWorkflowError("calibration artifact has an unsupported status")
    temperature = float(calibration.get("T", 1.0))
    eligible = predictions.score_eligible.astype(bool)
    computable = predictions.computable.astype(bool)
    rows = predictions[eligible & computable]
    if rows.empty or rows.target_class.isna().any() or rows.scores.isna().any():
        raise EvaluationWorkflowError("validation export needs computable scores and labels")
    scores = np.vstack([_parse_scores(value) for value in rows.scores])
    probabilities = np.exp(log_probs(scores, temperature))
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == rows.target_class.to_numpy(dtype=np.int64)
    policy_identity = {
        **identity,
        "calibration_sha256": _sha256(Path(calibration_path)),
        "validation_score_hash": _hash_scores(scores, rows.decision_id.astype(str).tolist()),
        "calibration_status": calibration["status"],
    }
    policy, curve = select_policy(
        confidence,
        correct,
        int(eligible.sum()),
        risk_limit,
        minimum_accepted_count,
        minimum_accepted_coverage,
        identity=policy_identity,
    )
    save_policy(output_path, policy)
    return policy, curve


def _load_evaluation(
    directory: Path, required_partition: str
) -> tuple[dict[str, Any], pd.DataFrame]:
    provenance_path = directory / "provenance.json"
    if not provenance_path.is_file():
        raise EvaluationWorkflowError(f"missing evaluation provenance: {provenance_path}")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if provenance.get("partition") != required_partition:
        raise EvaluationWorkflowError(
            f"workflow requires partition={required_partition!r}, got {provenance.get('partition')!r}"
        )
    csv_path = directory / "predictions.csv"
    parquet_path = directory / "predictions.parquet"
    if csv_path.is_file():
        predictions = pd.read_csv(csv_path)
    elif parquet_path.is_file():
        predictions = pd.read_parquet(parquet_path)
    else:
        raise EvaluationWorkflowError(
            "evaluation export needs predictions.csv or predictions.parquet"
        )
    required = {"decision_id", "score_eligible", "computable", "scores", "target_class"}
    if not required <= set(predictions.columns):
        raise EvaluationWorkflowError(
            f"prediction export missing columns {sorted(required - set(predictions.columns))}"
        )
    return provenance, predictions


def load_saved_postprocessing(
    run_dir: Path,
    input_hashes: dict[str, str],
    score_kind: str,
    calibration_path: str | Path | None,
    policy_path: str | Path | None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Load immutable postprocessing only when its saved identities match this run/data."""
    if policy_path is not None and calibration_path is None:
        raise EvaluationWorkflowError("a saved policy requires its saved calibration artifact")
    if calibration_path is None:
        return None, None
    identity = _artifact_identity(run_dir, {"identities": input_hashes})
    calibration = json.loads(Path(calibration_path).read_text(encoding="utf-8"))
    for key, value in identity.items():
        if calibration.get("identity", {}).get(key) != value:
            raise EvaluationWorkflowError(f"calibration artifact identity differs on {key}")
    if calibration.get("score_kind") != score_kind:
        raise EvaluationWorkflowError("calibration score_kind differs from the model")
    if calibration.get("status") not in {"ok", "fit_failed"}:
        raise EvaluationWorkflowError("unsupported calibration status")
    temperature = float(calibration.get("T", 0.0))
    if not np.isfinite(temperature) or temperature <= 0:
        raise EvaluationWorkflowError("calibration temperature must be finite and positive")
    if calibration["status"] == "fit_failed" and temperature != 1.0:
        raise EvaluationWorkflowError("failed calibration may only use the explicit T=1 fallback")
    policy = None
    if policy_path is not None:
        policy = json.loads(Path(policy_path).read_text(encoding="utf-8"))
        for key, value in {
            **identity,
            "calibration_sha256": _sha256(Path(calibration_path)),
        }.items():
            if policy.get("identity", {}).get(key) != value:
                raise EvaluationWorkflowError(f"policy artifact identity differs on {key}")
        if policy.get("status") not in {"selected", "disabled_no_constraints"}:
            raise EvaluationWorkflowError("saved policy is not selected or explicitly disabled")
        threshold = policy.get("tau")
        if policy["status"] == "selected" and (
            threshold is None or not np.isfinite(float(threshold)) or not 0 <= float(threshold) <= 1
        ):
            raise EvaluationWorkflowError("selected policy needs a finite threshold in [0,1]")
        if policy["status"] == "disabled_no_constraints" and threshold is not None:
            raise EvaluationWorkflowError("disabled policy cannot declare a threshold")
    return calibration, policy


def _artifact_identity(run_dir: Path, provenance: dict[str, Any]) -> dict[str, str]:
    class_map = json.loads((run_dir / "class_map.json").read_text(encoding="utf-8"))
    class_map_payload = json.dumps(
        class_map, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return {
        "run_id": _load_run_id(run_dir),
        "checkpoint_sha256": _sha256(run_dir / "weights.pt"),
        "class_map_sha256": hashlib.sha256(class_map_payload.encode("utf-8")).hexdigest(),
        "dataset_manifest_hash": str(
            provenance.get("identities", {}).get("data.manifest_path", "")
        ),
        "partition_hash": str(provenance.get("identities", {}).get("data.partitions_path", "")),
    }


def _load_run_id(run_dir: Path) -> str:
    resolved = yaml.safe_load((run_dir / "resolved_config.yaml").read_text(encoding="utf-8"))
    return str(resolved["run"]["run_id"])


def _parse_scores(value: Any) -> np.ndarray:
    if isinstance(value, str):
        scores = np.fromstring(value, sep=";", dtype=np.float64)
    else:
        scores = np.asarray(value, dtype=np.float64)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all():
        raise EvaluationWorkflowError("scores must be finite one-dimensional vectors")
    return scores


def _hash_scores(scores: np.ndarray, decision_ids: list[str]) -> str:
    digest = hashlib.sha256()
    for decision_id, row in zip(decision_ids, scores):
        digest.update(decision_id.encode("utf-8"))
        digest.update(np.asarray(row, dtype=np.float64).tobytes())
    return digest.hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
