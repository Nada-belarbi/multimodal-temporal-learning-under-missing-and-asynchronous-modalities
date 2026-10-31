"""Small data-preparation services connecting audited adapters and prepared storage."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from mmt.applications.pamap2 import PAMAP2_CLASS_MAP, PAMAP2_SOURCE_COLUMNS, Pamap2Adapter
from mmt.storage import write_prepared

from .audit import inventory_dataset_files, write_audit_bundle
from .splits import build_pamap2_window_decisions, validate_membership


@dataclass(frozen=True)
class PreparedDatasetResult:
    prepared_path: Path
    audit_dir: Path
    audit: Mapping[str, Any]


@dataclass(frozen=True)
class Pamap2AuditResult:
    audit_dir: Path
    audit: Mapping[str, Any]


def audit_pamap2_dataset(
    raw_root: str | Path,
    audit_root: str | Path,
    recording_paths: Mapping[str, str | Path],
    membership: pd.DataFrame,
    source_provenance: Mapping[str, Mapping[str, str]],
    *,
    chunksize: int = 100_000,
    window_seconds: float = 5.0,
    stride_seconds: float = 1.0,
    max_label_age_seconds: float = 0.02,
) -> Pamap2AuditResult:
    """Audit explicitly listed PAMAP2 inputs without publishing prepared arrays."""
    raw_root = Path(raw_root).expanduser().resolve()
    if not recording_paths or set(recording_paths) != set(membership.sequence_id):
        raise ValueError("recording sequence IDs must exactly match approved membership")
    split_report = validate_membership(list(recording_paths), membership, require_all_splits=True)
    inputs = []
    resolved_paths = {}
    for sequence_id, configured_path in recording_paths.items():
        path = Path(configured_path).expanduser()
        path = (raw_root / path).resolve() if not path.is_absolute() else path.resolve()
        if not path.is_relative_to(raw_root):
            raise ValueError(f"recording path escapes raw_root: {path}")
        provenance = source_provenance.get(sequence_id)
        if provenance is None:
            raise ValueError(f"missing source provenance for {sequence_id!r}")
        inputs.append({"relative_path": path.relative_to(raw_root).as_posix(), **provenance})
        resolved_paths[sequence_id] = path
    inventory = inventory_dataset_files(raw_root, inputs)
    adapter = Pamap2Adapter(chunksize=chunksize)
    recording_audits = {}
    decision_count = 0
    eligible_count = 0
    exclusions: dict[str, int] = {}
    for sequence_id, path in sorted(resolved_paths.items()):
        recording = adapter.read_recording(path, sequence_id)
        recording_audits[sequence_id] = recording.audit
        decisions, targets = build_pamap2_window_decisions(
            sequence_id,
            recording.timestamps,
            recording.raw_labels,
            membership,
            PAMAP2_CLASS_MAP,
            window_seconds,
            stride_seconds,
            max_label_age_seconds,
        )
        decision_count += len(decisions)
        eligible_count += int(targets.score_eligible.sum())
        for reason, count in targets.exclusion_reason.fillna("none").value_counts().items():
            exclusions[str(reason)] = exclusions.get(str(reason), 0) + int(count)
    audit = {
        "application": "pamap2",
        "status": "audit_pending_review",
        "availability_mode": "simulated",
        "class_map": {str(key): value for key, value in PAMAP2_CLASS_MAP.items()},
        "membership": split_report,
        "inventory": inventory,
        "recordings": recording_audits,
        "decision_counts": {
            "scheduled": decision_count,
            "eligible": eligible_count,
            "excluded_by_reason": exclusions,
        },
    }
    return Pamap2AuditResult(write_audit_bundle(audit_root, inventory, audit), audit)


def prepare_pamap2_dataset(
    raw_root: str | Path,
    prepared_root: str | Path,
    audit_root: str | Path,
    recording_paths: Mapping[str, str | Path],
    membership: pd.DataFrame,
    source_provenance: Mapping[str, Mapping[str, str]],
    preparation_id: str,
    *,
    chunksize: int = 100_000,
    window_seconds: float = 5.0,
    stride_seconds: float = 1.0,
    max_label_age_seconds: float = 0.02,
) -> PreparedDatasetResult:
    """Prepare only inventoried PAMAP2 recordings; group membership is always supplied externally."""
    raw_root = Path(raw_root).expanduser().resolve()
    if not recording_paths:
        raise ValueError("recording_paths must explicitly list at least one recording")
    if set(recording_paths) != set(membership.sequence_id):
        raise ValueError("recording sequence IDs must exactly match approved membership")
    split_report = validate_membership(list(recording_paths), membership, require_all_splits=True)
    inputs = []
    resolved_paths: dict[str, Path] = {}
    for sequence_id, configured_path in recording_paths.items():
        path = Path(configured_path).expanduser()
        path = (raw_root / path).resolve() if not path.is_absolute() else path.resolve()
        if not path.is_relative_to(raw_root):
            raise ValueError(f"recording path escapes raw_root: {path}")
        provenance = source_provenance.get(sequence_id)
        if provenance is None:
            raise ValueError(f"missing source provenance for {sequence_id!r}")
        inputs.append({"relative_path": path.relative_to(raw_root).as_posix(), **provenance})
        resolved_paths[sequence_id] = path

    inventory = inventory_dataset_files(raw_root, inputs)
    adapter = Pamap2Adapter(chunksize=chunksize)
    sequences: dict[str, dict[str, dict[str, Any]]] = {}
    decisions = []
    targets = []
    recording_audits = {}
    for sequence_id in sorted(recording_paths):
        recording = adapter.read_recording(resolved_paths[sequence_id], sequence_id)
        sequences[sequence_id] = recording.sources
        recording_audits[sequence_id] = recording.audit
        seq_decisions, seq_targets = build_pamap2_window_decisions(
            sequence_id,
            recording.timestamps,
            recording.raw_labels,
            membership,
            PAMAP2_CLASS_MAP,
            window_seconds,
            stride_seconds,
            max_label_age_seconds,
        )
        decisions.append(seq_decisions)
        targets.append(seq_targets)

    decision_table = pd.concat(decisions, ignore_index=True) if decisions else pd.DataFrame()
    target_table = pd.concat(targets, ignore_index=True) if targets else pd.DataFrame()
    source_meta = {
        source_id: {
            "modality_family": "wearable_sensor",
            "source_columns_zero_based": list(columns),
            "feature_order": [f"column_{column}" for column in columns],
        }
        for source_id, columns in PAMAP2_SOURCE_COLUMNS.items()
    }
    class_map = {str(raw_id): class_index for raw_id, class_index in PAMAP2_CLASS_MAP.items()}
    audit = {
        "application": "pamap2",
        "status": "audit_pending_review",
        "availability_mode": "simulated",
        "class_map": class_map,
        "membership": split_report,
        "inventory": inventory,
        "recordings": recording_audits,
        "decision_counts": {
            "scheduled": int(len(decision_table)),
            "eligible": int(target_table.score_eligible.sum()) if len(target_table) else 0,
            "excluded_by_reason": target_table.exclusion_reason.fillna("none")
            .value_counts()
            .to_dict()
            if len(target_table)
            else {},
        },
    }
    audit_path = write_audit_bundle(audit_root, inventory, audit)
    prepared_path = write_prepared(
        prepared_root,
        "pamap2",
        preparation_id,
        sequences,
        source_meta,
        extra_manifest={
            "application": "pamap2",
            "availability_mode": "simulated",
            "decision_policy": {
                "window_seconds": window_seconds,
                "stride_seconds": stride_seconds,
                "max_label_age_seconds": max_label_age_seconds,
            },
            "class_map": class_map,
            "source_inventory_hashes": {
                entry["relative_path"]: entry["sha256"] for entry in inventory
            },
        },
        tables={
            "partitions.csv": membership,
            "decisions.csv": decision_table,
            "targets.csv": target_table,
        },
    )
    return PreparedDatasetResult(prepared_path, audit_path, audit)
