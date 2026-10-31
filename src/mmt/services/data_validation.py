"""Cross-member validation services over prepared dataset artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from mmt.evaluation.protocol import validate_protocol
from mmt.storage import PreparedError, load_manifest


def validate_prepared_splits(
    manifest_path: str | Path, partitions_path: str | Path | None = None
) -> dict[str, Any]:
    """Validate stored membership, decisions, targets and class map before training/evaluation."""
    manifest_path = Path(manifest_path).expanduser().resolve()
    prepared_root = manifest_path.parent
    manifest = load_manifest(prepared_root, verify=True)
    partitions_path = (
        Path(partitions_path).expanduser().resolve()
        if partitions_path is not None
        else prepared_root / "partitions.csv"
    )
    decisions_path = prepared_root / "decisions.csv"
    targets_path = prepared_root / "targets.csv"
    missing = [
        path for path in (partitions_path, decisions_path, targets_path) if not path.is_file()
    ]
    if missing:
        raise PreparedError(f"prepared split inputs are missing: {[str(path) for path in missing]}")
    membership = pd.read_csv(partitions_path)
    decisions = pd.read_csv(decisions_path)
    targets = pd.read_csv(targets_path)
    registry = decisions.merge(
        targets, on="decision_id", how="outer", validate="one_to_one", indicator=True
    )
    unmatched = registry[registry._merge != "both"]
    if len(unmatched):
        raise PreparedError("decisions.csv and targets.csv do not have exactly matching IDs")
    registry = registry.drop(columns="_merge")
    class_map = manifest.get("class_map")
    if not isinstance(class_map, dict) or not class_map:
        raise PreparedError("prepared manifest must contain an audited class_map")
    try:
        class_map = {int(raw): int(index) for raw, index in class_map.items()}
    except (TypeError, ValueError) as error:
        raise PreparedError("class_map keys and values must be integer-convertible") from error
    report = validate_protocol(membership, registry, class_map)
    report["manifest_hash"] = _sha256(manifest_path)
    report["partitions_hash"] = _sha256(partitions_path)
    report["decisions_hash"] = _sha256(decisions_path)
    report["targets_hash"] = _sha256(targets_path)
    return report


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()
