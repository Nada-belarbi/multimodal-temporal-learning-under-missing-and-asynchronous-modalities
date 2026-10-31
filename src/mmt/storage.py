"""Validated, content-addressed storage for prepared temporal datasets."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import uuid
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = "1.0"
ARRAYS = ("values", "valid_mask", "observation_ids", "event_start", "event_end", "available_at")
_SAFE_ID = re.compile(r"^[A-Za-z0-9_.-]+$")
_MANIFEST_KEYS = {
    "schema_version",
    "dataset",
    "preparation_id",
    "source_order",
    "source_meta",
    "source_dims",
    "sequence_stats",
    "checksums",
}


class PreparedError(ValueError):
    """Raised when prepared data violates its schema or integrity manifest."""


def _safe_id(value: str, name: str) -> str:
    if not isinstance(value, str) or value in (".", "..") or not _SAFE_ID.fullmatch(value):
        raise PreparedError(f"unsafe {name} {value!r}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_source(
    where: str, arrays: dict[str, Any]
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    missing = set(ARRAYS) - set(arrays)
    if missing:
        raise PreparedError(f"{where}: missing arrays {sorted(missing)}")

    values = np.asarray(arrays["values"], dtype=np.float32)
    mask = np.asarray(arrays["valid_mask"], dtype=bool)
    if values.ndim != 2 or values.shape[1] == 0:
        raise PreparedError(f"{where}: values must have shape [N,D] with D > 0, got {values.shape}")
    if mask.shape != values.shape:
        raise PreparedError(f"{where}: valid_mask {mask.shape} != values {values.shape}")
    if not np.isfinite(values[mask]).all():
        raise PreparedError(f"{where}: non-finite value where valid_mask is True")

    row_count, feature_count = values.shape
    nonfinite = int((~np.isfinite(values)).sum())
    clean = {
        "values": np.where(mask, values, 0.0).astype(np.float32),
        "valid_mask": mask,
    }
    for name in ("event_start", "event_end", "available_at"):
        array = np.asarray(arrays[name], dtype=np.float64)
        if array.shape != (row_count,) or not np.isfinite(array).all():
            raise PreparedError(f"{where}: {name} must be finite with shape ({row_count},)")
        clean[name] = array

    observation_ids = np.asarray(arrays["observation_ids"], dtype=np.int64)
    if observation_ids.shape != (row_count,):
        raise PreparedError(f"{where}: observation_ids must have shape ({row_count},)")
    if len(np.unique(observation_ids)) != row_count:
        raise PreparedError(f"{where}: duplicate observation_ids")
    clean["observation_ids"] = observation_ids

    if (clean["event_end"] < clean["event_start"]).any():
        raise PreparedError(f"{where}: event_end < event_start")
    if (clean["available_at"] < clean["event_end"]).any():
        raise PreparedError(f"{where}: available_at < event_end")

    stats = {
        "n_obs": int(row_count),
        "n_features": int(feature_count),
        "n_valid_entries": int(mask.sum()),
        "n_invalid_entries": int((~mask).sum()),
        "n_nonfinite_replaced": nonfinite,
        "n_all_invalid_rows": int((~mask.any(axis=1)).sum()),
    }
    return clean, stats


def write_prepared(
    root: str | Path,
    dataset: str,
    preparation_id: str,
    sequences: dict[str, dict[str, dict[str, Any]]],
    source_meta: dict[str, dict[str, Any]],
    extra_manifest: dict[str, Any] | None = None,
    tables: dict[str, Any] | None = None,
) -> Path:
    """Validate and atomically publish arrays, tables and a checksum manifest."""
    dataset = _safe_id(dataset, "dataset")
    preparation_id = _safe_id(preparation_id, "preparation_id")
    if not isinstance(source_meta, dict) or not source_meta:
        raise PreparedError("source_meta must be a nonempty mapping")
    for source_id in source_meta:
        _safe_id(source_id, "source_id")
    extra_manifest = extra_manifest or {}
    if not isinstance(extra_manifest, dict):
        raise PreparedError("extra_manifest must be a mapping")
    collisions = set(extra_manifest) & _MANIFEST_KEYS
    if collisions:
        raise PreparedError(f"extra_manifest cannot override reserved keys {sorted(collisions)}")

    table_names = tuple((tables or {}).keys())
    for name in table_names:
        if (
            not isinstance(name, str)
            or Path(name).name != name
            or name in ("", ".", "..")
            or not name.endswith(".csv")
            or name == "manifest.json"
        ):
            raise PreparedError(f"unsafe table filename {name!r}")

    final = Path(root) / dataset / preparation_id
    if final.exists():
        raise FileExistsError(f"{final} exists; use a new preparation_id")
    final.parent.mkdir(parents=True, exist_ok=True)
    temporary = final.parent / f".{preparation_id}.tmp-{uuid.uuid4().hex[:8]}"
    temporary.mkdir()
    try:
        dimensions: dict[str, int] = {}
        sequence_stats: dict[str, dict[str, dict[str, int]]] = {}
        for sequence_id, sources in sequences.items():
            _safe_id(sequence_id, "sequence_id")
            sequence_stats[sequence_id] = {}
            for source_id, arrays in sources.items():
                _safe_id(source_id, "source_id")
                if source_id not in source_meta:
                    raise PreparedError(f"{sequence_id}: unknown source {source_id!r}")
                clean, stats = _check_source(f"{sequence_id}/{source_id}", arrays)
                dimension = stats["n_features"]
                if dimensions.setdefault(source_id, dimension) != dimension:
                    raise PreparedError(
                        f"{sequence_id}/{source_id}: dimension {dimension} != {dimensions[source_id]}"
                    )
                source_dir = temporary / "sequences" / sequence_id / source_id
                source_dir.mkdir(parents=True)
                for name, array in clean.items():
                    np.save(source_dir / f"{name}.npy", array, allow_pickle=False)
                sequence_stats[sequence_id][source_id] = stats

        for name, table in (tables or {}).items():
            table.to_csv(temporary / name, index=False)

        checksums = {
            path.relative_to(temporary).as_posix(): _sha256(path)
            for path in sorted(temporary.rglob("*"))
            if path.is_file()
        }
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "dataset": dataset,
            "preparation_id": preparation_id,
            "source_order": list(source_meta),
            "source_meta": source_meta,
            "source_dims": dimensions,
            "sequence_stats": sequence_stats,
            "checksums": checksums,
            **extra_manifest,
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8"
        )
        temporary.rename(final)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return final


def load_manifest(base: str | Path, verify: bool = True) -> dict[str, Any]:
    base = Path(base)
    manifest_path = base / "manifest.json"
    if not manifest_path.is_file():
        raise PreparedError(f"{base}: no manifest.json (partial or missing artifact)")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PreparedError(f"{base}: invalid manifest: {error}") from error
    if not isinstance(manifest, dict) or manifest.get("schema_version") != SCHEMA_VERSION:
        version = manifest.get("schema_version") if isinstance(manifest, dict) else None
        raise PreparedError(f"incompatible schema_version {version!r}")
    if verify:
        checksums = manifest.get("checksums")
        if not isinstance(checksums, dict):
            raise PreparedError("manifest checksums must be a mapping")
        base_resolved = base.resolve()
        for relative_name, expected_digest in checksums.items():
            relative = Path(relative_name)
            if relative.is_absolute() or ".." in relative.parts:
                raise PreparedError(f"unsafe checksum path {relative_name!r}")
            path = base / relative
            if not path.resolve().is_relative_to(base_resolved):
                raise PreparedError(f"checksum path escapes prepared root: {relative_name!r}")
            if not path.is_file():
                raise PreparedError(f"missing file {relative_name}")
            if _sha256(path) != expected_digest:
                raise PreparedError(f"checksum mismatch for {relative_name}")
    return manifest


def load_sequence(base: str | Path, sequence_id: str) -> dict[str, dict[str, np.ndarray]]:
    sequence_id = _safe_id(sequence_id, "sequence_id")
    sequence_dir = Path(base) / "sequences" / sequence_id
    if not sequence_dir.is_dir():
        raise PreparedError(f"unknown sequence {sequence_id!r}")
    loaded: dict[str, dict[str, np.ndarray]] = {}
    for source_dir in sorted(sequence_dir.iterdir()):
        if not source_dir.is_dir():
            continue
        source_id = _safe_id(source_dir.name, "source_id")
        try:
            loaded[source_id] = {
                name: np.load(source_dir / f"{name}.npy", mmap_mode="r", allow_pickle=False)
                for name in ARRAYS
            }
        except (OSError, ValueError) as error:
            raise PreparedError(
                f"{sequence_id}/{source_id}: invalid prepared arrays: {error}"
            ) from error
    return loaded
