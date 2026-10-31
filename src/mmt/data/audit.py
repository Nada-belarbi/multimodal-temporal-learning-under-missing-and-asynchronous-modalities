"""Dataset file inventory and atomic audit evidence output."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence


class AuditError(ValueError):
    """Raised when dataset inventory inputs or audit evidence are incomplete."""


INVENTORY_FIELDS = (
    "relative_path",
    "bytes",
    "sha256",
    "provider_reference",
    "retrieved_at",
    "declared_version",
    "reader_status",
)


def inventory_dataset_files(
    data_root: str | Path, inputs: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Hash explicitly inventoried files; never recursively discover or download data."""
    root = Path(data_root).expanduser().resolve()
    if not root.is_dir():
        raise AuditError(f"data root does not exist: {root}")
    if not inputs:
        raise AuditError("at least one explicitly inventoried input is required")
    entries = []
    seen: set[str] = set()
    for item in inputs:
        relative_name = item.get("relative_path")
        if not isinstance(relative_name, str) or not relative_name:
            raise AuditError("each input needs a relative_path")
        relative_path = Path(relative_name)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise AuditError(f"unsafe dataset path {relative_name!r}")
        portable_name = relative_path.as_posix()
        if portable_name in seen:
            raise AuditError(f"duplicate inventory path {portable_name!r}")
        seen.add(portable_name)
        path = (root / relative_path).resolve()
        if not path.is_relative_to(root):
            raise AuditError(f"dataset path escapes data root: {portable_name!r}")
        if not path.is_file():
            raise AuditError(f"inventoried file does not exist: {portable_name!r}")

        provider = item.get("provider_reference")
        retrieved_at = item.get("retrieved_at")
        declared_version = item.get("declared_version")
        reader_status = item.get("reader_status")
        for name, value in (
            ("provider_reference", provider),
            ("retrieved_at", retrieved_at),
            ("declared_version", declared_version),
            ("reader_status", reader_status),
        ):
            if not isinstance(value, str) or not value.strip():
                raise AuditError(f"{portable_name}: {name} must be declared and nonempty")
        try:
            datetime.fromisoformat(retrieved_at.replace("Z", "+00:00"))
        except ValueError as error:
            raise AuditError(f"{portable_name}: retrieved_at must be ISO 8601") from error

        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b""):
                digest.update(chunk)
        entries.append(
            {
                "relative_path": portable_name,
                "bytes": path.stat().st_size,
                "sha256": digest.hexdigest(),
                "provider_reference": provider,
                "retrieved_at": retrieved_at,
                "declared_version": declared_version,
                "reader_status": reader_status,
            }
        )
    return sorted(entries, key=lambda entry: entry["relative_path"])


def write_audit_bundle(
    output_dir: str | Path, inventory: Sequence[Mapping[str, Any]], audit: Mapping[str, Any]
) -> Path:
    """Publish inventory.csv and audit.json together without overwriting prior evidence."""
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"audit output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=destination.name + ".tmp-", dir=destination.parent))
    try:
        with (temporary / "inventory.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=INVENTORY_FIELDS, extrasaction="raise")
            writer.writeheader()
            writer.writerows(inventory)
        (temporary / "audit.json").write_text(
            json.dumps(audit, sort_keys=True, indent=2, allow_nan=False), encoding="utf-8"
        )
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination
