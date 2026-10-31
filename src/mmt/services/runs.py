"""Atomic, immutable run-directory lifecycle for long-running project workflows."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

_SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9_.-]+$")


class RunLifecycleError(RuntimeError):
    """Raised when an output run violates lifecycle or required-artifact rules."""


class RunDirectory:
    def __init__(self, output_root: str | Path, run_id: str) -> None:
        if (
            not isinstance(run_id, str)
            or run_id in {".", ".."}
            or not _SAFE_RUN_ID.fullmatch(run_id)
        ):
            raise ValueError(f"unsafe run_id {run_id!r}")
        self.output_root = Path(output_root).expanduser().resolve()
        self.run_id = run_id
        self.final_path = self.output_root / run_id
        self.temporary_path: Path | None = None
        self._state = "uninitialized"

    @property
    def path(self) -> Path:
        if self.temporary_path is None:
            raise RunLifecycleError("run directory has not been started")
        return self.temporary_path

    def start(self) -> Path:
        if self._state != "uninitialized":
            raise RunLifecycleError(f"cannot start run from state {self._state!r}")
        if self.final_path.exists():
            raise FileExistsError(f"run {self.run_id!r} already exists and is immutable")
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.temporary_path = Path(
            tempfile.mkdtemp(prefix=f".{self.run_id}.tmp-", dir=self.output_root)
        )
        self._state = "running"
        self._write_status("running", error=None)
        return self.temporary_path

    def complete(self, required_files: Iterable[str] = ()) -> Path:
        self._require_running()
        missing = []
        for name in required_files:
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts:
                raise RunLifecycleError(f"unsafe required artifact path {name!r}")
            path = self.path / relative
            if not path.is_file() or path.stat().st_size == 0:
                missing.append(name)
        if missing:
            raise RunLifecycleError(f"required output artifacts are missing or empty: {missing}")
        self._write_status("completed", error=None)
        self._publish()
        return self.final_path

    def fail(self, error: BaseException, *, incomplete: bool = False) -> Path:
        self._require_running()
        status = "incomplete" if incomplete else "failed"
        self._write_status(
            status,
            error={
                "type": type(error).__name__,
                "message": str(error),
            },
        )
        self._publish()
        return self.final_path

    def _write_status(self, status: str, error: dict[str, str] | None) -> None:
        payload: dict[str, Any] = {
            "schema_version": "1.0",
            "run_id": self.run_id,
            "status": status,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "error": error,
        }
        status_tmp = self.path / ".status.json.tmp"
        status_tmp.write_text(
            json.dumps(payload, sort_keys=True, allow_nan=False, indent=2), encoding="utf-8"
        )
        os.replace(status_tmp, self.path / "status.json")

    def _publish(self) -> None:
        if self.final_path.exists():
            raise FileExistsError(f"run {self.run_id!r} already exists and is immutable")
        os.replace(self.path, self.final_path)
        self._state = "published"

    def _require_running(self) -> None:
        if self._state != "running":
            raise RunLifecycleError(f"run must be running, current state is {self._state!r}")

    def __enter__(self) -> RunDirectory:
        self.start()
        return self

    def __exit__(self, error_type: Any, error: BaseException | None, traceback: Any) -> bool:
        if self._state == "running":
            if error is None:
                self.fail(RunLifecycleError("context exited without complete()"), incomplete=True)
            else:
                self.fail(error, incomplete=True)
        return False


def write_sha256sums(run_path: str | Path, relative_files: Iterable[str]) -> Path:
    root = Path(run_path).resolve()
    entries = {}
    for name in sorted(relative_files):
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise RunLifecycleError(f"unsafe checksum path {name!r}")
        path = root / relative
        if not path.is_file():
            raise RunLifecycleError(f"cannot hash missing artifact {name!r}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        entries[relative.as_posix()] = digest
    output = root / "sha256sums.json"
    output.write_text(json.dumps(entries, sort_keys=True, indent=2), encoding="utf-8")
    return output


def list_runs(output_root: str | Path) -> list[dict[str, Any]]:
    root = Path(output_root).expanduser().resolve()
    if not root.is_dir():
        return []
    runs = []
    for path in sorted(root.iterdir(), key=lambda item: item.name):
        status_path = path / "status.json"
        if not path.is_dir() or not status_path.is_file():
            continue
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        runs.append(
            {
                "run_id": path.name,
                "path": str(path),
                "status": status.get("status"),
                "updated_at_utc": status.get("updated_at_utc"),
                "error": status.get("error"),
            }
        )
    return runs


def verify_run(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir).expanduser().resolve()
    status_path = root / "status.json"
    checksums_path = root / "sha256sums.json"
    if not root.is_dir() or not status_path.is_file() or not checksums_path.is_file():
        raise RunLifecycleError("run requires status.json and sha256sums.json")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    checksums = json.loads(checksums_path.read_text(encoding="utf-8"))
    if not isinstance(checksums, dict) or not checksums:
        raise RunLifecycleError("sha256sums.json must contain at least one artifact")
    mismatches = []
    for name, expected in checksums.items():
        relative = Path(name)
        artifact = (root / relative).resolve()
        if relative.is_absolute() or ".." in relative.parts or not artifact.is_relative_to(root):
            mismatches.append(name)
            continue
        if not artifact.is_file() or hashlib.sha256(artifact.read_bytes()).hexdigest() != expected:
            mismatches.append(name)
    return {
        "run_id": root.name,
        "status": status.get("status"),
        "checked_artifacts": len(checksums),
        "mismatches": mismatches,
        "integrity": "valid" if not mismatches else "invalid",
    }
