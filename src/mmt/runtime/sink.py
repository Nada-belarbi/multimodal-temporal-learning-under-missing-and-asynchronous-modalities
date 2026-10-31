"""Incremental replay output with explicit incomplete status and atomic completion."""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .engine import InferenceEngine, PredictionRecord
from .replay import replay


class ReplayOutputSink:
    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir).resolve()
        if self.output_dir.exists():
            raise FileExistsError(f"replay output already exists: {self.output_dir}")
        self.output_dir.parent.mkdir(parents=True, exist_ok=True)
        self.temporary_dir = self.output_dir.parent / (
            f".{self.output_dir.name}.incomplete-{uuid.uuid4().hex[:10]}"
        )
        self.temporary_dir.mkdir()
        self.predictions_path = self.temporary_dir / "predictions.jsonl"
        self.traces_path = self.temporary_dir / "selected_ids.jsonl"
        self.row_count = 0
        self._closed = False
        self._write_status("incomplete", expected=None, error=None)

    def append(self, prediction: PredictionRecord) -> None:
        if self._closed:
            raise RuntimeError("replay sink is closed")
        prediction_payload = asdict(prediction)
        trace_payload = {
            "run_id": prediction.run_id,
            "scenario_id": prediction.scenario_id,
            "sequence_id": prediction.sequence_id,
            "decision_id": prediction.decision_id,
            "selected_ids": prediction.selected_ids,
        }
        try:
            self._append_json_line(self.predictions_path, prediction_payload)
            self._append_json_line(self.traces_path, trace_payload)
        except BaseException as error:
            self._write_status(
                "incomplete", expected=None, error=f"{type(error).__name__}: {error}"
            )
            raise
        self.row_count += 1
        self._write_status("incomplete", expected=None, error=None)

    def finalize(self, expected_count: int | None = None) -> Path:
        if self._closed:
            raise RuntimeError("replay sink is closed")
        if expected_count is not None and self.row_count != expected_count:
            error = f"expected {expected_count} predictions, wrote {self.row_count}"
            self._write_status("incomplete", expected=expected_count, error=error)
            raise ValueError(error)
        self._write_status("completed", expected=expected_count, error=None)
        os.replace(self.temporary_dir, self.output_dir)
        self._closed = True
        return self.output_dir

    def abort(self, error: BaseException) -> Path:
        if self._closed:
            return self.temporary_dir
        self._write_status("incomplete", expected=None, error=f"{type(error).__name__}: {error}")
        self._closed = True
        return self.temporary_dir

    @staticmethod
    def _append_json_line(path: Path, payload: dict[str, Any]) -> None:
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(serialized + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def _write_status(self, status: str, expected: int | None, error: str | None) -> None:
        payload = {
            "status": status,
            "rows_written": self.row_count,
            "expected_rows": expected,
            "error": error,
        }
        temporary_status = self.temporary_dir / ".status.json.tmp"
        temporary_status.write_text(
            json.dumps(payload, sort_keys=True, allow_nan=False), encoding="utf-8"
        )
        os.replace(temporary_status, self.temporary_dir / "status.json")


def replay_to_directory(
    engine: InferenceEngine,
    arrivals: Any,
    decisions: Any,
    output_dir: str | Path,
    expected_count: int | None = None,
) -> Path:
    """Run replay incrementally, publishing only complete output and retaining failures."""
    sink = ReplayOutputSink(output_dir)
    try:
        for prediction in replay(engine, arrivals, decisions):
            sink.append(prediction)
        return sink.finalize(expected_count)
    except BaseException as error:
        sink.abort(error)
        raise
