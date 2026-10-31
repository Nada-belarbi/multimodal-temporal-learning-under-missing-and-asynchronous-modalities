"""Render reports exclusively from immutable saved evaluation exports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from mmt.evaluation.report import evaluate_acceptance, final_status, render_report


class ReportServiceError(ValueError):
    """Raised when saved evaluation data is incomplete for report rendering."""


def render_saved_report(
    evaluation_dirs: Mapping[str, str | Path],
    output_path: str | Path,
    title: str,
    *,
    acceptance_rules: Sequence[Mapping[str, Any]] | None = None,
    limitations: Sequence[str] = (),
    provenance: Mapping[str, Any] | None = None,
    curve_csv: str | Path | None = None,
) -> Path:
    if not evaluation_dirs:
        raise ReportServiceError("at least one completed evaluation directory is required")
    saved_results = {}
    completed = True
    for scenario_id, directory in evaluation_dirs.items():
        directory = Path(directory).expanduser().resolve()
        metrics_path = directory / "metrics.json"
        status_path = directory / "status.json"
        if not metrics_path.is_file() or not status_path.is_file():
            raise ReportServiceError(f"missing metrics/status export in {directory}")
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("status") != "completed":
            raise ReportServiceError(f"evaluation {scenario_id!r} is not complete")
        completed &= True
        saved_results[scenario_id] = json.loads(metrics_path.read_text(encoding="utf-8"))

    if acceptance_rules is None:
        acceptance_rows: list[dict[str, Any]] = []
        qualification = "not_qualified"
    else:
        acceptance_rows, qualification = evaluate_acceptance(acceptance_rules, saved_results)
    output_path = Path(output_path).expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(f"report output already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    status_record = final_status(completed, qualification)
    return Path(
        render_report(
            {
                scenario_id: Path(directory).expanduser().resolve()
                for scenario_id, directory in evaluation_dirs.items()
            },
            output_path,
            title,
            status_record,
            acceptance_rows,
            limitations,
            provenance,
            None if curve_csv is None else str(curve_csv),
        )
    )
