"""Streamlit reviewer over saved artifacts; it never trains or recomputes metrics."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st
import yaml


def _safe_child(root: Path, child_name: str) -> Path | None:
    child = (root / child_name).resolve()
    return child if child.is_relative_to(root.resolve()) else None


def _completed_artifacts(root: Path, marker: str) -> dict[str, Path]:
    if not root.is_dir():
        return {}
    root = root.resolve()
    runs = {}
    for marker_path in sorted(root.rglob(marker)):
        path = marker_path.parent
        relative = path.relative_to(root)
        if any(part.startswith(".") or ".incomplete-" in part for part in relative.parts):
            continue
        if not path.resolve().is_relative_to(root) or not marker_path.resolve().is_relative_to(
            root
        ):
            continue
        status_path = path / "status.json"
        if not status_path.is_file():
            continue
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
            if not isinstance(status, dict):
                continue
            if marker == "metrics.json" and not isinstance(
                json.loads(marker_path.read_text()), dict
            ):
                continue
        except (OSError, json.JSONDecodeError):
            continue
        if status.get("status") == "completed":
            runs[relative.as_posix() if relative.parts else path.name] = path
    return runs


def _completed_runs(root: Path) -> dict[str, Path]:
    return _completed_artifacts(root, "metrics.json")


def _completed_replays(root: Path) -> dict[str, Path]:
    return _completed_artifacts(root, "predictions.jsonl")


def _prepared_datasets(root: Path) -> dict[str, Path]:
    if not root.is_dir():
        return {}
    root = root.resolve()
    candidates = {}
    for manifest in sorted(root.rglob("manifest.json")):
        path = manifest.parent
        relative = path.relative_to(root)
        if any(
            part.startswith(".") or ".incomplete-" in part for part in relative.parts
        ) or not path.resolve().is_relative_to(root):
            continue
        payload, error = _read_manifest(path)
        if not error and payload is not None and payload.get("source_dims"):
            try:
                indices = sorted(int(value) for value in payload["class_map"].values())
                for key in payload["class_map"]:
                    int(key)
            except (KeyError, ValueError, TypeError):
                continue
            if indices == list(range(len(indices))) and len(indices) > 1:
                candidates[relative.as_posix() if relative.parts else path.name] = path
    return candidates


def _load_metrics(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))


def _load_predictions(run_dir: Path) -> pd.DataFrame:
    csv_path = run_dir / "predictions.csv"
    parquet_path = run_dir / "predictions.parquet"
    if csv_path.is_file():
        return pd.read_csv(csv_path)
    if parquet_path.is_file():
        return pd.read_parquet(parquet_path)
    jsonl_path = run_dir / "predictions.jsonl"
    if jsonl_path.is_file():
        with jsonl_path.open(encoding="utf-8") as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
        return pd.DataFrame(rows)
    return pd.DataFrame()


def _load_selected_ids(
    run_dir: Path, sequence_id: str | None = None
) -> dict[str, dict[str, list[int]]]:
    trace_path = run_dir / "selected_ids.jsonl"
    if not trace_path.is_file():
        return {}
    traces = {}
    with trace_path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            if sequence_id is not None and record.get("sequence_id") != sequence_id:
                continue
            traces[str(record["decision_id"])] = record.get("selected_ids", {})
    return traces


def _read_manifest(prepared_root: Path) -> tuple[dict[str, Any] | None, str | None]:
    manifest_path = prepared_root / "manifest.json"
    if not manifest_path.is_file():
        return None, None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            return None, "Prepared manifest must be a JSON object."
        return manifest, None
    except (OSError, json.JSONDecodeError) as error:
        return None, str(error)


def _replay_command(
    project_config: str,
    run_dir: Path,
    prediction: dict[str, Any],
    output_dir: str,
    schedule_path: str | None = None,
    calibration_path: Path | None = None,
    policy_path: Path | None = None,
) -> str | None:
    scenario_id = str(prediction.get("scenario_id") or "clean")
    if scenario_id != "clean" and not schedule_path:
        return None
    args = [
        "mmt",
        "replay",
        "--config",
        project_config,
        "--run-dir",
        str(run_dir),
        "--sequence",
        str(prediction.get("sequence_id", "")),
        "--scenario",
        scenario_id,
        "--output-dir",
        output_dir,
    ]
    if schedule_path:
        args.extend(["--schedule", schedule_path])
    if calibration_path is not None:
        args.extend(["--calibration", str(calibration_path)])
    if policy_path is not None:
        args.extend(["--policy", str(policy_path)])
    return shlex.join(args)


def _startup_paths() -> tuple[str, str]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--prepared-root", default="data/prepared")
    options, _ = parser.parse_known_args(sys.argv[1:])
    return options.runs_root, options.prepared_root


def _checkpoint_config(runs_root: Path, run_id: str) -> tuple[Path | None, Path | None]:
    root = next(
        (path for path in [runs_root, *runs_root.parents] if (path / "pyproject.toml").is_file()),
        None,
    )
    if root is None:
        return None, None
    checkpoint = _safe_child(root / "runs", run_id)
    if checkpoint is None or not (checkpoint / "resolved_config.yaml").is_file():
        return None, None
    try:
        config = yaml.safe_load((checkpoint / "resolved_config.yaml").read_text())
        application = config["data"]["application"]
        model = config["run"]["model_name"]
        candidate = root / "configs" / f"{application}-{model}-seed-11.yaml"
        return (candidate if candidate.is_file() else None), checkpoint
    except (KeyError, TypeError, ValueError, OSError, yaml.YAMLError):
        return None, None


def _replay_export_rows(predictions: pd.DataFrame, traces: dict) -> list[dict[str, Any]]:
    hidden = {
        "target_class",
        "target_time",
        "score_eligible",
        "exclusion_reason",
        "group_id",
        "raw_label",
        "raw_sentiment",
    }
    output = []
    for row in predictions.to_dict(orient="records"):
        record = {key: _display_value(value) for key, value in row.items() if key not in hidden}
        record["selected_ids"] = traces.get(str(record.get("decision_id")), {})
        output.append(record)
    return output


def _postprocessing_for_run(runs_root: Path, run_id: str) -> tuple[Path | None, Path | None]:
    candidates = []
    for path in sorted(runs_root.rglob("calibration.json")):
        if any(
            part.startswith(".") or ".incomplete-" in part
            for part in path.relative_to(runs_root).parts
        ):
            continue
        try:
            artifact = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if artifact.get("identity", {}).get("run_id") == run_id:
            candidates.append(path)
    if len(candidates) != 1:
        return None, None
    calibration = candidates[0]
    policy = calibration.parent / "policy.json"
    return calibration, policy if policy.is_file() else None


def _move_replay(index_key: str, delta: int | None, last_index: int) -> None:
    current = int(st.session_state.get(index_key, 0))
    st.session_state[index_key] = 0 if delta is None else max(0, min(current + delta, last_index))


def main() -> None:
    st.set_page_config(page_title="MMT Reviewer", layout="wide")
    st.title("MMT Reviewer")
    default_runs_root, default_prepared_root = _startup_paths()
    runs_root = (
        Path(st.sidebar.text_input("Runs directory", default_runs_root)).expanduser().resolve()
    )
    project_config = st.sidebar.text_input(
        "Config override", "", placeholder="Automatic from checkpoint"
    )
    prepared_root_base = Path(st.sidebar.text_input("Prepared dataset root", default_prepared_root))
    prepared_root_base = prepared_root_base.expanduser().resolve()
    prepared_candidates = _prepared_datasets(prepared_root_base)
    if (prepared_root_base / "manifest.json").is_file():
        prepared_root = prepared_root_base
    elif prepared_candidates:
        preparation = st.sidebar.selectbox("Preparation", list(prepared_candidates))
        prepared_root = prepared_candidates[preparation]
    else:
        prepared_root = prepared_root_base
    run_dirs = _completed_runs(runs_root)
    replay_dirs = _completed_replays(runs_root)
    tab_audit, tab_replay, tab_compare, tab_export = st.tabs(
        ["Data audit", "Replay", "Comparison", "Export"]
    )

    with tab_audit:
        manifest, error = _read_manifest(prepared_root)
        if error:
            st.error(error)
        elif manifest is None:
            st.info("No prepared manifest at the selected path.")
        else:
            st.write(
                {
                    "application": manifest.get("application", manifest.get("dataset")),
                    "schema_version": manifest.get("schema_version"),
                    "preparation_id": manifest.get("preparation_id"),
                    "availability_mode": manifest.get("availability_mode"),
                }
            )
            st.subheader("Sources")
            st.dataframe(
                pd.DataFrame(
                    [
                        {"source": source, "features": dimension}
                        for source, dimension in manifest.get("source_dims", {}).items()
                    ]
                ),
                hide_index=True,
                width="stretch",
            )
            st.subheader("Sequence audit")
            rows = []
            for sequence_id, sources in manifest.get("sequence_stats", {}).items():
                for source_id, stats in sources.items():
                    rows.append({"sequence": sequence_id, "source": source_id, **stats})
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    with tab_replay:
        if not replay_dirs:
            st.info("No completed replay runs found.")
        else:
            run_id = st.selectbox("Replay run", list(replay_dirs))
            run_dir = replay_dirs[run_id]
            predictions = _load_predictions(run_dir)
            if predictions.empty:
                st.warning("This run has no prediction export.")
            else:
                if "run_id" in predictions and predictions.run_id.nunique() > 1:
                    checkpoint_id = st.selectbox(
                        "Replay checkpoint", sorted(predictions.run_id.dropna().unique())
                    )
                    predictions = predictions[predictions.run_id == checkpoint_id]
                sequence_ids = sorted(predictions.sequence_id.dropna().astype(str).unique())
                sequence_id = st.selectbox("Sequence", sequence_ids)
                rows = predictions[predictions.sequence_id.astype(str) == sequence_id].copy()
                if "decision_time" in rows:
                    rows = rows.sort_values("decision_time", kind="stable").reset_index(drop=True)
                index_key = (
                    f"replay-index:{run_id}:{sequence_id}:{predictions.iloc[0].get('run_id', '')}"
                )
                st.session_state.setdefault(index_key, 0)
                st.session_state[index_key] = min(
                    st.session_state[index_key], max(len(rows) - 1, 0)
                )
                previous, next_step, reset = st.columns(3)
                previous.button(
                    "Previous",
                    disabled=st.session_state[index_key] == 0,
                    on_click=_move_replay,
                    args=(index_key, -1, len(rows) - 1),
                    icon=":material/chevron_left:",
                    help="Previous decision",
                )
                next_step.button(
                    "Next",
                    disabled=st.session_state[index_key] >= len(rows) - 1,
                    on_click=_move_replay,
                    args=(index_key, 1, len(rows) - 1),
                    icon=":material/chevron_right:",
                    help="Next decision",
                )
                reset.button(
                    "Reset",
                    on_click=_move_replay,
                    args=(index_key, None, len(rows) - 1),
                    icon=":material/restart_alt:",
                    help="First decision",
                )
                row = rows.iloc[st.session_state[index_key]].to_dict()
                hidden_targets = {
                    "target_class",
                    "target_time",
                    "score_eligible",
                    "exclusion_reason",
                    "group_id",
                    "raw_label",
                    "raw_sentiment",
                }
                visible = {
                    key: _display_value(value)
                    for key, value in row.items()
                    if key not in hidden_targets and key != "scores"
                }
                traces = _load_selected_ids(run_dir, sequence_id)
                st.json({**visible, "selected_ids": traces.get(str(row["decision_id"]), {})})
                if {"decision_time", "confidence"} <= set(rows.columns):
                    chart_data = rows[["decision_time", "confidence"]].dropna()
                    if not chart_data.empty:
                        st.plotly_chart(
                            px.line(chart_data, x="decision_time", y="confidence"),
                            width="stretch",
                        )

    with tab_compare:
        comparisons = []
        for run_id, run_dir in run_dirs.items():
            metrics = _load_metrics(run_dir)
            counts, coverage = metrics.get("counts", {}), metrics.get("coverage", {})
            quality, probability = metrics.get("quality_S", {}), metrics.get("probability", {})
            try:
                provenance = json.loads((run_dir / "provenance.json").read_text())
            except (OSError, json.JSONDecodeError):
                provenance = {}
            comparisons.append(
                {
                    "run": run_id,
                    "application": "mosi"
                    if "mosi" in run_id.lower()
                    else "pamap2"
                    if "pamap2" in run_id.lower()
                    else "unknown",
                    "partition": provenance.get("partition", "unknown"),
                    "scenario": provenance.get("scenario_id", "unknown"),
                    "eligible": counts.get("eligible"),
                    "computable": counts.get("computable"),
                    "availability_coverage": coverage.get("availability_coverage"),
                    "macro_f1": quality.get("macro_f1"),
                    "balanced_accuracy": quality.get("balanced_accuracy"),
                    "nll": probability.get("nll"),
                    "brier": probability.get("brier"),
                    "ece": probability.get("ece"),
                    "selective_risk": coverage.get("selective_risk"),
                }
            )
        comparison_frame = pd.DataFrame(comparisons)
        if comparison_frame.empty:
            st.info("No completed evaluations found.")
        else:
            filters = st.columns(3)
            for container, column in zip(filters, ("application", "partition", "scenario")):
                selected = container.multiselect(
                    column.title(),
                    sorted(comparison_frame[column].unique()),
                    key=f"compare-{column}",
                )
                if selected:
                    comparison_frame = comparison_frame[comparison_frame[column].isin(selected)]
            st.dataframe(comparison_frame, hide_index=True, width="stretch")
            plotted = comparison_frame.dropna(subset=["macro_f1"])
            if not plotted.empty:
                st.plotly_chart(
                    px.bar(plotted, x="run", y="macro_f1", color="application"), width="stretch"
                )

    with tab_export:
        if run_dirs:
            run_id = st.selectbox("Run to inspect", list(run_dirs), key="export-run")
            run_dir = run_dirs[run_id]
            status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
            metrics = _load_metrics(run_dir)
            st.json(
                {
                    "run_id": run_id,
                    "status": status,
                    "metrics_schema_version": metrics.get("schema_version"),
                    "files": sorted(path.name for path in run_dir.iterdir() if path.is_file()),
                }
            )
            resolved = run_dir / "resolved_config.yaml"
            if not resolved.is_file():
                try:
                    provenance = json.loads((run_dir / "provenance.json").read_text())
                    _, checkpoint = _checkpoint_config(runs_root, str(provenance.get("run_id", "")))
                    if checkpoint is not None:
                        resolved = checkpoint / "resolved_config.yaml"
                except (OSError, json.JSONDecodeError):
                    pass
            if resolved.is_file():
                st.download_button(
                    "Download resolved config",
                    resolved.read_bytes(),
                    file_name=f"{run_id}-resolved.yaml",
                )
        if replay_dirs:
            replay_id = st.selectbox("Replay to export", list(replay_dirs), key="export-replay")
            replay_dir = replay_dirs[replay_id]
            predictions = _load_predictions(replay_dir)
            if "run_id" in predictions:
                selected_run = st.selectbox(
                    "Export checkpoint", sorted(predictions.run_id.dropna().unique())
                )
                predictions = predictions[predictions.run_id == selected_run]
            selected_sequence = st.selectbox(
                "Export sequence", sorted(predictions.sequence_id.dropna().unique())
            )
            predictions = predictions[predictions.sequence_id == selected_sequence]
            traces = _load_selected_ids(replay_dir, selected_sequence)
            exported_rows = _replay_export_rows(predictions, traces)
            payload = {
                "replay_id": replay_id,
                "predictions": exported_rows,
            }
            st.download_button(
                "Download replay trace",
                json.dumps(payload, sort_keys=True, indent=2, allow_nan=False),
                file_name=f"{replay_dir.name}-{selected_sequence}-trace.json",
                mime="application/json",
            )
            if exported_rows:
                first = exported_rows[0]
                output_dir = st.text_input(
                    "New replay output", f"runs/ui-replay-{replay_dir.name}-{selected_sequence}"
                )
                scenario_id = str(first.get("scenario_id") or "clean")
                schedule_path = None
                if scenario_id != "clean":
                    schedule_path = (
                        st.text_input(
                            "Saved scenario schedule", "", key=f"schedule-path:{replay_id}"
                        )
                        or None
                    )
                automatic_config, checkpoint = _checkpoint_config(
                    runs_root, str(first.get("run_id", ""))
                )
                calibration_path, policy_path = _postprocessing_for_run(
                    runs_root, str(first.get("run_id", ""))
                )
                command = (
                    None
                    if checkpoint is None or (not project_config and automatic_config is None)
                    else _replay_command(
                        project_config or str(automatic_config),
                        checkpoint,
                        first,
                        output_dir,
                        schedule_path,
                        calibration_path,
                        policy_path,
                    )
                )
                if command is None:
                    st.warning("A matching checkpoint/config and saved schedule are required.")
                else:
                    st.code(command, language="sh")


def _display_value(value: Any) -> Any:
    if isinstance(value, float) and pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


if __name__ == "__main__":
    main()
