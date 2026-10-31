"""Initial command-line entry point; workflow commands are added with their services."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import typer

from . import __version__
from .config import ConfigError, resolve_config, resolve_pamap2_preparation
from .data.preparation import audit_pamap2_dataset, prepare_pamap2_dataset
from .services.data_validation import validate_prepared_splits
from .services.evaluation import evaluate_run
from .services.evaluation_workflows import calibrate_saved_scores, select_saved_policy
from .services.reporting import render_saved_report
from .services.runs import list_runs, verify_run
from .services.runtime import replay_sequence
from .services.training import train_prepared

app = typer.Typer(
    name="mmt",
    help="Multimodal temporal learning tools.",
    no_args_is_help=True,
    invoke_without_command=True,
)
config_app = typer.Typer(
    name="config", help="Validate and inspect project configurations.", no_args_is_help=True
)
app.add_typer(config_app, name="config")


@app.callback()
def main(
    version: bool = typer.Option(False, "--version", help="Show the installed package version."),
) -> None:
    if version:
        typer.echo(__version__)
        raise typer.Exit()


@config_app.command("validate")
def config_validate(
    config_path: Path = typer.Option(
        ...,
        "--config",
        exists=True,
        readable=True,
        dir_okay=False,
        help="YAML configuration to validate.",
    ),
    model: str | None = typer.Option(None, "--model", help="Declared model-name override."),
    seed: int | None = typer.Option(None, "--seed", min=0, help="Declared run-seed override."),
) -> None:
    overrides = {}
    if model is not None:
        overrides["run.model_name"] = model
    if seed is not None:
        overrides["run.seed"] = seed
    try:
        resolved = resolve_config(config_path, overrides)
    except ConfigError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=2) from error
    typer.echo(
        json.dumps(
            {
                "status": "valid",
                "config_sha256": resolved.config_sha256,
                "input_hashes": resolved.input_hashes,
            },
            sort_keys=True,
        )
    )


@config_app.command("show")
def config_show(
    config_path: Path = typer.Option(
        ...,
        "--config",
        exists=True,
        readable=True,
        dir_okay=False,
        help="YAML configuration to inspect.",
    ),
    resolved: bool = typer.Option(False, "--resolved", help="Show canonical resolved JSON."),
) -> None:
    try:
        result = resolve_config(config_path)
    except ConfigError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=2) from error
    if resolved:
        typer.echo(result.canonical_json)
    else:
        typer.echo(f"config_sha256: {result.config_sha256}")


@app.command("audit")
def audit_command(
    config_path: Path = typer.Option(
        ..., "--config", exists=True, readable=True, dir_okay=False, help="PAMAP2 preparation YAML."
    ),
) -> None:
    try:
        resolved = resolve_pamap2_preparation(config_path)
        membership = pd.read_csv(resolved.membership_path)
        result = audit_pamap2_dataset(
            resolved.raw_root,
            resolved.audit_root / f"audit-{resolved.config.preparation_id}",
            resolved.recording_paths,
            membership,
            resolved.source_provenance,
            chunksize=resolved.config.chunksize,
            window_seconds=resolved.config.window_seconds,
            stride_seconds=resolved.config.stride_seconds,
            max_label_age_seconds=resolved.config.max_label_age_seconds,
        )
    except (ConfigError, ValueError, OSError) as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=2) from error
    typer.echo(
        json.dumps(
            {
                "status": result.audit["status"],
                "audit_dir": str(result.audit_dir),
                "recordings": len(result.audit["recordings"]),
            },
            sort_keys=True,
        )
    )


@app.command("prepare")
def prepare_command(
    config_path: Path = typer.Option(
        ..., "--config", exists=True, readable=True, dir_okay=False, help="PAMAP2 preparation YAML."
    ),
) -> None:
    try:
        resolved = resolve_pamap2_preparation(config_path)
        membership = pd.read_csv(resolved.membership_path)
        result = prepare_pamap2_dataset(
            resolved.raw_root,
            resolved.prepared_root,
            resolved.audit_root / f"prepare-{resolved.config.preparation_id}",
            resolved.recording_paths,
            membership,
            resolved.source_provenance,
            resolved.config.preparation_id,
            chunksize=resolved.config.chunksize,
            window_seconds=resolved.config.window_seconds,
            stride_seconds=resolved.config.stride_seconds,
            max_label_age_seconds=resolved.config.max_label_age_seconds,
        )
    except (ConfigError, ValueError, OSError) as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=2) from error
    typer.echo(
        json.dumps(
            {
                "status": "prepared",
                "prepared_path": str(result.prepared_path),
                "audit_dir": str(result.audit_dir),
            },
            sort_keys=True,
        )
    )


@app.command("validate-splits")
def validate_splits_command(
    config_path: Path = typer.Option(
        ...,
        "--config",
        exists=True,
        readable=True,
        dir_okay=False,
        help="Project config referencing prepared artifacts.",
    ),
) -> None:
    try:
        resolved = resolve_config(config_path)
        report = validate_prepared_splits(
            resolved.config.data.manifest_path, resolved.config.data.partitions_path
        )
    except (ConfigError, ValueError, OSError, KeyError) as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=2) from error
    typer.echo(json.dumps({"status": "valid", **report}, sort_keys=True))


@app.command("train")
def train_command(
    config_path: Path = typer.Option(
        ...,
        "--config",
        exists=True,
        readable=True,
        dir_okay=False,
        help="Project run configuration.",
    ),
    model: str | None = typer.Option(None, "--model", help="Declared model-name override."),
    seed: int | None = typer.Option(None, "--seed", min=0, help="Declared run-seed override."),
    run_id: str | None = typer.Option(None, "--run-id", help="New immutable run identity."),
    device: str = typer.Option("cpu", "--device", help="Torch execution device, e.g. cpu or mps."),
) -> None:
    overrides = {}
    if model is not None:
        overrides["run.model_name"] = model
    if seed is not None:
        overrides["run.seed"] = seed
    if run_id is not None:
        overrides["run.run_id"] = run_id
    try:
        resolved = resolve_config(config_path, overrides)
        run_path = train_prepared(resolved, device=device)
    except (ConfigError, ValueError, OSError, RuntimeError) as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=2) from error
    typer.echo(json.dumps({"status": "completed", "run_path": str(run_path)}, sort_keys=True))


@app.command("evaluate")
def evaluate_command(
    config_path: Path = typer.Option(
        ...,
        "--config",
        exists=True,
        readable=True,
        dir_okay=False,
        help="Project run/data configuration.",
    ),
    run_dir: Path = typer.Option(
        ..., "--run-dir", exists=True, file_okay=False, help="Completed V0, V1 or V2 training run."
    ),
    partition: str = typer.Option(
        ..., "--partition", help="train, validation, calibration or test."
    ),
    output_dir: Path = typer.Option(..., "--output-dir", help="New evaluation output directory."),
    scenario_id: str = typer.Option("clean", "--scenario", help="Saved scenario identity."),
    device: str = typer.Option("cpu", "--device", help="Torch execution device, e.g. cpu or mps."),
    calibration_path: Path | None = typer.Option(
        None, "--calibration", exists=True, dir_okay=False
    ),
    policy_path: Path | None = typer.Option(None, "--policy", exists=True, dir_okay=False),
) -> None:
    try:
        resolved = resolve_config(config_path)
        metrics, _ = evaluate_run(
            resolved,
            run_dir,
            partition,
            output_dir,
            scenario_id=scenario_id,
            device=device,
            calibration_path=calibration_path,
            policy_path=policy_path,
        )
    except (ConfigError, ValueError, OSError, RuntimeError) as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=2) from error
    typer.echo(
        json.dumps(
            {
                "status": "completed",
                "output_dir": str(output_dir),
                "counts": metrics["counts"],
                "coverage": metrics["coverage"],
            },
            sort_keys=True,
        )
    )


@app.command("calibrate")
def calibrate_command(
    run_dir: Path = typer.Option(..., "--run-dir", exists=True, file_okay=False),
    evaluation_dir: Path = typer.Option(
        ...,
        "--evaluation-dir",
        exists=True,
        file_okay=False,
        help="Evaluation export from the calibration partition.",
    ),
    output_path: Path = typer.Option(..., "--output", help="New calibration JSON artifact."),
) -> None:
    try:
        artifact = calibrate_saved_scores(run_dir, evaluation_dir, output_path)
    except (ValueError, OSError, RuntimeError) as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=2) from error
    typer.echo(
        json.dumps(
            {"status": artifact["status"], "T": artifact["T"], "output": str(output_path)},
            sort_keys=True,
        )
    )


@app.command("select-policy")
def select_policy_command(
    run_dir: Path = typer.Option(..., "--run-dir", exists=True, file_okay=False),
    validation_dir: Path = typer.Option(
        ...,
        "--validation-dir",
        exists=True,
        file_okay=False,
        help="Evaluation export from the validation partition.",
    ),
    calibration_path: Path = typer.Option(..., "--calibration", exists=True, dir_okay=False),
    output_path: Path = typer.Option(..., "--output", help="New policy JSON artifact."),
    risk_limit: float | None = typer.Option(None, "--risk-limit", min=0.0, max=1.0),
    minimum_accepted_count: int | None = typer.Option(None, "--minimum-accepted-count", min=1),
    minimum_accepted_coverage: float | None = typer.Option(
        None,
        "--minimum-accepted-coverage",
        min=0.0,
        max=1.0,
    ),
) -> None:
    try:
        policy, _ = select_saved_policy(
            run_dir,
            validation_dir,
            calibration_path,
            output_path,
            risk_limit=risk_limit,
            minimum_accepted_count=minimum_accepted_count,
            minimum_accepted_coverage=minimum_accepted_coverage,
        )
    except (ValueError, OSError, RuntimeError) as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=2) from error
    typer.echo(
        json.dumps(
            {"status": policy["status"], "tau": policy["tau"], "output": str(output_path)},
            sort_keys=True,
        )
    )


@app.command("replay")
def replay_command(
    config_path: Path = typer.Option(
        ...,
        "--config",
        exists=True,
        readable=True,
        dir_okay=False,
        help="Project run/data configuration.",
    ),
    run_dir: Path = typer.Option(..., "--run-dir", exists=True, file_okay=False),
    sequence_id: str = typer.Option(..., "--sequence", help="Prepared sequence identifier."),
    output_dir: Path = typer.Option(..., "--output-dir", help="New replay output directory."),
    scenario_id: str = typer.Option("clean", "--scenario", help="Saved scenario identifier."),
    schedule_path: Path | None = typer.Option(
        None,
        "--schedule",
        exists=True,
        dir_okay=False,
        help="Saved deterministic scenario schedule.",
    ),
    temperature: float = typer.Option(1.0, "--temperature"),
    threshold: float | None = typer.Option(None, "--threshold"),
    device: str = typer.Option("cpu", "--device", help="Torch execution device, e.g. cpu or mps."),
    calibration_path: Path | None = typer.Option(
        None, "--calibration", exists=True, dir_okay=False
    ),
    policy_path: Path | None = typer.Option(None, "--policy", exists=True, dir_okay=False),
) -> None:
    try:
        resolved = resolve_config(config_path)
        result = replay_sequence(
            resolved,
            run_dir,
            sequence_id,
            output_dir,
            scenario_id=scenario_id,
            schedule_path=schedule_path,
            temperature=temperature,
            threshold=threshold,
            device=device,
            calibration_path=calibration_path,
            policy_path=policy_path,
        )
    except (ConfigError, ValueError, OSError, RuntimeError) as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=2) from error
    typer.echo(json.dumps({"status": "completed", "output_dir": str(result)}, sort_keys=True))


@app.command("report")
def report_command(
    evaluation_dir: Path = typer.Option(..., "--evaluation-dir", exists=True, file_okay=False),
    output_path: Path = typer.Option(..., "--output", help="New HTML report path."),
    scenario_id: str = typer.Option(
        "clean", "--scenario", help="Scenario label for the saved export."
    ),
    title: str = typer.Option("MMT Evaluation Report", "--title"),
    limitations: list[str] = typer.Option([], "--limitation", help="Explicit report limitation."),
    acceptance_path: Path | None = typer.Option(
        None,
        "--acceptance",
        exists=True,
        dir_okay=False,
        help="Optional frozen JSON acceptance rules.",
    ),
) -> None:
    try:
        rules = (
            None
            if acceptance_path is None
            else json.loads(acceptance_path.read_text(encoding="utf-8"))
        )
        if rules is not None and not isinstance(rules, list):
            raise ValueError("acceptance JSON must be a list of rules")
        report_path = render_saved_report(
            {scenario_id: evaluation_dir},
            output_path,
            title,
            acceptance_rules=rules,
            limitations=limitations,
        )
    except (ValueError, OSError, RuntimeError) as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=2) from error
    typer.echo(json.dumps({"status": "rendered", "report": str(report_path)}, sort_keys=True))


@app.command("ui")
def ui_command(
    runs_root: Path = typer.Option(Path("runs"), "--runs-root"),
    prepared_root: Path = typer.Option(Path("data/prepared"), "--prepared-root"),
    address: str = typer.Option("127.0.0.1", "--address"),
    port: int = typer.Option(8501, "--port"),
) -> None:
    app_path = Path(__file__).parent / "ui" / "app.py"
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.address",
        address,
        "--server.port",
        str(port),
        "--",
        "--runs-root",
        str(runs_root),
        "--prepared-root",
        str(prepared_root),
    ]
    try:
        result = subprocess.run(command, check=False)
    except FileNotFoundError as error:
        typer.echo("Streamlit is not installed; install the optional 'ui' extra.", err=True)
        raise typer.Exit(code=2) from error
    raise typer.Exit(code=result.returncode)


runs_app = typer.Typer(
    name="runs", help="List and verify immutable training runs.", no_args_is_help=True
)
app.add_typer(runs_app, name="runs")


@runs_app.command("list")
def runs_list_command(
    root: Path = typer.Option(
        Path("runs"), "--root", file_okay=False, help="Root directory containing run directories."
    ),
) -> None:
    typer.echo(json.dumps(list_runs(root), sort_keys=True))


@runs_app.command("verify")
def runs_verify_command(
    run_dir: Path = typer.Option(..., "--run-dir", exists=True, file_okay=False),
) -> None:
    try:
        result = verify_run(run_dir)
    except (OSError, ValueError) as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=2) from error
    typer.echo(json.dumps(result, sort_keys=True))
    if result["integrity"] != "valid":
        raise typer.Exit(code=3)
