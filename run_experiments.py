"""Run the frozen clean V0/V1/V2 experiment matrix without test-driven selection."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
APPLICATIONS = ("mosi", "pamap2")
MODELS = ("v0", "v1", "v2")
SEEDS = (11, 22, 33)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def make_plan(campaign: str) -> dict[str, Any]:
    from mmt.config import resolve_config

    entries = []
    for application in APPLICATIONS:
        for model in MODELS:
            config = ROOT / f"configs/{application}-{model}-seed-11.yaml"
            for seed in SEEDS:
                run_id = f"{campaign}-{application}-{model}-seed-{seed}"
                resolved = resolve_config(config, {"run.seed": seed, "run.run_id": run_id})
                if resolved.config.run.training.get("max_epochs") != 100:
                    raise ValueError(f"expected full 100-epoch maximum: {config}")
                if resolved.config.run.training.get("patience") != 10:
                    raise ValueError(f"expected patience=10: {config}")
                entries.append(
                    {
                        "application": application,
                        "model": model,
                        "seed": seed,
                        "run_id": run_id,
                        "config": config.relative_to(ROOT).as_posix(),
                        "config_sha256": resolved.config_sha256,
                        "input_hashes": dict(resolved.input_hashes),
                    }
                )
    code = [*sorted((ROOT / "src/mmt").rglob("*.py")), Path(__file__).resolve()]
    return {
        "campaign": campaign,
        "created_at_utc": now(),
        "entries": entries,
        "lock_sha256": sha256(ROOT / "uv.lock"),
        "code_sha256": {path.relative_to(ROOT).as_posix(): sha256(path) for path in code},
        "protocol": {
            "regime": "clean",
            "max_epochs": 100,
            "patience": 10,
            "seeds": list(SEEDS),
            "device": "cpu",
            "threads": 4,
            "checkpoint_selection": "validation only; joint macro-F1 then CE; source minimum CE",
            "normalizer": "train only; no refit on validation/calibration/test",
            "test": "all selected checkpoints frozen before any descriptive test evaluation",
            "risk_constraints": None,
            "predictive_qualification": "not_qualified",
            "limitations": [
                "MOSI offline; one development video absent in mirror",
                "PAMAP2 simulated arrivals; one calibration participant",
                "no robust-training or frozen-encoder ablations in this campaign",
            ],
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
        },
    }


def verify_plan(plan: dict[str, Any]) -> None:
    from mmt.config import resolve_config

    if sha256(ROOT / "uv.lock") != plan["lock_sha256"]:
        raise ValueError("lockfile changed since campaign freeze")
    for name, digest in plan["code_sha256"].items():
        if sha256(ROOT / name) != digest:
            raise ValueError(f"campaign code changed: {name}")
    for entry in plan["entries"]:
        resolved = resolve_config(
            ROOT / entry["config"],
            {
                "run.seed": entry["seed"],
                "run.run_id": entry["run_id"],
            },
        )
        if resolved.config_sha256 != entry["config_sha256"]:
            raise ValueError(f"run configuration or inputs changed: {entry['run_id']}")


def run_stage(directory: Path, state: dict[str, Any], key: str, arguments: list[str]) -> None:
    record = state["stages"].get(key)
    if record and record["status"] == "completed":
        return
    log_path = directory / "logs" / (key.replace("/", "--") + ".log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [str(Path(sys.executable).with_name("mmt")), *arguments]
    record = {
        "status": "running",
        "started_at_utc": now(),
        "command": command,
        "log": log_path.relative_to(ROOT).as_posix(),
    }
    state["stages"][key] = record
    write_json(directory / "status.json", state)
    print(json.dumps({"stage": key, "status": "running", "at": now()}), flush=True)
    environment = {
        **os.environ,
        "OMP_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4",
        "OPENBLAS_NUM_THREADS": "4",
        "PYTHONUNBUFFERED": "1",
    }
    start = time.perf_counter()
    try:
        with log_path.open("a", encoding="utf-8") as log:
            result = subprocess.run(
                command,
                cwd=ROOT,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        record["seconds"] = time.perf_counter() - start
        record["exit_code"] = result.returncode
        record["ended_at_utc"] = now()
        record["status"] = "completed" if result.returncode == 0 else "failed"
        if result.returncode != 0:
            raise RuntimeError(f"stage failed ({result.returncode}): {key}; see {log_path}")
    except BaseException as error:
        record["status"] = "failed"
        record["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        write_json(directory / "status.json", state)
    print(
        json.dumps({"stage": key, "status": "completed", "seconds": round(record["seconds"], 2)}),
        flush=True,
    )


def summarize(directory: Path, plan: dict[str, Any]) -> None:
    import numpy as np

    rows = []
    for entry in plan["entries"]:
        base = directory / "results" / entry["run_id"]
        for partition in ("validation", "test"):
            path = base / partition / "metrics.json"
            if not path.is_file():
                continue
            metrics = json.loads(path.read_text())
            row = {
                "application": entry["application"],
                "model": entry["model"],
                "seed": entry["seed"],
                "partition": partition,
                "run_id": entry["run_id"],
            }
            for section in ("counts", "coverage", "quality_S", "quality_A", "probability"):
                for name, value in metrics.get(section, {}).items():
                    if value is None or isinstance(value, (int, float, str)):
                        row[f"{section}.{name}"] = value
            rows.append(row)
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with (directory / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    aggregates = []
    for application in APPLICATIONS:
        for model in MODELS:
            for partition in ("validation", "test"):
                group = [
                    row
                    for row in rows
                    if row["application"] == application
                    and row["model"] == model
                    and row["partition"] == partition
                ]
                if not group:
                    continue
                aggregate = {
                    "application": application,
                    "model": model,
                    "partition": partition,
                    "completed_seeds": len(group),
                }
                for key in fields:
                    values = [row[key] for row in group if isinstance(row.get(key), (int, float))]
                    if key == "seed" or not values:
                        continue
                    aggregate[key] = {
                        "mean": float(np.mean(values)),
                        "sample_std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                    }
                aggregates.append(aggregate)
    write_json(
        directory / "summary.json",
        {
            "rows": rows,
            "aggregates": aggregates,
            "status": "descriptive_not_qualified",
            "seed_selection": "none; report all runs",
        },
    )


def execute(directory: Path, plan: dict[str, Any]) -> None:
    state_path = directory / "status.json"
    state = (
        json.loads(state_path.read_text())
        if state_path.exists()
        else {
            "status": "running",
            "started_at_utc": now(),
            "stages": {},
        }
    )
    state["status"] = "running"
    try:
        verify_plan(plan)
        for entry in plan["entries"]:
            run_id, config = entry["run_id"], entry["config"]
            run = ROOT / "runs" / run_id
            base = directory / "results" / run_id
            run_stage(
                directory,
                state,
                f"{run_id}/train",
                [
                    "train",
                    "--config",
                    config,
                    "--seed",
                    str(entry["seed"]),
                    "--run-id",
                    run_id,
                    "--device",
                    "cpu",
                ],
            )
            run_stage(
                directory, state, f"{run_id}/verify", ["runs", "verify", "--run-dir", str(run)]
            )
            for partition in ("validation", "calibration"):
                run_stage(
                    directory,
                    state,
                    f"{run_id}/{partition}",
                    [
                        "evaluate",
                        "--config",
                        config,
                        "--run-dir",
                        str(run),
                        "--partition",
                        partition,
                        "--output-dir",
                        str(base / partition),
                        "--device",
                        "cpu",
                    ],
                )
            run_stage(
                directory,
                state,
                f"{run_id}/calibrate",
                [
                    "calibrate",
                    "--run-dir",
                    str(run),
                    "--evaluation-dir",
                    str(base / "calibration"),
                    "--output",
                    str(base / "calibration.json"),
                ],
            )
            run_stage(
                directory,
                state,
                f"{run_id}/policy",
                [
                    "select-policy",
                    "--run-dir",
                    str(run),
                    "--validation-dir",
                    str(base / "validation"),
                    "--calibration",
                    str(base / "calibration.json"),
                    "--output",
                    str(base / "policy.json"),
                ],
            )
            summarize(directory, plan)
        verify_plan(plan)
        freeze = {
            entry["run_id"]: {
                "weights": sha256(ROOT / "runs" / entry["run_id"] / "weights.pt"),
                "normalizer": sha256(ROOT / "runs" / entry["run_id"] / "normalizer.npz"),
                "calibration": sha256(directory / "results" / entry["run_id"] / "calibration.json"),
                "policy": sha256(directory / "results" / entry["run_id"] / "policy.json"),
            }
            for entry in plan["entries"]
        }
        frozen_path = directory / "frozen_artifacts.json"
        if frozen_path.exists() and json.loads(frozen_path.read_text())["runs"] != freeze:
            raise ValueError("selected artifacts changed after freeze")
        if not frozen_path.exists():
            write_json(
                frozen_path,
                {
                    "frozen_at_utc": now(),
                    "runs": freeze,
                    "predictive_qualification": "not_qualified",
                },
            )
        for entry in plan["entries"]:
            run_id, config = entry["run_id"], entry["config"]
            run = ROOT / "runs" / run_id
            base = directory / "results" / run_id
            run_stage(
                directory,
                state,
                f"{run_id}/test",
                [
                    "evaluate",
                    "--config",
                    config,
                    "--run-dir",
                    str(run),
                    "--partition",
                    "test",
                    "--output-dir",
                    str(base / "test"),
                    "--calibration",
                    str(base / "calibration.json"),
                    "--policy",
                    str(base / "policy.json"),
                    "--device",
                    "cpu",
                ],
            )
            run_stage(
                directory,
                state,
                f"{run_id}/report",
                [
                    "report",
                    "--evaluation-dir",
                    str(base / "test"),
                    "--output",
                    str(base / "report.html"),
                    "--title",
                    f"{entry['application']} {entry['model']} seed {entry['seed']} clean",
                    "--limitation",
                    "Descriptive clean experiment; no predictive qualification bounds.",
                ],
            )
            summarize(directory, plan)
        state["status"] = "completed"
        state["completed_at_utc"] = now()
    except BaseException as error:
        state["status"] = "incomplete"
        state["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        write_json(state_path, state)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", default="clean-long-20261027")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()
    if not args.campaign or any(
        char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_."
        for char in args.campaign
    ):
        parser.error("campaign must be a path-safe identifier")
    if args.campaign in {".", ".."}:
        parser.error("invalid campaign identity")
    directory = ROOT / "runs" / args.campaign
    if args.status:
        path = directory / "status.json"
        print(path.read_text() if path.exists() else json.dumps({"status": "not_started"}))
        return 0
    plan_path = directory / "plan.json"
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
        verify_plan(plan)
    else:
        plan = make_plan(args.campaign)
        write_json(plan_path, plan)
    print(
        json.dumps(
            {
                "campaign": args.campaign,
                "runs": len(plan["entries"]),
                "plan": str(plan_path),
                "execute": args.execute,
            }
        ),
        flush=True,
    )
    if args.execute:
        execute(directory, plan)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
