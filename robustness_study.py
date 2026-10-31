"""Frozen-checkpoint robustness, baseline and group-bootstrap study."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from mmt.config import resolve_config
from mmt.evaluation.bootstrap import group_confusions, intersection_ids, paired_bootstrap
from mmt.evaluation.perturbations import all_subsets, build_schedule, load_schedule, save_schedule
from mmt.services.evaluation import evaluate_run
from mmt.services.runs import verify_run
from mmt.storage import load_sequence

ROOT = Path(__file__).resolve().parent
CLEAN = ROOT / "runs/clean-long-20261027-run1"
OUTPUT = ROOT / "runs/robustness-20261031"


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def catalogue(sources, application):
    reference = "segment_duration" if application == "mosi" else 5.0
    profile = "segment" if application == "mosi" else "window"
    result = [{"id": "clean", "kind": "clean", "sources": [], "param": None}]
    for subset in all_subsets(sources):
        if len(subset) in {0, len(sources) - 1}:
            result.append(
                {
                    "id": "subset-" + ("-".join(subset) or "empty"),
                    "kind": "subset",
                    "sources": list(subset),
                    "param": None,
                }
            )
    for kind, levels in (("removal", (0.1, 0.3, 0.5)), ("delay", (0.1, 0.25, 0.5))):
        for selected in [list(sources)]:
            for level in levels:
                result.append(
                    {
                        "id": f"{kind}-{'-'.join(selected)}-{level}",
                        "kind": kind,
                        "sources": selected,
                        "param": level,
                    }
                )
    return [{**scenario, "R": reference, "profile": profile} for scenario in result]


def load_predictions(path: Path) -> pd.DataFrame:
    frame = (
        pd.read_parquet(path / "predictions.parquet")
        if (path / "predictions.parquet").exists()
        else pd.read_csv(path / "predictions.csv")
    )
    if "pred" not in frame:
        frame["pred"] = frame.predicted_class
    return frame


def entries(application):
    return [
        entry
        for entry in json.loads((CLEAN / "plan.json").read_text())["entries"]
        if entry["application"] == application
    ]


def evaluate_study():
    torch.set_num_threads(4)
    summaries = []
    for application in ("mosi", "pamap2"):
        runs = entries(application)
        config = resolve_config(ROOT / runs[0]["config"])
        prepared = config.config.data.manifest_path.parent
        partitions = pd.read_csv(config.config.data.partitions_path)
        sequence_ids = sorted(partitions.loc[partitions.split == "test", "sequence_id"])
        sources = {
            (sequence, source): arrays
            for sequence in sequence_ids
            for source, arrays in load_sequence(prepared, sequence).items()
        }
        scenarios = catalogue(config.config.data.feature_order, application)
        dump(OUTPUT / application / "catalogue.json", scenarios)
        for scenario in scenarios:
            schedule_path = OUTPUT / application / "schedules" / (scenario["id"] + ".json")
            schedule_path.parent.mkdir(parents=True, exist_ok=True)
            if not schedule_path.exists():
                save_schedule(
                    schedule_path,
                    build_schedule(sources, scenario, seed=1701, protocol_id="robustness-20261031"),
                )
            schedule = load_schedule(schedule_path)
            print(f"{application}: {scenario['id']}", flush=True)
            for entry in runs:
                run_dir = ROOT / "runs" / entry["run_id"]
                integrity = verify_run(run_dir)
                if integrity["integrity"] != "valid" or integrity["status"] != "completed":
                    raise ValueError(f"invalid trained run: {run_dir}")
                resolved = resolve_config(ROOT / entry["config"])
                references = [None]
                if entry["model"] == "v0" and scenario["kind"] == "clean":
                    references += list(resolved.config.data.feature_order)
                    if entry["seed"] == 11:
                        references += ["majority"]
                for reference in references:
                    label = entry["model"] if reference is None else f"source-{reference}"
                    if reference == "majority":
                        label = "majority"
                    directory = (
                        OUTPUT
                        / application
                        / "evaluations"
                        / scenario["id"]
                        / f"{label}-seed-{entry['seed']}"
                    )
                    start = time.perf_counter()
                    if not (directory / "status.json").exists():
                        kwargs = (
                            {}
                            if reference is not None
                            else {
                                "calibration_path": CLEAN
                                / "results"
                                / entry["run_id"]
                                / "calibration.json",
                                "policy_path": CLEAN / "results" / entry["run_id"] / "policy.json",
                            }
                        )
                        metrics, _ = evaluate_run(
                            resolved,
                            run_dir,
                            "test",
                            directory,
                            scenario_id=scenario["id"],
                            schedule=schedule,
                            reference=reference,
                            batch_size=32,
                            **kwargs,
                        )
                    else:
                        if (
                            json.loads((directory / "status.json").read_text())["status"]
                            != "completed"
                        ):
                            raise ValueError(f"incomplete export: {directory}")
                        metrics = json.loads((directory / "metrics.json").read_text())
                    summaries.append(
                        {
                            "application": application,
                            "scenario": scenario["id"],
                            "kind": scenario["kind"],
                            "severity": scenario["param"],
                            "model": label,
                            "seed": entry["seed"],
                            "macro_f1": metrics["quality_S"]["macro_f1"],
                            "balanced_accuracy": metrics["quality_S"]["balanced_accuracy"],
                            "coverage": metrics["coverage"]["availability_coverage"],
                            "eligible": metrics["counts"]["eligible"],
                            "computable": metrics["counts"]["computable"],
                            "nll": metrics["probability"].get("nll"),
                            "brier": metrics["probability"].get("brier"),
                            "ece": metrics["probability"].get("ece"),
                            "seconds": time.perf_counter() - start,
                            "schedule_hash": schedule["schedule_hash"],
                            "rows_removed": schedule["stats"]["rows_removed"],
                            "closure_availability_test": schedule["stats"][
                                "closure_availability_test"
                            ],
                        }
                    )
            pd.DataFrame(summaries).to_csv(OUTPUT / "evaluation_summary.csv", index=False)
        del sources
    dump(
        OUTPUT / "evaluation_status.json",
        {
            "status": "completed",
            "evaluations": len(summaries),
            "scope": "étude de robustesse partielle: all test decisions/checkpoints; empty and leave-one-source-out; global removal/delay at three levels; clean baselines",
            "qualification": "descriptive_not_qualified",
            "retraining": False,
        },
    )


def bootstrap_study():
    def balanced_accuracy(confusion):
        support = confusion.sum(axis=1)
        supported = support > 0
        if not supported.any():
            return None
        return float(np.mean(np.diag(confusion)[supported] / support[supported]))

    results = []
    for application in ("mosi", "pamap2"):
        classes = 2 if application == "mosi" else 12
        eval_root = OUTPUT / application / "evaluations"
        for scenario_dir in sorted(eval_root.iterdir()):
            for seed in (11, 22, 33):
                frames = {
                    path.name.removesuffix(f"-seed-{seed}"): load_predictions(path)
                    for path in scenario_dir.glob(f"*-seed-{seed}")
                }
                majority = scenario_dir / "majority-seed-11"
                if majority.exists():
                    frames["majority"] = load_predictions(majority)
                for name in sorted(frames):
                    for reference in ("v0", "majority"):
                        if name == reference or reference not in frames:
                            continue
                        intersection = intersection_ids(frames[name], frames[reference])
                        cms_a = group_confusions(frames[name], classes, intersection)
                        cms_b = group_confusions(frames[reference], classes, intersection)
                        independent_groups = sorted(
                            set(frames[name].group_id) | set(frames[reference].group_id)
                        )
                        for group in independent_groups:
                            cms_a.setdefault(group, np.zeros((classes, classes), dtype=np.int64))
                            cms_b.setdefault(group, np.zeros((classes, classes), dtype=np.int64))
                        result = paired_bootstrap(cms_a, cms_b, classes, B=1000, seed=2026)
                        balanced = paired_bootstrap(
                            cms_a, cms_b, classes, metric=balanced_accuracy, B=1000, seed=2026
                        )
                        path = (
                            OUTPUT
                            / application
                            / "bootstrap"
                            / scenario_dir.name
                            / f"{name}-vs-{reference}-seed-{seed}.json"
                        )
                        dump(path, result)
                        dump(path.with_name(path.stem + "-balanced-accuracy.json"), balanced)
                        ci = result.get("ci95") or [None, None]
                        balanced_ci = balanced.get("ci95") or [None, None]
                        results.append(
                            {
                                "application": application,
                                "scenario": scenario_dir.name,
                                "seed": seed,
                                "model": name,
                                "reference": reference,
                                "common_decisions": len(intersection),
                                "groups": result["n_groups"],
                                "difference_macro_f1": result.get("observed_diff"),
                                "difference_balanced_accuracy": balanced.get("observed_diff"),
                                "balanced_ci_low": balanced_ci[0],
                                "balanced_ci_high": balanced_ci[1],
                                "ci_low": ci[0],
                                "ci_high": ci[1],
                                "valid_replicates": result.get("valid_replicates", 0),
                                "status": result["status"],
                                "warning": result.get("note", ""),
                            }
                        )
    pd.DataFrame(results).to_csv(OUTPUT / "paired_bootstrap.csv", index=False)
    dump(
        OUTPUT / "bootstrap_status.json",
        {
            "status": "completed",
            "comparisons": len(results),
            "replicates": 1000,
            "seed": 2026,
            "scope": "conditional on each trained run; common computable decisions; independent group draws",
            "pamap2_warning": "two test participants; descriptive and unstable confidence intervals",
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--bootstrap", action="store_true")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if args.evaluate:
        evaluate_study()
    if args.bootstrap:
        bootstrap_study()


if __name__ == "__main__":
    main()
