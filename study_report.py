"""Render saved robustness, paired-comparison and replay evidence, without recomputation."""

from __future__ import annotations

import html
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from robustness_study import OUTPUT, dump


def render(preview=False):
    required = [
        OUTPUT / "evaluation_status.json",
        OUTPUT / "replay_status.json",
        OUTPUT / "bootstrap_status.json",
    ]
    complete = all(
        path.exists() and json.loads(path.read_text())["status"] == "completed" for path in required
    )
    if not complete and not preview:
        raise RuntimeError("evaluation and replay must both complete before the final report")
    evaluation = pd.read_csv(OUTPUT / "evaluation_summary.csv")
    paired = pd.read_csv(OUTPUT / "paired_bootstrap.csv")
    replay = pd.read_csv(OUTPUT / "replay_summary.csv")
    total_timing_rows = []
    for application in ("mosi", "pamap2"):
        proof_root = "replay-batch-proof" if application == "mosi" else "replay-shared-proof"
        for path in (OUTPUT / application / proof_root).glob("*/timings.jsonl"):
            if not (path.parent / "status.json").exists():
                continue
            by_run = {}
            for chunk in pd.read_json(path, lines=True, chunksize=10000):
                for run_id, group in chunk.groupby("run_id"):
                    by_run.setdefault(run_id, []).extend(
                        (group.decision_ms + group.shared_insert_ms).tolist()
                    )
            for run_id, values in by_run.items():
                total_timing_rows.append(
                    {
                        "application": application,
                        "scenario": path.parent.name,
                        "run_id": run_id,
                        "timed_decisions": len(values),
                        "ingestion_prediction_p50_ms": float(np.quantile(values, 0.5)),
                        "ingestion_prediction_p95_ms": float(np.quantile(values, 0.95)),
                    }
                )
    pd.DataFrame(total_timing_rows).to_csv(OUTPUT / "replay_total_latency.csv", index=False)
    clean_reference = evaluation[evaluation.kind == "clean"][
        ["application", "model", "seed", "macro_f1", "coverage"]
    ].rename(columns={"macro_f1": "clean_macro_f1", "coverage": "clean_coverage"})
    deltas = evaluation.merge(
        clean_reference, on=["application", "model", "seed"], validate="many_to_one"
    )
    deltas["delta_macro_f1"] = deltas.macro_f1 - deltas.clean_macro_f1
    deltas["delta_coverage"] = deltas.coverage - deltas.clean_coverage
    deltas.to_csv(OUTPUT / "degradation_deltas.csv", index=False)
    figures = OUTPUT / "figures"
    figures.mkdir(exist_ok=True)
    graphs = []
    tables = []
    for application in ("mosi", "pamap2"):
        frame = evaluation[evaluation.application == application]
        clean = frame[frame.kind == "clean"]
        summary = clean.groupby("model", sort=False).agg(
            seeds=("seed", "count"),
            macro_f1=("macro_f1", "mean"),
            f1_seed_std=("macro_f1", "std"),
            balanced_accuracy=("balanced_accuracy", "mean"),
            coverage=("coverage", "mean"),
            nll=("nll", "mean"),
            brier=("brier", "mean"),
            ece=("ece", "mean"),
        )
        tables.append(
            f"<h2>{application.upper()} — clean and baselines</h2>"
            + summary.to_html(float_format=lambda x: f"{x:.4f}")
        )
        fig, axis = plt.subplots(figsize=(10, 4))
        means = summary.macro_f1
        errors = summary.f1_seed_std.fillna(0)
        axis.bar(np.arange(len(means)), means, yerr=errors, capsize=3, color="#267978")
        axis.set_xticks(np.arange(len(means)), means.index, rotation=30, ha="right")
        axis.set_ylabel("Macro-F1 (mean ± seed SD)")
        axis.set_ylim(0, 1)
        axis.set_title(f"{application.upper()} clean: fusion and saved baselines")
        fig.tight_layout()
        name = f"{application}-baselines.png"
        fig.savefig(figures / name, dpi=140)
        plt.close(fig)
        graphs.append(name)
        for kind in ("removal", "delay"):
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            subset = frame[frame.kind == kind]
            for model in ("v0", "v1", "v2"):
                group = (
                    subset[subset.model == model]
                    .groupby("severity")
                    .agg(
                        f1=("macro_f1", "mean"),
                        sd=("macro_f1", "std"),
                        coverage=("coverage", "mean"),
                    )
                )
                axes[0].errorbar(
                    group.index, group.f1, yerr=group.sd.fillna(0), marker="o", label=model
                )
                axes[1].plot(group.index, group.coverage, marker="o", label=model)
            axes[0].set_ylabel("Macro-F1 on computable scored decisions")
            if subset.macro_f1.isna().all():
                axes[0].set_xlim(0.08, 0.52)
                axes[0].text(
                    0.5,
                    0.5,
                    "Quality undefined: no computable predictions",
                    ha="center",
                    va="center",
                    transform=axes[0].transAxes,
                )
            axes[1].set_ylabel("Availability coverage / eligible decisions")
            for axis in axes:
                axis.set_xlabel(
                    "Removal probability" if kind == "removal" else "Added delay / support duration"
                )
                axis.set_ylim(0, 1)
                axis.legend()
                axis.grid(alpha=0.2)
            fig.suptitle(f"{application.upper()} global {kind}")
            fig.tight_layout()
            name = f"{application}-{kind}.png"
            fig.savefig(figures / name, dpi=140)
            plt.close(fig)
            graphs.append(name)
        subset = frame[frame.kind == "subset"]
        heat = subset.pivot_table(
            index="scenario", columns="model", values="macro_f1", aggfunc="mean"
        )
        heat = heat.reindex(index=sorted(subset.scenario.unique()), columns=["v0", "v1", "v2"])
        fig, axis = plt.subplots(figsize=(7, 4))
        image = axis.imshow(heat.to_numpy(), vmin=0, vmax=1, cmap="viridis", aspect="auto")
        axis.set_xticks(np.arange(len(heat.columns)), heat.columns)
        axis.set_yticks(np.arange(len(heat.index)), heat.index)
        fig.colorbar(image, ax=axis, label="Macro-F1 (null if no computable input)")
        axis.set_title(f"{application.upper()} source absence")
        fig.tight_layout()
        name = f"{application}-absence.png"
        fig.savefig(figures / name, dpi=140)
        plt.close(fig)
        graphs.append(name)
        comparison = paired[(paired.application == application) & (paired.scenario == "clean")]
        forest = comparison[comparison.reference == "v0"].dropna(subset=["ci_low", "ci_high"])
        fig, axis = plt.subplots(figsize=(10, max(4, len(forest) * 0.3)))
        for position, row in enumerate(forest.itertuples()):
            axis.plot([row.ci_low, row.ci_high], [position, position], color="#555555")
            axis.plot(row.difference_macro_f1, position, "o", color="#267978")
        axis.axvline(0, color="#a4434e", linestyle="--")
        axis.set_yticks(
            range(len(forest)), [f"{row.model} seed {row.seed}" for row in forest.itertuples()]
        )
        axis.set_xlabel("Paired Macro-F1 difference vs V0; conditional 95% group bootstrap")
        axis.set_title(
            application.upper()
            + (
                " — only TWO test participants: unstable descriptive intervals"
                if application == "pamap2"
                else " — source-video grouping"
            )
        )
        fig.tight_layout()
        name = f"{application}-paired-bootstrap.png"
        fig.savefig(figures / name, dpi=140)
        plt.close(fig)
        graphs.append(name)
        tables.append(
            f"<h2>{application.upper()} — paired clean comparisons</h2>"
            + comparison[
                [
                    "seed",
                    "model",
                    "reference",
                    "common_decisions",
                    "groups",
                    "difference_macro_f1",
                    "ci_low",
                    "ci_high",
                    "difference_balanced_accuracy",
                    "balanced_ci_low",
                    "balanced_ci_high",
                    "valid_replicates",
                    "warning",
                ]
            ].to_html(index=False, float_format=lambda x: f"{x:.4f}")
        )
    timings = replay.groupby(["application", "scenario", "model"]).agg(
        seeds=("seed", "count"),
        sequences=("sequences", "max"),
        decisions=("decisions", "max"),
        parity_failures=("parity_failures", "sum"),
        p50_ms=("prediction_p50_ms", "mean"),
        p95_ms=("prediction_p95_ms", "mean"),
        shared_insert_p95_ms=("insertion_p95_ms", "mean"),
        peak_rss_bytes=("process_peak_rss_bytes", "max"),
        buffer_bytes=("buffer_high_water_bytes", "max"),
    )
    tables.append(
        "<h2>Recorded replay: whole test set, all checkpoints</h2>"
        + timings.to_html(float_format=lambda x: f"{x:.3f}")
    )
    tables.append(
        "<h2>Ingestion + prediction, per-decision quantiles</h2>"
        + pd.DataFrame(total_timing_rows).to_html(index=False, float_format=lambda x: f"{x:.3f}")
    )
    fig, axis = plt.subplots(figsize=(10, 4))
    clean_timing = (
        replay[replay.scenario == "clean"]
        .groupby(["application", "model"])
        .prediction_p95_ms.mean()
    )
    axis.bar(np.arange(len(clean_timing)), clean_timing, color="#a4434e")
    axis.set_xticks(
        np.arange(len(clean_timing)),
        [" ".join(key) for key in clean_timing.index],
        rotation=25,
        ha="right",
    )
    axis.set_ylabel("Mean p95 prediction latency per seed (ms)")
    axis.set_title("CPU replay, batch=1, 20 warm-ups; insertion/I/O excluded")
    fig.tight_layout()
    fig.savefig(figures / "replay-latency.png", dpi=140)
    plt.close(fig)
    graphs.append("replay-latency.png")
    content = """<!doctype html><html lang="en"><meta charset="utf-8"><title>Frozen-checkpoint robustness study</title>
<style>body{font-family:Georgia,serif;max-width:1200px;margin:30px auto;color:#222;padding:16px}table{border-collapse:collapse;font-size:13px;display:block;overflow:auto}th,td{border:1px solid #ccc;padding:7px}img{max-width:100%}h2{margin-top:35px}</style>
<h1>Étude de robustesse partielle — frozen trained checkpoints</h1>
<p>No retraining. All test decisions and all 18 trained fusion checkpoints are evaluated.
Scopes: clean, empty/leave-one-source-out, global observation removal 10/30/50%, global delay 10/25/50% of support.
Outages, out-of-order arrivals, held-out combinations and multi-source subset combinations are not included.</p>
<p>Schedules are saved once per application and reused for all compared runs. Seed SD is training variability,
not a bootstrap confidence interval. Paired Macro-F1 intervals use 1,000 identical group draws on common computable
decision IDs. PAMAP2 has only two test participants: intervals are descriptive and unstable, not evidence of significance.
MOSI grouping is by source video, not verified speaker identity.</p>
<p>MOSI vectors arrive at segment closure in this offline profile: any positive global delay removes all sources
at the decision. Quality is then undefined (null), not zero, and coverage is zero. Silence/OOV and natural masks remain.
No confidence is invented for the label-only majority. Source-only probability metrics are uncalibrated; fusion metrics
use each run's existing calibration (including disclosed T=1 fallbacks), so do not interpret probability rankings as matched calibration.</p>
<p>Replay uses all test sequences, 20 warm-up forwards, CPU/4 threads, batch size one, a shared ingestion buffer for nine
frozen predictors per application, and independent model calls. Prediction timing includes selection, normalization,
collation, model and postprocessing; insertion, stateless parity reference and trace I/O are outside that interval.
Process RSS is a cumulative lifetime high-water measurement, not an isolated per-model memory benchmark.
Evaluation and replay ran concurrently on this machine; latency is measured under that workload, not a certified idle-machine SLA.
Total latency quantiles are computed from each insertion+prediction observation, not by adding separate percentiles. Trace I/O remains excluded.
Persisted batch32/replay numerical discrepancies are counted and rechecked against stateless batch1 at unchanged CPU tolerances;
class/status mismatches are never accepted. The selected-ID trace also records validity-mask SHA-256 values.
No physical sensor or feature-extraction latency is measured. No accepted production threshold or qualification is claimed.</p>"""
    content += "".join(tables)
    for name in graphs:
        content += f'<h2>{html.escape(name)}</h2><img src="figures/{html.escape(name)}" alt="{html.escape(name)}">'
    content += "<h2>Raw evidence</h2><p><a href='evaluation_summary.csv'>Evaluation rows</a> · <a href='degradation_deltas.csv'>Scenario-minus-clean deltas</a> · <a href='paired_bootstrap.csv'>Paired comparisons</a> · <a href='replay_summary.csv'>Replay/resource rows</a></p></html>"
    (OUTPUT / "report.html").write_text(content, encoding="utf-8")
    dump(
        OUTPUT / "report_status.json",
        {
            "status": "completed" if complete else "preview_incomplete",
            "graphs": len(graphs),
            "evaluations": len(evaluation),
            "paired_comparisons": len(paired),
            "replay_runs": len(replay),
            "scope": "étude de robustesse partielle",
            "qualification": "not_qualified",
        },
    )


if __name__ == "__main__":
    render()
