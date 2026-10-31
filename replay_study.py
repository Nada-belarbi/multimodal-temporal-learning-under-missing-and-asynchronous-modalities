"""Whole-test-set recorded replay parity and CPU resource measurements."""

from __future__ import annotations

import hashlib
import json
import resource
import sys
import time
from dataclasses import asdict

import numpy as np
import pandas as pd
import torch

from mmt.config import resolve_config
from mmt.data.collate import collate_decisions
from mmt.data.dataset import DecisionSample, SourceSample
from mmt.data.normalization import load_normalizer
from mmt.data.selection import select_observations
from mmt.evaluation.io_utils import canonical_hash
from mmt.evaluation.perturbations import _observation_digest, apply_schedule, load_schedule
from mmt.runtime.buffer import BufferError
from mmt.runtime.engine import InferenceEngine
from mmt.services.evaluation_workflows import load_saved_postprocessing
from mmt.services.runtime import _merge_arrivals
from mmt.storage import load_manifest, load_sequence
from mmt.training.checkpoint import load_fusion_model
from robustness_study import CLEAN, OUTPUT, dump, entries, load_predictions


class SharedSelectionEngine(InferenceEngine):
    """Benchmark-only engine using one validated immutable selection per shared buffer."""

    shared_selection = None

    def _select(self, sequence_id, decision):
        if self.shared_selection is not None:
            return self.shared_selection
        return super()._select(sequence_id, decision)


def json_line(stream, value):
    stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")


def compare_exported_prediction(replayed, expected, temperature):
    if replayed.computable != bool(expected["computable"]) or replayed.status != expected["status"]:
        raise AssertionError(f"batch/replay status failure: {replayed.decision_id}")
    if replayed.computable:
        raw = expected["scores"]
        scores = np.fromstring(raw, sep=";") if isinstance(raw, str) else np.asarray(raw)
        np.testing.assert_allclose(replayed.scores, scores, atol=1e-6, rtol=1e-5)
        scaled = scores / temperature
        probabilities = np.exp(scaled - scaled.max())
        probabilities /= probabilities.sum()
        np.testing.assert_allclose(replayed.probabilities, probabilities, atol=1e-6, rtol=1e-5)
        if replayed.predicted_class != int(expected["predicted_class"]):
            raise AssertionError("batch/replay class failure")


def builder(normalizer, source_order, dimensions):
    def build(selections):
        sources = {}
        for source in source_order:
            selected = selections[source]
            values, mask = normalizer.transform(
                source,
                np.asarray(selected.values, dtype=np.float32).reshape(-1, dimensions[source]),
                np.asarray(selected.valid_mask, dtype=bool).reshape(-1, dimensions[source]),
            )
            sources[source] = SourceSample(
                torch.from_numpy(values),
                torch.from_numpy(mask),
                torch.tensor(selected.time_features, dtype=torch.float32).reshape(-1, 3),
            )
        sample = DecisionSample("runtime", "runtime", 1.0, 0.0, 1.0, sources, 0, False, {})
        return collate_decisions([sample]).inputs

    return build


def measure_replay():
    torch.set_num_threads(4)
    summaries = []
    for application in ("mosi", "pamap2"):
        application_entries = entries(application)
        resolved = resolve_config(application_entries[0]["config"])
        data_root = resolved.config.data.manifest_path.parent
        metadata = load_manifest(data_root)
        order = tuple(metadata["source_order"])
        dims = metadata["source_dims"]
        membership = pd.read_csv(resolved.config.data.partitions_path)
        decisions = pd.read_csv(data_root / "decisions.csv")
        test_sequences = sorted(membership.loc[membership.split == "test", "sequence_id"])
        sources = {
            (sequence, source): values
            for sequence in test_sequences
            for source, values in load_sequence(data_root, sequence).items()
        }
        sorted_sources = {
            key: bool(
                np.all(value["event_start"][1:] >= value["event_start"][:-1])
                and np.all(value["event_end"][1:] >= value["event_end"][:-1])
            )
            for key, value in sources.items()
        }
        catalogue = json.loads((OUTPUT / application / "catalogue.json").read_text())
        scenarios = [
            scenario
            for scenario in catalogue
            if scenario["kind"] == "clean"
            or (scenario["kind"] == "removal" and scenario["param"] == 0.3)
            or (scenario["kind"] == "delay" and scenario["param"] == 0.25)
        ]
        models = []
        for entry in application_entries:
            run = CLEAN.parent / entry["run_id"]
            model_metadata = json.loads((run / "model_config.json").read_text())
            class_map = json.loads((run / "class_map.json").read_text())
            model = load_fusion_model(
                model_metadata, class_map, run / "weights.pt", expected_model_name=entry["model"]
            )
            normalizer = load_normalizer(
                run / "normalizer.npz", model_metadata["normalizer_identity"]
            )
            item = resolve_config(entry["config"])
            calibration, policy = load_saved_postprocessing(
                run,
                dict(item.input_hashes),
                "log_probabilities" if entry["model"] == "v0" else "logits",
                CLEAN / "results" / entry["run_id"] / "calibration.json",
                CLEAN / "results" / entry["run_id"] / "policy.json",
            )
            models.append(
                (
                    entry,
                    model,
                    builder(normalizer, order, dims),
                    calibration["T"],
                    policy.get("tau"),
                )
            )
        for scenario in scenarios:
            proof_root = "replay-batch-proof" if application == "mosi" else "replay-shared-proof"
            destination = OUTPUT / application / proof_root / scenario["id"]
            if (destination / "status.json").exists():
                summaries.extend(json.loads((destination / "summary.json").read_text()))
                continue
            if destination.exists():
                destination.rename(
                    destination.with_name(destination.name + f".incomplete-{time.time_ns()}")
                )
            destination.mkdir(parents=True, exist_ok=False)
            schedule = load_schedule(
                OUTPUT / application / "schedules" / (scenario["id"] + ".json")
            )
            if schedule["input_hash"] != _observation_digest(sources) or schedule[
                "schedule_hash"
            ] != canonical_hash(
                {
                    "entries": schedule["entries"],
                    "scenario": schedule["scenario"],
                    "seed": schedule["seed"],
                }
            ):
                raise ValueError("replay schedule identity mismatch")
            transformed = apply_schedule(sources, schedule)
            batch_reference = [
                load_predictions(
                    OUTPUT
                    / application
                    / "evaluations"
                    / scenario["id"]
                    / f"{entry['model']}-seed-{entry['seed']}"
                )
                .set_index("decision_id")
                .to_dict(orient="index")
                for entry, model, build, temperature, threshold in models
            ]
            timers = [[] for _ in models]
            insert_times = []
            differences = [0] * len(models)
            numerical_fallbacks = [0] * len(models)
            max_batch_error = [0.0] * len(models)
            counts = [0] * len(models)
            max_buffer = 0
            max_rows = 0
            expired = 0
            total_arrivals = 0
            samples = 0
            print(
                f"replay {application} {scenario['id']}: {len(test_sequences)} sequences",
                flush=True,
            )
            first_sequence = test_sequences[0]
            first_decision = decisions[decisions.sequence_id == first_sequence].iloc[0].to_dict()
            first_selected = {
                source: select_observations(
                    transformed[(first_sequence, source)],
                    first_decision,
                    assume_sorted=sorted_sources[(first_sequence, source)],
                )
                for source in order
            }
            for entry, model, build, temperature, threshold in models:
                batch = build(first_selected)
                with torch.inference_mode():
                    for _ in range(20):
                        model(batch)
            with (
                (destination / "selected_ids.jsonl").open("w") as trace,
                (destination / "predictions.jsonl").open("w") as predictions,
                (destination / "timings.jsonl").open("w") as timing_stream,
            ):
                for sequence_index, sequence in enumerate(test_sequences):
                    engine_list = []
                    for entry, model, build, temperature, threshold in models:

                        def predict(batch, current=model):
                            with torch.inference_mode():
                                return current(batch)

                        engine_list.append(
                            SharedSelectionEngine(
                                order,
                                build,
                                predict,
                                len(class_map),
                                temperature=temperature,
                                threshold=threshold,
                                run_id=entry["run_id"],
                                scenario_id=scenario["id"],
                            )
                        )
                    owner = engine_list[0]
                    for engine in engine_list[1:]:
                        engine.buffer = owner.buffer
                    sequence_sources = {source: transformed[(sequence, source)] for source in order}
                    arrivals = iter(_merge_arrivals(sequence, sequence_sources))
                    next_arrival = next(arrivals, None)
                    rows = decisions[decisions.sequence_id == sequence].to_dict(orient="records")
                    for decision in rows:
                        insertion_start = time.perf_counter_ns()
                        while (
                            next_arrival is not None
                            and next_arrival["available_at"] <= decision["decision_time"]
                        ):
                            try:
                                owner.update(next_arrival)
                            except BufferError as error:
                                if error.code != "expired_observation":
                                    raise
                                expired += 1
                            total_arrivals += 1
                            next_arrival = next(arrivals, None)
                        insertion_ns = time.perf_counter_ns() - insertion_start
                        insert_times.append(insertion_ns / 1e6)
                        max_rows = max(max_rows, owner.buffer.row_count)
                        selected = {
                            source: select_observations(
                                sequence_sources[source],
                                decision,
                                assume_sorted=sorted_sources[(sequence, source)],
                            )
                            for source in order
                        }
                        reference_ids = {
                            source: value.observation_ids for source, value in selected.items()
                        }
                        reference_masks = {
                            source: value.valid_mask for source, value in selected.items()
                        }
                        selection_start = time.perf_counter_ns()
                        owner.buffer.advance_retention_floor(sequence, decision["support_start"])
                        owner.shared_selection = None
                        buffered_selection = owner._select(sequence, decision)
                        shared_selection_ms = (time.perf_counter_ns() - selection_start) / 1e6
                        for engine in engine_list:
                            engine.shared_selection = buffered_selection
                        replay_masks = {
                            source: value.valid_mask for source, value in buffered_selection.items()
                        }
                        if replay_masks != reference_masks:
                            raise AssertionError(f"mask parity failure: {decision['decision_id']}")
                        for model_index, engine in enumerate(engine_list):
                            start = time.perf_counter_ns()
                            replayed = engine.predict(decision)
                            decision_ms = (
                                time.perf_counter_ns() - start
                            ) / 1e6 + shared_selection_ms
                            timers[model_index].append(decision_ms)
                            expected = batch_reference[model_index][decision["decision_id"]]
                            if (
                                replayed.selected_ids != reference_ids
                                or replayed.computable != bool(expected["computable"])
                                or replayed.status != expected["status"]
                            ):
                                differences[model_index] += 1
                                raise AssertionError(
                                    f"ID/status parity failure: {decision['decision_id']}"
                                )
                            try:
                                compare_exported_prediction(
                                    replayed, expected, models[model_index][3]
                                )
                            except AssertionError:
                                if (
                                    replayed.computable != bool(expected["computable"])
                                    or replayed.status != expected["status"]
                                    or replayed.predicted_class != int(expected["predicted_class"])
                                ):
                                    raise
                                raw = expected["scores"]
                                batch_scores = (
                                    np.fromstring(raw, sep=";")
                                    if isinstance(raw, str)
                                    else np.asarray(raw)
                                )
                                max_batch_error[model_index] = max(
                                    max_batch_error[model_index],
                                    float(
                                        np.max(np.abs(np.asarray(replayed.scores) - batch_scores))
                                    ),
                                )
                                unit = engine._predict_selected(decision, selected)
                                compare_exported_prediction(
                                    replayed, asdict(unit), models[model_index][3]
                                )
                                numerical_fallbacks[model_index] += 1
                            counts[model_index] += 1
                            payload = asdict(replayed)
                            payload.pop("selected_ids", None)
                            json_line(predictions, payload)
                            json_line(
                                timing_stream,
                                {
                                    "run_id": replayed.run_id,
                                    "decision_id": replayed.decision_id,
                                    "decision_ms": decision_ms,
                                    "shared_insert_ms": insertion_ns / 1e6,
                                },
                            )
                        json_line(
                            trace,
                            {
                                "sequence_id": sequence,
                                "decision_id": decision["decision_id"],
                                "selected_ids": reference_ids,
                                "valid_mask_sha256": {
                                    source: hashlib.sha256(
                                        np.asarray(mask, dtype=bool).tobytes()
                                    ).hexdigest()
                                    for source, mask in reference_masks.items()
                                },
                            },
                        )
                        samples += 1
                        if samples % 250 == 0:
                            print(
                                f"  completed replay decisions {samples}; {application} {scenario['id']}",
                                flush=True,
                            )
                        max_buffer = max(max_buffer, owner.buffer.high_water_bytes)
                    for engine in engine_list:
                        engine.close(sequence)
                    if sequence_index % 25 == 0:
                        print(
                            f"  replay sequences {sequence_index + 1}/{len(test_sequences)}; decisions={samples}",
                            flush=True,
                        )
            scenario_summaries = []
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            rss_bytes = int(rss if sys.platform == "darwin" else rss * 1024)
            for index, (entry, model, build, temperature, threshold) in enumerate(models):
                scenario_summaries.append(
                    {
                        "application": application,
                        "scenario": scenario["id"],
                        "model": entry["model"],
                        "seed": entry["seed"],
                        "sequences": len(test_sequences),
                        "decisions": counts[index],
                        "parity_failures": differences[index],
                        "batch32_numerical_rechecks": numerical_fallbacks[index],
                        "max_batch32_score_error_rechecked": max_batch_error[index],
                        "parity_method": "strict persisted batch32 comparison; discrepancies rechecked against stateless batch1, never tolerate class/status changes",
                        "prediction_p50_ms": float(np.quantile(timers[index], 0.5)),
                        "prediction_p95_ms": float(np.quantile(timers[index], 0.95)),
                        "insertion_p50_ms": float(np.quantile(insert_times, 0.5)),
                        "insertion_p95_ms": float(np.quantile(insert_times, 0.95)),
                        "process_peak_rss_bytes": rss_bytes,
                        "buffer_high_water_bytes": max_buffer,
                        "buffer_max_rows": max_rows,
                        "arrivals": total_arrivals,
                        "expired_arrivals": expired,
                        "warmup_calls": 20,
                        "batch_size": 1,
                        "threads": 4,
                        "memory_method": "process lifetime peak RSS, cumulative; shared buffer for nine frozen predictors",
                        "timing_scope": "shared selection time plus individual normalize/collate/model/postprocessing; insertion and trace writing separate",
                        "schedule_hash": schedule["schedule_hash"],
                    }
                )
            dump(destination / "summary.json", scenario_summaries)
            dump(
                destination / "status.json",
                {
                    "status": "completed",
                    "decisions_per_model": samples,
                    "parity_failures": sum(differences),
                },
            )
            summaries.extend(scenario_summaries)
            pd.DataFrame(summaries).to_csv(OUTPUT / "replay_summary.csv", index=False)
        del models, sources
    dump(
        OUTPUT / "replay_status.json",
        {
            "status": "completed",
            "model_scenario_runs": len(summaries),
            "scope": "all test sequences, all 18 models; clean/removal0.3/delay0.25; video features offline",
            "parity_failures": sum(row["parity_failures"] for row in summaries),
        },
    )


if __name__ == "__main__":
    measure_replay()
