# Member 4 — Temporal Selection, Inference and Replay

Source: MMT-WP-004 v1.0. This is the text implementation specification transcribed from the approved work package. Names describe responsibilities; individual assignments are pending team selection. Numerical defaults are development settings, not measured performance.

## 1 Responsibility and deliverable boundary

Member 4 shall implement the shared temporal selector, bounded observation buffers, model inference engine and deterministic replay of asynchronous arrivals. The required invariant is that batch evaluation and replay select identical observation IDs at the same decision time and produce equivalent predictions using the same artifacts.

**Owned modules** — src/mmt/data/selection.py; runtime/buffer.py, engine.py, replay.py and diagnostics.py; configs/runtime.yaml; selector, buffer, inference and replay tests.

**Member 1 interface** — Consumes canonical observations and decisions. Supplies the selector used by Dataset and normalizer fitting. Reuses member 1 normalization and collate implementations.

**Member 2 interface** — Uses model registry and strict artifact loader for source-only references, V0/V1/V2 and B-GRID. Does not implement a second model or trainer.

**Member 3 interface** — Applies saved scenarios upstream of reception ordering. Loads temperature and abstention policy; exports immutable predictions, diagnostics and timing evidence.

**Member 5 interface** — Publishes Predictor and replay APIs through the shared contracts. Supplies engine events and snapshots for CLI/UI integration; the UI does not own temporal logic.

Use NumPy for source selection and temporal channels, Python dataclasses and explicit state objects for runtime records, and PyTorch for model execution. Start with a single synchronous execution owner; no message broker, distributed service or background worker is required. PyArrow output integration remains compatible with member 3’s report schema.

### Scope

Support recorded feature-stream replay and reusable local inference APIs. Raw audiovisual extraction, physical sensor drivers, clock synchronization, online learning and hard real-time scheduling are outside this release. Numeric observations carry a declared common timebase. All capacities below are engineering limits to test, not measured throughput guarantees.

## 2 Runtime architecture and public API

```text
load_predictor(artifact_dir, runtime_config) -> Predictor
Predictor.update(observation) -> UpdateResult
Predictor.predict(decision) -> Prediction
Predictor.predict_batch(decisions, observations) -> list[Prediction]
Predictor.reset(sequence_id) -> None
replay(predictor, arrivals, decisions) -> Iterator[Prediction]
```

Predictor owns immutable model/normalizer/calibration/policy artifacts and a map of sequence state. update validates and retains an arrival but does not predict. predict closes the specified logical decision time, selects observations, computes an output and records that decision as emitted. predict_batch is stateless with respect to streaming buffers and clocks.

Implement select_observations(source, decision, max_source_age) as a pure function returning selected IDs, values, masks, times and diagnostics. Buffer.select delegates to this function; Dataset calls the same function over canonical arrays. Neither call reads labels.

Artifact loading completes before any sequence starts. A model or policy change requires a new predictor/run identity. No runtime path fits normalization, updates weights, fits temperature or chooses a confidence threshold.

## 3 Observation decision and clock contracts

**Observation identity** — sequence_id, source_id and observation_id identify one canonical observation. observation_id is int64 unique within sequence/source. Source name and feature dimension must match the artifact schema.

**Observation payload** — values float32 [D]; valid_mask bool [D]; event_start, event_end and available_at finite float64 seconds. Valid entries must be finite. Invalid entries use zero and a false mask.

**Time constraints** — event_start <= event_end <= available_at. A point observation has equal start/end. The adapter must also establish availability after all feature extraction dependencies.

**Decision** — decision_id, sequence_id, decision_time, support_start, support_end with support_start <= support_end <= decision_time. Labels and participant identity never enter model features.

**Clock ownership** — Arrival calls within a sequence are nondecreasing in available_at. Prediction time is at least the last admitted arrival time and strictly greater than the previous emitted decision time.

**Time representation** — Compare canonical float64 seconds directly with inclusive boundaries. Do not add an epsilon that admits future data. Generate periodic times from start + integer_index * stride to avoid cumulative addition drift.

The current arrival clock is the timestamp of the event being processed. update must reject event_end greater than that clock. After a prediction at t is emitted, a new nonduplicate arrival claiming available_at <= t is invalid arrival order; a genuinely late observation carries a later available_at and may still have an earlier event time.

At time t, replay delivers all arrivals with available_at <= t before calling predict. The external caller has the same responsibility: predict seals that timestamp. This explicit boundary prevents an arrival at exactly t from being applied after an already-emitted prediction.

Use seconds relative to each sequence’s declared origin. Different sequences do not share a logical clock unless an adapter supplies a global timebase; no wall-clock conversion is inferred by the core.

## 4 Shared selection and temporal channels

```text
keep = (available_at <= decision_time)
keep &= (event_start >= support_start)
keep &= (event_end <= support_end)
keep &= valid_mask.any(axis=1)
rows = stable_sort(rows[keep],
    key=(event_end, event_start, observation_id))
```

Support is a closed interval. An interval crossing either support boundary is excluded in full; do not clip its timestamps. Repeated event times are allowed. Event-order sorting is independent of reception order. Missing sources return an empty result with a reason, never a fabricated observation.

### Freshness

max_source_age is an optional nonnegative scalar in seconds per source; null disables extra freshness filtering. After selection, let latest_end be the maximum event_end. If decision_time−latest_end > max_source_age, mark the entire source stale and return it as unavailable. Equality remains usable. Do not silently change this into a per-row filter.

```text
delta[0] = event_end[0] - support_start
delta[1:] = event_end[1:] - event_end[:-1]
age = decision_time - event_end
duration = event_end - event_start
time_features = log1p(stack(delta, age, duration))
```

Compute differences in float64, verify nonnegativity and convert channels to float32 [N,3]. Recompute them after every removal, delay selection or freshness decision. Padding is introduced only by collate and has zero temporal channels. Never normalize timestamps using feature statistics.

### Diagnostics

Return candidate_count, selected_count before and after freshness, latest_event_end, latest_age, valid_feature_fraction and source_status. Define valid_feature_fraction as true mask entries divided by selected row count times D; return null when no rows remain. Use source_status in {present, empty, no_eligible_rows, stale}. Offline diagnostics may additionally count future rows; runtime diagnostics cannot claim knowledge of arrivals not yet received.

This selector is the first deliverable because member 1 depends on it for eligible training observations, windows and normalization. Publish fixtures before implementing the stateful engine.

## 5 Buffer implementation duplicates and atomic updates

Implement one SourceBuffer per sequence/source: a dictionary from observation_id to owned immutable records and a sorted list of keys (event_end, event_start, observation_id). Use bisect for locating insertion/range positions [2]. Reject caller mutations by copying payload arrays on successful insertion.

A Python sorted list has linear insertion cost despite logarithmic search. This implementation is adequate only within the tested operating envelope; record insertion cost under out-of-order bursts. A chunked array index can replace it later without changing contracts if profiling justifies the work.

```text
insert(observation):
  validate schema, timestamps and canonical payload
  resolve retained duplicate identity
  reject expired record or invalid arrival order
  evict only records proven unusable by future decisions
  check projected row and byte limits
  commit record, index and accounting together
```

**Identical duplicate** — Same retained identity and exact canonical timestamps, values and mask: return duplicate_ignored without changing state, capacity or model input. Check this before arrival-order rejection so a retransmission is harmless.

**Conflicting duplicate** — Same retained identity with different canonical payload: return duplicate_conflict; preserve the original record. No overwrite or averaging.

**Expired identity** — Duplicate lookup covers retained records only. Records outside the retention floor are rejected as expired_observation rather than reintroduced. Do not keep an unbounded lifetime ID set.

**Transaction boundary** — A failed insert cannot leave a dictionary entry without its index, corrupt byte counters or advance the admitted-arrival clock. Safe prior eviction may remain committed.

**Isolation** — A source missing from one sequence does not modify another sequence. Enforce max_active_sequences before allocating new state.

No label table, model hidden state or prediction history is stored in SourceBuffer. Keep emitted records in the output sink, not an ever-growing in-memory list.

## 6 Retention limits and sequence lifecycle

**Defaults** — max_active_sequences=1; max_observations_per_source=100000; max_buffer_bytes=268435456; max_source_age=null; strict_errors=true. These inherited limits are configuration values, not capacity measurements.

**Byte scope** — max_buffer_bytes covers all active observation buffers and indexes. Count owned payload bytes once plus metadata/container sizes under a versioned estimator. Model weights, transient batches and process RSS are measured separately.

**Eviction proof** — For supported chronological profiles, future support_start is nondecreasing. Let floor be the next decision’s support_start; discard records with event_start < floor because full-interval containment can never hold later.

**Before next decision** — PAMAP2 obtains floor from its next scheduled window. MOSI keeps floor=segment start until its closure decision. Advance the floor before processing arrivals for that next decision.

**Unknown future support** — If no safe floor is available, retain observations until a later decision establishes one; enforce capacity. Never guess a horizon and silently invalidate future queries.

**Overflow** — After safe eviction, if either limit would be exceeded, reject the new observation with buffer_overflow and stop strict replay. No oldest-row truncation or silent downsampling.

### Lifecycle

A registered sequence starts EMPTY and becomes COLLECTING after its first accepted observation. Scheduled decisions may still return unavailable_no_data while EMPTY. Prediction does not close a sequence. Explicit close after the final scheduled decision releases buffers; reset clears observations, duplicate indexes, clocks, retention floor and last-decision state.

reset starts a new generation under the replay session identity, so new decisions cannot overwrite previous output keys. Retain no unbounded closed-sequence cache. The orchestrator owns the known sequence list and calls close/reset explicitly. A source can recover after an outage without resetting the sequence or reloading weights.

Reject stateful decisions whose support start moves behind the retained floor. Historical arbitrary queries use predict_batch over canonical data. If max_active_sequences is increased, maintain separate clocks and floors and test interleaving; buffers still share the configured global byte limit.

## 7 Artifact loading and model execution

**Load checks** — Validate model configuration, source names/order/dimensions, class map, normalizer identity and manifest compatibility. Use member 2’s state_dict loader with strict=True and weights_only=True.

**Calibration and policy** — Verify exact checkpoint, score_kind and class-map identities against member 3 artifacts. A missing optional calibrator uses explicit T=1 uncalibrated status. An enabled policy cannot reference a different or missing calibration artifact.

**Execution mode** — Call model.eval() and execute forwards inside torch.inference_mode(). inference_mode does not itself disable dropout; both are required [1]. No AMP in the initial inference profile.

**Tensor preparation** — Call member 1’s saved normalizer and collate functions. Preserve feature-validity, time-padding and source-presence masks. Move numeric model inputs to the configured device once per batch.

**Temporal state** — Recompute the complete selected support at every decision. Initialize model recurrent state through its ordinary forward contract. Never carry a GRU hidden state across overlapping windows or insert late rows into an old hidden state.

```text
selected = shared_selector(buffer_snapshot, decision)
batch = collate(normalizer.transform(selected))
with torch.inference_mode():
    output = model(batch)
prediction = apply_saved_calibration_and_policy(output)
```

Use the model’s computable flag, not merely any-source-present, because a source-only reference or B-GRID may be unable to compute from an otherwise nonempty selection. Scores on computable=false rows are internal placeholders and must never be converted into probabilities.

V0 returns log probability scores; V1/V2 return logits. Member 3’s adapter normalizes scores/T for either kind. Label-only B-MAJORITY returns a class with probability and confidence fields null; confidence abstention is not applicable to this reference.

A model exception or non-finite computable output fails the run. It must not be converted to ordinary no-data status. Retain the failing decision ID and artifact identity in the error record.

## 8 Prediction records and batch parity

**Identity** — session_id, generation, run_id, scenario_id, sequence_id, decision_id, decision_time and support bounds. Include model, normalizer, calibration, policy and configuration hashes.

**Prediction fields** — computable, score_kind, uncalibrated scores, probabilities when applicable, predicted_class, confidence, accepted and status. No-data rows use null class/probabilities/confidence, not uniform scores.

**Statuses** — accepted; abstained_low_confidence; unavailable_no_data. Record an additional reason such as all_sources_absent, stale_sources or model_input_unusable. Fatal execution errors belong to failed-run records.

**Source evidence** — Ordered presence pattern, per-source selected count, latest age, validity fraction and source status. Export selected observation IDs in a separate trace keyed by decision for reproducibility.

**Policy behavior** — For probabilistic predictions, accept when policy is disabled or confidence >= tau. On abstention, retain the candidate class and probabilities for inspection but accepted=false. Label-only references explicitly mark policy not applicable.

predict_batch receives complete source sequences and decisions, selects using each decision’s own availability cutoff, and does not mutate streaming state. Reuse the same normalization, collate, model adapter and calibration path. Restore original decision order after any batching; do not sort outputs silently.

Batch/replay parity requires exact equality of selected IDs, masks, computable flags and status semantics. In the fixed CPU environment compare scores/probabilities with atol=1e-6 and rtol=1e-5. Near an abstention boundary, inspect numerical differences explicitly; do not add an undocumented confidence tolerance.

Test batch size one first, then mixed lengths and missing-source patterns. For final replay, use batch size one to retain straightforward chronological semantics. Model batching can be benchmarked separately without relabeling its latency as one-decision streaming latency.

The runtime never loads targets. Member 3 joins predictions to the authoritative label/eligibility registry. Member 5 may show ground truth in a reviewer view through that separate evaluation path.

## 9 Deterministic replay algorithm

```text
for decision in chronological_decisions:
    set_safe_retention_floor(decision.support_start)
    while next_arrival.available_at <= decision.time:
        predictor.update(next_arrival)
        advance_arrival_iterator()
    result = predictor.predict(decision)
    output_sink.append(result)
close_sequence_after_final_decision()
```

For each sequence, sort arrivals by (available_at, source_id, observation_id). Sort decisions by decision_time and reject duplicate times in this initial profile. Availability ties are ingested completely before predicting. End-of-stream does not cancel remaining scheduled decisions: emit their statuses with whatever eligible history remains.

Apply member 3’s saved schedule to canonical observations before sorting reception order. Delays modify available_at only; outages/removal determine which rows arrive. Do not shuffle already-selected tensors or randomly regenerate a schedule inside replay.

### Bounded data access

Prepare the deterministic arrival index as an on-disk artifact, then iterate it without loading the whole sequence into the buffer. Use memory-mapped canonical arrays or bounded row batches from member 1 storage. Index preparation time and memory are preprocessing costs, separate from runtime buffer capacity. A one-item lookahead is sufficient for merging arrivals with decisions.

### Logical and display clocks

Run headless replay without sleeps. An optional paced UI mode waits according to logical-time differences but uses identical eligibility and ordering. Pause does not advance logical time. Rewind or a changed scenario starts a new session from a known initial point and reproduces events; it does not edit emitted records.

Persist outputs incrementally to a temporary run directory and finalize only after all expected decisions are written and checked. An interrupted replay leaves status=incomplete and a completed-decision count. Resume-from-middle is outside this release; deterministic restart avoids an undocumented buffer checkpoint format.

A late arrival can influence a later window only if its event interval is still inside that window. No correction, backfill or retrospective change is applied to an earlier prediction.

## 10 Application profiles and temporal limitations

**PAMAP2 scheduling** — Member 1 provides five-second windows at one-second stride, first decision at sequence start + 5 s. Replay retains missing activity targets; the evaluator handles scoring eligibility.

**PAMAP2 availability** — Base available_at=event_end is a simulated reception policy, not a measured network trace. Preserve availability_mode=simulated in every run and apply delays to that declared base.

**PAMAP2 retention** — Use the next scheduled support start as floor. Gaps remain gaps: no interpolation, synthetic observations or holding a hidden state across an absent sensor period.

**MOSI scheduling** — One completed-segment decision at closure; sources are audited language/acoustic/visual feature sequences. Keep the entire segment until its decision, then close and release it.

**MOSI offline mode** — Member 1 may declare vectors available at closure for offline analysis. Replay can execute that profile but must label it offline; native interval timestamps do not establish causal feature production.

**MOSI causal mode** — Allow only a feature profile with audited extraction dependency timing. Unknown timing yields unsupported_profile for a causal request. Do not invent word-level availability times.

If all MOSI vectors are available at closure, a positive injected reception delay makes the affected source unavailable at that decision. Several delay levels may therefore collapse to the same input. Export that condition to member 3 instead of describing the run as graded real-time sentiment tracking.

The core shall contain no PAMAP2/MOSI branches. Profile configuration supplies decision schedule, source schema, availability mode and retention behavior. Unsupported custom schedules fail validation or use the stateless batch path when a bounded retention proof is unavailable.

PAMAP2 supports activity classification during a recording. MOSI in this scope supports classification of a completed segment. Neither profile implies raw microphone, camera or physical sensor acquisition support. Feature-level inference cost excludes extraction unless separately measured and reported.

## 11 Errors diagnostics and resource measurement

**Structured errors** — invalid_data; incompatible_artifact; duplicate_conflict; expired_observation; buffer_overflow; invalid_arrival_order; invalid_decision_order; unsupported_profile; model_execution_failed; output_write_failed.

**Error policy** — Malformed data, conflicting duplicates, overflow, execution and sink errors stop strict replay. Identical duplicates are counted and ignored. Expected expired arrivals are counted, rejected and recorded without changing past decisions.

**Event log** — error_code, sequence/source/observation or decision identity, logical time and concise reason. Stream logs to disk with bounded in-memory counters; no unbounded list of errors.

**Timing** — Use time.perf_counter_ns for elapsed durations [3]. Measure insert, select, normalize/collate, model forward, calibration/policy and decision total separately. Record sink-write time separately.

**Benchmark setup** — Warm up 20 representative forwards, reset runtime state, then replay the full measured sequence. CUDA timing requires synchronization around device work. State device, batch size, thread count, dtype, lengths and timing count.

**Memory evidence** — Export buffer accounted bytes, row counts and high-water marks plus process RSS and device peak allocated memory where supported. Buffer accounting is not a process memory cap.

Report p50 and p95 using member 3’s linear quantile convention, along with maxima and sample counts for diagnosing bursts. Exclude file loading, arrival-index creation, model loading and UI pacing from per-decision compute latency; report their costs separately. Do not claim a latency SLA before validation on a declared machine.

Inspect long replay for stable buffer bounds and for output/log accumulation outside the buffer. Selection and batch construction allocate temporary arrays; record their contribution through process/device measurements rather than claiming max_buffer_bytes bounds the entire process.

A prediction is immutable once successfully emitted. If its sink write fails, stop the run and mark it incomplete. Do not retry a stateful predict call and create a different decision from newly arrived data.

## 12 Verification matrix and release gates

**Temporal boundaries** — Exactly available at t is included; next representable float after t is excluded. Closed support endpoints included; boundary-crossing intervals excluded; repeated times sorted by the full key.

**Channels and masks** — Hand-compute delta, age and duration before/after a removed row. Real zero with mask=true stays valid; all-invalid rows vanish; stale equality remains present.

**Duplicates** — Identical retained duplicate leaves state unchanged; conflict fails without overwrite; an expired old record cannot restore history. Test capacity counters after rejection.

**Retention** — No observation eligible for any future supported decision is evicted. Point at floor retained; interval starting before floor excluded. MOSI retains early segment rows until closure.

**Arrival and decision order** — Arrivals at t precede its prediction; post-seal arrival claiming <=t rejected; genuine late arrival affects only a later eligible decision. Backward stateful queries fail.

**Missingness and recovery** — All sources absent yields no probability vector; one source returns without reload; stale sources reported; B-GRID/source-only computability respected.

**Isolation and reset** — Reset empties buffers, indexes and clocks; generation changes output key. Two interleaved sequences, when enabled, match independent runs and respect global capacity.

**Batch and replay** — Exact selected IDs and masks; numerically compatible eval outputs; identical artifact/policy semantics. Changing future unseen observations cannot change an earlier prediction.

**Capacity and failure** — Forced low row/byte limits produce explicit overflow; no silent truncation. Model failure and sink failure mark incomplete runs. Long replay cannot accumulate prediction history in RAM.

Use pytest with deterministic synthetic records for these software properties, then run both audited application profiles through the actual model loader and shared data pipeline. CPU score tolerance is atol=1e-6, rtol=1e-5; probability sums must be within 1e-5 of one on computable probabilistic outputs.

Exit requires all mandatory software tests passing, complete decision/status exports, reproducible headless replay, bounded-state evidence and documented performance measurements. Predictive quality and acceptance thresholds remain member 3’s responsibility. These tests do not establish real-world sensor performance.

## 13 Development sequence and team handoff

**R01 Shared selector** — Implement the pure selector and temporal channels first. Deliver boundary fixtures to member 1. Gate: training Dataset and direct selector select identical IDs.

**R02 Stateless predictor** — Load one V0 artifact and reuse normalization/collate. Gate: a contract batch yields a valid record and no-data behavior without any runtime buffer.

**R03 Source buffers** — Implement immutable insertion, event-order index, duplicate handling, retention and capacity accounting. Gate: transaction, expiry and overflow tests pass.

**R04 Stateful engine** — Add per-sequence clocks, decision sealing, reset and artifact/policy integration. Gate: deterministic single-sequence predictions match stateless evaluation.

**R05 Replay iterator** — Merge arrivals and decisions; connect saved scenarios and incremental sinks. Gate: all scheduled decisions emitted and future-data invariant passes.

**R06 Application runs** — Execute PAMAP2 window replay and MOSI segment replay in declared availability modes. Gate: both profiles run without dataset-specific core branches.

**R07 Qualification** — Measure latency/memory, exercise failures and optional interleaving, integrate CLI/UI. Gate: clean-environment replay and handoff checks complete.

```text
mmt replay --run RUN --sequence SEQUENCE
  --runtime-config configs/runtime.yaml --scenario SCENARIO
mmt replay --run RUN --sequence SEQUENCE
  --runtime-config configs/runtime.yaml --availability offline
```

These command forms are integration targets for member 5. Resolve exact option names centrally and document defaults; scenario is a saved member 3 artifact, not an unseeded runtime randomization. An explicit causal request must validate the profile rather than overriding an offline label.

Deliver source code, runtime configuration schema, API usage guide, selector fixtures, error catalogue, replay records and selected-ID traces, test evidence, resource measurements and two application demonstration manifests. Include reset/restart instructions and declared unsupported conditions.

Member 1 verifies selection/normalization parity; member 2 verifies artifact execution; member 3 checks scenario and policy parity plus report ingestion; member 5 checks the same engine drives CLI and UI. No separate UI-only prediction implementation is accepted.

## 14 References and implementation decisions

[1] PyTorch inference_mode documentation. Inference execution context and the need to set evaluation mode separately.
https://docs.pytorch.org/docs/2.14/generated/torch.autograd.grad_mode.inference_mode.html

[2] Python bisect documentation. Sorted-list insertion and its linear insertion cost.
https://docs.python.org/3/library/bisect.html

[3] Python time documentation. perf_counter_ns for elapsed-time measurement.
https://docs.python.org/3/library/time.html#time.perf_counter_ns

### Authority and evidence

Parent contracts: MMT-TS-001 Technical Specification v1.0; MMT-WP-001 Data Engineering; MMT-WP-002 Models and Training; MMT-WP-003 Evaluation, Robustness and Uncertainty. The current work package specifies the runtime implementation of their shared timing, data and artifact interfaces.

**Existing framework support** — NumPy and PyTorch supply numerical and inference primitives. Python supplies indexing and timing utilities. The project implements the temporal eligibility, bounded state and replay contracts itself.

**Engineering choices** — Synchronous ownership, event-order indexes, full-window recomputation, initial capacities and error policy are concrete release decisions. They do not assert optimality or measured speed.

**Evidence not yet available** — Dataset audit results, model accuracy, tested maximum throughput, real reception delay distributions and hardware qualification must be produced by the assigned members.

**Extension boundary** — Incremental recurrent caching, distributed serving, physical acquisition, crash-resume snapshots and raw feature extraction require separate designs and tests. They are not implied by asynchronous feature replay.

Lock actual dependency versions with member 5 and test the selected environment. Documentation references establish API behavior; they do not certify project integration. No numeric performance result in this document is presented as measured evidence.
