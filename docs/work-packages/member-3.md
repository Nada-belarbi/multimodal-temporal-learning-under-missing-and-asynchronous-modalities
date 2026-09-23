# Member 3 — Evaluation, Robustness and Uncertainty

Source: MMT-WP-003 v1.0. This is the text implementation specification transcribed from the approved work package. Names describe responsibilities; individual assignments are pending team selection. Numerical defaults are development settings, not measured performance.

## 1 Responsibility and implementation boundary

Member 3 shall implement the experimental protocol, reproducible corruption schedules, predictive metrics, probability calibration, confidence-based abstention and evaluation reports. The evaluator must distinguish prediction quality from the fraction of decisions for which the system can produce a prediction.

**Owned package** — evaluation/protocol.py, runner.py, perturbations.py, metrics.py, calibration.py, abstention.py, bootstrap.py and report.py; configs/scenarios.yaml; protocol.yaml; acceptance.yaml; evaluation tests.

**Member 1 provides** — Audited dataset manifest, explicit partitions, canonical observation IDs, decision registry, labels and normalizer identity. Member 1 applies the supplied transformations in the data pipeline.

**Member 2 provides** — Selected model artifacts, score_kind, scores, training histories and model cost measurements. Member 3 specifies experiments; member 2 owns training.

**Member 4 provides** — Shared temporal eligibility, replay execution, computable status and end-to-end timing. Do not duplicate the selector inside the evaluator.

**Member 5 integrates** — Shared schemas, public CLI, dependency lock and report viewer. Member 3 produces machine-readable artifacts and a static HTML report.

### Required result

A single evaluation command shall verify artifact identities, execute a named scenario over a fixed decision registry, preserve every decision status, compute the defined metrics and export a reproducible report. Invalid inputs fail explicitly; empty predictions never become uniform probabilities.

Framework choices: NumPy for numerical aggregation and schedules; scikit-learn for classification metrics; PyTorch for temperature fitting; PyArrow/Parquet for decision records; JSON and CSV for summaries; Matplotlib and Jinja2 for static charts and HTML; pytest for verification. Resolve and lock compatible versions with member 5. This specification does not claim a tested dependency environment or measured dataset results.

## 2 Protocol and application bindings

**PAMAP2 task** — Protocol subset; classes [1,2,3,4,5,6,7,12,13,16,17,24] mapped to indices 0–11. Sources: hand, chest, ankle and heart rate. These are source streams, not four unrelated sensing modalities.

**PAMAP2 decision** — Five-second history, one-second stride, first decision at sequence start + 5 s. Use the latest label at or before decision time with age at most 0.02 s. Activity 0 and unavailable labels are unscored; retain their decisions in replay counts.

**MOSI task** — Completed-segment binary sentiment. Negative annotation maps to 0, positive to 1; zero is excluded. Language, acoustic and visual feature dimensions come from the audited manifest.

**MOSI timing** — One decision at segment closure. Offline feature availability does not prove online availability. Causal claims require documented extraction dependencies and arrival times.

**Partition roles** — Train fits model and normalizer. Validation selects model/configuration and later abstention policy. Calibration fits temperature after checkpoint selection. Test evaluates frozen artifacts only.

protocol.yaml shall name the dataset manifest hash, class map, decision policy, group field, explicit group membership for every partition and exclusion reasons. PAMAP2 partitions contain whole participants. MOSI retains the audited supplied split and reserves calibration groups from development data before tuning; never move final-test groups to obtain calibration data.

Audit group overlap, observation leakage and training class support before executing experiments. If verified identity information conflicts with a supplied split, stop and create a named revised protocol with its comparability limitation. Do not invent participant IDs or split percentages. Missing training classes fail qualification.

Window length, stride and label-age limit are inherited engineering settings. Their suitability must be assessed on validation data. No test-driven changes may be presented as results under the original frozen protocol.

## 3 Architecture and evaluator contracts

```text
evaluate(protocol, model_artifact, scenario, partition)
  -> EvaluationBundle
fit_temperature(calibration_scores, artifact_identity)
  -> CalibrationArtifact
select_policy(validation_scores, calibration, constraints)
  -> AbstentionPolicy
```

Prediction rows are unique on (run_id, scenario_id, decision_id). Required fields: sequence_id, group_id, decision_time, support_start, support_end, source_presence, computable, status, score_kind and scores. Scores have C entries in the fixed class order when computable. Label-only references instead provide predicted_class. Labels and eligibility come from a separate authoritative decision registry.

Left-join predictions onto that registry, never the reverse. Assert exactly one returned status for each scheduled decision; reject duplicates, unknown IDs, class-order mismatches and missing rows. A model exception is a failed run, not an unavailable observation. An incomplete export cannot yield a successful final report.

The runner verifies model, normalizer, dataset, partition and scenario hashes before execution. It requests predictions from member 2/4 through public interfaces, then attaches labels inside evaluation. Internal zero score placeholders for computable=false are discarded before probability conversion.

## 4 Perturbation catalogue and scope

Clean means no injected degradation; natural missingness remains. Scenario settings below are initial engineering stress levels, not estimates of real sensor failure frequency. Generate schedules before model execution and reuse them across compared models.

**Source subsets** — Evaluate all 16 PAMAP2 and all 8 MOSI source-presence subsets, including empty and full. Remove excluded sources for the whole sequence. Source-only models remain unavailable when their source is absent.

**Observation removal** — Independent row removal with p in [0,0.1,0.3,0.5], applied separately to each selected source. Removal clears row availability; do not replace removed values with valid zeros.

**Contiguous outage** — One interval [a,a+L) per selected source and sequence. L = min(r * R, sequence duration), r in [0.1,0.3,0.5]. R = 5 s for PAMAP2 or segment duration for MOSI. Remove rows whose event_end lies in the interval.

**Reception delay** — available_at becomes original available_at + d, with d/R in [0,0.1,0.25,0.5]. Event timestamps and target labels stay unchanged.

**Out-of-order arrival** — Independent per-row delay Uniform(0,d_max), d_max/R in [0.1,0.25,0.5]. Process reception order but select and encode event order. Record actual arrival inversions.

**Held-out combination** — Outage r=0.3 plus recovery delay d/R=0.25 on surviving rows with event_end at or after outage end. Reserve this composition from robust training; freeze its source cases before testing.

For removal, outage, delay and out-of-order tests, instantiate one case per individual source plus one case affecting all sources. Deduplicate identical clean configurations. Do not build an uncontrolled Cartesian product. Record selected sources, intended severity, actual removed rows and affected decision fraction.

Robust training uses source dropout p=0.2 and observation dropout p=0.1 as specified for member 2. These are augmentation settings, separate from test scenario levels; the held-out composition must not be silently added to training.

## 5 Reproducible schedule implementation

```text
build_schedule(observation_index, scenario, schedule_seed)
  -> schedule.parquet + schedule.json
apply_schedule(sequence, schedule) -> transformed_sequence
```

Use schedule_seed=1701 as the initial protocol value, distinct from model seeds. Derive per-sequence/source generator seeds from SHA-256 of canonical JSON containing protocol ID, scenario ID, seed, sequence ID and source name. Use sorted keys and UTF-8; convert the first eight digest bytes to an unsigned big-endian integer and initialize NumPy PCG64. Do not use Python hash().

Sort canonical rows by observation ID before drawing row-level random numbers. Draw an outage start uniformly from [sequence_start, sequence_end−L]; if L equals the duration, use sequence_start. Persist the resulting affected IDs, outage endpoints and availability overrides, not only the random seed. Save generator/library versions and input/schedule hashes.

Generate observation transformations on the full canonical sequence before overlapping windows are constructed. An observation shared by windows receives the same corruption. Training row dropout is keyed additionally by epoch; training source dropout is keyed by epoch, decision ID and source. Both schedules are model-independent.

### Temporal integration

Apply transformations before the shared selector. Member 4 retains only observations inside the support interval and available by decision time, then sorts by event time and recomputes temporal channels. Do not alter timestamps to mimic reception delay or retrospectively revise an earlier prediction after a late arrival.

### MOSI limitation to implement explicitly

If all pre-extracted vectors become available at segment closure, any positive added delay makes the affected source unavailable at that closure. Multiple delay severities may therefore produce identical outcomes. Label these as closure-availability tests; do not claim graded streaming robustness. Enable causal delay experiments only when audited feature dependency times permit them.

If an out-of-order schedule creates no actual arrival inversion, mark that case as not exercising reordering. A short outage within a long PAMAP2 recording can affect few decisions: report the observed exposure instead of describing every window as corrupted.

## 6 Classification metrics and denominators

```text
E = label-eligible scheduled decisions
S = decisions in E with computable predictions
A = decisions in S accepted by the frozen policy
availability_coverage = len(S) / len(E)
accepted_coverage = len(A) / len(E)
selective_risk = incorrect_predictions(A) / len(A)
```

Retain total scheduled count separately from eligible count, with exclusions grouped by reason. If E is empty, coverage is null with status=no_eligible_labels. If S is empty, predictive metrics are null and availability coverage is zero. If A is empty, selective risk is null; it is never reported as zero error.

**Macro F1** — Use f1_score(y, pred, labels=range(C), average="macro", zero_division=0). The class list stays fixed even when some classes are absent in the scored partition [2].

**Balanced accuracy** — Compute mean recall over classes with nonzero ground-truth support in S. Export included classes and absent classes explicitly; this denominator differs from fixed-class macro F1.

**Per-class evidence** — Export precision, recall, F1 and support for every configured class, plus the C by C confusion matrix. Undefined per-class precision/recall use zero with support metadata.

**Decision quality** — Report metrics on S and, separately, on A when abstention is enabled. Derive argmax with ties resolved to the lowest class index.

**Comparison subset** — For paired model quality comparisons, also score the intersection of computable decision IDs. Show each model’s original coverage and the intersection size alongside those scores.

Do not drop difficult rows from the denominator because a source is missing. B-GRID may have different computability from event-based models; preserve this difference. B-MAJORITY remains a label-only reference, subject to the agreed no-data guard, without fabricated confidence values.

The primary aggregate pools eligible decisions within each partition. Also export per-group results so long recordings cannot conceal failures on shorter groups. Group-balanced averages, if added, must have a different metric name and explicit weighting.

## 7 Probabilities calibration metrics and uncertainty

For score_kind=logits, compute log_p=log_softmax(scores/T). For score_kind=log_probabilities, apply the same normalization to the supplied log scores divided by T. V0 supplies log(clamp(mean_probability, 1e-8)); do not apply a second logarithm. T=1 defines the uncalibrated reference.

```text
p = exp(log_p)
confidence = max(p)
NLL = mean(-log_p[row, target])
Brier = mean(sum((p - one_hot(target))**2, axis=1))
entropy = -sum(p * log_p, axis=1)
```

Use float64 metric accumulation and stable log_softmax for NLL. The specified multiclass Brier sums over classes, including in the binary task; it is twice the single-positive-class binary convention. Label-only outputs have null probability metrics with reason=not_available.

### Expected calibration error

Use 15 equal-width confidence bins spanning [0,1]. Bin j contains confidence in [j/15,(j+1)/15), with the final bin including 1. For each nonempty bin compute mean confidence, empirical accuracy and count. ECE is the sum of count/len(S) times the absolute difference between those two means. Save edges and bin statistics; empty bins have null empirical values.

Generate reliability plots for uncalibrated and calibrated predictions on the same S. Always report NLL and Brier alongside ECE; changing binning can change ECE. The 15-bin choice is a reporting convention, not a validated optimum.

### Interpretation limits

Confidence and entropy describe the model’s predictive distribution. They do not provide an epistemic/aleatoric decomposition or an individual correctness guarantee. Three independent training seeds are repeated runs; they are not an ensemble unless a separate ensemble predictor is implemented and evaluated.

Measure calibration on each stress scenario using the already-fitted clean calibration artifact. Do not refit temperature on corrupted test labels. A deterioration under missingness is a result to report rather than a reason to silently recalibrate.

## 8 Temperature fitting and artifact persistence

Fit one positive scalar temperature per selected model run on computable, label-eligible calibration decisions. The model weights, normalizer and class order are frozen. Temperature scaling follows the post-processing approach of Guo et al. [1]; its benefit for this framework remains to be measured.

```text
log_T = torch.nn.Parameter(torch.zeros((), dtype=float64))
T = exp(log_T)
loss = cross_entropy(frozen_scores / T, targets)
optimizer = torch.optim.LBFGS([log_T], lr=1.0,
    max_iter=50, history_size=10,
    tolerance_grad=1e-7, tolerance_change=1e-9,
    line_search_fn="strong_wolfe")
```

Run on CPU float64. The closure clears gradients, recomputes loss and calls backward; pass it to optimizer.step [3]. Detach input scores and never include model parameters in the optimizer. Values above are initial numerical settings, not calibrated performance requirements.

**Preconditions** — Nonempty calibration S; finite scores; labels in the fixed class map; correct checkpoint and partition hashes. Export calibration class support, including missing classes.

**Postconditions** — Finite positive T and finite final NLL, no worse than T=1 baseline by more than 1e-8. Record initial/final loss, gradient, iterations and termination evidence. Hitting the iteration limit alone does not establish convergence.

**Failure handling** — Return fit_failed on empty input, numerical failure or rejected postconditions. An explicit identity fallback T=1 may be exported as uncalibrated; never label the fallback a successful calibration.

**Saved artifact** — schema_version, T, status, score_kind, checkpoint hash, class-map hash, calibration partition/score hashes, optimizer settings, fit diagnostics and software versions.

Use a single clean-data temperature for all source subsets and stress scenarios in this release. Separate temperatures per missingness pattern would require a different protocol and sufficient calibration support; do not fit them implicitly.

Positive temperature preserves the ordering of finite class scores, so argmax labels remain unchanged apart from numerical ties. Test this invariant. Reject a calibration artifact if any identity field differs from the model being evaluated.

## 9 Abstention policy and operating constraints

After temperature fitting, score the validation partition with the frozen model and calibration artifact. A decision is accepted exactly when computable=true and max probability is at least tau. With tau=null, abstention is disabled; no-data decisions remain unavailable.

```text
candidate_thresholds = sorted({0.0} | set(confidences))
for tau in candidate_thresholds:
    accepted = computable & (confidence >= tau)
    evaluate accepted_count, coverage and empirical_risk
choose highest coverage among feasible nonempty sets
```

Feasibility requires the configured empirical risk limit and any minimum accepted count/coverage. Tie-break by lower empirical risk, then lower threshold. Thresholds preserve whole confidence ties; do not arbitrarily split tied decisions to improve apparent coverage. If no nonempty set is feasible, return no_feasible_policy.

**acceptance.yaml inputs** — risk_limit, minimum_accepted_count and minimum_accepted_coverage; each starts null until the team defines the application operating requirement using validation evidence.

**Unset constraints** — Export descriptive risk–coverage curves and disabled policy. Do not claim an accepted production operating point when required constraints are missing.

**Policy artifact** — tau; comparison operator >=; constraints; model and calibration hashes; validation score hash; selected validation count/risk/coverage; selection status.

**Runtime outputs** — accepted, abstained_low_confidence or unavailable_no_data. Preserve predicted probabilities for audit on computable abstentions; user-facing behavior belongs to member 4/5.

Export validation and test risk–coverage curves, but apply only the validation-selected threshold in the test operating-point report. A test curve is descriptive and must not be used to select a new release threshold.

Report matched-coverage comparisons only where threshold-attainable coverage levels match; otherwise show neighboring attainable points. Reusing validation for model and policy selection can create optimism. Empirical validation risk is not a finite-sample safety guarantee and may change under missingness or distribution shift.

## 10 Model comparison and sampling uncertainty

Evaluate source-only references, B-MAJORITY, B-GRID, V0, V1 and V2 under the frozen protocol. Use model seeds 11,22,33 for neural runs and the same saved evaluation schedules for every run. Separate clean and robust training regimes. Include the agreed frozen-encoder V1/V2 ablations to distinguish fusion effects from end-to-end encoder adaptation.

### Comparison table

For each application, model, training regime, seed and scenario, retain macro F1, balanced accuracy, availability coverage, NLL, Brier, ECE and accepted risk/coverage. Report scenario-minus-clean changes with both denominators. Do not rank variants solely by accuracy on their differing computable subsets.

### Group bootstrap

Use 1,000 bootstrap replicates, generator seed 2026 and percentile 95% intervals. Resample independent group IDs with replacement; retain all decisions for each sampled group, including multiplicity. Use participant for PAMAP2. For MOSI, use the audited independent grouping; if only source-video identity is available, label the resulting limitation.

For paired differences, use the identical group draw for both models and identical scenario decision registries. Compute each metric anew inside a replicate. Fixed-class F1 retains the configured class list; do not discard a replicate because a class is absent. Undefined metrics produce null replicates with an explicit valid-replicate count; do not invent interval endpoints when all are undefined.

With fewer than two groups, suppress the interval. With few groups, disclose that bootstrap intervals are descriptive and unstable. Overlapping windows are not independent resampling units. Save the resampled group indices for exact reconstruction.

### Separate training and sampling variability

Report the three seed results plus their arithmetic mean and sample standard deviation with ddof=1. Bootstrap intervals are conditional on a fixed trained run; do not pool seed-by-window rows as independent observations. A seed-averaged paired statistic may use the same group draw across seeds, explicitly described as conditional on those trained runs.

Freeze the release candidate using validation evidence before inspecting final test metrics. Report all prescribed seed runs, including failures, without selecting the best test seed.

## 11 Performance measurement and acceptance rules

**Timing protocol** — Use 20 warm-up calls, then time the fixed evaluation decision list. Record batch size, actual source lengths, dtype, device, thread settings, timed count and cache/I/O policy. Synchronize CUDA before and after timed device sections.

**Measurements** — Separate preprocessing, model forward, calibration/policy and total decision latency. Export p50/p95 using linear-interpolation quantiles. Member 4 supplies replay end-to-end measurements; member 2 supplies training cost.

**Memory** — Record peak process RSS and, where applicable, device peak allocated memory, with measurement method and baseline. Do not compare machines or batch policies as if they were identical.

**Software gates** — All contract, temporal, metric, schedule, calibration, policy and reporting tests pass; no missing decision rows; no non-finite accepted probability; explicit empty-input behavior; complete artifact identities.

**Predictive gates** — acceptance.yaml defines each metric, direction, scenario scope, minimum coverage and numerical bound. Derive bounds from operating needs and measured validation baselines, then freeze them before final test.

**Unset or failed gates** — Missing mandatory quality bounds yields not_qualified, not pass. Report pass/fail/not_applicable per configured rule with measured value, denominator and artifact reference.

Do not insert an arbitrary F1 target or latency promise. First measure the references on validation, identify the allowable quality–coverage–cost trade-off, record the selected bound and its rationale, then freeze it. A relative rule shall identify the exact reference run and whether comparison uses own coverage or common computable decisions.

The test runner requires a frozen manifest covering protocol, partitions, model weights, normalizer, scenario catalogue, calibration, policy and acceptance configuration. Any subsequent change creates a new version and a disclosed re-evaluation. A changed protocol is not silently substituted into an earlier result.

Completion has two explicit statuses: implementation_verified means the software checks pass; predictive_qualification records whether the frozen real-data requirements are met. Passing the first does not imply the second.

## 12 Report files and public execution workflow

**predictions.parquet** — One row per scheduled decision per run/scenario, including exclusions, unavailability, source pattern, scores/probabilities where applicable, predicted class, acceptance and identity fields.

**metrics.json** — Schema version; run/scenario identities; counts; metric values or null reasons; configuration hashes; status. Never serialize NaN or Infinity as JSON numbers.

**class_metrics.csv** — One row per configured class and run/scenario, with support, precision, recall and F1. confusion.csv uses true-class rows and predicted-class columns.

**reliability.csv** — Calibration status, bin edges, count, mean confidence and empirical accuracy. Same bins for all compared models.

**risk_coverage.csv** — Threshold, eligible/computable/accepted counts, coverage and risk. Mark the frozen operating point; preserve confidence ties.

**comparison.csv** — Model/regime/seed/scenario, own and intersection coverage, paired changes and group-bootstrap results. Do not replace per-run records by seed means.

**report.html** — Static report with provenance, counts, quality tables, degradation curves, reliability diagrams, risk–coverage curves, resource costs and explicit limitations.

**provenance.json** — Resolved configuration, dataset/model/calibration/policy hashes, code revision, environment versions, timing context and saved schedule references.

```text
mmt evaluate --protocol protocol.yaml --partition validation
  --model-artifact RUN --scenarios configs/scenarios.yaml
mmt calibrate --model-artifact RUN --partition calibration
mmt select-policy --model-artifact RUN --partition validation
mmt freeze-evaluation --protocol protocol.yaml
mmt evaluate --frozen-manifest RELEASE --partition test
```

Commands are implementation targets integrated with member 5. Arguments must resolve to a single application and artifact set. Write outputs to a new run directory atomically; never overwrite a completed result silently. The report renderer consumes exported files and must not rerun training, recalibration or threshold selection.

## 13 Verification fixtures and exit tests

**Metric arithmetic** — Analytical fixture: y=[0,1], p=[[0.8,0.2],[0.3,0.7]]. NLL=−log(0.56)/2; multiclass Brier=0.13; 15-bin ECE=0.25. These are unit-test expectations, not dataset results.

**Abstention arithmetic** — For that fixture, tau=0.75 accepts one of two decisions: coverage=0.5 and risk=0. Add a no-data eligible row: coverage becomes 1/3, not 1/2.

**Empty cases** — All unavailable: zero availability, null predictive metrics. All abstained: zero accepted coverage and null risk. No eligible labels: null coverage with exclusion counts.

**Class handling** — Absent class retained in fixed-class macro F1; supported-class balanced accuracy documented. Check invalid target and permuted class-order rejection.

**Schedules** — Identical inputs yield identical saved schedules; overlapping windows agree; p=0 removes nothing; forced p=1 test removes all rows; exact outage boundaries follow [a,a+L).

**Causality** — Delay changes availability only; arrivals after decision time are excluded; intervals crossing support boundaries are not clipped into validity; MOSI closure delays flag collapsed severity cases.

**Calibration** — T=1 matches baseline; positive T preserves argmax; V0 adapter uses log scores once; empty/non-finite fit fails; changed checkpoint rejects artifact.

**Policy** — Confidence equal to tau is accepted; tied confidences remain together; no feasible policy is explicit; test labels cannot enter threshold selection.

**Bootstrap and reporting** — Resample groups with multiplicity, pair identical draws, retain null-replicate counts. Recomputed summary matches exported decisions. Reject duplicates and missing decision records.

Use pytest parameterization for both class counts and source schemas. Compare deterministic CPU arithmetic in float64 with atol=1e-10, rtol=1e-8; these are software numerical tolerances. Stochastic schedules are checked through exact saved identities and effects, not by requiring a small sample to match its theoretical probability exactly.

Integration tests shall execute one full validation/calibration/policy/report cycle on a small contract fixture, then one audited real-data run for each application. A fixture verifies implementation; it cannot establish real-data predictive quality.

## 14 Development order and handoff

**E01 Protocol validator** — Implement explicit partition/decision identity checks first. Exit: overlap and class-map failures are caught before evaluation starts.

**E02 Metrics and registry join** — Implement counts, fixed-class metrics and analytical tests before calling a model. Exit: empty/no-data/abstained cases produce the required denominators.

**E03 Saved scenarios** — Implement deterministic sequence-level schedules and member 1 integration. Exit: overlap consistency and shared-selector causality tests pass.

**E04 Evaluation runner** — Connect one selected V0 artifact through member 2/4 interfaces. Exit: every scheduled decision has a status and exported provenance.

**E05 Calibration** — Fit and persist scalar temperature on calibration only. Exit: fresh-process reload and identity mismatch tests pass.

**E06 Abstention** — Implement validation threshold selection and policy serialization. Exit: risk/coverage curves and deployed threshold use identical acceptance semantics.

**E07 Comparisons** — Add prescribed models/seeds/scenarios, paired group bootstrap and resource imports. Exit: denominators and failed runs remain visible.

**E08 Reports and release** — Generate CSV/JSON/HTML, freeze evaluation manifest and execute final test. Exit: every acceptance rule has evidence or an explicit not-qualified status.

First executable milestone: given a decision registry and saved scores, produce correct metrics.json and predictions.parquet without access to model internals. Then attach scenario execution. This keeps metric correctness independently testable while other members implement their components.

Handoff includes evaluation source code, configuration schemas, protocol and schedules, calibration/policy artifacts, test results, report exports, environment lock reference and a short command guide. Member 1 signs off transformation placement; member 2 confirms score semantics; member 4 verifies runtime policy parity; member 5 verifies packaged command and report operation.

Do not declare this work package complete with only notebooks or charts. The deliverable is a reusable tested evaluator, reproducible artifacts and documented real-data evidence for both application profiles.

## 15 Sources and technical decision register

[1] Guo, Pleiss, Sun and Weinberger. On Calibration of Modern Neural Networks. ICML / PMLR 70, 2017. Reference for scalar temperature scaling; no project-specific improvement is inferred.
https://proceedings.mlr.press/v70/guo17a.html

[2] scikit-learn. f1_score API documentation. Reference for labels, macro averaging and zero_division behavior.
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html

[3] PyTorch. LBFGS API documentation. Reference for closure-based optimization and optimizer parameters.
https://docs.pytorch.org/docs/stable/generated/torch.optim.LBFGS.html

### Project decisions and evidence status

**Inherited contracts** — MMT-TS-001 v1.0; MMT-WP-001 Member 1; MMT-WP-002 Member 2. Dataset bindings, model score conventions and temporal selection remain shared contracts.

**Engineering defaults** — Scenario severities, schedule seed, 15 ECE bins, 1,000 bootstrap replicates and temperature optimizer settings specify the first implementation. They are not externally validated optimal settings.

**Application evidence** — Dataset access, manifest identities, actual group counts, class support and MOSI feature timing require member 1’s audit. No audit outcome is invented in this document.

**Results to produce** — Real-data quality, degradation, calibration changes, accepted risk/coverage and resource use must be measured. No performance numbers in this document represent an experiment.

**Scope limitation** — No raw-feature extraction, new model architecture, GUI implementation or uncertainty decomposition is assigned to member 3. Those require the corresponding owner or a separately approved scope change.

This work package is an implementation specification. Configuration schemas and API names are development targets; they must be integrated into the shared repository and verified before being described as existing framework capabilities.
