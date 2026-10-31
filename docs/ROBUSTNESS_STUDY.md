# Etude de Robustesse Partielle

## Final Results

All three requested blocks completed: 230 evaluation exports, 54 model/scenario
replay runs and 204 paired comparisons, each with 1,000 saved group-bootstrap
draws for Macro-F1 and balanced accuracy. The report contains 11 nonblank PNG
graphs and tables. No checkpoint was retrained or changed.

The detailed HTML report, per-run evaluation rows, paired comparisons and replay
measurements remain local under the ignored `runs/` directory. The versioned
[consolidated notebook and exported tables/figures](NOTEBOOK_REPORT_README.md)
contain the shareable results. Mean Macro-F1 over the three nominal seeds:

| Application / Condition | V0 | V1 | V2 | Coverage |
| --- | ---: | ---: | ---: | ---: |
| MOSI clean | 0.5674 | 0.6534 | 0.6927 | 1.0 |
| MOSI global removal 30% | 0.5283 | 0.6345 | 0.6557 | 1.0 |
| MOSI global delay 25% of duration | null | null | null | 0.0 |
| PAMAP2 clean | 0.7417 | 0.7056 | 0.7594 | 1.0 |
| PAMAP2 global removal 30% | 0.7237 | 0.7046 | 0.7591 | 1.0 |
| PAMAP2 global delay 25% of support | 0.7089 | 0.6900 | 0.7532 | 1.0 |

The null MOSI delay metric is expected offline closure behavior, not a zero
classification score. Remaining rates and leave-one-source-out results, seed
variability and coverage are in the HTML/CSV artifacts.

Clean baselines: MOSI majority=0.2969, language=0.6548, audio=0.3320,
visual=0.4988. PAMAP2 majority=0.0170, hand=0.6316, chest=0.5048,
ankle=0.5156, heart-rate=0.2498. Source-only figures are three-seed means;
majority is fitted once, not treated as three independent neural runs.

Whole-test replay produced 236,655 model/decision records: every one of the
686 MOSI segments and 8,079 PAMAP2 scheduled decisions, all nine models per
application, in each of three conditions. Strict stateless-unit/streaming
checks passed without ID, mask, class, status or probability mismatches.
There were 563 numerical batch32 discrepancies, with maximum rechecked score
error 0.000020264; every such case passed the unchanged batch1 tolerance.
This is not a claim of bitwise or strict-tolerance equivalence across batch sizes.

Clean prediction p95, averaged over seed-specific p95s: MOSI V0/V1/V2
22.974/23.341/23.819 ms; PAMAP2 12.697/12.677/13.433 ms. Actual
ingestion+prediction per-decision quantiles are saved separately; adding two
individual p95s would be incorrect. The peak process RSS across conditions
was approximately 1.28 GB; the largest accounted shared observation buffer
was 1,926,155 bytes. These are measured workload-specific values, not an SLA.

V2-minus-V0 clean Macro-F1 differences on MOSI were positive for all three
seeds (0.0594, 0.1268, 0.1897). PAMAP2 differences varied by seed
(0.1340, -0.0204, -0.0608); only two independent test participants are
available, so the bootstrap intervals are unstable and not a reliable
general superiority/significance claim. All observations remain descriptive
on these fixed partitions; no predictive qualification is granted.

## Scope

This is a post-clean, descriptive frozen-checkpoint evaluation. Test outcomes from
the clean campaign were already observed; no model hyperparameters, partitions,
weights, calibration temperatures or thresholds are tuned in this extension.
No retraining is performed. It is not a new blind test or a qualified release.

All test decisions are used: 686 MOSI segments (656 scored) and 8,079 PAMAP2
decisions from both complete test recordings (5,133 scored). All V0/V1/V2
checkpoints from nominal seeds 11/22/33 are included.

Each application has these conditions:

- Clean, natural missingness retained.
- All sources absent and each source removed individually (leave-one-source-out).
- Global independent observation removal with probabilities 0.1, 0.3 and 0.5.
- Global added reception delay / support duration of 0.1, 0.25 and 0.5.

There are 11 MOSI and 12 PAMAP2 conditions. The remaining source combinations,
per-source removal/delay, contiguous outages, out-of-order arrivals, recovery-delay
compositions, robust training and frozen-encoder ablations are not included.
Therefore every result/report is named **etude de robustesse partielle**.

## Shared Schedules and Denominators

[robustness_study.py](../robustness_study.py) saves each schedule once, using seed
1701 and protocol ID `robustness-20261031`. Identical schedules are loaded for every
compared checkpoint; source observation identities and schedule hashes are verified.
Delay changes only `available_at`; labels and event intervals remain unchanged.
Corruption acts on canonical rows before the shared selector, not on padded tensors.

PAMAP2 uses reference support duration 5 seconds. MOSI uses each segment's actual
duration. Its feature availability is offline closure: any positive global delay
makes all sources unavailable at closure. Different delay levels consequently
have identical no-data results; do not claim graded streaming robustness.

Every scheduled decision is exported. Macro-F1 and balanced accuracy are reported
on computable scored decisions; coverage uses all eligible decisions. No-data
quality is null, not zero. Scenario-minus-clean changes and coverage are reported
separately; never rank models on changing computable subsets alone.

Fusion probability metrics use the original per-run calibration, including existing
T=1 fallbacks. No temperature is refitted on corrupted or test data.

## Baselines

Majority and each source-only reference are evaluated on clean test data. Majority
is deterministic from eligible train targets only, with low-index tie-breaking and
the same no-input guard. It returns labels without invented confidence/probabilities.

Source-only references use the selected V0 source classifier weights, with the same
normalizer and class order. Exact equality to all 21 constituent checkpoints across
the six V0 runs is recorded in `baseline_provenance.json`. Internal source training
seeds are nominal seed plus source index; these values are recorded explicitly.
Their probability metrics are uncalibrated: do not attribute differences versus
calibrated fusion probabilities solely to the model architecture.

## Paired Bootstrap

Pairwise comparisons use the intersection of computable eligible decision IDs,
retaining each model's original coverage. References are V0 and, in clean, majority.
The statistic is fixed-class Macro-F1. Each comparison uses 1,000 PCG64 draws,
seed 2026, resampling participant/video groups with multiplicity and the same
draw for both models. Per-group confusion matrices are summed per replicate.
Draw indices, observed differences, valid replicate counts and percentile 95%
intervals are saved. Undefined replicates remain null, not invented zero errors.

Intervals are conditional on each trained seed; mean/sample SD between training
seeds is a different quantity. PAMAP2 has only two test participants: bootstrap
intervals are descriptive and unstable, not a sound significance claim. MOSI
grouping is source video; speaker independence is not verified.

## Global Recorded Replay

[replay_study.py](../replay_study.py) processes every test sequence for all 18
fusion checkpoints in clean, global removal 0.3 and global delay 0.25. This is
54 model/scenario runs, not one recording or a few cherry-picked segments.

Reception order is chronological. A shared bounded buffer ingests each sequence
once for nine frozen predictors; model calls and states are separate. This saves
duplicate ingestion but is not a nine-process memory benchmark. There is one
active sequence at a time; the byte/row capacity limits are retained.

Streaming selected IDs and feature masks are compared with the pure selector
on canonical transformed sequences. Scores, classes, probabilities, computability
and statuses are compared with persisted batch evaluation exports. CPU tolerance
is atol=1e-6, rtol=1e-5. Batch32 numerical discrepancies are counted and rechecked
against stateless batch1 with the unchanged tolerance; class/status differences
are never tolerated. The ID trace also includes per-source mask SHA-256 values.

The initial same-process reference-forward replay and a batch32 numerical failure
are preserved in old/`.incomplete-*` directories for diagnosis. Final evidence
uses `mosi/replay-batch-proof` and `pamap2/replay-shared-proof`; incomplete
attempts are not final results. PAMAP2 shares one immutable selected buffer
view across nine predictors, with that selection cost included in each
prediction measurement. Model forwards remain independent.

Timing uses `perf_counter_ns`, float32, CPU, four threads, batch size one and
20 warm-up forwards per model/condition. Prediction time includes selection,
normalization, collation, forward and postprocessing; ingestion is timed separately.
Stateless parity checks and trace-file writing are excluded. Reported p50/p95 are
per-run quantiles, with sample/decision counts. File I/O and raw-feature extraction
are not claimed as timed online latency.

Memory records buffer estimated bytes/row high-water marks and process lifetime
peak RSS (`resource.getrusage`, bytes on macOS). RSS is cumulative across the
process, not an isolated per-model or per-condition allocation. Buffer accounting
is not total process memory. No physical sensor or causal MOSI acquisition is tested.
Evaluation and replay overlapped on this machine; measured latency describes
that workload rather than an idle-machine benchmark.

## Outputs and Verification

Output root: `runs/robustness-20261031/`.

- Application catalogues and `schedules/`: saved transformations and hashes.
- `evaluations/`: per-model/condition predictions, metrics, provenance and class tables.
- `evaluation_summary.csv`: all fusion and clean-baseline rows.
- `degradation_deltas.csv`: scenario-minus-clean quality/coverage changes.
- `paired_bootstrap.csv` and `bootstrap/`: common-ID comparisons and saved draws.
- `replay-batch-proof/`: predictions, selected IDs/mask hashes, timings and status.
- `replay_summary.csv`: resource measurements and parity results.
- `report.html` and `figures/`: consolidated tables and graphics.
- Status JSON files distinguish completed blocks from preview/incomplete output.

```sh
uv run --locked python robustness_study.py --evaluate
uv run --locked python robustness_study.py --bootstrap
uv run --locked python replay_study.py
uv run --locked python study_report.py
```

The final report requires evaluation, bootstrap and replay status=completed.
An earlier preview is marked incomplete. The new clean fusion exports were checked
against the original 18 test results: Macro-F1 difference was exactly zero, and
trained weights remained identical to the frozen campaign hashes.