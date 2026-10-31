# Experiment and Data Readiness Log

## 2026-10-31 — Reviewer UI Tested and Corrected

The old 8501 server returned HTTP 500. A corrected test instance on 8502 loads
both prepared applications, 284 nested evaluation exports and nine completed
replay directories. Discovery, multi-model replay filtering, navigation callbacks,
scoped label-free export, matched config/postprocessing and malformed JSON handling
were corrected and tested. A UI-generated headless MOSI segment command matched
the original IDs/scores/probabilities/status. The suite now passes 124 tests and
Ruff. Browser interaction tested four tabs at desktop/mobile CSS viewports;
native download completion and screenshot dimensions remain unverified.
See [UI_TEST_REPORT.md](UI_TEST_REPORT.md) for precise evidence and limitations.

## 2026-10-31 — Robustness, Replay and Baselines Completed

The frozen trained checkpoints were reused without retraining for an explicitly
partial robustness study: clean, empty and leave-one-source-out, global removal
10/30/50%, and global reception delay 10/25/50% of support. All test decisions
and all 18 fusion checkpoints were evaluated with identical saved schedules
(seed 1701). Source-only and majority baselines were evaluated in clean.

Final verified outputs: 230 evaluation exports, 54 complete replay model/scenario
runs over all test sequences, 204 paired Macro-F1/BA comparisons with 1,000
saved group draws each (seed 2026), and 11 graphs in a consolidated report.
Clean Macro-F1 matched the original 18 run results exactly; all 21 source-only
checkpoint tensors matched their V0 constituents exactly. Trained weights stayed
unchanged. The baseline probability metrics remain uncalibrated and label-only
majority confidence metrics remain unavailable.

Replays include clean/removal0.3/delay0.25 on every test recording/segment for
all models/seeds, with 236,655 model/decision records. Selected IDs, validity masks,
scores, probabilities, class and status were checked. 563 batch32 numerical
discrepancies were recorded and successfully rechecked against stateless batch1
at the existing CPU tolerances; batch-size equivalence is not claimed falsely.
Previous interrupted/failed attempts are retained and excluded from final results.
Prediction and insertion quantiles, cumulative process peak RSS and shared-buffer
high-water marks are available, with measurement scope and concurrent load disclosed.

Results, commands, tables and scientific limitations are in
[ROBUSTNESS_STUDY.md](ROBUSTNESS_STUDY.md) and the
The consolidated notebook and shareable tables/figures are described in
[NOTEBOOK_REPORT_README.md](NOTEBOOK_REPORT_README.md); full HTML/CSV run outputs
remain under the ignored local `runs/` directory.
The complete perturbation catalogue, robust training, frozen-encoder ablations,
B-GRID workflow and predictive qualification remain outside this study.

## Clean Campaign — Final Results

Campaign `clean-long-20261027-run1` completed successfully: 144/144 stages,
18/18 trained runs, 18 test exports and HTML reports, plus 18 calibration and
policy artifacts. Every run used seed 11, 22 or 33. The campaign status is
`completed`; predictive qualification remains `not_qualified` because no
application acceptance bounds were approved.

Macro-F1, balanced accuracy (BA), NLL, multiclass Brier, ECE and coverage below
are test means over all three seeds; `±` is the sample standard deviation
(`ddof=1`). No best seed was selected. Every model had 100% availability coverage
on these clean test partitions.

| Application | Model | Macro-F1 | BA | NLL | Brier | ECE | Coverage |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MOSI | V0 | 0.5674 ± 0.0458 | 0.6291 ± 0.0201 | 0.6815 ± 0.0269 | 0.4871 ± 0.0257 | 0.0917 ± 0.0315 | 1.000 |
| MOSI | V1 | 0.6534 ± 0.0103 | 0.6634 ± 0.0146 | 1.1365 ± 0.8976 | 0.4959 ± 0.1140 | 0.1375 ± 0.1427 | 1.000 |
| MOSI | V2 | 0.6927 ± 0.0258 | 0.7004 ± 0.0155 | 0.6835 ± 0.0947 | 0.4430 ± 0.0400 | 0.1327 ± 0.0742 | 1.000 |
| PAMAP2 | V0 | 0.7417 ± 0.0909 | 0.7378 ± 0.0884 | 0.8352 ± 0.0652 | 0.3395 ± 0.0529 | 0.0656 ± 0.0061 | 1.000 |
| PAMAP2 | V1 | 0.7056 ± 0.1026 | 0.7129 ± 0.0883 | 1.0968 ± 0.3498 | 0.4317 ± 0.1323 | 0.1578 ± 0.0554 | 1.000 |
| PAMAP2 | V2 | 0.7594 ± 0.0383 | 0.7650 ± 0.0406 | 0.9193 ± 0.3708 | 0.3637 ± 0.0968 | 0.1084 ± 0.0914 | 1.000 |

Validation, all per-seed rows, denominators, confusion/class metrics and
provenance are in local `runs/clean-long-20261027-run1/results.csv` and
`runs/clean-long-20261027-run1/summary.json` (excluded from Git). The versioned
consolidated notebook preserves shareable result tables. The frozen checkpoint,
normalizer, calibration and policy hashes are in `frozen_artifacts.json`.
Per-run HTML reports and test exports are under `results/<run_id>/`.

Calibration status was `ok` for 12 runs and `fit_failed` for 6; failed fits
use the explicit T=1 fallback. Because no risk/coverage constraints were
approved, all 18 abstention policies are `disabled_no_constraints`. Thus the
table's confidence metrics for those six runs are uncalibrated. MOSI remains an
offline mirror profile with one official development video missing and
redistribution conditions unverified. PAMAP2 uses the user-requested,
label-informed participant split, one calibration participant, and simulated
arrival times. These results are descriptive on those fixed partitions, not a
claim of official-split comparability, causal streaming performance, or release
qualification. Robust-training, frozen-encoder ablations and the full baseline/
perturbation matrix were not part of this campaign.

## 2026-10-27 — Long Clean Experiment Campaign Started

Campaign ID: `clean-long-20261027-run1`. Actual launch time:
`2026-10-27T16:26:54Z`. This section records the pre-run protocol and artifacts;
final results and qualification caveats appear above.

### Frozen Execution Plan

[run_experiments.py](../run_experiments.py) executes 18 runs: MOSI and PAMAP2,
V0/V1/V2, seeds 11/22/33. The full prepared train/validation partitions are used;
there is no subsampling, reduced epoch limit or test-based choice of seed.
The maximum is 100 epochs with patience 10, batch size 32, hidden size 64,
dropout 0.1 and the existing AdamW settings. Early stopping is intentional:
a run can finish before 100 epochs without being a smoke test.

Each source-only V0 checkpoint is selected by its validation cross-entropy.
Joint V1/V2 selection uses validation macro-F1, then validation loss. The
normalizer uses train observations only. Calibration uses calibration labels;
policy selection uses validation only. Unset operating constraints produce an
explicitly disabled policy, not an invented production threshold.

All 18 selected models, normalizers and postprocessing artifacts are frozen
before the first test-quality evaluation. Final test results are descriptive
clean experiments: acceptance bounds are still unset, so predictive qualification
remains `not_qualified`. No model is selected by test scores. Robust training,
frozen-encoder ablations and the full baseline/scenario matrix are not covered
by this clean campaign.

The immutable plan records code SHA-256 values, the lockfile hash, resolved
configuration hashes and prepared input hashes. Code or input changes prevent
reusing that plan. Completed run directories are not overwritten. The older
incomplete PAMAP2 seed-11 run is preserved and is not included in this campaign.

### Hardware Probe and Software Checks

Machine reports 10 CPU cores, 16 GiB RAM and available Apple MPS. A real MOSI
train batch of 32 decisions had shapes language `[32,42,300]`, audio
`[32,1299,74]`, visual `[32,391,35]`. A disposable V1 forward/backward probe
measured 0.631 seconds on CPU with one thread, 0.415 seconds on CPU with four
threads, and 5.191 seconds on MPS with one thread. All losses were finite.
These are single-call hardware probes, not warmed-up benchmarks, model-quality
results or runtime latency claims. Probe weights were discarded and no optimizer
update was applied to an experimental model.

The campaign uses CPU with four requested OMP/MKL/OpenBLAS threads, float32 and
sequential runs to avoid memory competition. Python is 3.11.16 and PyTorch is
2.14.1. Before launch, all 118 tests and Ruff passed, including a test proving
that result aggregation retains F1 from `quality_S`/`quality_A` and reports
all seeds with sample standard deviation rather than the best seed.

### Commands and Outputs

```sh
uv run --locked python run_experiments.py --campaign clean-long-20261027-run1 --execute
uv run --locked python run_experiments.py --campaign clean-long-20261027-run1 --status
```

The first command was launched and ran the campaign sequentially. The campaign
is complete; do not rerun it under the same immutable output identity. The preliminary plan under
`runs/clean-long-20261027/` was a dry-run plan made before final formatting;
only `clean-long-20261027-run1` is the active campaign identity.

Artifacts under `runs/clean-long-20261027-run1/`:

- `plan.json`: frozen matrix, hashes, protocol and environment.
- `status.json`: persisted stage status, command, elapsed time and failure details.
- `logs/`: per-command stdout/stderr logs.
- `results/<run_id>/`: validation/calibration/test exports, calibration, policy and HTML report.
- `results.csv` and `summary.json`: incrementally generated real result tables;
   these appear only after evaluations complete. Mean and sample standard deviation
   are computed over the available seeds, with the completed seed count explicit.
- `frozen_artifacts.json`: selected artifact hashes written before any test-quality evaluation.

Training weights and checkpoints live separately at `runs/<run_id>/`. While a
run is active its temporary directory is hidden under `runs/.<run_id>.tmp-*/`.
Source/V0 and joint training now append `history.jsonl` once per completed epoch.
These logs include train loss, validation loss/F1/coverage and update counts;
completed runs include their checksums. Stage wall times are recorded separately.

At the first progress check, MOSI V0 seed 11 had completed the language-source
training (14 epochs) and 10 audio-source epochs. This proves actual training
progress; it is not the final fused V0 metric or a completed 18-run campaign.
Do not quote interim last-epoch values as selected-checkpoint or test results.

During execution, an interrupted neural run would be retained as incomplete;
the launcher can skip completed stages but does not implement automatic resume
of an incomplete neural run. This completed campaign has no failed or incomplete
stages.

## 2026-10-27 — V1/V2 End-to-End Integration

Clean joint training is now integrated for V1/V2, in addition to independent-source
V0 training. The service seeds construction, reuses the train-only normalizer,
DataLoader and generic trainer, selects on validation macro-F1/loss, and publishes
best/last checkpoints, selected inference weights, architecture metadata, training
history, environment, source/class schemas and checksums atomically.

Evaluation and replay strictly reconstruct the saved architecture and source order;
V1/V2 scores are exported as logits. Calibration/policy files can now be supplied
through `--calibration` and `--policy` to both commands, with checkpoint, class-map,
dataset and partition identity validation. Evaluation moves batches to the requested
device and uses the existing fast selector for validated monotonic streams.
The prior V0 Python entry-point names remain compatibility aliases.

Four real-data configs were added for PAMAP2/MOSI x V1/V2, with seed 11 defaults.
All resolve against the existing prepared artifacts. Overrides for seeds 22/33 and
distinct `--run-id` values were validated. No long real-data training was launched
as part of this software integration.

Validation executed:

```sh
uv run --locked --extra dev --extra ui python -m unittest discover -s tests -p test_training_service.py -v
uv run --locked --extra dev --extra ui python -m unittest discover -s tests -p test_models.py -v
uv run --locked --extra dev --extra ui python -m unittest discover -s tests
uv run --locked --extra dev --extra ui ruff check src tests
```

Result: 117 tests pass on Python 3.11.16; Ruff passes. The expanded fixture chain
covers both applications and both models through training, evaluation, calibration,
policy, report and replay, including absent sources, score parity, saved calibration,
wrong-artifact rejection and exact repeat weights at fixed seed. Fresh-process
reload verifies source ordering and non-default V2 constructor options.

See [FUSION_WORKFLOW.md](FUSION_WORKFLOW.md) for executable commands. Robust-training
schedules, frozen-encoder ablations, baseline workflow completion and final experiment
matrix orchestration remain outside this integration. Robust training is rejected
explicitly. No real-data predictive metric or qualification is claimed.

## 2026-10-27 — MOSI Ready for Local Offline Training

The user explicitly authorized a frozen pretrained GloVe lookup for timestamped
words. This is a named extension of the original data work package, not an
unannounced replacement of the official SDK word vectors. This entry supersedes
the earlier numeric-language-feature and local partition blockers. Formal data
review, mirror redistribution conditions and predictive qualification remain open.

### Profile and Inputs

Profile: `native-offline-glove6b300-covarep74-facet42-35`.
Language is fixed GloVe Wikipedia 2014/Gigaword (6B tokens), 300 dimensions;
audio is COVAREP, 74 dimensions; visual is FACET 4.2, 35 dimensions.
Audio and visual streams retain their own observation counts and intervals;
neither is averaged or interpolated onto words. Time origins are converted from
video-relative to segment-relative seconds. All admitted observations are made
available at segment closure, explicitly `availability_mode=offline`.

- Raw MultiBench HDF5 SHA-256:
   `4915dbd1ad174ebd4967843188d8a678c4c3c0be4e7dce99edb6f50b89c3c29b`.
- SDK fold source revision: `4f2eadcd7e7b9e20e83b868cdad385b41830285c`.
- Fold file SHA-256:
   `027ca94558d1f2ac0b30b7a773f2b65439dc437fa42634c41e1fcf635d56127d`.
- GloVe provider: `piskvorky/gensim-data`, release `glove-wiki-gigaword-300`;
   gzip word2vec-text format, 400,000 entries. Published MD5
   `29e9329ac2241937d55b852e8284e89b` verified.
- GloVe SHA-256:
   `0a7aebbe49097dc6e5ffff7a25e9aa20181a6e862050ca68bd2f36e056739e00`.
- The GloVe distributor lists PDDL for these vectors. This does not confer any
   license on the MOSI mirror; redistribution rights for that input remain unverified.

Words are decoded and lowercased for fixed-vector lookup. Silence (`sp`, `<sp>`,
empty tokens) and out-of-vocabulary words receive zero values with false masks;
they do not become valid zero-valued measurements. The existing segment adapter
omits all-invalid observations from canonical inputs; original word-row IDs and
per-segment exclusion counts remain in the audit. No embeddings or tokenizer are
learned on MOSI. Collecting the input vocabulary only selects entries from the
external pretrained dictionary; it does not fit preprocessing statistics.

Word audit: 32,084 observations, 26,674 matched, 5,390 silence, 20 unknown.
Coverage among non-silence word occurrences is 99.9251%. COVAREP contains 119
non-finite feature entries, masked before preparation; FACET contains none.
Native intervals do not establish causal feature-extraction arrival times.

### Partitions

The official SDK train and test assignments are retained for every available
video. The mirror has 92 videos, not all 93 listed by the SDK: development video
`c5xsKMxpXnc` is missing. It is not replaced with test data or synthesized.

Five of the nine available development videos form calibration; the other four
remain validation. All 126 possible five-video subsets were enumerated. The
criterion is eligible-segment-count-weighted Jensen-Shannon divergence of the
two sentiment distributions from the combined development distribution, with
both classes required in each group and lexicographic video-ID tie-breaking.
The selected objective is 0.00000029342397. This is a project-specific,
development-label-informed split, not the untouched standard validation protocol.
No model results or final-test labels are used to choose calibration membership.

Calibration videos: `WKA5OygbEKI`, `X3j2zQgwYgE`, `ZAIRrfG22O0`,
`ZUXBRvtny7o`, `atnd_PF-Lbs`.

| Partition | Videos | Scheduled Segments | Eligible | Negative | Positive |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 52 | 1,283 | 1,230 | 552 | 678 |
| validation | 4 | 117 | 111 | 43 | 68 |
| calibration | 5 | 97 | 90 | 35 | 55 |
| test | 31 | 686 | 656 | 379 | 277 |

All 2,183 segments are retained; 96 zero-sentiment targets are excluded from
supervised scoring. Negative scores map to class 0, positive scores to class 1.
Every video belongs to exactly one partition, and every partition has both classes.
Grouping proves source-video disjointness only; speaker identity is not verified.
Calibration and validation are small; report uncertainty and class support.

### Artifacts and Execution

- Script: [audit_mosi.py](../audit_mosi.py).
- Preparation config: [configs/mosi-preparation.yaml](../configs/mosi-preparation.yaml).
- Versioned segment membership: [configs/mosi-membership.csv](../configs/mosi-membership.csv).
- Prepared root: `data/prepared/mosi/native-offline-glove6b300-2026-10-27-v2/`.
- Prepared manifest SHA-256:
   `56b342f9bebccc6cda0dc2b670527937e054f27d1b10370c6cc0daf6aebd64c9`.
- Audit and fitted normalizer:
   `data/raw/mosi/provenance/2026-10-27/native-offline-glove6b300-2026-10-27-v2/`.
- Canonical tables include decisions, targets, partitions, original video/segment
   joins, and per-segment timing/word/non-finite-feature audit records.
- V0 run configs exist for seeds 11, 22 and 33. All resolved successfully.

Commands executed:

```sh
uv run --locked python audit_mosi.py --acquire
uv run --locked python audit_mosi.py --prepare
uv run --locked mmt validate-splits --config configs/mosi-v0-seed-11.yaml
uv run --locked --extra dev --extra ui ruff check audit_mosi.py src tests
uv run --locked --extra dev --extra ui python -m unittest discover -s tests
```

The first preparation was rejected by the shared validator because textual class-map
keys were not integer-convertible. It was retained for diagnosis and superseded
without overwriting by `v2`, whose sign-based class map is `-1 -> 0`, `1 -> 1`.
Only `v2` is referenced by run configs. The script refuses to overwrite completed
outputs; use a new preparation ID and evidence destination for a new profile.

Verification: shared split validation passed; a normalizer was fitted on train
sources only and saved with dataset/membership identities; real DataLoader batches
of up to 32 decisions were checked for shapes and finite values in all four splits.
An eight-decision real train batch also passed untrained V0, V1 and V2 forwards
with finite `[8,2]` scores and eight computable rows. These forwards are integration
checks, not predictive metrics. The full suite passes 115 tests under Python 3.11.16;
Ruff passes. No MOSI weights were trained and no test quality metrics were computed.

To start the first real V0 run after reviewing this local profile:

```sh
uv run --locked mmt train --config configs/mosi-v0-seed-11.yaml --device cpu
```

The training service currently refits its own train-only normalizer; the separately
saved normalizer proves the preparation path and is not silently treated as a trained
model artifact. Status: `training_ready_local_offline`, not `predictively_qualified`.

## 2026-10-27 — Deterministic PAMAP2 Partitions

At the user's request, a participant-disjoint partition was selected using the existing
raw-data audit. This supersedes the earlier missing-membership blocker for local
exploratory PAMAP2 preparation, but is not independent protocol review or release qualification.
MOSI blockers remain unchanged.

### Selection Method

`audit_pamap2.py` verifies the original nine files against audited sizes and SHA-256
checksums, then exhaustively enumerates assignments with 4 train, 2 validation,
1 calibration and 2 test participants. These sizes are an explicit project choice,
not an official PAMAP2 split: they retain four training participants and two groups
for each validation/test comparison while reserving a separate calibration subject.
Subject 109 is short and contributes only activity 24, so four train participants
do not represent four equally informative recordings.

Every split must contain all 12 classes. Of 3,780 assignments, 1,510 meet that constraint.
The selected assignment minimizes the sum of Jensen-Shannon divergences between each
split's class distribution and the global distribution, weighted by selected-class row
counts (activity 0 excluded). Natural logarithms are used; the objective is
0.00149052688447. Scores are rounded to 14 decimal places for deterministic comparison,
then tied candidates are ordered lexicographically by subject IDs in split order.
No random generator or model accuracy enters selection.

This is label-informed group stratification, including held-out label counts, not a
previously untouched official benchmark split. It must be disclosed in the report.
Do not revise these groups after viewing model outcomes. Class presence does not
guarantee equal class support or generalization to other participants.

### Membership and Decision Counts

The versioned assignment is [configs/pamap2-membership.csv](../configs/pamap2-membership.csv).
Each participant appears exactly once, with `sequence_id=group_id`.

| Partition | Participants | Scheduled Decisions | Eligible Decisions |
| --- | --- | ---: | ---: |
| train | 103, 105, 107, 109 | 9,478 | 6,858 |
| validation | 101, 104 | 7,051 | 4,817 |
| calibration | 108 | 4,076 | 2,621 |
| test | 102, 106 | 8,079 | 5,133 |

There are 28,684 scheduled decisions and 19,429 scored targets; 9,255 are excluded
for activity 0. The following counts are actual eligible decisions after the
5-second window, 1-second stride and 0.02-second label-age rule, not raw row counts.

| Raw Activity | Train | Validation | Calibration | Test |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 713 | 503 | 242 | 468 |
| 2 | 679 | 489 | 229 | 454 |
| 3 | 685 | 465 | 251 | 498 |
| 4 | 949 | 542 | 316 | 582 |
| 5 | 283 | 213 | 166 | 321 |
| 6 | 473 | 463 | 254 | 456 |
| 7 | 549 | 479 | 289 | 564 |
| 12 | 422 | 327 | 117 | 306 |
| 13 | 396 | 291 | 96 | 264 |
| 16 | 663 | 430 | 243 | 417 |
| 17 | 905 | 486 | 330 | 667 |
| 24 | 141 | 129 | 88 | 136 |

All 12 classes remain present in every split. Calibration has only one participant,
so participant-bootstrap intervals there must be suppressed. Test has only two
participants; any group-level interval is descriptive and unstable. Overlapping
windows are not independent samples.

### Execution Evidence

1. `uv run --locked python audit_pamap2.py` verified all nine input files and previewed the assignment.
2. `uv run --locked python audit_pamap2.py --write` saved `membership.csv`,
   `partition_report.json` and `partition_report.md` under
   `data/raw/pamap2/provenance/2026-10-26/partition-proposal/`.
   Existing output directories are never overwritten; use `--output-dir` for a new proposal.
3. `uv run --locked mmt audit --config configs/pamap2-preparation.yaml` audited nine
   recordings with the new membership; its formal status remains `audit_pending_review`.
4. `uv run --locked mmt prepare --config configs/pamap2-preparation.yaml` published
   `data/prepared/pamap2/pamap2/protocol-grouped-2026-10-27/`, with masks, separate
   targets, decisions, partitions and checksums. The repeated `pamap2` path component
   follows the existing storage API and is used consistently in the run configs.
5. `uv run --locked mmt validate-splits --config configs/pamap2-v0-seed-11.yaml`
   returned `status=valid`, with no group overlap or missing training class.
   The seed-11/22/33 run configurations all validated; seeds 22/33 were not trained.

Prepared manifest SHA-256:
`97ad1163a65ceadd2158cf286b6de72db6f50456c050e3950a965a9a24360e74`.

### Training Attempt and Performance Fix

The first seed-11 training attempt was stopped during normalization: each window
rescanned an entire recording. The normalizer now unions eligible observations and
merges float64 statistics without repeated full scans. The Dataset uses an opt-in
binary-search path in the shared selector after checking monotonic timestamps once.
Reference behavior remains available for unordered sources.

Focused tests check overlap deduplication, masks, boundary-crossing intervals,
delays, ordered/unordered parity and deterministic class-covered partitions.
The actual normalizer fit over all 9,478 train decisions took 0.714 seconds in the
local measured invocation; this is preprocessing cost, not model latency.
The source-valid counts were 945,834 hand, 948,600 chest, 945,850 ankle per feature,
and 86,714 heart-rate observations. This follows the existing service convention
of all scheduled train decisions, including decisions with excluded targets.

A second seed-11 CPU attempt reached the hand-source checkpoints and was interrupted
explicitly with Ctrl-C (exit 130) to complete the partition delivery. Its output is
incomplete, not a trained V0 or a scientific result; no final-test predictions were
computed. Preserve the partial run for diagnosis and use a new run ID for a fresh
training command. No calibration, robustness matrix or final report is complete.

### Final Verification

- `uv run --locked --extra dev --extra ui ruff check audit_pamap2.py src tests`: passed.
- `uv run --locked --extra dev --extra ui python -m unittest discover -s tests`:
   111 tests passed on Python 3.11.16.
- The versioned membership was compared with the prepared `partitions.csv` using
   `pandas.testing.assert_frame_equal`; it matches exactly. All four splits pass
   the project's participant-disjoint membership validation.
- The local `.venv` now uses the project-target Python 3.11.16 with dev/UI extras.
   This supersedes the earlier 2026-10-26 environment-restoration entry.

## 2026-10-26 — Current readiness

### Data evidence

- PAMAP2: the nine official Protocol recordings were read and schema-audited. The audit covers 2,872,533 rows and reports `files-readable-review-and-partitions-pending`. No participant membership file or approved partitions are present. The raw-data notice records class-support gaps and simulated, not measured, availability.
- MOSI: `data/raw/mosi/multibench/mosi_unalign.hdf5` was downloaded from the public MultiBench Drive folder and structurally inspected. It is a mirror candidate, not the official SDK acquisition. Its six streams share 2,183 segment IDs and contain feature intervals; COVAREP, FACET, OpenFace and OpenSmile dimensions are 74, 35, 430 and 1,585. The `words` stream is byte-string text, not model-ready numeric embeddings. COVAREP contains 119 non-finite values. The hash, size and detailed limitations are recorded in `data/raw/mosi/README.md`.
- MOSI remains blocked for a full three-source experiment: mirror redistribution terms and original usage permission have not been verified; numeric language features and approved group partitions are missing. Native event intervals do not establish extraction-arrival times, so only an explicitly offline profile could be considered after approval.
- There are currently no checked-in YAML experiment configurations and no prepared real-data manifests, partition tables, normalizers or real-data runs.

### Executed checks

1. `uv run --python /tmp/mmt-python311/bin/python --locked python -m unittest discover -s tests -p 'test_training_service.py' -v`
   - Passed: the fixture integration trains independent source models, assembles V0, evaluates test/validation/calibration fixture partitions, attempts calibration and policy selection, exercises replay/no-source handling, and verifies run integrity.
   - This is a synthetic one-epoch smoke test. It is not a PAMAP2/MOSI experiment and its scores are not scientific results.
2. `uv sync --locked --extra dev --extra ui --python /tmp/mmt-python311/bin/python && uv run --python /tmp/mmt-python311/bin/python --locked python -m unittest discover -s tests -v`
   - Passed: 106 tests on Python 3.11.16 after installing the documented development and UI extras.
3. Restored the workspace `.venv` to Python 3.14.7 with `uv sync --locked --extra dev --extra ui --python 3.14`.

### Why real experiments were not started

- Member 1's D02/D05 work owns data acquisition/audit and the MOSI adapter. The PAMAP2 audit explicitly awaits review and approved participant-level membership. The work package forbids inventing splits.
- Member 3 owns the explicit group partitions and experimental protocol. Its guide requires fixed train, validation, calibration and test membership plus acceptance criteria before scored experiments.
- The integrated `train` service currently supports only V0. Although model components and generic training code exist, the documented baseline/V1/V2 matrix, robust-training schedules and frozen-encoder ablations are not wired into a complete real-data run workflow.
- Therefore no real weights, validation metrics, calibration or robustness claims were produced. Starting them now would use an unapproved protocol or incomplete MOSI profile.

### Next gates, in order

1. Member 1 reviews/accepts the PAMAP2 audit and resolves the MOSI feature bindings, numeric language-feature source and usage/redistribution conditions; otherwise MOSI stays out of the experiment matrix.
2. Member 3 and the team approve group-disjoint PAMAP2 partitions and the MOSI supplied split/calibration policy, and freeze the protocol/acceptance rules. Do not infer subject splits from file order.
3. Member 1 prepares canonical data and train-only normalizers; validate membership, class support, manifests and checksums before training.
4. Add versioned run configs and complete the missing train/evaluate integration for the required baselines, V1/V2, robust regimes and frozen-encoder ablations. Run the V0 integration milestone first, then the approved seeds and saved scenarios.
5. Fit calibration on the calibration partition, select any abstention policy on validation only, run frozen test evaluation once, then complete replay parity, timing/memory and release qualification evidence.

Current status: `software_fixture_verified`; `real_data_experiments_blocked_pending_data_review_and_protocol`; `predictive_qualification=not_qualified`.