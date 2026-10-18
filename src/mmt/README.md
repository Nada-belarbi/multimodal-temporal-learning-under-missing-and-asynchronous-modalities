# nada
This document describes the current implementation in `src/mmt/models`, `src/mmt/training`, `src/mmt/data` and `src/mmt/runtime`. It records code that exists in this workspace; it is not a claim that either work package is fully integrated or qualified.

## Status

I have executable prototypes and synthetic tests. The current suite contains 26 tests and passes in the local virtual environment. No real PAMAP2/MOSI training, replay or performance measurement has been completed, and no predictive result is claimed.

The project specification targets Python 3.11. The current local `.venv` uses Python 3.14.7 with NumPy 2.5.4 and PyTorch 2.14.1. Dependency versions are not yet declared in project packaging metadata or locked for a clean installation.

## Models and Training

### Implemented

| Module | Current responsibility |
| --- | --- |
| `models/source_encoder.py` | `SourceBatch`, `EncoderOutput` and `ModelOutput` records; feature-mask validation; projection plus packed, unidirectional GRU; source-only classifier. |
| `models/v0_late_fusion.py` | Independent source classifiers combined by equal averaging of available source probabilities; emits log-probability scores and marks all-absent rows uncomputable. |
| `models/fusion.py` | V1 learned source gates and V2 hierarchical temporal/source attention; absent sources are masked and all-absent rows bypass fusion. |
| `models/baselines.py` | Training-majority label-only reference and the B-GRID GRU classifier. |
| `models/registry.py` | Schema-driven constructors for unimodal, B-GRID, V0, V1 and V2 models; validates source dimensions and contiguous class indices. |
| `data/grid.py` | Deterministic common-grid construction using the latest valid value at or before each grid point, with a hold limit and no future interpolation. |
| `training/trainer.py` | Generic CPU/GPU training loop, computability and target masks, cross-entropy, gradient clipping, validation macro-F1 selection, patience and history. |
| `training/checkpoint.py` | Atomic best/last checkpoint writing and strict `state_dict` loading with `weights_only=True`. |

V0 reports `score_kind="log_probabilities"`; V1, V2, B-GRID and source classifiers report logits. The majority model is label-only and does not fabricate confidence scores.

### Still outstanding

- Integrate the actual Member 1 data batches, normalizer, class map, partitions and audited source dimensions.
- Add epoch-boundary resume, robust-training schedule integration, frozen-encoder ablations and full experiment runs.
- Verify all model architectures and training behavior on both audited application profiles.
- Export the training artifacts and score records using the final shared contracts agreed with Members 3 and 5.

## Temporal Selection, Inference and Replay

### Implemented

| Module | Current responsibility |
| --- | --- |
| `data/selection.py` | Pure selector for availability cutoff, closed support containment, event-time ordering, all-invalid row removal, freshness and `log1p` temporal channels. |
| `runtime/buffer.py` | Owned immutable observation copies, duplicate detection, event-order snapshots, per-sequence arrival ordering, explicit retention floor and row/byte limits. |
| `runtime/engine.py` | Injected batch/model adapter, stateless batch prediction, stateful update/predict/reset, saved temperature/threshold application, source evidence and prediction identities. |
| `runtime/replay.py` | Single-sequence replay iterator; arrivals available at a decision timestamp are delivered before that decision. Inputs must be chronologically sorted. |

No-data decisions return null scores and probabilities. Model exceptions remain execution failures rather than being converted to unavailable predictions. Stateful decisions enforce increasing decision time; reset advances a generation identifier.

### Still outstanding

- Connect selection to the real Member 1 normalizer and collator, and verify selected-ID parity on prepared application data.
- Add the persistent/incremental replay output sink and integrate Member 3 saved scenarios, calibration and policy artifacts.
- Implement and qualify PAMAP2-window and MOSI-segment profiles, including their distinct availability limitations.
- Measure replay latency, memory high-water marks, overflow and failure behavior on a declared machine.
- Extend replay to orchestrated multi-sequence runs if required; the current iterator is deliberately single-sequence.

## Verification

From the repository root, with the local environment installed:

```sh
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests
```

The 26 tests cover synthetic selector boundaries, temporal-grid hold behavior, buffer identity/capacity rules, replay ordering and parity, model missing-source behavior, majority tie-breaking, and a small CPU train/checkpoint/reload cycle. They do not exercise real datasets, final CLI/UI flows, GPU training, or performance qualification.

The local Mac reports MPS available through PyTorch. The trainer accepts an explicit `device` argument, but the tests above use CPU; MPS correctness and speed have not been measured.

## Integration Boundary

These modules use local batch records and injected callbacks where the shared Member 1/5 contracts are not available yet. Expect interface adjustments when the data schema, configuration resolver, CLI and artifact contracts are agreed. Such adjustments should be tested as integration work; the current prototypes should not be treated as frozen public APIs.