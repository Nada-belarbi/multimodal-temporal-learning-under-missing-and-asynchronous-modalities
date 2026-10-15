# mmt.evaluation: evaluation, robustness and uncertainty (Member 3)

Python package that checks an experiment design, injects reproducible faults, scores predictions honestly,
calibrates probabilities and decides when to abstain. It never trains a model.

**Status: working prototype, tested on synthetic data with a stand-in model only. No real-data result exists and
no performance number is claimed.** Parts were drafted with AI assistance; the author is reviewing them.

## What is done
| Module | Does |
| --- | --- |
| `protocol.py` | Checks the split before any run: a participant in two splits, duplicate IDs, an empty split, a class missing from training. Lists every problem at once. |
| `metrics.py` | Joins predictions onto labels, then computes coverage, macro-F1, balanced accuracy, NLL, Brier, ECE (15 bins), confusion matrix, per-group results. |
| `perturbations.py` | Fault catalogue (missing sources, row removal, outage, delay, out-of-order, combination). Randomness is drawn once, saved and replayed exactly. |
| `runner.py` | Runs one scenario: verifies hashes, asks the model for every scheduled decision, joins labels inside the evaluator, writes results atomically. |
| `calibration.py` | One temperature T fitted on the calibration split only. |
| `abstention.py` | Confidence threshold chosen on validation; risk-coverage curve. |
| `bootstrap.py` | Paired bootstrap over participants (not windows), seed mean and standard deviation. |
| `report.py` | Frozen manifest, acceptance rules, two separate statuses, static HTML report. |
| `io_utils.py` | Hashes, JSON without NaN, never-overwrite result folders. |

Checked with 32 automated tests (kept outside this folder) that passed when run, including the specification's
hand-computed example: for `y=[0,1]`, `p=[[0.8,0.2],[0.3,0.7]]` the NLL is 0.2899, Brier 0.13, ECE 0.25.

## What is not done yet
- Nothing has run on real PAMAP2 or MOSI data, or with a real trained model.
- Real participant lists and group counts: not available, so no real split exists. None are invented here.
- Acceptance bounds (for example a minimum F1) are not set. They must be measured on validation, then frozen.
- The PyTorch version of the temperature fit was never run and is not included; SciPy is used instead.
- Parquet output was never run here; predictions are written as CSV when `pyarrow` is missing.
- Results for clean versus robust training, three seeds and the full 81-scenario PAMAP2 catalogue do not exist yet.

## How to run
Python 3.11 or newer (developed on 3.12). Install, then run from the repository root with `src` on the path:
```
pip install numpy pandas scikit-learn scipy matplotlib jinja2
```
Scores-only example (made-up numbers, runs in a second):
```python
import sys; sys.path.insert(0, 'src')
import numpy as np
from mmt.evaluation.calibration import fit_temperature
from mmt.evaluation.abstention import select_policy
from mmt.evaluation.metrics import probability_metrics, log_probs

rng = np.random.default_rng(0); n, C = 600, 3
y = rng.integers(0, C, n); S = rng.normal(size=(n, C)); S[np.arange(n), y] += 2.0; S *= 6.0   # over-confident
cal = fit_temperature(S[:300], y[:300], C, dict(model='demo'), 'logits')                    # calibration rows
print('status', cal['status'], 'T', round(cal['T'], 2))
before, _ = probability_metrics(S[300:], y[300:], 1.0); after, _ = probability_metrics(S[300:], y[300:], cal['T'])
print('NLL', round(before['nll'], 3), '->', round(after['nll'], 3), '| ECE', round(before['ece'], 3), '->', round(after['ece'], 3))
P = np.exp(log_probs(S[300:], cal['T'])); ok = P.argmax(1) == y[300:]
pol, curve = select_policy(P.max(1), ok, len(ok), risk_limit=0.05)
print('policy', pol['status'], 'tau', None if pol['tau'] is None else round(pol['tau'], 3))
```
The full pipeline calls, in order: `validate_protocol` -> `scenario_catalogue` / `build_schedule` -> `evaluate`
(needs a `predict_fn(row, sources) -> dict(computable, scores, predicted_class, presence)`) -> `fit_temperature` ->
`select_policy` -> `paired_bootstrap` -> `freeze_manifest` / `evaluate_acceptance` / `render_report`.

## What to expect
- `evaluate` writes `predictions.csv`, `metrics.json`, `class_metrics.csv`, `confusion.csv`, `reliability.csv`,
  `per_group.csv`, `provenance.json`, `status.json`. An existing result folder is never overwritten; a model crash
  publishes nothing.
- Over-confident scores give T above 1; NLL and ECE fall after calibration; predicted classes do not change.
- Rows with no data lower availability coverage; they are never hidden. Undefined values are `null`, not 0.
- With an unset acceptance bound the verdict is `not_qualified`, even when the software checks pass.
- With fewer than 10 groups the bootstrap is flagged as unstable; with fewer than 2 it gives no interval.

## Possible final outcome
When real data and a real model are available: a frozen protocol per application; tables of macro-F1, availability,
NLL, Brier and ECE across the fault scenarios; degradation and reliability plots; one calibrated temperature; an
abstention threshold if the team defines a risk limit; paired model comparisons with participant-level intervals;
and two verdicts, `implementation_verified` and `predictive_qualification`. The second may well stay
`not_qualified` or fail: measured results decide, and targets are never changed afterwards.

## Limits
Specification settings (seed 1701, 15 bins, 1000 replicates, scenario levels) are initial choices, not validated
optima. Delay scenarios on pre-extracted features collapse into closure-availability tests. Saved schedules are JSON,
which is large for long recordings. Status names used: `accepted`, `abstained_low_confidence`, `unavailable_no_data`.
