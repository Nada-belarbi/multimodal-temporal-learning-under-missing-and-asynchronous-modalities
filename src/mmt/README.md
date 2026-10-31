# Package Status

This package contains the shared multimodal temporal learning implementation.
It is a research prototype with completed local data workflows and experiments,
not a qualified production release. The repository root README and
[`docs/EXPERIMENT_LOG.md`](../../docs/EXPERIMENT_LOG.md) are the authoritative
project-level status and evidence summaries.

## Implemented Workflows

- **Contracts and configuration:** Pydantic records, YAML validation, declared
  overrides, component registries, run lifecycle and artifact checksums.
- **Data:** PAMAP2 audit and participant-grouped preparation; MOSI segment
  preparation from the documented public mirror with frozen GloVe lookup;
  canonical storage, train-only normalization, datasets and batching.
- **Models and training:** source encoders, unimodal references, V0 late fusion,
  V1 learned gates, V2 temporal/source attention, majority and B-GRID model
  constructors; V0/V1/V2 training and checkpoint loading.
- **Evaluation:** metrics, saved perturbation schedules, calibration,
  abstention-policy selection, paired group bootstrap and HTML reporting.
- **Runtime and reviewer:** stateless/stateful selection and prediction,
  bounded buffers, replay traces, CLI commands and a Streamlit saved-artifact UI.

## Current Limits

The 18-run clean campaign and a frozen-checkpoint partial robustness study are
complete. Robust training, frozen-encoder ablations, the full baseline and
perturbation catalogue, the complete B-GRID experiment workflow, formal release
manifest/qualification, and team review/acceptance remain outstanding. Six clean
calibration fits use the explicit T=1 fallback; abstention policies are disabled
without approved constraints. Results are descriptive on fixed partitions, not a
claim of predictive qualification.

PAMAP2 replay availability is simulated. MOSI uses an offline mirror, has one
missing development video, does not establish speaker-disjoint grouping, and has
unverified redistribution terms. See the experiment log and
[`docs/ROBUSTNESS_STUDY.md`](../../docs/ROBUSTNESS_STUDY.md) before interpreting
results.

## Verification

From the repository root with Python 3.11 and uv:

```sh
uv sync --locked --extra dev --extra ui
uv run --locked python -m unittest discover -s tests
uv run --locked --extra dev --extra ui ruff check src tests
uv run --locked mmt --help
```