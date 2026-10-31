# Multimodal Temporal Learning under Missing and Asynchronous Modalities

Reusable multimodal temporal learning framework for heterogeneous temporal inputs, missing observations, asynchronous availability and prediction uncertainty.

**Current status:** the 18-run clean campaign (V0/V1/V2 × PAMAP2/MOSI × three seeds) completed, including test exports, reports, calibration and policy artifacts. Results and limitations are in the [experiment log](docs/EXPERIMENT_LOG.md). Predictive qualification remains `not_qualified`: acceptance bounds are unset, six calibrations fell back to T=1, and all abstention policies are disabled. Robust training, frozen-encoder ablations and the full baseline/perturbation matrix remain incomplete. See the [fusion workflow](docs/FUSION_WORKFLOW.md) for reproduction. MOSI is offline; its mirror lacks one development video and redistribution conditions remain unverified.

The [partial robustness study](docs/ROBUSTNESS_STUDY.md) now includes 230 real
evaluation exports with saved shared schedules, clean majority/source-only baselines,
204 paired/bootstrap comparisons and 54 whole-test replay proofs with timing/memory
measurements. Its 11 graphs and publishable result tables are in the
[consolidated report and exports](docs/NOTEBOOK_REPORT_README.md). This is not
exhaustive robustness or predictive qualification.

## Quick Start

Requires Python 3.11 and uv. From the repository root:

```sh
uv sync --locked --extra dev --extra ui
uv run --locked python -m unittest discover -s tests
uv run --locked mmt --help
uv run --locked --extra ui mmt ui --port 8502
```

The saved-result [reviewer UI test report](docs/UI_TEST_REPORT.md) documents
the tested browser flows and their limits. The English
[consolidated experiment notebook](docs/NOTEBOOK_REPORT_README.md) links the
reproducible report and its exported tables/figures. The final clean and partial
robustness evidence is summarized in the [experiment log](docs/EXPERIMENT_LOG.md).

## Team workspace

- [GitHub Project — Development](https://github.com/users/Nada-belarbi/projects/2)
- [41 development tasks and dependencies](docs/TASKS.md)
- [Assignment, progress comments and review workflow](CONTRIBUTING.md)

| Role | Responsibility | Specification |
| --- | --- | --- |
| Member 1 | Data engineering: acquisition, audit, adapters, splits, normalization and batching | [Guide](docs/work-packages/member-1.md) |
| Member 2 | Unimodal references, V0/V1/V2 fusion models, training and checkpoints | [Guide](docs/work-packages/member-2.md) |
| Member 3 | Evaluation, robustness scenarios, calibration, abstention and reporting | [Guide](docs/work-packages/member-3.md) |
| Member 4 | Temporal selection, stateless prediction, bounded runtime and replay | [Guide](docs/work-packages/member-4.md) |
| Member 5 | Shared contracts, configuration, CLI, reviewer UI, packaging and release | [Guide](docs/work-packages/member-5.md) |

Individual assignments are pending the team's choice. Set Assignees on issues after roles are agreed. Start with I01, D01, M01 and D02 in the task index.

## Application scope

- **PAMAP2 activity recognition:** temporal sensor windows; documented simulated availability for replay.
- **CMU MOSI sentiment classification:** completed opinion segments using audited language/audio/visual features; explicit offline/causal limitations.

## Tested implementation stack

Python 3.11, PyTorch, NumPy, pandas, h5py, Pydantic, Typer, Streamlit, Plotly,
pytest, Ruff and uv are declared in `pyproject.toml` and locked in `uv.lock`.
The test suite and lint checks are run locally; see the verification section above
and [CONTRIBUTING.md](CONTRIBUTING.md) for the development workflow.

## Repository policy

Keep datasets, secrets and generated runs outside version control. Publish only
permitted fixtures and results. The ignored raw/prepared datasets and run outputs
are local inputs, not part of this source delivery. A public repository does not
by itself select a software license; no license has been selected for this project.
