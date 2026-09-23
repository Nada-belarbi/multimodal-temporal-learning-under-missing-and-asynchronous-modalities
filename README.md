# Multimodal Temporal Learning under Missing and Asynchronous Modalities

Reusable multimodal temporal learning framework for heterogeneous temporal inputs, missing observations, asynchronous availability and prediction uncertainty.

**Current status:** specifications and development backlog are available. Model implementation, tests and experiments remain to be completed; no performance result is claimed.

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

## Planned implementation stack

Python 3.11, PyTorch, NumPy, pandas, h5py, Pydantic, Typer, Streamlit, Plotly, pytest, Ruff and uv. Dependency compatibility and exact versions must be locked and tested in I01/I07. Installation and execution commands become operational as the corresponding issues are completed.

## Repository policy

Keep datasets, secrets and generated runs outside version control. Publish only permitted fixtures and results. A public repository does not by itself select a software license; licensing remains a team decision.
