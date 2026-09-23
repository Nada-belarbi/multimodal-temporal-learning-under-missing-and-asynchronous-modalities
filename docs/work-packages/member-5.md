# Member 5 — Integration, CLI, UI and Release

Source: MMT-WP-005 v1.0. This is the text implementation specification transcribed from the approved work package. Names describe responsibilities; individual assignments are pending team selection. Numerical defaults are development settings, not measured performance.

## 1 Responsibility and deliverable boundary

Member 5 shall integrate the four technical work packages into one installable and inspectable framework. The deliverable is the contract layer, configuration system, command line workflows, reviewer interface, packaging and release evidence. Member 5 does not implement dataset adapters, model architectures, evaluation mathematics or a second runtime engine.

**Owned modules** — src/mmt/contracts.py, config.py, cli.py, registry.py, ui/app.py; packaging metadata; schemas; integration tests; CI and release scripts.

**Member 1 interface** — Registers data adapters and consumes validated DataConfig. The CLI exposes audit, prepare and split validation without duplicating adapter logic.

**Member 2 interface** — Registers model constructors and train/load functions. The CLI resolves model names and passes immutable RunConfig; it does not inspect model internals.

**Member 3 interface** — Exposes evaluation, calibration, policy and report commands. The UI reads saved artifacts and never recomputes metrics silently.

**Member 4 interface** — Exposes Predictor/replay APIs and runtime events. CLI and UI call the same engine and preserve its status/error semantics.

Use Python 3.11 as the initial project target, uv with a committed uv.lock, pyproject.toml, Pydantic for configuration boundary validation, Typer for the CLI, Streamlit and Plotly for inspection, pytest and Ruff for verification, and PyArrow/Parquet plus JSON/CSV for artifacts. These are implementation choices to lock and test; this document contains no measured package compatibility claim.

### Release outcome

A fresh environment shall install the wheel, show mmt --help, validate a configuration, run a small redistributable fixture through audit/prepare/train/evaluate/replay, open the reviewer interface and reproduce the same outputs headlessly. Every run receives a unique identity and a complete resolved configuration.

## 2 Integration architecture and ownership

```text
contracts -> config resolver -> CLI command
                         |
                 shared service API
              data / model / evaluation / runtime
                         |
              artifacts -> UI and reports
```

The application layer is a thin orchestration boundary. Commands resolve a named application profile, validate referenced files and hashes, call one service function, write an atomic run directory and return a structured status. The Streamlit interface calls those same service functions or reads saved artifacts; it must not contain alternative selection, model, calibration or metric code.

Create an explicit registry for adapters, model constructors, evaluators and report renderers. Registration keys are stable strings such as pamap2, mosi, v0, v1, v2, b_grid and majority. Unknown keys fail before loading data. A plugin may add a key only by supplying contracts and integration tests.

**Layer** — Allowed responsibility

**contracts.py** — Typed records, enums, schema versions, serialization and cross-component identity fields.

**config.py** — YAML load, approved path resolution, Pydantic validation, defaults, canonical serialization and hash.

**services** — Orchestrate existing member modules; no private duplicate business logic.

**cli.py** — Argument parsing, user-facing errors, exit codes, run directory selection and command dispatch.

**ui/app.py** — Artifact selection, visualization, step/reset controls and export; no hidden mutation of completed runs.

Use dependency injection for filesystem, clock and service objects in tests. Avoid module-level global model state, implicit current working directory and mutable singleton configuration.

## 3 Shared contracts and schema versions

Define all cross-package records as dataclasses or Pydantic models with explicit schema_version. JSON boundary models reject unknown fields. Internal tensors remain NumPy/PyTorch objects until a service boundary converts them.

**DataConfig** — application, manifest_path, partitions_path, source_schema, decision_policy, availability_mode, feature_order and normalizer reference.

**RunConfig** — run_id, application, model_name, seed, regime, partition hashes, model config, optimizer/training settings and output root.

**ScenarioRef** — scenario_id, schedule_path, schedule_hash, transformation parameters, schedule_seed and producer revision.

**ArtifactRef** — artifact_type, path, sha256, schema_version, producer, created_at_utc and parent hashes.

**PredictionRecord** — run/scenario/sequence/decision identity, computable, status, score_kind, scores/probabilities or null, predicted_class, confidence and diagnostics.

**ReleaseManifest** — release_id, code revision, lockfile hash, dataset/model/calibration/policy hashes, commands, tests, hardware and qualification status.

Use UTC ISO 8601 for artifact timestamps and float64 seconds for application event times. IDs are opaque strings at the CLI boundary. JSON null represents unavailable or undefined; never serialize NaN or Infinity.

The resolver computes canonical JSON with sorted keys, normalized portable paths and UTF-8 encoding before SHA-256 hashing. Save the resolved form beside the original YAML. A reader accepts only declared schema versions or an explicit migration; it never silently drops or reinterprets fields.

## 4 Configuration resolver and validation

```text
mmt config validate --config configs/pamap2.yaml
mmt audit --config configs/pamap2.yaml
mmt train --config configs/pamap2.yaml --model v0 --seed 11
mmt evaluate --run RUN --partition validation --scenario clean
mmt replay --run RUN --sequence SEQUENCE --scenario SCENARIO
```

Implement load_config(path, overrides) -> ResolvedConfig. Load YAML, apply only declared CLI overrides, validate Pydantic models, resolve paths relative to the configuration file, calculate input hashes and write resolved_config.yaml plus config.sha256. Unknown keys, missing fields, invalid ranges and incompatible application/model pairs fail before data or weights are opened.

**PAMAP2 profile** — application=pamap2; decision mode window; source schema from manifest; window_seconds=5.0; stride_seconds=1.0; max_label_age_seconds=0.02; fixed audited class map.

**MOSI profile** — application=mosi; decision mode segment; strict negative/positive rule with zero excluded; source feature keys/dimensions from audited manifest; availability profile explicit.

**Runtime profile** — max_active_sequences, max_observations_per_source, max_buffer_bytes, max_source_age and strict_errors.

**Acceptance profile** — metric rules, coverage requirements, risk limit if defined, hardware/load profile and frozen test identity.

**Path policy** — Allow paths under configured project/data roots or explicit existing inputs. Use portable relative paths in manifests; record private absolute paths only locally.

Seed, scenario and partition are mandatory for stochastic or scored workflows. Re-running a completed run requires a new run identity. Provide `mmt config show --resolved` so developers can inspect the exact values passed to members 1–4.

## 5 CLI command design and exit behavior

**audit** — Read permitted inputs and produce audit.json, data dictionary, timing provenance and exclusion counts. Does not prepare tensors or train.

**prepare** — Convert audited inputs into canonical storage, decisions, targets, partitions and manifest. Refuses audit status that is not qualified.

**validate-splits** — Check group disjointness, class support, decision coverage, hashes and train-only normalization prerequisites.

**train** — Resolve model registry entry, call member 2 trainer for one application/model/seed/regime and write immutable run artifacts.

**evaluate** — Load a run, scenario and partition; call member 3 evaluator and write predictions, metrics and provenance.

**calibrate** — Fit member 3 calibration on the declared calibration partition and attach the exact checkpoint hash.

**replay** — Load member 4 predictor and replay a sequence or batch; write prediction records and selected-ID traces.

**report** — Render saved member 3 exports without rerunning training, calibration or policy selection.

**ui** — Start the Streamlit reviewer against a read-only root; writes only new scenario exports or annotations.

Use exit code 0 only for a completed command whose outputs pass schema validation. Use distinct nonzero codes for configuration, input, incompatible artifact, contract, runtime, incomplete-output and qualification failures. Print a concise error to stderr, write full structured error.json and print one final machine-readable result JSON line to stdout.

```text
mmt --version
mmt config validate --config configs/mosi.yaml
mmt audit --config configs/mosi.yaml
mmt evaluate --run RUN --partition test --frozen-manifest RELEASE
mmt report --run RUN --format html
```

The CLI shall never select a best test seed, refit calibration, mutate a completed run or generate a random scenario without a saved seed and schedule.

## 6 Run directories and artifact lifecycle

```text
runs/<run_id>/
  resolved_config.yaml  config.sha256  environment.json
  manifest.json         partitions.sha256
  source_schema.json     class_map.json
  model_config.json      normalizer.npz
  weights.pt             calibration.json
  scenario_schedule/    predictions/
  metrics/               reports/
  status.json            error.json
```

Create a temporary sibling directory, write and flush files, validate required schemas, then atomically rename to the final run directory. status.json progresses from created to running to completed, failed or incomplete. A completed directory is immutable; new execution receives a new run ID.

**Provenance** — Code revision, lockfile hash, Python/package versions, OS/device, command argv, resolved config hash, dataset/partition/model/normalizer/calibration/policy hashes and UTC timestamps.

**Integrity** — Write sha256sums.json for portable outputs after completion and validate hashes before reports or replay consume an artifact.

**Failed run** — Retain error code, failing component, affected identity and completed-output count. Partial metrics are never a successful report.

**Portable run** — Use relative paths in manifests. Restricted datasets and credentials remain outside the release bundle and are recorded as declared unavailable inputs.

**Management** — Provide `mmt runs list` and `mmt runs verify`; never automatically delete old runs.

Member 5 owns lifecycle mechanics; semantic contents belong to their producing member. A renderer cannot repair a missing prediction row or reinterpret null.

## 7 Integration tests and CI gates

**Unit contracts** — Round-trip schemas, reject unknown fields, invalid timestamps, duplicate IDs and non-finite boundaries. Check canonical hash stability under YAML key reordering.

**CLI smoke** — Fresh environment: help, config validation, audit and prepare on a fixture; verify exit codes, stdout JSON and expected files.

**Cross-member integration** — Fixture prepare -> train source/V0 -> calibrate -> evaluate -> replay -> report. Use public APIs only.

**Parity** — UI-exported scenario rerun headlessly yields identical scenario hash, selected IDs, statuses and scores within member 4 tolerances.

**Failure injection** — Missing artifact, mismatched class map, corrupt YAML, read failure, model exception, overflow and interrupted output produce structured failure and no completed status.

**Package tests** — Build wheel in a clean temporary environment; install from wheel; imports, CLI entry point and version metadata work outside the repository.

**Reproducibility** — Run the fixture twice in separate processes and compare canonical artifacts and deterministic outputs. Timestamps are compared separately from content hashes.

CI stages run lint/format checks, schema checks, unit tests, fixture integration, wheel build/install and a lightweight UI import test. Full PAMAP2/MOSI runs are separate jobs because data access and resources differ. Store reports and environment metadata as CI artifacts.

Use pytest markers `unit`, `integration`, `fixture`, `dataset`, `slow` and `ui`. The default check completes without external datasets. Dataset tests report unavailable-input explicitly when permitted files are absent; they do not silently skip required validation.

## 8 Streamlit reviewer interface

Build one read-only Streamlit application over the public service APIs and saved artifacts. The interface inspects experiments and replay behavior; it does not train or recalibrate models.

**Data Audit** — Choose application and manifest; show source dimensions, event/availability ranges, missingness, exclusions, groups and audit status with hashes.

**Replay** — Choose existing run, sequence and saved scenario. Provide play, pause, step, reset and speed controls; show logical time, source presence, selected counts, age, status, class, confidence, entropy and abstention.

**Comparison** — Select model/scenario/partition and show member 3 metric tables, source-pattern coverage, confusion, reliability, risk–coverage and resource summaries.

**Export** — Write a new scenario reference, resolved UI state and selected-ID/prediction trace to a new directory. Return the exact headless CLI command and hashes.

**Safety** — Disable controls when identities are incomplete. A changed scenario starts a new replay generation; prior output is immutable. Runtime view does not expose targets.

Use session state only for widget state. Cache immutable artifact loading by content hash and clear cache when selection changes. Deterministically downsample long plots while preserving raw export and labeling the plot as downsampled. Render explicit unavailable, abstained and error statuses; blank charts are not explanations.

## 9 Packaging dependency and security boundaries

**Build metadata** — pyproject.toml defines package/version, Python constraint, dependencies, optional UI/dev extras, console script mmt and package data.

**Locking** — Commit uv.lock. CI installs with `uv sync --frozen`; release build records its hash.

**Package layout** — src/mmt, config templates, schemas and UI assets are included explicitly. No raw data or credentials enter the wheel.

**Versioning** — Use project version plus schema versions. Breaking contract/artifact changes require migration or explicit failure.

**Build evidence** — Build wheel/source distribution in a clean directory and record SHA-256 and supported platform metadata.

**Input safety** — Treat YAML, manifests, run records and report text as data. Refuse path traversal and symlink escapes from configured roots.

Load model artifacts through member 2’s state_dict/weights-only path. Do not unpickle arbitrary dataset or run objects. Escape UI data values before HTML/Markdown rendering. The initial release is local and single-user; authentication, remote stores and network serving require a separate design.

## 10 Release manifest and qualification

**Software readiness** — Clean wheel install, CLI smoke, unit/integration tests, UI import and report rendering pass.

**Data readiness** — PAMAP2/MOSI audit, permitted-file provenance, source dimensions, class maps, groups and timing limitations are present.

**Model readiness** — Selected artifacts load strictly in a fresh process; source/class/normalizer hashes match; failed runs are retained.

**Evaluation readiness** — Calibration/policy artifacts, schedules, metric denominators, bootstrap configuration and frozen test manifest are present.

**Runtime readiness** — Batch/replay parity, overflow/error behavior, selected-ID traces, timing protocol and memory evidence pass.

**Release status** — Set software, data, model, evaluation, runtime and overall statuses independently. overall=qualified only when mandatory statuses pass.

**Known limits** — List offline MOSI mode, simulated PAMAP2 availability, unsupported causal profiles and hardware envelope.

Create release_manifest.json only after all referenced hashes resolve. Freeze acceptance.yaml, code revision, lockfile and dataset manifests before final test evaluation. Never edit criteria after observing final test results; create a new candidate when criteria change.

Release archive contains wheel, lockfile, configs, schemas, documentation, selected artifacts, reports, tests and a redistributable fixture. Restricted datasets remain external. Distinguish implementation_verified from predictive_qualification.

## 11 Development sequence and handoff

**I01 Project skeleton** — Create pyproject.toml, src/mmt, tests, configs and CLI entry point. Gate: clean wheel install and mmt --help.

**I02 Contracts** — Implement schemas, canonical serialization, hashes and error codes. Gate: round-trip/rejection tests pass.

**I03 Config resolver** — Implement YAML/Pydantic resolver, path policy, overrides and provenance. Gate: PAMAP2/MOSI/runtime configs validate and invalid cases fail.

**I04 Registries/services** — Register adapters, models, evaluator, runtime and reports. Gate: fixture service chain works without CLI.

**I05 CLI lifecycle** — Implement audit/prepare/splits/train/evaluate/calibrate/replay/report and atomic run directories. Gate: fixture pipeline completes.

**I06 UI** — Implement audit, replay and comparison views over the same APIs. Gate: UI export reproduces headlessly.

**I07 CI/package** — Add lint, markers, wheel build/install, lock verification and release manifest. Gate: clean CI fixture run.

**I08 Handoff** — Document commands, schemas, errors, profiles and release checklist. Gate: members 1–4 sign interface parity.

```text
uv sync --frozen
uv run mmt config validate --config configs/pamap2.yaml
uv run mmt audit --config configs/pamap2.yaml
uv run mmt prepare --config configs/pamap2.yaml
uv run mmt train --config configs/pamap2.yaml --model v0 --seed 11
uv run mmt evaluate --run <run_id> --partition validation
uv run mmt replay --run <run_id> --sequence <sequence_id>
uv run mmt report --run <run_id> --format html
uv run streamlit run src/mmt/ui/app.py
```

Angle-bracket values are produced IDs, not defaults. Repeat model seeds and MOSI commands after the fixture path is stable. Each member verifies its handoff: member 1 schemas, member 2 registry/load, member 3 report/policy artifacts and member 4 replay/status parity.

Deliver source, pyproject.toml, uv.lock, schemas, templates, CLI/API guide, UI, tests and reports, package hashes, release manifest, fixture and troubleshooting guide.

## 12 Verification matrix and acceptance gates

**Contract gate** — Boundary models serialize/deserialize; schema/hash mismatches fail; class/source order cannot change silently.

**Configuration gate** — Unknown fields, invalid ranges, missing files, path escape and incompatible profile/model combinations fail before side effects.

**CLI gate** — Every command has deterministic exit code, final stdout JSON, error.json on failure and immutable run status.

**Integration gate** — A new process completes fixture audit through report and loads outputs; no service bypasses shared contracts.

**Runtime/UI gate** — UI and headless replay select identical IDs and statuses; reset starts a new generation; no UI-only prediction logic.

**Packaging gate** — Wheel installation outside repository works; version, console script, schemas and UI assets are present; lockfile honored.

**Release gate** — Hashes, tests, evidence and limitations appear in release_manifest.json; restricted data stays external.

**Predictive boundary** — No software gate contains an invented F1, calibration or latency score. Member 3 supplies frozen criteria and measured results.

Use deterministic fixture tolerances defined by members 2–4. Compare canonical content hashes and numerical arrays with documented tolerances. Test failure paths as carefully as successful paths: partial or ambiguous output is not a qualified release.

The first release may be software-ready while overall qualification is pending permitted datasets or frozen predictive criteria. The package must state the missing evidence and the command that will produce it.

## 13 Public documentation and operational guide

**README** — Install, supported Python/platforms, fixture quickstart, full command sequence, output locations and troubleshooting.

**Configuration guide** — Every field, default, allowed range, profile restriction and generated hash; complete PAMAP2, MOSI and runtime templates.

**Developer guide** — How to add an adapter, model, metric or report with registry key, schema tests and fixture integration.

**Artifact guide** — Run fields, status transitions, identity hashes, null semantics, restricted-data handling and fresh-process loading.

**Reviewer guide** — Audit, replay, comparison, export and headless reproduction of a UI scenario.

**Release guide** — Lockfile, clean install, tests, manifests, qualification states, archive contents and rollback to immutable run.

Documentation examples use the fixture or clearly labeled placeholders for produced run IDs. Every command in documentation must match the Typer signature tested in CI. Contract changes update schemas, migrations, tests and command examples together.

## 14 References and technical decision register

[1] Pydantic documentation. Data validation and serialization boundary reference.
https://docs.pydantic.dev/latest/

[2] Typer documentation. Typed command line application reference.
https://typer.tiangolo.com/

[3] Streamlit documentation. Application and session-state reference.
https://docs.streamlit.io/

[4] Python Packaging User Guide. pyproject.toml and wheel distribution guidance.
https://packaging.python.org/en/latest/

### Project decisions and evidence status

**Existing component reuse** — Members 1–4 remain owners of data, models, evaluation and runtime semantics. Member 5 composes public interfaces and does not duplicate them.

**Engineering choices** — Python 3.11 target, uv lock, Pydantic, Typer, Streamlit, Plotly, atomic run directories, content hashes and local single-user execution define the initial release.

**Evidence still required** — Dependency compatibility, clean-install success, audited access, model scores, robustness, calibration, replay cost and hardware limits require tests and runs.

**Extension boundary** — Remote serving, authentication, multi-user persistence, online learning, raw extraction and distributed orchestration require separate designs.

This work package is a technical implementation guide. API names and paths are development targets until integrated and verified in the shared repository. No numerical quality or performance result is asserted here.
