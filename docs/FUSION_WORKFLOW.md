# V0, V1 and V2 Workflow

## Supported Scope

The public CLI supports clean prepared-data training, strict checkpoint reload,
evaluation, temperature calibration, validation-only policy selection, single-sequence
replay and saved HTML reports for V0, V1 and V2.

V0 trains independent source classifiers and averages probabilities. V1/V2 jointly
train their encoders and fusion modules. Joint checkpoints are selected by highest
clean-validation macro-F1, then lowest validation cross-entropy, retaining the earlier
epoch on an exact tie. `weights.pt` is a copy of the selected `checkpoints/best.pt`,
not the last epoch. `checkpoints/last.pt` retains optimizer/RNG state for the generic
trainer API; CLI resume is not implemented.

This integration does not implement robust-training augmentation, frozen-encoder
ablations, the complete baseline matrix, experiment orchestration or final release
qualification. `regime: robust` is rejected explicitly instead of running clean training
under a misleading label. The application-specific data limitations remain in
[EXPERIMENT_LOG.md](EXPERIMENT_LOG.md).

## Configurations

- [configs/pamap2-v1-seed-11.yaml](../configs/pamap2-v1-seed-11.yaml)
- [configs/pamap2-v2-seed-11.yaml](../configs/pamap2-v2-seed-11.yaml)
- [configs/mosi-v1-seed-11.yaml](../configs/mosi-v1-seed-11.yaml)
- [configs/mosi-v2-seed-11.yaml](../configs/mosi-v2-seed-11.yaml)

V1 accepts `hidden_size` and `dropout`; V2 also accepts `n_heads` and
`feedforward_size`. `epsilon` belongs to V0 only. Use the architecture-specific
configuration instead of overriding a V0 config that still contains `epsilon`.
All supplied configs use hidden size 64, dropout 0.1, AdamW lr 0.001,
weight decay 0.0001, batch size 32, at most 100 epochs and patience 10.

## Training and Validation

Example for MOSI V1, from the project root:

```sh
CONFIG=configs/mosi-v1-seed-11.yaml
RUN=runs/mosi-v1-offline-glove6b300-seed-11-2026-10-27
OUT=runs/mosi-v1-seed-11-evidence

uv run --locked mmt config validate --config "$CONFIG"
uv run --locked mmt validate-splits --config "$CONFIG"
uv run --locked mmt train --config "$CONFIG" --device cpu
uv run --locked mmt runs verify --run-dir "$RUN"
uv run --locked mmt evaluate --config "$CONFIG" --run-dir "$RUN" --partition validation --output-dir "$OUT/validation"
uv run --locked mmt evaluate --config "$CONFIG" --run-dir "$RUN" --partition calibration --output-dir "$OUT/calibration"
uv run --locked mmt calibrate --run-dir "$RUN" --evaluation-dir "$OUT/calibration" --output "$OUT/calibration.json"
uv run --locked mmt select-policy --run-dir "$RUN" --validation-dir "$OUT/validation" --calibration "$OUT/calibration.json" --output "$OUT/policy.json"
```

Without application risk/coverage constraints, policy selection explicitly exports a
disabled policy. Do not invent bounds or describe this as an approved operating point.
Failed temperature fitting may export only the explicit uncalibrated T=1 fallback.

For seed repetitions, provide a new immutable run ID as well as the seed:

```sh
uv run --locked mmt train --config configs/mosi-v1-seed-11.yaml --seed 22 --run-id mosi-v1-offline-glove6b300-seed-22-2026-10-27 --device cpu
uv run --locked mmt train --config configs/mosi-v1-seed-11.yaml --seed 33 --run-id mosi-v1-offline-glove6b300-seed-33-2026-10-27 --device cpu
```

Evaluation/replay reconstruct the saved architecture and identify the actual saved run;
seed overrides do not relabel outputs as the template's seed-11 run. Each repeated run
needs its own calibration and policy artifacts. Never reuse them across checkpoints.

## Frozen Test and Reports

Only after freezing the model, partition/protocol choices and acceptance criteria,
evaluate final test. CLI execution alone does not enforce or certify that scientific
freeze, and a report without mandatory acceptance bounds remains not qualified.

```sh
uv run --locked mmt evaluate --config "$CONFIG" --run-dir "$RUN" --partition test --output-dir "$OUT/test" --calibration "$OUT/calibration.json" --policy "$OUT/policy.json"
uv run --locked mmt report --evaluation-dir "$OUT/test" --output "$OUT/report.html" --title "MOSI V1 Offline"
```

Do not select a seed or change the model from test outcomes. Source scores remain
uncalibrated in the prediction export; confidence, status and metrics use the loaded
temperature/policy. V0 declares log-probabilities; V1/V2 declare logits.

## Replay

For one MOSI test segment:

```sh
SEQUENCE=$(uv run --locked python -c 'import pandas as pd; p=pd.read_csv("configs/mosi-membership.csv"); print(p.loc[p.split=="test", "sequence_id"].iloc[0])')
uv run --locked mmt replay --config "$CONFIG" --run-dir "$RUN" --sequence "$SEQUENCE" --output-dir "$OUT/replay" --calibration "$OUT/calibration.json" --policy "$OUT/policy.json"
```

For PAMAP2 choose a recording ID from its membership file, e.g. `subject102`.
Non-clean replay additionally requires an explicit saved `--schedule` and matching
`--scenario`. A changed schedule receives a new output directory. Saved postprocessing
cannot be mixed with scalar `--temperature`/`--threshold` overrides.

## Verification Evidence

117 tests pass under Python 3.11.16 with dev/UI extras; Ruff passes. Fixture integration
tests execute train, checkpoint verification, validation/calibration/test evaluation,
calibration, policy, reports, replay and all-sources-absent replay for both joint models
on both application profiles. They check exact repeat weights at fixed seed and compare
evaluation/replay scores and calibrated probabilities within CPU tolerances.

Separate fresh-process tests reload both architectures, including non-alphabetical
source order and non-default V2 attention options. Incompatible architectures,
score kinds and calibration checkpoints are rejected. The V0 workflow remains covered.
These are software checks, not completed long real-data experiments or predictive results.