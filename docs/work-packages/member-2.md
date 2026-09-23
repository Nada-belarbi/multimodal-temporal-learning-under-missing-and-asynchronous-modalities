# Member 2 — Models and Training

Source: MMT-WP-002 v1.0. This is the text implementation specification transcribed from the approved work package. Names describe responsibilities; individual assignments are pending team selection. Numerical defaults are development settings, not measured performance.

## 1 Assignment and deliverable boundary

Member 2 shall implement the learning components of the multimodal temporal framework: required reference models, V0/V1/V2, training, checkpointing and reloadable model artifacts. The same model interfaces shall support PAMAP2 activity classification and CMU MOSI segment sentiment classification, with separate trained weights per application.

This work package specifies an engineering implementation. The architecture sizes and training values are initial project settings, not published optimal settings or measured results. PyTorch API behavior is grounded in the cited documentation. No predictive score or claim of superiority is assigned to a model before experiments.

**Owned code** — models/source_encoder.py, baselines.py, v0_late_fusion.py, v1_gated_fusion.py, v2_attention_fusion.py, registry.py; training/trainer.py and checkpoint.py; configs/models and model/training tests.

**Member 1 input** — Prepared source batches, fixed class map, normalizer, partition identity and feature schema. Models do not read raw PAMAP2 or MOSI files.

**Member 3 interface** — Consumes exported scores for evaluation and calibration; supplies perturbation schedules and experimental protocol. Member 2 runs agreed training and ablations.

**Member 4 interface** — Consumes models through forward and artifact loading; owns temporal eligibility and replay. Models receive only already-selected observations.

**Member 5 interface** — Maintains shared contracts, configuration validation, CLI and package integration. Member 2 supplies model constructors and train/load entry points.

### Required deliverables

Deliver B-MAJORITY, each source-only reference, B-GRID, V0, V1 and V2; shared source encoders; a tested trainer; resumable checkpoints; model configurations; training logs; and reloadable inference artifacts. Provide validation runs for both applications and an implementation note identifying every model difference.

Calibration fitting, threshold selection and final comparative conclusions belong to member 3. The first completed V0 training run is an integration milestone. Delivery also requires the other specified models and their missing-source, numerical and artifact tests.

## 2 Ordered tasks and expected outputs

**M01 Input contract** — Agree SourceBatch, ModelOutput, class order and source order with members 1 and 5. Output: synthetic batches covering unequal lengths, partial validity and missing sources.

**M02 Shared encoder** — Implement projection, explicit dropout and packed unidirectional GRU. Output: temporal states and last valid states with zero-length handling and padding tests.

**M03 Source models and V0** — Train each source independently; implement probability averaging over usable sources. Output: source checkpoints, V0 assembly metadata and forward tests.

**M04 Trainer and persistence** — Implement optimizer loop, validation selection, logs, best/last checkpoints and fresh-process reload. Output: complete PAMAP2 V0 integration run.

**M05 Required baselines** — Implement training-majority predictor and common-grid GRU reference. Output: executable references sharing partitions and decision IDs.

**M06 Learned fusion V1** — Implement source gate networks, masked weights and classifier. Output: joint-training model and every-source-subset tests.

**M07 Attention fusion V2** — Implement temporal attention pooling, source embeddings and source self-attention. Output: joint-training model and attention-mask tests.

**M08 Controlled experiments** — Run clean/robust training and frozen-encoder ablations through both application profiles. Output: versioned runs, validation scores and resource measurements.

**M09 Handoff** — Deliver reloadable artifacts, complete configuration, resume evidence and integration guide. Output: accepted model package for evaluation and replay.

Implement M01–M04 first. Build M05 before claiming the baseline comparison is complete. V1 and V2 depend on a verified encoder, shared batch contract and trainer. Evaluate all variants under the member 3 protocol; validation is allowed during development, but final-test outcomes shall not drive implementation tuning.

Keep architecture modules independent of dataset names. PAMAP2 supplies dimensions [6,6,6,1] and C=12. MOSI supplies three audited feature dimensions and C=2. The registry receives these values from the source schema and class map.

## 3 Environment and public model contracts

Use Python 3.11 and PyTorch in the shared uv environment. Use torch.nn.Module, ModuleDict, Linear, GRU, MultiheadAttention and TransformerEncoderLayer. Use the project trainer rather than introducing a second training framework. NumPy supports seed/state handling; pytest and torch.testing.assert_close support verification. Member 5 locks dependency versions after CPU and target-device smoke tests; the cited API version does not assert a tested installation.

**SourceBatch per source** — values float32 [B,T_m,D_m]; feature_mask bool of the same shape; time_features float32 [B,T_m,3]; lengths int64 [B]; padding_mask bool [B,T_m]; present bool [B].

**Presence invariant** — present=(lengths>0). Padding is true where no row exists; feature validity is separate. The all-absent source representation may have T_m=1 containing padding only.

**ModelOutput** — scores float32 [B,C]; computable bool [B]; score_kind=logits or log_probabilities; optional diagnostics. Zero placeholder score rows where computable=false are internal storage only, never a probability distribution.

**Auxiliary metadata** — Source order, dimensions and class map are fixed at construction. Decision IDs remain external diagnostics. No target, subject ID or raw sentiment score enters forward.

**Validity gate** — Return computable=false when no effective input is usable. Filter those rows before any all-masked attention, fusion softmax, loss or probability conversion.

```text
build_model(model_config, source_schema, class_map) -> nn.Module
SourceEncoder.forward(source_batch) -> EncoderOutput
Model.forward(inputs) -> ModelOutput
train(run_config, datasets, model) -> RunArtifact
load_model(artifact_dir, device) -> nn.Module
```

EncoderOutput contains states [B,T_m,H], last_state [B,H] and presence. Scores are uncalibrated; member 3 applies the saved temperature, and member 4 applies the acceptance/abstention policy. No model class shall silently invoke calibration or read evaluation labels.

Registry names are majority, unimodal, b_grid, v0, v1 and v2. unimodal requires source_id. The majority reference exposes a label-only baseline adapter and is excluded from probability metrics rather than supplying artificial confidence.

## 4 Shared source encoder implementation

Each source owns separate weights. Set H=64. Replace invalid input entries with zero using the feature mask before concatenation, even when the data loader already filled them. This makes invalid-value invariance testable. The three temporal channels are supplied by the selector: log1p(delta), log1p(age) and log1p(duration).

```text
x = torch.cat([values.masked_fill(~feature_mask, 0),
               feature_mask.float(), time_features], dim=-1)
# [B,T_m,2*D_m+3]
projection = nn.Linear(2*D_m + 3, 64, bias=True)
input_dropout = nn.Dropout(p=0.1)
gru = nn.GRU(64, 64, num_layers=1, batch_first=True,
             bidirectional=False, dropout=0.0, bias=True)
x = input_dropout(torch.relu(projection(x)))
```

### Packed sequence procedure

Select batch indices where lengths>0. Pass those rows to pack_padded_sequence with lengths.cpu(), batch_first=True and enforce_sorted=False. Run the GRU with default zero initial state. Unpack using pad_packed_sequence with batch_first=True and total_length=T_m. Obtain the last state from h_n[0], retaining the original selected-row order [1].

Scatter valid outputs back into zero-initialized full-batch tensors with a differentiable indexing operation. If the source is absent from the entire batch, return zeros and presence=false without calling pack or GRU. Never select the final padded tensor position as the last valid state.

**PAMAP2 projection** — D=6 → 15 input channels for each movement source. D=1 → 5 channels for heart rate. These dimensions follow 2D+3.

**MOSI projection** — For audited dimension D_m, construct Linear(2D_m+3,64). Feature dimensions are required schema fields, not inferred from a zero-filled missing batch.

**Initialization** — Use PyTorch module defaults for Linear and GRU, with the run seed set before construction. Record changes to initialization as a new configuration.

**Dropout placement** — Use explicit dropout after projection/ReLU. The GRU itself has dropout=0.0; one recurrent layer has no inter-layer dropout [1].

Use a unidirectional encoder and recompute each selected support independently. This encoder supplies ordered temporal states while the external selector enforces availability. It is a fixed basis for the fusion comparison, not evidence that GRU is the best architecture for these datasets.

## 5 V0 independent models and probability fusion

### Source model training

For each source m, implement SourceClassifier = SourceEncoder_m + Linear(64,C). Train independently using cross-entropy on rows with a valid target and that source present. Each source has its own optimizer, early-stopping state and best checkpoint. Select its checkpoint by the lowest source-validation cross-entropy; exact ties keep the earlier epoch. If no validation decision has that source, fail selection explicitly.

### Fusion implementation

```text
p_m = softmax(source_logits_m, dim=-1)
# Evaluate only source-present rows and scatter zeros otherwise.
p = sum(p_m for usable sources) / usable_source_count
scores = log(clamp(p, min=1e-8))
score_kind = "log_probabilities"
```

Apply division only to computable rows. If one source is present, the averaged probability equals that source’s probability. With none, scores are internal placeholders and computable=false. A downstream softmax of log-clamped scores renormalizes the small numerical floor. Record epsilon=1e-8 in configuration.

V0 has no joint fusion training and no learned fusion weights. Package the selected source weights together in a ModuleDict and persist source checkpoint identities. Report source-only predictions as separate baselines. The calibration interface consumes V0 log probability scores, not an undefined set of fused logits.

### Required checks

Test equal averaging with a small controlled probability fixture, exact reduction to one source before clamping, insensitivity to absent-source placeholders, finite output on mixed availability, and no fabricated probabilities for all-empty rows. All tensors must follow the configured class order.

## 6 V1 learned gated fusion

### Inputs to the gates

For source m, concatenate last_state_m [B,64], latest_event_age_log [B,1] and valid_fraction [B,1]. latest_event_age_log is the log1p(age) channel at its last valid row. valid_fraction is valid observed feature entries divided by lengths×D_m, excluding padding. Compute these only for present sources; absent placeholders are zero.

```text
gate_m = Linear(66,32) -> ReLU -> Linear(32,1)
g = stack(source_gate_scores, dim=source)   # [B,M]
g = g.masked_fill(~present, -inf)
alpha = softmax(g, dim=source)  # computable rows only
fused = sum(alpha_m * last_state_m)         # [B,64]
head = Linear(64,64) -> ReLU -> Linear(64,C)
```

Use one gate network per source stored in ModuleDict. Jointly optimize encoders, gates and the classification head with raw-logit cross-entropy. Set unavailable source weights to zero through masking and verify that usable weights sum to one. Do not apply softmax to class logits before passing them to CrossEntropyLoss [3].

### Initialization and diagnostics

Use module-default initialization for gates and head after seeding. Fresh training initializes the complete model; the frozen-encoder ablation loads specified V0 encoder weights. Export alpha only when diagnostics are requested, with source order and presence. Gate weights are model diagnostics, not a validated explanation of causality.

For all-empty rows, bypass the gate softmax; softmax over only −infinity values is undefined. Test all source subsets, one-source alpha=1, missing-source alpha=0, batch permutation consistency and gradient propagation to every participating encoder and gate.

## 7 V2 temporal and cross source attention

### Stage A temporal summary within each source

For each source, allocate a trainable query q_m [1,1,64], initialized from Normal(mean=0,std=0.02), and a separate MultiheadAttention(embed_dim=64,num_heads=4,dropout=0.1,batch_first=True). Expand q_m to [B_present,1,64]. Use GRU states as keys and values with the source padding mask; a true key-padding entry is ignored [2]. Absent sources bypass this stage.

### Stage B interactions between source summaries

Add a trainable source embedding e_m [64], initialized with Normal(0,0.02). Stack summaries in fixed source order to [B,M,64]. On computable rows, run one TransformerEncoderLayer with d_model=64, nhead=4, dim_feedforward=128, dropout=0.1, activation=relu, batch_first=True, norm_first=False, layer_norm_eps=1e-5 and bias=True. Use src_key_padding_mask=~present.

```text
z = source_transformer(summaries, src_key_padding_mask=~present)
z = z.masked_fill((~present).unsqueeze(-1), 0)
fused = z.sum(dim=1) / present.sum(dim=1, keepdim=True)
head = Linear(64,64) -> ReLU -> Linear(64,C)
```

Zero absent query outputs explicitly before pooling; masking keys alone does not suppress those output positions. Jointly train the entire model. No extra residual around temporal pooling or extra final LayerNorm is added in this configuration. Set need_weights=False normally; request weights only for diagnostics.

Attention operates on already admissible observations and then source summaries; no triangular temporal mask is required for a decision-level summary. This is a project-defined hierarchical fusion, not a MulT reproduction or pairwise alignment of raw modality tokens. Report its measured quality and cost; do not presume that attention improves V0 or V1.

## 8 Required majority and common grid references

### B MAJORITY

Count eligible training target classes and choose the most frequent, breaking ties by the lowest class index. Save counts and class map. Emit the selected label only when the shared input-availability guard permits a decision; report coverage separately. This is a label-only baseline, excluded from NLL, Brier and calibration analyses. It has no optimizer or fitted neural weights.

### B GRID representation

Implement a deterministic grid transform in models/baselines.py before a dedicated grid classifier. The transform consumes canonical selected observations plus their event endpoints, feature validity and decision bounds from the shared selector. These endpoints are required in an auxiliary batch view; the three logarithmic time channels alone are insufficient to reconstruct them reliably. Agree this small interface addition with members 1 and 4.

```text
grid_step_seconds = 0.05
hold_limit_seconds = 0.5
# a=support_start, b=support_end
grid = a + k*step for integer k>=1 where a+k*step<=b
append b if not already present
for each grid point g and each source feature:
  choose latest valid selected value with event_end<=g
  if g-event_end<=hold_limit: use it with mask=True
  else: normalized_value=0, mask=False, age=hold_limit
```

If support has zero duration, reject the application decision; otherwise the grid contains at least its final endpoint. Resolve same-time candidates with the selector’s deterministic ID order. Selected rows already satisfy available_at≤decision_time. A late arrival may fill a historical slot for the current decision; no earlier emitted decision is changed. No future interpolation is used.

### Grid classifier

Per source concatenate normalized values, feature masks and per-feature log1p(age). Append one log1p(decision_time−g) channel to the concatenated source representation. Input dimension is 3×sum(D_m)+1. Apply Linear(input_dim,64), ReLU, Dropout(0.1), one unidirectional GRU(64,64) and Linear(64,C) at the last actual grid point. Pack across variable grid lengths and exclude entirely unusable grid rows from prediction/loss.

The grid includes real empty time slots represented by masks; those are not batch padding. Report coverage after grid construction and any information lost by the hold/grid choices. Train using the joint-model protocol. Grid step and hold limit are initial engineering settings, not dataset constants.

## 9 Training loop and checkpoint selection

### Initial training procedure

```text
model.train()
for batch in train_loader:
  optimizer.zero_grad(set_to_none=True)
  output = model(batch.inputs)
  use = output.computable & batch.target_valid
  if not use.any(): count_skipped(batch); continue
  loss = cross_entropy(output.scores[use], batch.targets[use])
  require_finite(loss)
  loss.backward()
  clip_grad_norm_(trainable_parameters, 1.0,
                  error_if_nonfinite=True)
  optimizer.step()
```

The joint loop applies to V1, V2 and B-GRID; source-only training applies the source-presence filter. V0 is assembled after independent source training and is never passed through a fused optimization loop. All-empty rows do not contribute loss or fabricated targets. If an epoch has zero eligible training rows, fail the run and report counts.

### Loss and validation

Use CrossEntropyLoss(weight=None,reduction="mean",label_smoothing=0.0) with raw class logits and integer target indices [3]. Accumulate epoch losses weighted by eligible decision counts, not an unweighted mean of batch means. At epoch end use model.eval() and torch.no_grad(); never evaluate with training dropout active.

For joint models, select the highest clean-validation macro-F1 over computable scored decisions, using the fixed class map and zero_division=0 metric supplied by member 3. Break an F1 tie by lower validation cross-entropy, then retain the earlier epoch. Report availability coverage alongside the metric. No computable validation data is a selection error, not an F1 of zero.

For each source classifier, use minimum source-validation cross-entropy. Early stopping counts completed epochs without improvement. Save best.pt when the selection improves and last.pt at each complete epoch for resume. Do not choose checkpoints or seeds from final-test scores.

### Failure reporting

Reject non-finite model inputs, output scores on computable rows, loss or gradients. Include run ID, epoch, decision IDs and source lengths in the diagnostic. Reject unexpected target ranges, changed class maps and incompatible feature dimensions. Do not silently truncate sequences or skip failing batches.

## 10 Configuration and initial hyperparameters

**Optimizer** — AdamW; lr=0.001; weight_decay=0.0001; betas=(0.9,0.999); eps=1e-8; amsgrad=False. Optimize all trainable parameters; no special no-decay groups initially.

**Schedule and duration** — scheduler=null; batch_size=32; max_epochs=100; early_stopping_patience=10; min_delta=0; gradient_clip_norm=1.0.

**Precision and loading** — float32; AMP=False; gradient_accumulation_steps=1; num_workers=0 initially; drop_last=False. Train shuffle=True, validation shuffle=False with recorded generators.

**Encoder** — hidden_size=64; projection_size=64; recurrent_layers=1; bidirectional=False; explicit input_dropout=0.1.

**V1** — Gate dimensions 66→32→1 per source; classification head 64→64→C; ReLU between linear layers.

**V2** — Temporal heads=4; source heads=4; source layers=1; feedforward=128; dropout=0.1; query/source-embedding init std=0.02.

**Reference defaults** — V0 epsilon=1e-8. B-GRID step=0.05 seconds and hold limit=0.5 seconds.

**Repetition** — Model seeds 11,22,33. Record Python, NumPy, PyTorch, DataLoader generator and device RNG states.

### Registry and configuration checks

Reject hidden dimensions not divisible by attention head count, dropout outside [0,1), nonpositive grid step, negative hold limits, nonpositive epoch/batch settings, unknown source IDs and output dimensions differing from the class map. Build all source modules before constructing the optimizer.

Use configs/models/v0_late_fusion.yaml, v1_gated_fusion.yaml, v2_attention_fusion.yaml and b_grid.yaml. Save the resolved application/model/training combination in each run. Model constructors receive source_schema and class_map rather than embedding PAMAP2/MOSI branches.

### Parameter status

These settings define a concrete first implementation and comparison budget. They are not claimed to be optimal or sufficient for an accuracy target. Change them only in a named validation experiment and record the change. Confirm device memory needs by a full-size batch smoke run; if batch size changes, record a new comparable run configuration rather than hiding the change.

## 11 Missing inputs and augmentation integration

### Mandatory model behavior

The model shall use remaining sources when some are absent, preserve partially valid feature masks, and return computable=false for complete absence. Source disappearance is represented through lengths/present and selected rows, not merely by writing zeros into observed values. A returning source requires no model reload; the next admissible batch can include it.

Test every presence subset: sixteen for four-source PAMAP2 and eight for three-source MOSI. These counts enumerate availability combinations, not measured experiments. A row with some missing features remains usable if the selector retained it. The model must not reinterpret a valid numeric zero as missing.

### Two training regimes

**clean** — No injected source or observation removal. Retain naturally invalid values and their masks.

**robust** — Initial source dropout probability=0.2 per epoch/decision/source; observation dropout probability=0.1 per epoch/canonical observation. Other outages or delays are explicit additional schedules, not hidden defaults.

**Schedule ownership** — Member 3 supplies saved schedules. Member 1’s data pipeline applies them before the shared selector and normalization/batching; member 2 passes epoch/scenario identity and records schedule hashes.

**Overlapping history** — Observation-level corruption uses stable observation IDs so overlapping windows agree within an epoch. Decision-level source dropout is explicitly an augmentation convention.

**No-data augmentation** — Skip its loss, increment counters and retain its status in evaluation. Do not force a random source back into the sample without defining a different augmentation regime.

Recompute temporal channels after perturbation/selection through the shared selector. A reception delay changes availability, not the underlying event time. Member 2 must not implement a separate tensor-only approximation of time delay inside forward.

### Epoch interface

Provide a dataset/scenario set_epoch(epoch) hook agreed with member 1 and an explicit schedule seed distinct from model seed. Models receive the same perturbation schedule IDs for a comparison. Log optimizer update counts and skipped decisions, since source-specific and joint training may have different effective sample counts.

## 12 Checkpoints reload and deterministic resume

**Inference artifact** — model_config.json; source_schema.json; class_map.json; normalizer identity and file; weights.pt with state_dict; manifest/partition/config hashes; code and environment versions.

**Epoch checkpoint** — Model and optimizer state; scheduler state or explicit null; completed epoch; next epoch; optimizer update count; best metric/epoch; patience counter; model/data/scenario seeds and RNG states.

**V0 assembly** — Source names/order, each source configuration/checkpoint identity and all source state_dict entries. Verify identical class order and normalization profile before assembly.

**Atomic writing** — Write a temporary checkpoint on the same filesystem, flush/close, then rename. Mark a run complete only after its selected artifact passes reload checks.

### Loading contract

```text
model = build_model(config, source_schema, class_map)
state = torch.load(weights_path, map_location="cpu",
                   weights_only=True)
model.load_state_dict(state, strict=True)
model.to(device)
model.eval()
```

Use state_dict serialization rather than pickling the entire model object [4]. Store checkpoint metadata and random state using tensors and primitive containers supported by the selected loader; convert NumPy-specific RNG objects explicitly. Verify this serialization in the locked environment. Do not solve loading errors by silently enabling unrestricted deserialization.

### Resume scope and procedure

Support resume at a completed epoch boundary. Reload architecture, weights and optimizer; restore the recorded completed epoch, patience and best metric. Restore all RNG/generator states after construction and before the next DataLoader iteration; restore the scenario epoch. Mid-epoch cursor recovery is outside this implementation and shall not be implied.

The same configuration, prepared data identity, ordered source schema, class map and partition are mandatory for exact resume. Changes create a new run. Test uninterrupted versus interrupted-at-epoch-boundary training on a small CPU fixture under deterministic operations. Compare parameters and next-epoch outputs at documented tolerances; do not promise bitwise reproducibility across devices or library versions.

The reference reload test uses atol=1e-6 and rtol=1e-5 for CPU evaluation outputs. These are numerical test tolerances, not predictive quality thresholds.

## 13 Experiments and resource evidence

### Run matrix

For both application profiles, train every source-only reference and B-GRID, assemble V0, and train V1 and V2. Use seeds 11,22,33 and the agreed clean/robust regimes. B-MAJORITY is deterministic once training targets are fixed and does not require neural retraining per seed. Member 3 owns the final scenario/report matrix.

### Frozen encoder ablation

Load identical selected V0 source encoder weights into V1 and V2 for a controlled fusion comparison. Freeze their parameters and keep the encoders in eval mode, even after calling model.train() on the fusion model. This disables encoder input dropout as well as gradients. Train only gates/attention/classifier parameters; persist the source checkpoint hashes and the frozen flag.

Compare those runs with fully trainable V1/V2, keeping partitions, dimensions, schedules and budgets fixed. V0 trains sources independently while V1/V2 train jointly; report that difference rather than attributing every gain to the fusion operator. Use the same source initialization checkpoints for the corresponding frozen comparisons.

### Outputs supplied to member 3

Export decision_id, source-presence pattern, computable flag, score_kind and uncalibrated scores for validation and calibration through the approved workflow. Join labels only inside the evaluator. Provide training histories, checkpoint selection reason, parameter counts, effective sample/update counts and total training duration. Failed runs retain their error record.

### Resource measurement

Measure model forward and training-step cost separately from dataset I/O. Record device, batch size, actual source lengths, dtype, warm-up count and timed sample count. Synchronize CUDA around device timing. Use twenty warm-up forwards followed by the agreed evaluation decisions; provide p50/p95 to member 3. Report trainable and total parameter counts and peak device memory.

No latency, memory or F1 result is pre-filled. Runtime end-to-end replay measurements belong to member 4. A model that is more expensive and does not satisfy the agreed validation trade-off remains a comparison artifact rather than becoming the selected release model.

## 14 Verification matrix and exit gates

**Encoder and padding** — Unequal lengths; all-absent source; differentiable scatter; correct last state; adding padding or changing padded values does not change eval scores. Compare batched versus individual eval outputs.

**Feature validity** — Changing masked feature values has no effect after zeroing. A valid zero remains a valid feature. Reject non-finite unmasked input values.

**V0 fusion** — Controlled probability average; reduction to one source; deterministic source order; log-clamp score convention; no all-empty probability vector.

**V1 gates** — Absent alpha=0, one-source alpha=1, usable weights sum to one; no softmax on all-masked rows; gradients reach participating modules.

**V2 attention** — Padding never contributes as keys; absent source query outputs are zeroed before pooling; all-absent rows bypass attention; finite gradients through query and source embeddings.

**B GRID** — Exact endpoint and hold-limit fixtures; no future interpolation; event/availability eligibility preserved; variable-length grid padding; explicit coverage when the grid is unusable.

**Trainer** — One valid batch updates weights; all-empty batch skips without stepping; non-finite loss fails; weighted epoch loss and checkpoint tie rules are correct.

**Artifacts and resume** — Fresh-process strict reload; reject changed dimensions/class order; epoch-boundary resume matches the uninterrupted deterministic CPU fixture.

**Integration** — Both application schemas pass forward and training smoke tests; source subsets complete; scores can be consumed by member 3 and loaded by member 4 without private model access.

Probability checks apply only where computable=true: finite values in [0,1] and sum within 1e-5 of one after the appropriate conversion. Compare invariant eval outputs at atol=1e-6, rtol=1e-5 on the fixed CPU test environment. Attention diagnostics with dropout enabled are not required to sum as evaluation probabilities.

All software gates shall pass before a selected artifact is handed to replay. Predictive acceptance is a separate member 3 decision under the frozen protocol. Passing unit tests does not establish real-data accuracy or robustness.

## 15 Development sequence and team handoff

### First executable milestone

Create a source encoder and source-only classifier on synthetic contract batches, then train and reload one PAMAP2 source model. Add independent models for the remaining sources and assemble V0. Confirm that member 4 can load the assembled artifact and member 3 can read its scores before adding more fusion complexity.

```text
Implementation order
  1  contracts and synthetic batches
  2  SourceEncoder and SourceClassifier
  3  trainer, best/last checkpoints and strict reload
  4  V0 assembly and reference predictions
  5  majority and B-GRID baselines
  6  V1 gated fusion
  7  V2 hierarchical attention
  8  robust augmentation and frozen-encoder ablations
  9  complete both application run matrices
 10  handoff and qualification evidence
```

### Handoff checklist

Deliver source code, model YAML files, resolved run configurations, class/source schemas, selected state_dict files, V0 source identities, normalizer references, training histories, validation score exports, test results and a short load/train/resume guide. Preserve failed-run diagnostics and data limitations affecting a model.

Member 1 confirms source dimensions and normalization identity. Member 3 confirms experiment comparability and calibration input convention. Member 4 verifies eval-mode loading and no-data behavior. Member 5 exposes the public train/load workflows in the CLI and packages the dependency lock.

### Commands to expose through the shared CLI

```text
mmt train --config configs/pamap2.yaml --model v0 --seed 11
mmt train --config configs/pamap2.yaml --model v1 --seed 11
mmt train --config configs/pamap2.yaml --model v2 --seed 11
mmt train --config configs/mosi.yaml --model b_grid --seed 11
```

These entry points are implementation targets jointly integrated with member 5. For V0, train orchestrates independent source runs then assembles their selected checkpoints. The run configuration selects clean/robust regime and ablation flags. Resume accepts an explicit checkpoint reference and validates its configuration before continuing.

## 16 References and decision register

[1] PyTorch GRU documentation. Packed sequence support, hidden-state shapes and recurrent-layer dropout behavior.
https://docs.pytorch.org/docs/2.14/generated/torch.nn.GRU.html

[2] PyTorch MultiheadAttention documentation. Query/key/value dimensions and padding-mask semantics.
https://docs.pytorch.org/docs/2.14/generated/torch.nn.MultiheadAttention.html

[3] PyTorch CrossEntropyLoss documentation. Class-logit input and target conventions.
https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html

[4] PyTorch serialization documentation. state_dict and weights_only loading.
https://docs.pytorch.org/docs/2.14/notes/serialization.html

### Status of technical choices

**Library facts** — The references support API behavior. The implementation environment shall still be resolved and tested; no dependency version is claimed to have passed project integration.

**Project architecture** — GRU hidden size, gate sizes, hierarchical attention composition, initialization choices and grid settings define the implementation to evaluate. They are not extracted performance claims from a paper.

**Dataset bindings** — PAMAP2 dimensions/classes follow the parent specification. MOSI dimensions and feature versions come from member 1’s audited manifest, not from a guessed published configuration.

**Evidence still to produce** — Model quality, calibration under degradation, inference cost, memory usage and the final selected variant require the specified validation runs.

Parent documents: MMT-TS-001 Technical Specification and Development Guide v1.0; MMT-WP-001 Member 1 Data Engineering Work Package v1.0. This work package specifies member 2’s implementation while retaining the shared temporal and data contracts. Additional model families or raw-feature encoders require a named extension after the required comparison is executable.
