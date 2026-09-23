# Member 1 — Data Engineering

Source: MMT-WP-001 v1.0. This is the text implementation specification transcribed from the approved work package. Names describe responsibilities; individual assignments are pending team selection. Numerical defaults are development settings, not measured performance.

## 1 Assignment and boundaries

Member 1 shall deliver the data engineering layer for PAMAP2 activity recognition and CMU MOSI completed-segment sentiment classification. The result is a tested Python package that converts permitted source files into validated canonical observations, fixed decisions and training-ready batches. Both applications shall use the same storage, normalization and batch contracts.

This work package implements the data portion of MMT-TS-001. It defines work to build, not completed experiments. Listed numerical settings are initial implementation defaults. Source dimensions, usable feature files and partition membership must come from audited evidence; they are not fabricated to complete a configuration.

**Owned modules** — applications/pamap2.py; applications/mosi.py; data/audit.py; storage.py; splits.py; decisions.py; normalization.py; dataset.py; collate.py; matching unit tests and data documentation.

**Shared contracts** — Member 5 maintains contracts.py and config.py. Member 1 supplies data fields, dtypes, validation requirements and representative fixtures before the contracts are frozen.

**Temporal selection** — Member 4 owns data/selection.py, including decision-time eligibility, freshness and temporal channels. Member 1 calls this component and tests the integration; no second selector is implemented.

**Experimental protocol** — Member 3 defines and approves explicit group partitions, scoring exclusions and perturbation protocols. Member 1 materializes and validates those rules.

**Training consumer** — Member 2 consumes source batches and target tensors. Member 1 delivers an executable DataLoader smoke run before full model training starts.

### Required final outcome

A clean run shall read prepared data, load fixed partitions, fit or load the training normalizer, build PAMAP2 and MOSI batches, and emit a verification report. Labels and participant/video grouping metadata shall never occur in model input tensors. Every derived artifact shall identify its input checksums and preparation configuration.

### Work outside this assignment

Model design, optimization, calibration, perturbation generation, the production replay engine and Streamlit are owned by the other members. Dataset acquisition is included; raw-video feature extraction and construction of a new language embedding pipeline require a separate approved scope change.

## 2 Ordered tasks and expected results

**D01 Contract fixtures** — Write small PAMAP2 and MOSI fixtures with valid zeros, missing features, unequal lengths and temporal intervals. Agree the contracts with members 2, 4 and 5. Result: schema fixture and expected output assertions.

**D02 Acquisition and audit** — Inventory permitted files, verify readers and inspect sources, labels, times and group metadata. Result: audit report, file inventory and data dictionary for each dataset.

**D03 Storage and validation** — Implement atomic prepared-data writes, schema validation and load/round-trip checks. Result: reusable storage.py and rejection diagnostics.

**D04 PAMAP2 adapter** — Read Protocol files, extract the specified columns and preserve labels separately. Result: four canonical source streams per recording and the class map.

**D05 MOSI adapter** — Inspect CSD keys, bind actual feature files, preserve native intervals and map segment labels. Result: three source streams per segment and feature provenance.

**D06 Groups and decisions** — Apply the explicit membership supplied by member 3; construct supported decisions and target records. Result: partitions.csv, decisions.csv, targets.csv and exclusion counts.

**D07 Normalization** — Fit statistics on unique eligible training observations and save them. Result: reloadable normalizer and training-only provenance checks.

**D08 Dataset and batching** — Call the shared selector, normalize selected values and collate source-specific sequences. Result: tested PyTorch Dataset and DataLoader for both profiles.

**D09 Handoff** — Run contract, adapter, split, normalization and batch tests; deliver a preparation walkthrough and manifests. Result: signed-off data handoff with known limitations and remaining audit gates.

Implement D01–D04 first to unblock the initial PAMAP2 training chain. Inspect MOSI access and feature availability during D02 so a data dependency is discovered early; complete D05 before claiming the second application is supported. D06 requires the approved membership file, and D08 requires the member 4 selector interface.

No duration is assigned without the team’s availability and audited dataset conditions. A task is complete when its result and verification evidence exist, not when its code file has been created.

## 3 Development environment and module interfaces

**Runtime** — Python 3.11 in the shared uv project. Member 5 owns pyproject.toml and uv.lock; request dependency changes there and use uv sync --frozen after the lock is committed.

**Reading and arrays** — pandas for delimited metadata and PAMAP2 chunks; NumPy for numeric arrays and memory mapping; h5py for validated CSD/HDF5 access.

**MOSI support** — Use a pinned CMU SDK checkout when needed to resolve dataset recipes and computational-sequence metadata. Record its commit; test the actual reader against the acquired files.

**Batching and validation** — torch.utils.data.Dataset/DataLoader; PyTorch tensor dtypes; shared Pydantic configuration validation; pytest and numpy.testing for tests.

**Initial I/O settings** — PAMAP2 read_chunksize=100000 rows. DataLoader batch_size=32, num_workers=0, pin_memory=False, drop_last=False for the initial smoke workflow. Enable performance options only after correctness checks.

```text
audit_dataset(config) -> AuditReport
prepare_dataset(config) -> DatasetManifest
load_sequence(manifest, sequence_id) -> CanonicalSequence
validate_partitions(manifest, membership) -> SplitReport
build_decisions(manifest, profile) -> DecisionTables
fit_normalizer(dataset, train_decisions, selector) -> Normalizer
DecisionDataset.__getitem__(index) -> DecisionSample
collate_decisions(samples) -> Batch
```

These are interfaces to implement, not assertions that the functions already exist. Read and validate YAML through the shared config module. Resolve all relative file paths against an explicitly declared data root. Reject unknown source keys and contradictory dimensions before reading the full dataset.

### Configuration delivered by this member

configs/pamap2.yaml shall specify the raw/prepared roots, Protocol file inventory, source columns, class map and decision settings. configs/mosi.yaml shall specify the audited source file paths, actual root/feature/interval keys, dimensions, label source, time conversion and availability profile. Feature bindings are mandatory audit outputs; unresolved bindings cause validation to fail.

Use independent configuration hashes for preparation and training. Changing a feature order, class map, time interpretation or partition requires a new prepared artifact identity; never overwrite an incompatible dataset in place.

## 4 Canonical schema and prepared storage

```text
data/prepared/<dataset>/<preparation_id>/
  manifest.json          inventory.csv
  partitions.csv        decisions.csv
  targets.csv           exclusions.csv
  audit.json            data_dictionary.csv
  sequences/<sequence_id>/<source_id>/
    values.npy          valid_mask.npy
    observation_ids.npy
    event_start.npy     event_end.npy
    available_at.npy
```

**Observation arrays** — values float32 [N,D]; valid_mask bool [N,D]; observation_ids int64 [N]; event_start, event_end and available_at float64 [N]. IDs are unique within a sequence/source.

**Identity and semantics** — Manifest identifies application, source_id, modality family, feature names/order/units and sequence time origin. Source dimensions may differ. Point events have event_start=event_end.

**Decisions** — decision_id, sequence_id, decision_time, support_start, support_end. No labels or group features. Times use the same seconds and origin as observations.

**Targets** — decision_id, target_class nullable, target_time nullable, score_eligible bool, exclusion_reason nullable. Preserve original labels separately for audit.

**Membership** — sequence_id, group_id, split. Allowed split values: train, validation, calibration, test. One sequence belongs to exactly one group assignment and one split.

**Manifest provenance** — schema version; preparation ID; source file hashes; feature schema; class map; code/config hashes; time units and offsets; extraction context; availability_mode; usage-condition references; output checksums.

Reject mismatched lengths, duplicate conflicting IDs, unknown sources, invalid shapes and non-finite times. Require event_start≤event_end and available_at≥event_end. Causal profiles also enforce documented extraction dependencies; the offline profile uses the declared access convention in Section 7. Retain all-invalid rows for audit; the selector removes them from model input.

Store invalid feature entries as zero with mask=false; retain genuine measured zeros with mask=true. Record NaN/Inf counts and source row/key references before replacement. Use np.load(..., mmap_mode="r", allow_pickle=False). Write to a temporary preparation directory, validate its manifest and checksums, then atomically publish the completed directory. Never mark a partial preparation as complete.

## 5 Acquisition and audit procedure

### D02 implementation sequence

1. Resolve official dataset/recipe references and document applicable usage conditions. Acquire only the selected files into data/raw; retain their original names and bytes. Record retrieval date, provider/version or recipe commit, file size and SHA256. Do not place credentials or access tokens in manifests.

2. Open every file type with the intended reader. For PAMAP2 inspect row width, label IDs and source columns. For MOSI enumerate computational-sequence roots and metadata, feature/interval datasets, source-video keys and segment label joins. Document actual keys; do not assume that every CSD file uses an identical internal path.

3. Compute counts per file/sequence/source: rows, dimensions, valid values, invalid values, all-invalid rows, duplicate time intervals, non-monotonic times and minimum/median/maximum sequence lengths. Report quantiles of positive inter-observation intervals. Same-time records are counted separately, not deduplicated by timestamp.

4. Build the data dictionary with feature name, source, modality, unit, original column/key, dimension, dtype, invalid-value policy and extraction context. Check which timing represents measurement time, representation support or actual availability.

5. Audit participant/video identities and class support before proposing group partitions. Record natural missingness separately from synthetic experiments. No synthetic outage or delay is inserted into the base prepared arrays.

**inventory.csv** — relative_path, bytes, sha256, provider_reference, retrieved_at, declared_version and reader_status.

**audit.json** — Machine-readable counts and violations by sequence/source, class support, join failures, temporal distributions and feature-context status.

**data_dictionary.csv** — Feature order and interpretation sufficient to reconstruct every values column.

**audit_report.md** — Concise findings, suitability decision, limitations and unresolved blocking issues. Include exact paths/keys for any rejected resource.

### Pass or block

Pass when required files are readable, feature/label joins are consistent and the available timing supports the assigned experiment. If MOSI exposes only aligned vectors or undocumented future-context features, record the limitation and block the affected native-time or causal profile. Do not generate artificial timestamps to conceal missing temporal information.

## 6 PAMAP2 adapter and decision rules

Implement Pamap2Adapter with pandas.read_csv using sep=r"\s+", header=None, explicit numeric conversion and chunksize=100000. Verify exactly 54 columns before extraction. Read only inventoried Protocol recordings. Provider metadata documents NaN missingness, three IMUs and a heart-rate source [1]. Preserve each recording as a separate sequence; map its participant identifier in metadata.

**Metadata columns** — Zero-based index 0: seconds; index 1: activity ID. Neither is copied into the predictive feature matrix.

**Movement sources** — hand=[4,5,6,10,11,12]; chest=[21,22,23,27,28,29]; ankle=[38,39,40,44,45,46]. Each D=6: ±16g acceleration then gyroscope.

**Heart rate** — heart_rate=[2], D=1. Keep published timestamps; do not deduplicate repeated values or infer undocumented acquisition times.

**Class mapping** — Raw IDs [1,2,3,4,5,6,7,12,13,16,17,24] map to indices 0–11 in that order. Store raw ID and class name in class_map metadata.

**Initial decision settings** — window_seconds=5.0; stride_seconds=1.0; max_label_age_seconds=0.02. These are development settings, not validated optimal values.

**Availability** — event_start=event_end=published timestamp. Base replay available_at=event_end, explicitly marked simulated; no claim of measured network arrival.

### Implementation details

Assign observation_ids from the original recording row number, stable across chunk sizes. Use np.isfinite for each selected feature mask. Retain all source rows in canonical storage, including invalid rows; maintain a source-row reference in audit metadata. Exclude temperature, alternate acceleration range, magnetometer and orientation from this profile.

Build decision times as session_start+window+k×stride for integer k≥0 while t≤session_end. Support is [t−window,t]. Use searchsorted on sorted label times to find the latest label at or before t. Accept it only if its age≤0.02 seconds. A missing, invalid or older label produces an unscored target with an explicit reason.

Activity 0 remains in history and replay, but its target is excluded from supervised training and primary scoring. Unknown nonzero labels trigger an audit/configuration error. Windows may span activity transitions; do not replace the target with the window-majority label. Preserve gaps and do not forward-fill labels across them.

Gate: a fixture with known column values proves each selected feature order; the target mapping and age boundary tests pass; changing read chunk size does not change canonical arrays or observation IDs.

## 7 CMU MOSI adapter and temporal provenance

### Bind the audited feature profile

Implement MosiAdapter using inspected CSD/HDF5 paths or the pinned CMU SDK. The SDK represents features with temporal intervals in computational sequences [2]. Bind one selected vector sequence for language, audio and visual; read each actual dimension from its array. Record extractor and version where available. This work package does not prescribe fictitious feature names or assume that a recipe download is accessible.

Join feature records and opinion-segment labels by their documented video/segment identity. Require one unambiguous label per segment. Missing modalities are allowed and counted; conflicting labels or ambiguous joins block preparation. Record the original key alongside each prepared sequence ID.

### Time conversion

Keep source-native observation intervals. For video-relative times, subtract the supplied segment_start from both event endpoints and known availability times; store the offset in the manifest. Canonical segment support becomes [0, segment_end−segment_start]. If files already use segment-relative times, validate and record that convention without subtracting the offset a second time.

A precomputed vector crossing the segment support shall not be clipped into a new vector. Exclude it from that segment input and log its original interval and reason. Do not average audio/visual features onto words in this native-time profile. Previously aligned inputs require an explicitly named profile and an audit limitation.

**Target** — score<0 → negative class 0; score>0 → positive class 1; score=0 → excluded. Preserve original score separately. This strict sign convention is used consistently [3].

**Decision** — One decision at supplied segment closure; support [0,duration] after canonical conversion. No partial-segment target and no inferred boundary.

**Offline profile** — Set effective available_at to segment closure for admitted rows and mark availability_mode=offline. This expresses offline access, not physical causal availability.

**Causal feature replay** — Use only documented dependency timing. Segment-context features can be available at closure; later-video-context features cannot. Unknown dependency context blocks causal claims but can remain in the offline profile.

Gate: dimensions and joins match the manifest; zeros are excluded consistently; interval offsets round-trip to original times; an interval crossing a boundary is logged; the adapter preserves unequal modality lengths.

## 8 Group partitions and target construction

### Implement membership supplied by member 3

Member 3 supplies explicit group membership and the grouping policy. Member 1 implements deterministic mapping and checks. For PAMAP2, place all sessions from a participant in the same split. For MOSI, import supplied dataset membership, reserve calibration groups from development data and preserve final test membership under the agreed protocol. Audit whether grouping is by participant or only by source video.

No default percentage or invented participant allocation is authorized by this work package. If supplied MOSI groups conflict with a stronger identity-disjoint policy, report the conflict to member 3 for a named protocol revision; do not silently move groups. Hash the final membership and record any departure from the supplied benchmark protocol.

```text
validate_partitions(manifest, membership):
  require every retained sequence exactly once
  require split in {train, validation, calibration, test}
  reject a group assigned to multiple splits
  reject unknown or duplicate sequence IDs
  report sequence, group and class counts by split
  fail if training lacks an intended class
  fail if required calibration partition is empty
```

### Construct decisions after group allocation

Generate decisions using the application policy, then inherit membership from sequence_id. Generate targets into a separate table. Save decision_id deterministically from application, sequence and scheduled index. Inference must remain possible when targets.csv is not loaded.

The training view includes only score_eligible decisions. Replay and evaluation views retain scheduled decisions with excluded targets so their counts and missing-data behavior remain visible. Do not drop an otherwise score-eligible decision merely because every source is currently unavailable: downstream training skips its loss and evaluation reports availability coverage.

### Required verification output

Write a split report with group counts, sequence counts, scored/excluded decision counts and per-class support for all four splits. Persist labels excluded for transition, zero sentiment, missing label and expired label separately. Verify that no window from a recording can cross its assigned split and no forbidden group occurs twice.

Gate: the membership file and split report have matching hashes; normalizer fitting can obtain only train decision IDs; validation/calibration/test decisions cannot enter the fit path.

## 9 Training normalization implementation

### Fit once on the unperturbed training view

normalization.py shall compute per-source, per-feature mean and population standard deviation from valid unique observations eligible for at least one training decision. Use the common selector from member 4 with the configured support/availability policy. Do not count an observation repeatedly because it appears in overlapping windows. Never fit on synthetic corruption realizations or on development/test observations.

```text
for each training sequence and source:
  eligible = union of observation IDs selected by train decisions
  for each eligible ID exactly once:
    update count, mean and M2 for valid features only
variance = M2 / count                 # population variance
scale = sqrt(max(variance, 0))
scale[scale == 0] = 1
```

Use float64 Welford or equivalent stable mergeable statistics; store counts as int64. Accumulate chunkwise to avoid loading full recordings. The deduplication key is (sequence_id, source_id, observation_id), not observation_id alone. A feature with zero valid training observations fails preparation until a versioned feature-profile decision resolves it.

### Transform

```text
normalized = zeros(values.shape, dtype=float32)
normalized[valid] = ((values - mean) / scale)[valid]
return normalized, valid_mask
```

Preserve real zero values according to their mask. Invalid entries remain zero after normalization and mask=false. Do not normalize masks or timestamps. Temporal channels are computed by the selector and supplied unchanged. No interpolation or carry-forward is introduced here; the separate B-GRID reference belongs to member 2.

### Saved artifact and checks

Save normalizer.npz with counts, means and scales; save metadata identifying source/feature order, schema version, manifest hash, training-membership hash and eligible-observation identity digest. Reject loading against a changed feature order or preparation identity. Validate finite transformed values after the float32 conversion.

Use a test feature with valid values [1,3,5]: mean=3 and population variance=8/3. Duplicate those observations through overlapping decisions and assert that the statistics are unchanged. Add invalid values and real zeros to test mask semantics. Changing any test-only value must leave fitted statistics identical.

## 10 Dataset and batch implementation

### Per-decision loading

DecisionDataset loads a decision and its canonical sources, calls select_observations from member 4, applies the saved normalizer, and returns a DecisionSample. Use memory-mapped NumPy arrays and lazily opened per-worker handles. Avoid serializing open HDF5 handles into DataLoader workers; raw-file conversion is completed before training loads prepared arrays.

A sample contains source tensors, diagnostics and decision metadata; its optional target is a separate object. Group identity and original sentiment score are not source tensors. The trainer passes only the input sub-object to model.forward. A missing source is represented by length zero with its declared feature dimension, not a fabricated measurement.

**values** — Per source: float32 [B,T_m,D_m]. Source m has its own padded temporal length T_m.

**feature_mask** — bool [B,T_m,D_m]; true for valid features. All padding entries are false.

**time_features** — float32 [B,T_m,3], supplied by the selector: log1p(delta), log1p(age), log1p(duration). Padding values are zero.

**lengths and present** — lengths int64 [B]; present bool [B] with present=(lengths>0). Padding does not increase lengths.

**padding_mask** — bool [B,T_m], true exactly at padded positions. Source-level absence is also exposed through present.

**Targets and metadata** — targets int64 [B] for fully supervised batches; separate target_valid mask for mixed/evaluation batches. Metadata stays outside model inputs.

### Collation rules

Pad each source independently to its batch maximum. If every sample lacks a source, allocate one padding position for that source, with lengths=0, present=false and padding_mask=true; this is an explicit batch representation, not a valid row. Member 2 excludes zero-length rows before packed GRU execution.

Retain fixed manifest source order. Reject mixed dimensions or incompatible schemas. Never truncate a long sequence silently; report its ID and dimensions if a configured allocation limit is exceeded. Start smoke tests with batch_size=32, num_workers=0, shuffle=False and pin_memory=False. The training owner enables deterministic training shuffling.

Gate: batches preserve source-specific lengths, are finite and contain no label-derived input; missing sources and entirely missing decisions collate without numerical errors.

## 11 Tests and acceptance results

**test_storage.py** — Save/load arrays and manifest; preserve masks, values, IDs and times. Reject partial artifacts, wrong checksums and incompatible schema versions.

**test_pamap2.py** — Check all zero-based columns with unique fixture values; verify raw IDs, transition exclusions, label age at/after the boundary and row IDs across read chunk sizes.

**test_mosi.py** — Verify feature/interval joins, actual dimension binding, strict score sign, zero exclusions, interval conversion and boundary-crossing exclusions.

**test_splits.py** — Reject shared groups and missing/duplicate membership. Assert that decision split follows its parent sequence and intended training classes exist.

**test_normalization.py** — Check analytic statistics, unique-observation counting, valid-zero preservation, invalid masks, constant-feature scale and unchanged fit after modifying test data.

**test_collate.py** — Assert shapes/dtypes, independent source padding and empty-source behavior. Sentinel values in padding never become valid observations.

**test_data_integration.py** — Compare IDs selected through Dataset and the shared selector at identical decision times. Load a batch without opening the target table in inference mode.

### Required result package

For each dataset, deliver the audit report, inventory, data dictionary, preparation manifest, fixed partitions, decision and target tables, exclusion counts and a batch verification report. Deliver normalizer artifacts with their training provenance. Dataset files are distributed only under their verified conditions; use synthetic fixtures for repository tests.

The batch verification report records run/config hashes, dependency environment, source order/dimensions, sample and batch counts, empty-source counts, tensor dtypes/shapes and test outcomes. Report actual measurements from the run. No predictive F1 or accuracy target belongs to this data-layer handoff.

### Completion criteria

All listed tests pass. Both adapters complete preparation on the selected real datasets. Members 2 and 4 can consume prepared observations through the shared contracts. No labels/groups enter predictive tensors; no test data affects fitting; all exclusions and synthetic availability assumptions are visible. Any unresolved data condition is listed with the affected profile and the exact stage it blocks.

## 12 Handoff guide and references

### First development sequence

```text
1. Agree contracts and create small synthetic fixtures.
2. Audit both datasets and bind feature paths/keys.
3. Implement storage and PAMAP2 conversion.
4. Validate group membership and build decisions.
5. Integrate the shared temporal selector.
6. Fit and save training normalization.
7. Implement Dataset, collate and a DataLoader smoke run.
8. Complete MOSI conversion and repeat the same pipeline.
9. Deliver reports, tests, configurations and handoff notes.
```

### Integration demonstration

With member 5’s CLI, execute audit, prepare and validate-splits using each application configuration. Provide a small data smoke entry point that loads train decisions, fits or loads the normalizer and builds one batch; then iterate the full prepared decision index for schema validation. Member 2 confirms the batch contract by performing a forward pass; member 4 confirms selected observation IDs at the same decision times.

The handoff notes shall include preparation commands, acquired file references, expected generated paths, source order/dimensions, class mapping, partition identity, timing/availability interpretation, normalization provenance, test invocation and known limitations. Commands that require integration with another member shall be labelled as such until that entry point exists.

### Primary sources

[1] UCI PAMAP2 dataset documentation. Column order, source layout and activity IDs.
https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring

[2] CMU Multimodal SDK. Computational sequences, feature/interval representation and feature recipes.
https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK

[3] CMU MOSI README. Sentiment task conventions.
https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK/blob/main/mmsdk/mmdatasdk/dataset/standard_datasets/CMU_MOSI/README.md

Project references: MMT-TS-001 Technical Specification and Development Guide v1.0; MMT-FS-001 Functional Specification v1.0; MMT-APP-001 and MMT-APP-002 Application Scope Briefs. This work package adds data-layer implementation detail while retaining their task and timing boundaries.
