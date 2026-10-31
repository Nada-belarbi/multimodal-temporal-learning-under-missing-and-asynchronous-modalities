"""Versioned data contracts shared by training, evaluation and runtime services."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    schema_version: Literal["1.0"] = "1.0"


class DataConfig(ContractModel):
    application: Literal["pamap2", "mosi"]
    manifest_path: Path
    partitions_path: Path
    source_schema: dict[str, int]
    decision_policy: Literal["window", "segment"]
    availability_mode: Literal["simulated", "offline", "causal"]
    feature_order: tuple[str, ...]
    normalizer_ref: Path | None = None

    @field_validator("source_schema")
    @classmethod
    def validate_source_schema(cls, value: dict[str, int]) -> dict[str, int]:
        if not value or any(not source or dimension <= 0 for source, dimension in value.items()):
            raise ValueError("source_schema must map nonempty source IDs to positive dimensions")
        return value

    @field_validator("feature_order")
    @classmethod
    def validate_feature_order(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value or any(not name for name in value) or len(set(value)) != len(value):
            raise ValueError("feature_order must contain unique nonempty names")
        return value

    @model_validator(mode="after")
    def validate_profile(self) -> DataConfig:
        expected_policy = {"pamap2": "window", "mosi": "segment"}[self.application]
        if self.decision_policy != expected_policy:
            raise ValueError(f"{self.application} requires decision_policy={expected_policy!r}")
        if self.application == "pamap2" and self.availability_mode != "simulated":
            raise ValueError("PAMAP2 availability must be explicitly marked simulated")
        return self


class RunConfig(ContractModel):
    run_id: str = Field(min_length=1)
    model_name: Literal["majority", "unimodal", "b_grid", "v0", "v1", "v2"]
    seed: int = Field(ge=0)
    regime: Literal["clean", "robust"]
    partition_hashes: dict[str, str]
    model_options: dict[str, Any] = Field(
        default_factory=dict,
        validation_alias=AliasChoices("model_config", "model_options"),
        serialization_alias="model_config",
    )
    training: dict[str, Any] = Field(default_factory=dict)
    output_root: Path

    @field_validator("partition_hashes")
    @classmethod
    def validate_partition_hashes(cls, value: dict[str, str]) -> dict[str, str]:
        if not value or any(not name or not digest for name, digest in value.items()):
            raise ValueError("partition_hashes must contain nonempty names and hashes")
        return value


class RuntimeConfig(ContractModel):
    max_active_sequences: int = Field(default=1, gt=0)
    max_observations_per_source: int = Field(default=100_000, gt=0)
    max_buffer_bytes: int = Field(default=268_435_456, gt=0)
    max_source_age: float | None = Field(default=None, ge=0)
    strict_errors: bool = True


class ProjectConfig(ContractModel):
    data: DataConfig
    run: RunConfig
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    allowed_input_roots: tuple[Path, ...] = ()


class DatasetFileRef(ContractModel):
    sequence_id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_.-]+$")
    relative_path: Path
    provider_reference: str = Field(min_length=1)
    retrieved_at: datetime
    declared_version: str = Field(min_length=1)
    reader_status: str = Field(min_length=1)

    @field_validator("relative_path")
    @classmethod
    def require_relative_safe_path(cls, value: Path) -> Path:
        if value.is_absolute() or ".." in value.parts:
            raise ValueError("relative_path must not be absolute or traverse parent directories")
        return value

    @field_validator("retrieved_at")
    @classmethod
    def require_utc_retrieval_time(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() != timezone.utc.utcoffset(value):
            raise ValueError("retrieved_at must be timezone-aware UTC")
        return value


class Pamap2PreparationConfig(ContractModel):
    application: Literal["pamap2"]
    raw_root: Path
    membership_path: Path
    prepared_root: Path
    audit_root: Path
    preparation_id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_.-]+$")
    recordings: tuple[DatasetFileRef, ...]
    window_seconds: float = Field(default=5.0, gt=0)
    stride_seconds: float = Field(default=1.0, gt=0)
    max_label_age_seconds: float = Field(default=0.02, ge=0)
    chunksize: int = Field(default=100_000, gt=0)

    @field_validator("recordings")
    @classmethod
    def require_unique_recordings(
        cls, value: tuple[DatasetFileRef, ...]
    ) -> tuple[DatasetFileRef, ...]:
        sequence_ids = [recording.sequence_id for recording in value]
        if not value or len(set(sequence_ids)) != len(sequence_ids):
            raise ValueError("recordings must be nonempty with unique sequence IDs")
        return value


class ScenarioRef(ContractModel):
    scenario_id: str = Field(min_length=1)
    schedule_path: Path
    schedule_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    transformation: dict[str, Any]
    schedule_seed: int = Field(ge=0)
    producer_revision: str = Field(min_length=1)


class ArtifactRef(ContractModel):
    artifact_type: str = Field(min_length=1)
    path: Path
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    schema_version: str = Field(min_length=1)
    producer: str = Field(min_length=1)
    created_at_utc: datetime
    parent_hashes: tuple[str, ...] = ()

    @field_validator("created_at_utc")
    @classmethod
    def require_utc_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() != timezone.utc.utcoffset(value):
            raise ValueError("created_at_utc must be timezone-aware UTC")
        return value

    @field_validator("parent_hashes")
    @classmethod
    def validate_parent_hashes(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not re.fullmatch(r"[0-9a-f]{64}", digest) for digest in value):
            raise ValueError("parent_hashes must contain SHA-256 hex digests")
        return value


class PredictionRecord(ContractModel):
    run_id: str = Field(min_length=1)
    scenario_id: str = Field(min_length=1)
    sequence_id: str = Field(min_length=1)
    decision_id: str = Field(min_length=1)
    decision_time: float = Field(ge=0)
    computable: bool
    status: Literal["accepted", "abstained_low_confidence", "unavailable_no_data"]
    score_kind: Literal["logits", "log_probabilities", "label_only"] | None = None
    scores: tuple[float, ...] | None = None
    probabilities: tuple[float, ...] | None = None
    predicted_class: int | None = Field(default=None, ge=0)
    confidence: float | None = Field(default=None, ge=0, le=1)
    diagnostics: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_prediction_semantics(self) -> PredictionRecord:
        if not self.computable:
            if self.status != "unavailable_no_data" or any(
                value is not None
                for value in (
                    self.scores,
                    self.probabilities,
                    self.predicted_class,
                    self.confidence,
                )
            ):
                raise ValueError("uncomputable predictions must be unavailable with null outputs")
        elif self.status == "unavailable_no_data":
            raise ValueError("computable predictions cannot have unavailable_no_data status")
        if self.score_kind == "label_only" and any(
            value is not None for value in (self.scores, self.probabilities, self.confidence)
        ):
            raise ValueError("label-only predictions cannot fabricate scores or confidence")
        if self.probabilities is not None:
            if not self.probabilities or any(not 0 <= value <= 1 for value in self.probabilities):
                raise ValueError("probabilities must be a nonempty vector in [0, 1]")
            if abs(sum(self.probabilities) - 1.0) > 1e-5:
                raise ValueError("probabilities must sum to one")
        return self


class ReleaseManifest(ContractModel):
    release_id: str = Field(min_length=1)
    code_revision: str = Field(min_length=1)
    lockfile_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset_hashes: dict[str, str]
    model_hashes: dict[str, str]
    calibration_hashes: dict[str, str] = Field(default_factory=dict)
    policy_hashes: dict[str, str] = Field(default_factory=dict)
    commands: tuple[str, ...]
    tests: dict[str, str]
    hardware: dict[str, Any]
    qualification_status: Literal["not_qualified", "qualified", "failed"]
