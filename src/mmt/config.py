"""Strict YAML configuration loading, path policy and canonical identities."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml
from pydantic import ValidationError

from .contracts import Pamap2PreparationConfig, ProjectConfig

ALLOWED_OVERRIDES = frozenset(
    {
        "run.model_name",
        "run.seed",
        "run.run_id",
        "run.regime",
        "run.output_root",
        "runtime.strict_errors",
    }
)


class ConfigError(ValueError):
    """Raised when configuration syntax, schema, inputs or paths are invalid."""


@dataclass(frozen=True)
class ResolvedConfig:
    config: ProjectConfig
    source_path: Path
    project_root: Path
    input_hashes: Mapping[str, str]
    config_sha256: str
    canonical_json: str


@dataclass(frozen=True)
class ResolvedPamap2Preparation:
    config: Pamap2PreparationConfig
    source_path: Path
    project_root: Path
    raw_root: Path
    membership_path: Path
    prepared_root: Path
    audit_root: Path
    recording_paths: Mapping[str, Path]
    source_provenance: Mapping[str, Mapping[str, str]]
    config_sha256: str


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _apply_overrides(raw: dict[str, Any], overrides: Mapping[str, Any]) -> None:
    unknown = set(overrides) - ALLOWED_OVERRIDES
    if unknown:
        raise ConfigError(f"undeclared overrides: {sorted(unknown)}")
    for dotted_name, value in overrides.items():
        section, field = dotted_name.split(".", maxsplit=1)
        raw.setdefault(section, {})[field] = value


def _path_from_config(value: Path, config_dir: Path) -> Path:
    return (config_dir / value).resolve() if not value.is_absolute() else value.resolve()


def resolve_config(
    path: str | Path,
    overrides: Mapping[str, Any] | None = None,
    *,
    require_inputs: bool = True,
) -> ResolvedConfig:
    """Load, validate and hash one YAML config without opening data or model artifacts."""
    source_path = Path(path).expanduser().resolve()
    if not source_path.is_file():
        raise ConfigError(f"configuration file does not exist: {source_path}")
    try:
        raw = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ConfigError(f"cannot read YAML config {source_path}: {error}") from error
    if not isinstance(raw, dict):
        raise ConfigError("configuration root must be a YAML mapping")
    _apply_overrides(raw, overrides or {})
    try:
        config = ProjectConfig.model_validate(raw)
    except ValidationError as error:
        raise ConfigError(f"configuration schema validation failed:\n{error}") from error

    config_dir = source_path.parent
    project_root = next(
        (
            parent.resolve()
            for parent in source_path.parents
            if (parent / "pyproject.toml").is_file()
        ),
        config_dir,
    )
    allowed_roots = tuple(
        _path_from_config(root, config_dir) for root in config.allowed_input_roots
    )
    data = config.data
    run = config.run
    paths = {
        "data.manifest_path": _path_from_config(data.manifest_path, config_dir),
        "data.partitions_path": _path_from_config(data.partitions_path, config_dir),
        "run.output_root": _path_from_config(run.output_root, config_dir),
    }
    if data.normalizer_ref is not None:
        paths["data.normalizer_ref"] = _path_from_config(data.normalizer_ref, config_dir)

    input_paths = {key: value for key, value in paths.items() if key != "run.output_root"}
    input_hashes: dict[str, str] = {}
    for key, resolved_path in input_paths.items():
        in_project = _inside(resolved_path, project_root)
        in_allowed_root = any(_inside(resolved_path, root) for root in allowed_roots)
        if not in_project and not in_allowed_root:
            raise ConfigError(
                f"{key} escapes the project and declared input roots: {resolved_path}"
            )
        if require_inputs:
            if not resolved_path.is_file():
                raise ConfigError(f"{key} is not an existing file: {resolved_path}")
            input_hashes[key] = _sha256_file(resolved_path)

    output_root = paths["run.output_root"]
    if not _inside(output_root, project_root):
        raise ConfigError(f"run.output_root must remain inside the project root: {output_root}")

    resolved_data = data.model_copy(
        update={
            "manifest_path": paths["data.manifest_path"],
            "partitions_path": paths["data.partitions_path"],
            "normalizer_ref": paths.get("data.normalizer_ref"),
        }
    )
    resolved_run = run.model_copy(update={"output_root": output_root})
    resolved = config.model_copy(
        update={"data": resolved_data, "run": resolved_run, "allowed_input_roots": allowed_roots}
    )

    payload = resolved.model_dump(mode="json", by_alias=True)

    def portable(path_key: str, resolved_path: Path) -> str:
        if _inside(resolved_path, project_root):
            return resolved_path.relative_to(project_root).as_posix()
        for index, root in enumerate(allowed_roots):
            if _inside(resolved_path, root):
                return f"@input_root/{index}/{resolved_path.relative_to(root).as_posix()}"
        raise ConfigError(f"cannot make {path_key} portable")

    payload["data"]["manifest_path"] = portable("data.manifest_path", paths["data.manifest_path"])
    payload["data"]["partitions_path"] = portable(
        "data.partitions_path", paths["data.partitions_path"]
    )
    if data.normalizer_ref is not None:
        payload["data"]["normalizer_ref"] = portable(
            "data.normalizer_ref", paths["data.normalizer_ref"]
        )
    payload["run"]["output_root"] = portable("run.output_root", output_root)
    payload["allowed_input_roots"] = [
        f"@input_root/{index}" for index, _ in enumerate(allowed_roots)
    ]
    payload["input_hashes"] = input_hashes
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )
    config_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return ResolvedConfig(resolved, source_path, project_root, input_hashes, config_hash, canonical)


def save_resolved_config(resolved: ResolvedConfig, output_dir: str | Path) -> tuple[Path, Path]:
    """Write a portable resolved YAML and its canonical SHA-256 into a new run directory."""
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    config_path = destination / "resolved_config.yaml"
    hash_path = destination / "config.sha256"
    if config_path.exists() or hash_path.exists():
        raise FileExistsError(f"resolved config already exists in {destination}")
    payload = json.loads(resolved.canonical_json)
    config_path.write_text(
        yaml.safe_dump(payload, sort_keys=True, allow_unicode=True), encoding="utf-8"
    )
    hash_path.write_text(f"{resolved.config_sha256}  resolved_config.yaml\n", encoding="ascii")
    return config_path, hash_path


def resolve_pamap2_preparation(path: str | Path) -> ResolvedPamap2Preparation:
    """Resolve an explicitly inventoried PAMAP2 preparation YAML without scanning directories."""
    source_path = Path(path).expanduser().resolve()
    if not source_path.is_file():
        raise ConfigError(f"configuration file does not exist: {source_path}")
    try:
        raw = yaml.safe_load(source_path.read_text(encoding="utf-8"))
        config = Pamap2PreparationConfig.model_validate(raw)
    except (OSError, yaml.YAMLError, ValidationError) as error:
        raise ConfigError(f"PAMAP2 preparation config is invalid: {error}") from error

    project_root = next(
        (
            parent.resolve()
            for parent in source_path.parents
            if (parent / "pyproject.toml").is_file()
        ),
        source_path.parent,
    )
    config_dir = source_path.parent
    raw_root = config.raw_root.expanduser()
    raw_root = (
        (config_dir / raw_root).resolve() if not raw_root.is_absolute() else raw_root.resolve()
    )
    if not raw_root.is_dir():
        raise ConfigError(f"raw_root does not exist or is not a directory: {raw_root}")

    membership = config.membership_path.expanduser()
    membership = (
        (config_dir / membership).resolve()
        if not membership.is_absolute()
        else membership.resolve()
    )
    if not membership.is_file() or not membership.is_relative_to(project_root):
        raise ConfigError(
            f"membership_path must be an existing file inside the project: {membership}"
        )

    def output_path(value: Path, field: str) -> Path:
        resolved = (config_dir / value).resolve() if not value.is_absolute() else value.resolve()
        if not resolved.is_relative_to(project_root):
            raise ConfigError(f"{field} must remain inside the project: {resolved}")
        return resolved

    prepared_root = output_path(config.prepared_root, "prepared_root")
    audit_root = output_path(config.audit_root, "audit_root")
    recordings: dict[str, Path] = {}
    provenance: dict[str, dict[str, str]] = {}
    for recording in config.recordings:
        resolved = (raw_root / recording.relative_path).resolve()
        if not resolved.is_relative_to(raw_root) or not resolved.is_file():
            raise ConfigError(f"recording {recording.sequence_id!r} is missing or escapes raw_root")
        recordings[recording.sequence_id] = resolved
        provenance[recording.sequence_id] = {
            "provider_reference": recording.provider_reference,
            "retrieved_at": recording.retrieved_at.isoformat(),
            "declared_version": recording.declared_version,
            "reader_status": recording.reader_status,
        }

    portable = config.model_dump(mode="json")
    portable["raw_root"] = "@declared_raw_root"
    portable["membership_path"] = membership.relative_to(project_root).as_posix()
    portable["prepared_root"] = prepared_root.relative_to(project_root).as_posix()
    portable["audit_root"] = audit_root.relative_to(project_root).as_posix()
    canonical = json.dumps(
        portable, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )
    config_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return ResolvedPamap2Preparation(
        config,
        source_path,
        project_root,
        raw_root,
        membership,
        prepared_root,
        audit_root,
        recordings,
        provenance,
        config_hash,
    )
