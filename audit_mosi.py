"""Acquire pinned MOSI prerequisites and prepare the native offline training profile."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from mmt.applications.mosi import MosiDataError
from mmt.config import resolve_config
from mmt.data.collate import collate_decisions
from mmt.data.dataset import DecisionDataset
from mmt.data.mosi_preparation import (
    SOURCE_DIMS,
    file_hash,
    prepare_mosi_dataset,
    read_official_folds,
)
from mmt.data.normalization import fit_normalizer, save_normalizer
from mmt.services.data_validation import validate_prepared_splits
from mmt.storage import load_manifest, load_sequence

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs/mosi-preparation.yaml"


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def local_path(value: str, config_dir: Path) -> Path:
    path = (config_dir / value).resolve()
    if not path.is_relative_to(ROOT):
        raise MosiDataError(f"path escapes the project: {path}")
    return path


def download(url: str, destination: Path) -> None:
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    subprocess.run(
        [
            "curl",
            "--fail",
            "--location",
            "--retry",
            "3",
            "--connect-timeout",
            "10",
            "--max-time",
            "600",
            "--continue-at",
            "-",
            "--output",
            str(partial),
            url,
        ],
        check=True,
    )
    os.replace(partial, destination)


def save_new(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"refusing to overwrite changed output: {path}")
        return
    with path.open("x", encoding="utf-8") as stream:
        stream.write(content)


def verify_glove(path: Path, expected_md5: str) -> None:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected_md5:
        raise MosiDataError("GloVe download does not match the provider's published MD5")


def verify_batches(prepared: Path, evidence: Path) -> dict[str, Any]:
    manifest = load_manifest(prepared, verify=True)
    decisions = pd.read_csv(prepared / "decisions.csv")
    targets = pd.read_csv(prepared / "targets.csv")
    partitions = pd.read_csv(prepared / "partitions.csv")
    joined = decisions.merge(
        partitions[["sequence_id", "split"]], on="sequence_id", validate="many_to_one"
    )
    train_decisions = joined[joined.split == "train"].to_dict(orient="records")
    sources = {
        (sequence_id, source_id): arrays
        for sequence_id in sorted({row["sequence_id"] for row in train_decisions})
        for source_id, arrays in load_sequence(prepared, sequence_id).items()
    }
    identity = {
        "manifest_hash": file_hash(prepared / "manifest.json"),
        "training_membership_hash": canonical_hash(
            partitions.sort_values("sequence_id").to_dict(orient="records")
        ),
    }
    normalizer = fit_normalizer(sources, train_decisions, manifest["source_order"], identity)
    evidence.mkdir(parents=True, exist_ok=True)
    normalizer_path = evidence / "normalizer.npz"
    if normalizer_path.exists():
        raise FileExistsError(f"normalizer already exists: {normalizer_path}")
    save_normalizer(normalizer_path, normalizer)
    checks = {}
    for split in ("train", "validation", "calibration", "test"):
        dataset = DecisionDataset(
            prepared,
            decisions,
            partitions,
            normalizer,
            manifest["source_order"],
            targets,
            split=split,
        )
        batch = next(
            iter(
                DataLoader(
                    dataset,
                    batch_size=32,
                    num_workers=0,
                    shuffle=False,
                    collate_fn=collate_decisions,
                )
            )
        )
        shapes = {}
        for source_id, source in batch.inputs.items():
            if (
                not torch.isfinite(source.values).all()
                or not torch.isfinite(source.time_features).all()
            ):
                raise MosiDataError(f"non-finite batch in {split}/{source_id}")
            shapes[source_id] = list(source.values.shape)
        checks[split] = {"dataset_decisions": len(dataset), "batch_shapes": shapes}
    return {
        "batches": checks,
        "normalizer_identity": identity,
        "normalizer_counts": {
            source: normalizer.counts[source].tolist() for source in manifest["source_order"]
        },
        "normalizer_sha256": file_hash(normalizer_path),
    }


def export_run_configs(prepared: Path, evidence: Path) -> None:
    membership = pd.read_csv(prepared / "partitions.csv")
    partition_hashes = {
        split: canonical_hash(group.sort_values("sequence_id").to_dict(orient="records"))
        for split, group in membership.groupby("split")
    }
    for seed in (11, 22, 33):
        config = {
            "data": {
                "application": "mosi",
                "manifest_path": os.path.relpath(prepared / "manifest.json", ROOT / "configs"),
                "partitions_path": os.path.relpath(prepared / "partitions.csv", ROOT / "configs"),
                "source_schema": SOURCE_DIMS,
                "decision_policy": "segment",
                "availability_mode": "offline",
                "feature_order": list(SOURCE_DIMS),
                "normalizer_ref": os.path.relpath(evidence / "normalizer.npz", ROOT / "configs"),
            },
            "run": {
                "run_id": f"mosi-v0-offline-glove6b300-seed-{seed}-2026-10-27",
                "model_name": "v0",
                "seed": seed,
                "regime": "clean",
                "partition_hashes": partition_hashes,
                "model_config": {"hidden_size": 64, "dropout": 0.1, "epsilon": 1e-8},
                "training": {
                    "batch_size": 32,
                    "max_epochs": 100,
                    "patience": 10,
                    "learning_rate": 0.001,
                    "weight_decay": 0.0001,
                    "gradient_clip_norm": 1.0,
                },
                "output_root": "../runs",
            },
        }
        path = ROOT / f"configs/mosi-v0-seed-{seed}.yaml"
        save_new(path, yaml.safe_dump(config, sort_keys=False))
        resolve_config(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--acquire", action="store_true", help="Download pinned prerequisites only."
    )
    parser.add_argument("--prepare", action="store_true", help="Prepare and verify training data.")
    args = parser.parse_args()
    try:
        config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        config_dir = args.config.resolve().parent
        raw_path = local_path(config["raw_path"], config_dir)
        resources = local_path(config["resources_root"], config_dir)
        revision = config["sdk_revision"]
        if len(revision) != 40 or any(char not in "0123456789abcdef" for char in revision):
            raise MosiDataError("SDK revision must be an immutable Git SHA")
        if config["glove_model"] != "glove-wiki-gigaword-300":
            raise MosiDataError("this profile requires GloVe Wikipedia/Gigaword 300D")
        folds = resources / f"cmu_mosi_std_folds-{revision}.py"
        glove = resources / "glove-wiki-gigaword-300.gz"
        fold_url = (
            "https://raw.githubusercontent.com/CMU-MultiComp-Lab/CMU-MultimodalSDK/"
            f"{revision}/mmsdk/mmdatasdk/dataset/standard_datasets/CMU_MOSI/cmu_mosi_std_folds.py"
        )
        glove_url = (
            "https://github.com/piskvorky/gensim-data/releases/download/"
            "glove-wiki-gigaword-300/glove-wiki-gigaword-300.gz"
        )
        if args.acquire:
            download(fold_url, folds)
            read_official_folds(folds)
            download(glove_url, glove)
        verify_glove(glove, config["glove_md5"])
        hashes = {
            name: file_hash(path)
            for name, path in {"raw": raw_path, "glove": glove, "folds": folds}.items()
        }
        expected_hashes = {
            "raw": config["raw_sha256"],
            "glove": config["glove_sha256"],
            "folds": config["folds_sha256"],
        }
        if hashes != expected_hashes:
            raise MosiDataError(
                "an input SHA-256 differs from the pinned preparation configuration"
            )
        provenance = {
            "sdk_revision": revision,
            "folds_url": fold_url,
            "glove_url": glove_url,
            "raw_url": "https://drive.google.com/uc?id=1UblAYI40H-ZQFWqo0-34QyN33sf0Pabk",
            "glove_provider_license": "http://opendatacommons.org/licenses/pddl/",
            "glove_metadata": "https://github.com/piskvorky/gensim-data/blob/master/list.json",
            "sdk_usage_reference": "https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK",
            "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
            "text_scope": "user-authorized frozen pretrained GloVe lookup; no embedding training",
            "license_status": "MOSI mirror redistribution not independently verified; local use only",
            "software": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "torch": torch.__version__,
            },
        }
        save_new(
            resources / "input_hashes.json", json.dumps(hashes, sort_keys=True, indent=2) + "\n"
        )
        if not args.prepare:
            print(json.dumps({"status": "inputs_verified", "sha256": hashes}, sort_keys=True))
            return 0
        evidence = local_path(config["evidence_root"], config_dir) / config["preparation_id"]
        if evidence.exists():
            raise FileExistsError(f"evidence already exists: {evidence}")
        print("Preparing native offline MOSI segments...", flush=True)
        prepared = prepare_mosi_dataset(
            raw_path,
            glove,
            folds,
            local_path(config["prepared_root"], config_dir),
            config["preparation_id"],
            expected_hashes=hashes,
            provenance=provenance,
            calibration_videos=config["calibration_videos"],
        )
        print("Verifying partitions, train-only normalizer and DataLoaders...", flush=True)
        split_report = validate_prepared_splits(
            prepared / "manifest.json", prepared / "partitions.csv"
        )
        checks = verify_batches(prepared, evidence)
        segment_audits = pd.read_csv(prepared / "segment_audit.csv")
        word_stats = {name: 0 for name in ("rows", "matched", "silence", "unknown")}
        invalid_counts = {"audio": 0, "visual": 0}
        for row in segment_audits.itertuples():
            audit = json.loads(row.audit_json)
            for name in word_stats:
                word_stats[name] += audit["word_coverage"][name]
            for name in invalid_counts:
                invalid_counts[name] += audit["nonfinite_features"][name]
        manifest = load_manifest(prepared, verify=False)
        report = {
            "status": "training_ready_local_offline",
            "predictive_qualification": "not_qualified",
            "prepared_path": prepared.relative_to(ROOT).as_posix(),
            "manifest_sha256": file_hash(prepared / "manifest.json"),
            "inputs": hashes,
            "split_report": split_report,
            "split_selection": manifest["split_audit"],
            "word_coverage": word_stats,
            "nonfinite_features": invalid_counts,
            "provenance": provenance,
            **checks,
        }
        save_new(
            evidence / "audit.json",
            json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n",
        )
        save_new(ROOT / "configs/mosi-membership.csv", (prepared / "partitions.csv").read_text())
        export_run_configs(prepared, evidence)
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "prepared_path": str(prepared),
                    "audit": str(evidence / "audit.json"),
                    "counts": split_report["counts"],
                    "word_coverage": word_stats,
                },
                sort_keys=True,
            )
        )
        return 0
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        print(f"audit_mosi: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
