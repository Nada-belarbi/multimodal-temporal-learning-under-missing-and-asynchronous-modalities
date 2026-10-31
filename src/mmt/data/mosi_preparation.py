"""Offline native-interval MOSI preparation with fixed pretrained word vectors."""

from __future__ import annotations

import ast
import gzip
import hashlib
import itertools
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np
import pandas as pd

from mmt.applications.mosi import MosiAdapter, MosiDataError
from mmt.data.splits import validate_membership
from mmt.storage import write_prepared

SOURCE_DIMS = {"language": 300, "audio": 74, "visual": 35}
STREAMS = {"audio": "COVAREP", "visual": "FACET_4.2"}
LABEL_STREAM = "Opinion Segment Labels"
SILENCE_WORDS = {"", "sp", "<sp>"}


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_official_folds(path: Path) -> dict[str, tuple[str, ...]]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    fields = {
        "standard_train_fold": "train",
        "standard_valid_fold": "validation",
        "standard_test_fold": "test",
    }
    folds = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            raise MosiDataError("fold source must contain only literal list assignments")
        target = node.targets[0]
        if not isinstance(target, ast.Name) or target.id not in fields:
            raise MosiDataError("unknown assignment in official folds")
        values = ast.literal_eval(node.value)
        if (
            not isinstance(values, list)
            or not values
            or any(not isinstance(value, str) or not value for value in values)
        ):
            raise MosiDataError("official fold must be a nonempty video-ID list")
        name = fields[target.id]
        if name in folds or len(values) != len(set(values)):
            raise MosiDataError("duplicate fold assignment or video ID")
        folds[name] = tuple(sorted(values))
    if set(folds) != set(fields.values()):
        raise MosiDataError("official train, validation and test folds are required")
    videos = [video for group in folds.values() for video in group]
    if len(videos) != len(set(videos)):
        raise MosiDataError("a video occurs in multiple official folds")
    return folds


def segment_identity(key: str) -> tuple[str, str]:
    match = re.fullmatch(r"([A-Za-z0-9_-]+)\[([0-9]+)\]", key)
    if match is None:
        raise MosiDataError(f"ambiguous video/segment identity: {key!r}")
    video, segment = match.groups()
    return video, f"mosi_{video}_{int(segment):04d}"


def choose_membership(
    sentiments: Mapping[str, float], folds: Mapping[str, tuple[str, ...]], calibration_videos: int
) -> tuple[pd.DataFrame, dict[str, Any]]:
    video_counts: dict[str, np.ndarray] = {}
    for key, sentiment in sentiments.items():
        if not math.isfinite(sentiment):
            raise MosiDataError(f"non-finite sentiment: {key}")
        video, _ = segment_identity(key)
        counts = video_counts.setdefault(video, np.zeros(2, dtype=np.int64))
        if sentiment != 0:
            counts[int(sentiment > 0)] += 1
    official = {video: split for split, videos in folds.items() for video in videos}
    unknown = set(video_counts) - set(official)
    if unknown:
        raise MosiDataError(f"videos missing from official folds: {sorted(unknown)}")
    development = tuple(video for video in folds["validation"] if video in video_counts)
    if not 0 < calibration_videos < len(development):
        raise MosiDataError("calibration must reserve fewer than all development videos")
    total = sum((video_counts[video] for video in development), np.zeros(2, dtype=np.int64))
    if (total == 0).any():
        raise MosiDataError("development data must contain negative and positive labels")

    def divergence(counts: np.ndarray) -> float:
        probability = counts / counts.sum()
        reference = total / total.sum()
        midpoint = (probability + reference) / 2
        return float(
            0.5 * np.sum(probability * np.log(probability / midpoint))
            + 0.5 * np.sum(reference * np.log(reference / midpoint))
        )

    candidates = []
    for selected in itertools.combinations(development, calibration_videos):
        calibration = sum((video_counts[video] for video in selected), np.zeros(2, dtype=np.int64))
        validation = total - calibration
        if (calibration == 0).any() or (validation == 0).any():
            continue
        objective = (
            calibration.sum() * divergence(calibration) + validation.sum() * divergence(validation)
        ) / total.sum()
        candidates.append((round(float(objective), 14), selected))
    if not candidates:
        raise MosiDataError("no development split covers both sentiment classes")
    objective, selected = min(candidates)
    rows = []
    for key in sorted(sentiments):
        video, sequence_id = segment_identity(key)
        split = "calibration" if video in selected else official[video]
        rows.append({"sequence_id": sequence_id, "group_id": video, "split": split})
    membership = pd.DataFrame(rows)
    split_report = validate_membership(membership.sequence_id.tolist(), membership)
    for split in ("train", "validation", "calibration", "test"):
        videos = membership.loc[membership.split == split, "group_id"].unique()
        support = sum((video_counts[video] for video in videos), np.zeros(2, dtype=np.int64))
        if (support == 0).any():
            raise MosiDataError(f"{split} lacks a sentiment class")
    return membership, {
        "membership": split_report,
        "calibration_videos": list(selected),
        "development_candidates": math.comb(len(development), calibration_videos),
        "feasible_candidates": len(candidates),
        "development_weighted_js": objective,
        "selection": "development-only label-count stratification; lexicographic video tie-break",
        "official_train_test_preserved": True,
        "missing_official_videos": sorted(set(official) - set(video_counts)),
        "grouping_limitation": "source-video disjointness; speaker identity is not verified",
    }


def load_word_vectors(path: Path, vocabulary: set[str], dimension: int) -> dict[str, np.ndarray]:
    found = {}
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        header = stream.readline().split()
        if len(header) != 2 or int(header[1]) != dimension:
            raise MosiDataError("expected word2vec-text GloVe header with audited dimension")
        count = 0
        for line in stream:
            count += 1
            word, separator, numbers = line.rstrip().partition(" ")
            if not separator:
                raise MosiDataError("malformed GloVe row")
            if word not in vocabulary:
                continue
            vector = np.fromstring(numbers, dtype=np.float32, sep=" ")
            if vector.shape != (dimension,) or not np.isfinite(vector).all() or word in found:
                raise MosiDataError(f"invalid or duplicate embedding: {word}")
            found[word] = vector
        if count != int(header[0]):
            raise MosiDataError("GloVe row count does not match header")
    if not found:
        raise MosiDataError("no MOSI words matched pretrained GloVe")
    return found


def embed_words(
    words: np.ndarray, intervals: np.ndarray, vectors: Mapping[str, np.ndarray], dimension: int
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    if words.ndim != 2 or words.shape[1] != 1 or intervals.shape != (len(words), 2):
        raise MosiDataError("words must be [N,1] and native intervals [N,2]")
    values = np.zeros((len(words), dimension), dtype=np.float32)
    mask = np.zeros_like(values, dtype=bool)
    stats = {"rows": len(words), "matched": 0, "silence": 0, "unknown": 0}
    for index, row in enumerate(words):
        word = bytes(row[0]).decode("utf-8", errors="strict").lower()
        if word in SILENCE_WORDS:
            stats["silence"] += 1
        elif word not in vectors:
            stats["unknown"] += 1
        else:
            values[index] = vectors[word]
            mask[index] = True
            stats["matched"] += 1
    return {
        "values": values,
        "valid_mask": mask,
        "observation_ids": np.arange(len(words), dtype=np.int64),
        "event_start": intervals[:, 0],
        "event_end": intervals[:, 1],
        "available_at": intervals[:, 1],
    }, stats


def prepare_mosi_dataset(
    raw_path: Path,
    glove_path: Path,
    folds_path: Path,
    output_root: Path,
    preparation_id: str,
    *,
    expected_hashes: Mapping[str, str],
    provenance: Mapping[str, Any],
    calibration_videos: int = 5,
) -> Path:
    paths = {"raw": raw_path, "glove": glove_path, "folds": folds_path}
    hashes = {name: file_hash(path) for name, path in paths.items()}
    if hashes != dict(expected_hashes):
        raise MosiDataError("input SHA-256 mismatch; cannot prepare changed MOSI inputs")
    folds = read_official_folds(folds_path)
    adapter = MosiAdapter()
    sequences, decisions, targets, audits, index = {}, [], [], [], []
    with h5py.File(raw_path, "r") as hdf:
        labels = hdf[LABEL_STREAM]
        keys = sorted(labels)
        for stream in ("words", *STREAMS.values()):
            if set(hdf[stream]) != set(keys):
                raise MosiDataError(f"segment joins differ for {stream}")
        sentiments = {}
        for key in keys:
            if labels[key]["features"].shape != (1, 1) or labels[key]["intervals"].shape != (1, 2):
                raise MosiDataError(f"ambiguous label: {key}")
            sentiments[key] = float(labels[key]["features"][0, 0])
        membership, split_audit = choose_membership(sentiments, folds, calibration_videos)
        vocabulary = {
            bytes(row[0]).decode("utf-8").lower()
            for key in keys
            for row in hdf["words"][key]["features"][()]
        } - SILENCE_WORDS
        vectors = load_word_vectors(glove_path, vocabulary, SOURCE_DIMS["language"])
        for key in keys:
            video, sequence_id = segment_identity(key)
            start, end = (float(value) for value in labels[key]["intervals"][0])
            words = hdf["words"][key]
            language, word_stats = embed_words(
                words["features"][()],
                words["intervals"][()].astype(np.float64),
                vectors,
                SOURCE_DIMS["language"],
            )
            features = {"language": language}
            invalid_counts = {}
            for source_id, stream in STREAMS.items():
                item = hdf[stream][key]
                values = np.asarray(item["features"], dtype=np.float32)
                intervals = np.asarray(item["intervals"], dtype=np.float64)
                if values.ndim != 2 or intervals.shape != (len(values), 2):
                    raise MosiDataError(f"invalid feature/interval shape: {stream}/{key}")
                mask = np.isfinite(values)
                invalid_counts[source_id] = int((~mask).sum())
                features[source_id] = {
                    "values": np.where(mask, values, 0.0),
                    "valid_mask": mask,
                    "observation_ids": np.arange(len(values), dtype=np.int64),
                    "event_start": intervals[:, 0],
                    "event_end": intervals[:, 1],
                    "available_at": intervals[:, 1],
                }
            segment = adapter.prepare_segment(
                {
                    "sequence_id": sequence_id,
                    "video_id": video,
                    "segment_start": start,
                    "segment_end": end,
                    "sentiment": sentiments[key],
                },
                features,
                SOURCE_DIMS,
                timestamps_relative=False,
                availability_mode="offline",
            )
            if sequence_id in sequences:
                raise MosiDataError(f"canonical sequence ID collision: {key}")
            sequences[sequence_id] = segment.sources
            decisions.append(segment.decision)
            targets.append(segment.target)
            audits.append(
                {
                    **segment.audit,
                    "word_coverage": word_stats,
                    "nonfinite_features": invalid_counts,
                    "original_key": key,
                }
            )
            index.append(
                {
                    "sequence_id": sequence_id,
                    "original_key": key,
                    "video_id": video,
                    "segment_start": start,
                    "segment_end": end,
                }
            )
    source_meta = {
        source_id: {
            "modality_family": source_id,
            "feature_order": [f"{source_id}_{number:03d}" for number in range(dimension)],
            "extractor": "GloVe Wikipedia/Gigaword 6B 300D"
            if source_id == "language"
            else STREAMS[source_id],
            "units": "precomputed feature units; see provider",
        }
        for source_id, dimension in SOURCE_DIMS.items()
    }
    return write_prepared(
        output_root,
        "mosi",
        preparation_id,
        sequences,
        source_meta,
        extra_manifest={
            "application": "mosi",
            "availability_mode": "offline",
            "class_map": {"-1": 0, "1": 1},
            "profile": "native-offline-glove6b300-covarep74-facet42-35",
            "source_inventory_hashes": hashes,
            "provenance": dict(provenance),
            "decision_policy": {"mode": "segment_closure", "zero_sentiment": "excluded"},
            "split_audit": split_audit,
        },
        tables={
            "partitions.csv": membership,
            "decisions.csv": pd.DataFrame(decisions),
            "targets.csv": pd.DataFrame(targets),
            "segment_index.csv": pd.DataFrame(index),
            "segment_audit.csv": pd.DataFrame(
                {
                    "sequence_id": item["sequence_id"],
                    "audit_json": json.dumps(item, sort_keys=True, allow_nan=False),
                }
                for item in audits
            ),
        },
    )
