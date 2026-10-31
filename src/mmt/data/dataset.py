"""Memory-mapped per-decision dataset using the shared selector and normalizer."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from mmt.storage import load_manifest, load_sequence

from .normalization import Normalizer
from .selection import SelectionResult, select_observations


class DatasetError(ValueError):
    """Raised when decision, membership, target or prepared-data identities disagree."""


@dataclass(frozen=True)
class SourceSample:
    values: torch.Tensor
    feature_mask: torch.Tensor
    time_features: torch.Tensor


@dataclass(frozen=True)
class DecisionSample:
    decision_id: str
    sequence_id: str
    decision_time: float
    support_start: float
    support_end: float
    sources: Mapping[str, SourceSample]
    target_class: int
    target_valid: bool
    metadata: Mapping[str, Any]


class DecisionDataset(Dataset[DecisionSample]):
    """Load one decision at a time; group and target values remain outside model inputs."""

    def __init__(
        self,
        prepared_root: str | Path,
        decisions: pd.DataFrame,
        membership: pd.DataFrame,
        normalizer: Normalizer,
        source_order: Sequence[str],
        targets: pd.DataFrame | None = None,
        split: str | None = None,
    ) -> None:
        self.prepared_root = Path(prepared_root).resolve()
        self.manifest = load_manifest(self.prepared_root, verify=True)
        manifest_hash = hashlib.sha256(
            (self.prepared_root / "manifest.json").read_bytes()
        ).hexdigest()
        if normalizer.identity.get("manifest_hash") != manifest_hash:
            raise DatasetError("normalizer was fitted against a different prepared manifest")
        self.decisions = decisions.reset_index(drop=True).copy()
        self.membership = membership.set_index("sequence_id")
        if not self.membership.index.is_unique:
            raise DatasetError("membership sequence IDs must be unique")
        self.normalizer = normalizer
        self.source_order = tuple(source_order)
        self.targets = None if targets is None else targets.set_index("decision_id")
        if self.targets is not None and not self.targets.index.is_unique:
            raise DatasetError("target decision IDs must be unique")
        required = {"decision_id", "sequence_id", "decision_time", "support_start", "support_end"}
        if not required <= set(self.decisions.columns):
            raise DatasetError(f"decisions require columns {sorted(required)}")
        if self.decisions.decision_id.duplicated().any():
            raise DatasetError("decision IDs must be unique")
        if not self.source_order or len(set(self.source_order)) != len(self.source_order):
            raise DatasetError("source_order must be nonempty and unique")
        unknown_sources = set(self.source_order) - set(self.manifest["source_dims"])
        if unknown_sources:
            raise DatasetError(f"sources missing from prepared manifest: {sorted(unknown_sources)}")
        if self.targets is not None:
            if set(self.targets.index) - set(self.decisions.decision_id):
                raise DatasetError("targets contain unknown decision IDs")
        missing_membership = set(self.decisions.sequence_id) - set(self.membership.index)
        if missing_membership:
            raise DatasetError(f"decisions have no membership: {sorted(missing_membership)}")
        if split is not None:
            if split not in {"train", "validation", "calibration", "test"}:
                raise DatasetError(f"unknown split {split!r}")
            self.decisions = self.decisions[
                self.decisions.sequence_id.map(self.membership.split) == split
            ].reset_index(drop=True)
        self._sequence_cache: dict[str, dict[str, dict[str, np.ndarray]]] = {}
        self._source_sorted: dict[tuple[str, str], bool] = {}

    def __len__(self) -> int:
        return len(self.decisions)

    def __getitem__(self, index: int) -> DecisionSample:
        row = self.decisions.iloc[index].to_dict()
        decision = {
            name: row[name]
            for name in (
                "decision_id",
                "sequence_id",
                "decision_time",
                "support_start",
                "support_end",
            )
        }
        sequence_id = str(decision["sequence_id"])
        if sequence_id not in self._sequence_cache:
            self._sequence_cache[sequence_id] = load_sequence(self.prepared_root, sequence_id)
        sequence_sources = self._sequence_cache[sequence_id]
        for source_id, source in sequence_sources.items():
            cache_key = (sequence_id, source_id)
            if cache_key not in self._source_sorted:
                starts = source["event_start"]
                ends = source["event_end"]
                self._source_sorted[cache_key] = bool(
                    np.all(starts[1:] >= starts[:-1]) and np.all(ends[1:] >= ends[:-1])
                )
        unknown = set(sequence_sources) - set(self.source_order)
        if unknown:
            raise DatasetError(
                f"sequence {sequence_id!r} contains unconfigured sources {sorted(unknown)}"
            )

        sources: dict[str, SourceSample] = {}
        for source_id in self.source_order:
            raw_source = sequence_sources.get(source_id)
            if raw_source is None:
                selection = _empty_selection()
            else:
                selection = select_observations(
                    raw_source,
                    decision,
                    assume_sorted=self._source_sorted[(sequence_id, source_id)],
                )
            values, mask = self.normalizer.transform(
                source_id,
                np.asarray(selection.values, dtype=np.float32).reshape(
                    -1, self.manifest["source_dims"][source_id]
                ),
                np.asarray(selection.valid_mask, dtype=bool).reshape(
                    -1, self.manifest["source_dims"][source_id]
                ),
            )
            sources[source_id] = SourceSample(
                torch.from_numpy(values),
                torch.from_numpy(mask),
                torch.tensor(selection.time_features, dtype=torch.float32).reshape(-1, 3),
            )

        target_class = 0
        target_valid = False
        if self.targets is not None and decision["decision_id"] in self.targets.index:
            target_row = self.targets.loc[decision["decision_id"]]
            target_valid = bool(target_row["score_eligible"])
            if target_valid:
                value = target_row["target_class"]
                if pd.isna(value):
                    raise DatasetError("score-eligible target has null target_class")
                target_class = int(value)

        membership = self.membership.loc[sequence_id]
        metadata = {
            "group_id": str(membership["group_id"]),
            "split": str(membership["split"]),
        }
        return DecisionSample(
            str(decision["decision_id"]),
            sequence_id,
            float(decision["decision_time"]),
            float(decision["support_start"]),
            float(decision["support_end"]),
            sources,
            target_class,
            target_valid,
            metadata,
        )


def _empty_selection() -> SelectionResult:
    return SelectionResult((), (), (), (), (), (), (), 0, 0, 0, None, None, None, "empty")
