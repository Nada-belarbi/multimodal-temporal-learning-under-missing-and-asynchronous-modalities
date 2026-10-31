"""Verify the PAMAP2 audit and propose class-covered participant partitions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_AUDIT = PROJECT_ROOT / "data/raw/pamap2/provenance/2026-10-26/audit.json"
DEFAULT_RAW_ROOT = PROJECT_ROOT / "data/raw/pamap2/uci_archive/extracted/PAMAP2_Dataset/Protocol"
DEFAULT_OUTPUT = PROJECT_ROOT / "data/raw/pamap2/provenance/2026-10-26/partition-proposal"
EXPECTED_SUBJECTS = tuple(f"subject{subject_id}" for subject_id in range(101, 110))
SPLIT_ORDER = ("train", "validation", "calibration", "test")
DEFAULT_SPLIT_SIZES = {"train": 4, "validation": 2, "calibration": 1, "test": 2}
EXPECTED_CLASSES = (1, 2, 3, 4, 5, 6, 7, 12, 13, 16, 17, 24)


class AuditError(ValueError):
    """Raised when an audited input or proposed partition is inconsistent."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_verified_audit(audit_path: Path, raw_root: Path) -> tuple[dict[str, Any], str]:
    try:
        audit_bytes = audit_path.read_bytes()
        audit = json.loads(audit_bytes)
    except (OSError, json.JSONDecodeError) as error:
        raise AuditError(f"cannot read audit JSON {audit_path}: {error}") from error

    if audit.get("application") != "pamap2":
        raise AuditError("audit application must be 'pamap2'")
    if audit.get("availability_mode") != "simulated":
        raise AuditError("PAMAP2 audit must declare simulated availability")
    recordings = audit.get("recordings")
    if not isinstance(recordings, dict) or set(recordings) != set(EXPECTED_SUBJECTS):
        raise AuditError(f"audit must list exactly {list(EXPECTED_SUBJECTS)}")

    raw_root = raw_root.resolve()
    for sequence_id in EXPECTED_SUBJECTS:
        record = recordings[sequence_id]
        relative_path = Path(record.get("relative_path", ""))
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise AuditError(f"unsafe relative_path for {sequence_id}: {relative_path}")
        recording_path = (raw_root / relative_path).resolve()
        if not recording_path.is_relative_to(raw_root) or not recording_path.is_file():
            raise AuditError(f"recording is missing or escapes raw_root: {recording_path}")
        actual_size = recording_path.stat().st_size
        if actual_size != record.get("bytes"):
            raise AuditError(
                f"size mismatch for {sequence_id}: expected {record.get('bytes')}, got {actual_size}"
            )
        actual_hash = sha256_file(recording_path)
        if actual_hash != record.get("sha256"):
            raise AuditError(f"SHA-256 mismatch for {sequence_id}: {recording_path}")

    return audit, hashlib.sha256(audit_bytes).hexdigest()


def jensen_shannon_to_global(
    split_support: dict[int, int], global_support: dict[int, int]
) -> float:
    split_total = sum(split_support.values())
    global_total = sum(global_support.values())
    divergence = 0.0
    for class_id, global_count in global_support.items():
        split_probability = split_support[class_id] / split_total
        global_probability = global_count / global_total
        midpoint = (split_probability + global_probability) / 2.0
        if split_probability:
            divergence += 0.5 * split_probability * math.log(split_probability / midpoint)
        if global_probability:
            divergence += 0.5 * global_probability * math.log(global_probability / midpoint)
    return divergence


def choose_partition(
    audit: dict[str, Any], split_sizes: dict[str, int]
) -> tuple[dict[str, tuple[str, ...]], dict[str, Any]]:
    recordings = audit["recordings"]
    subjects = tuple(sorted(recordings))
    for subject, record in recordings.items():
        support = record.get("class_support", {})
        if set(support) != {str(class_id) for class_id in EXPECTED_CLASSES}:
            raise AuditError(f"{subject}: class_support must contain the 12 PAMAP2 classes")
        counts = list(support.values())
        row_count = record.get("row_count")
        if (
            any(type(count) is not int or count < 0 for count in counts)
            or type(row_count) is not int
            or row_count <= 0
            or sum(counts) > row_count
        ):
            raise AuditError(f"{subject}: invalid audited class or row counts")
    class_ids = tuple(
        sorted(
            {
                int(class_id)
                for record in recordings.values()
                for class_id in record["class_support"]
            }
        )
    )
    global_support = {
        class_id: sum(
            int(record["class_support"].get(str(class_id), 0)) for record in recordings.values()
        )
        for class_id in class_ids
    }
    if any(count <= 0 for count in global_support.values()):
        raise AuditError("global class support must be positive for every audited class")
    if tuple(split_sizes) != SPLIT_ORDER:
        split_sizes = {name: split_sizes[name] for name in SPLIT_ORDER}
    if any(size <= 0 for size in split_sizes.values()) or sum(split_sizes.values()) != len(
        subjects
    ):
        raise AuditError(
            f"split sizes must be positive and sum to {len(subjects)} subjects: {split_sizes}"
        )

    total_class_rows = sum(global_support.values())
    total_assignments = math.factorial(len(subjects))
    for size in split_sizes.values():
        total_assignments //= math.factorial(size)

    best_key: tuple[Any, ...] | None = None
    best_groups: dict[str, tuple[str, ...]] | None = None
    best_support: dict[str, dict[int, int]] | None = None
    best_rows: dict[str, int] | None = None
    feasible_assignments = 0

    def evaluate(groups: dict[str, tuple[str, ...]]) -> None:
        nonlocal best_key, best_groups, best_support, best_rows, feasible_assignments
        support_by_split = {
            split: {
                class_id: sum(
                    int(recordings[subject]["class_support"].get(str(class_id), 0))
                    for subject in groups[split]
                )
                for class_id in class_ids
            }
            for split in SPLIT_ORDER
        }
        if any(count == 0 for support in support_by_split.values() for count in support.values()):
            return

        feasible_assignments += 1
        class_rows_by_split = {
            split: sum(support_by_split[split].values()) for split in SPLIT_ORDER
        }
        objective = sum(
            class_rows_by_split[split]
            / total_class_rows
            * jensen_shannon_to_global(support_by_split[split], global_support)
            for split in SPLIT_ORDER
        )
        tie_break = tuple(groups[split] for split in SPLIT_ORDER)
        key = (round(objective, 14), tie_break)
        if best_key is None or key < best_key:
            best_key = key
            best_groups = groups.copy()
            best_support = support_by_split
            best_rows = class_rows_by_split

    def assign(
        split_index: int, remaining: tuple[str, ...], groups: dict[str, tuple[str, ...]]
    ) -> None:
        if split_index == len(SPLIT_ORDER) - 1:
            final_split = SPLIT_ORDER[split_index]
            if len(remaining) == split_sizes[final_split]:
                groups[final_split] = remaining
                evaluate(groups)
                groups.pop(final_split)
            return

        split = SPLIT_ORDER[split_index]
        for selected in itertools.combinations(remaining, split_sizes[split]):
            selected_set = set(selected)
            groups[split] = selected
            assign(split_index + 1, tuple(s for s in remaining if s not in selected_set), groups)
            groups.pop(split)

    assign(0, subjects, {})
    if best_groups is None or best_support is None or best_rows is None or best_key is None:
        raise AuditError("no group-disjoint assignment covers every class in every split")

    report = {
        "status": "proposed_for_review",
        "application": "pamap2",
        "availability_mode": "simulated",
        "class_ids": list(class_ids),
        "class_support_unit": "rows from the 12 selected classes, excluding activity 0",
        "split_sizes_subjects": split_sizes,
        "subjects_by_split": {split: list(best_groups[split]) for split in SPLIT_ORDER},
        "class_rows_by_split": best_rows,
        "class_support_by_split": {
            split: {str(class_id): count for class_id, count in best_support[split].items()}
            for split in SPLIT_ORDER
        },
        "global_class_support": {str(key): value for key, value in global_support.items()},
        "selection": {
            "method": "exhaustive deterministic group assignment",
            "coverage_constraint": "every audited class has positive support in every split",
            "objective": "minimize split-row-weighted Jensen-Shannon divergence from global class distribution",
            "logarithm": "natural",
            "tie_break": "lexicographic subject IDs in train, validation, calibration, test order",
            "feasible_assignments": feasible_assignments,
            "evaluated_assignments": total_assignments,
            "best_weighted_divergence": best_key[0],
        },
        "limitations": [
            "This is a deterministic partition proposal, not independent scientific approval.",
            "Calibration contains one participant; uncertainty estimates at participant level are not reliable.",
            "Class support is audited row support; decision-level support must be recomputed after preparation.",
            "PAMAP2 availability is simulated, not measured reception timing.",
        ],
    }
    return best_groups, report


def write_proposal(
    output_dir: Path,
    audit_path: Path,
    audit: dict[str, Any],
    audit_hash: str,
    groups: dict[str, tuple[str, ...]],
    report: dict[str, Any],
) -> None:
    output_dir = output_dir.resolve()
    if not output_dir.is_relative_to(PROJECT_ROOT):
        raise AuditError(f"output directory must be inside the project: {output_dir}")
    if output_dir.exists():
        raise AuditError(f"refusing to overwrite existing output directory: {output_dir}")

    resolved_audit = audit_path.resolve()
    report["audit_path"] = (
        resolved_audit.relative_to(PROJECT_ROOT).as_posix()
        if resolved_audit.is_relative_to(PROJECT_ROOT)
        else resolved_audit.as_posix()
    )
    report["audit_sha256"] = audit_hash
    report["recording_sha256"] = {
        subject: audit["recordings"][subject]["sha256"] for subject in sorted(audit["recordings"])
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    try:
        membership_path = temporary_dir / "membership.csv"
        with membership_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=("sequence_id", "group_id", "split"))
            writer.writeheader()
            for split in SPLIT_ORDER:
                for subject in groups[split]:
                    writer.writerow({"sequence_id": subject, "group_id": subject, "split": split})

        (temporary_dir / "partition_report.json").write_text(
            json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        lines = [
            "# PAMAP2 Participant Partition Proposal",
            "",
            "This deterministic proposal is derived from the audited Protocol subject class counts.",
            "It is not a random split. Each participant belongs to exactly one split.",
            "",
            f"Audit SHA-256: `{audit_hash}`",
            f"Selection objective: `{report['selection']['objective']}`.",
            f"Feasible assignments: {report['selection']['feasible_assignments']} of "
            f"{report['selection']['evaluated_assignments']}.",
            "",
            "| Split | Participants | Labeled class rows |",
            "| --- | --- | ---: |",
        ]
        for split in SPLIT_ORDER:
            participants = ", ".join(groups[split])
            lines.append(f"| {split} | {participants} | {report['class_rows_by_split'][split]} |")
        lines.extend(
            [
                "",
                "## Class Support",
                "",
                "Counts below are audited labeled rows, not window/decision counts.",
                "",
                "| Raw class | " + " | ".join(SPLIT_ORDER) + " |",
                "| ---: | " + " | ".join("---:" for _ in SPLIT_ORDER) + " |",
            ]
        )
        for class_id in report["class_ids"]:
            counts = [
                report["class_support_by_split"][split][str(class_id)] for split in SPLIT_ORDER
            ]
            lines.append(f"| {class_id} | " + " | ".join(str(count) for count in counts) + " |")
        lines.extend(["", "## Limitations", ""])
        lines.extend(f"- {limitation}" for limitation in report["limitations"])
        lines.extend(
            [
                "",
                "Before real-data training, have the project protocol owner review this proposal, "
                "then validate decision-level class support and train-only normalization.",
                "",
            ]
        )
        (temporary_dir / "partition_report.md").write_text("\n".join(lines), encoding="utf-8")
        os.replace(temporary_dir, output_dir)
    except BaseException:
        if temporary_dir.exists():
            for path in temporary_dir.iterdir():
                path.unlink()
            temporary_dir.rmdir()
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify the PAMAP2 audit and generate a deterministic participant split proposal."
    )
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-subjects", type=int, default=DEFAULT_SPLIT_SIZES["train"])
    parser.add_argument(
        "--validation-subjects", type=int, default=DEFAULT_SPLIT_SIZES["validation"]
    )
    parser.add_argument(
        "--calibration-subjects", type=int, default=DEFAULT_SPLIT_SIZES["calibration"]
    )
    parser.add_argument("--test-subjects", type=int, default=DEFAULT_SPLIT_SIZES["test"])
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write membership.csv and partition reports; default prints a preview only.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        audit, audit_hash = load_verified_audit(args.audit.resolve(), args.raw_root.resolve())
        split_sizes = {
            "train": args.train_subjects,
            "validation": args.validation_subjects,
            "calibration": args.calibration_subjects,
            "test": args.test_subjects,
        }
        groups, report = choose_partition(audit, split_sizes)
        if args.write:
            write_proposal(args.output_dir, args.audit, audit, audit_hash, groups, report)
        summary = {
            "status": "written" if args.write else "preview",
            "subjects_by_split": report["subjects_by_split"],
            "class_ids": report["class_ids"],
            "feasible_assignments": report["selection"]["feasible_assignments"],
            "evaluated_assignments": report["selection"]["evaluated_assignments"],
            "weighted_divergence": report["selection"]["best_weighted_divergence"],
            "output_dir": str(args.output_dir.resolve()) if args.write else None,
        }
        print(json.dumps(summary, sort_keys=True))
        return 0
    except (AuditError, OSError, ValueError) as error:
        print(f"audit_pamap2: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
