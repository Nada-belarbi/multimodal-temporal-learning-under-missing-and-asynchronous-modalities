import gzip
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from mmt.applications.mosi import MosiAdapter, MosiDataError
from mmt.data.mosi_preparation import (
    SOURCE_DIMS,
    choose_membership,
    embed_words,
    file_hash,
    prepare_mosi_dataset,
    read_official_folds,
)
from mmt.services.data_validation import validate_prepared_splits
from mmt.storage import load_manifest, load_sequence


def feature(values, starts, ends, available, ids=None, mask=None):
    values = np.asarray(values, dtype=np.float32)
    count = values.shape[0]
    return {
        "values": values,
        "valid_mask": np.ones(values.shape, dtype=bool)
        if mask is None
        else np.asarray(mask, dtype=bool),
        "observation_ids": np.arange(count, dtype=np.int64) if ids is None else np.asarray(ids),
        "event_start": np.asarray(starts, dtype=np.float64),
        "event_end": np.asarray(ends, dtype=np.float64),
        "available_at": np.asarray(available, dtype=np.float64),
    }


class MosiAdapterTests(unittest.TestCase):
    def test_hdf5_reader_uses_audited_dataset_keys_and_offline_availability(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "features.csd"
            with h5py.File(path, "w") as hdf:
                hdf.create_dataset("/custom/lang/features", data=[[1.0, np.nan], [2.0, 3.0]])
                hdf.create_dataset("/custom/lang/intervals", data=[[0.0, 0.5], [0.5, 1.0]])
                hdf.create_dataset("/custom/lang/ids", data=[5, 6])
            bindings = {
                "language": {
                    "values": "/custom/lang/features",
                    "event_intervals": "/custom/lang/intervals",
                    "observation_ids": "/custom/lang/ids",
                }
            }

            features = MosiAdapter().read_hdf5_feature_bindings(
                path, bindings, availability_mode="offline"
            )

        self.assertEqual(features["language"]["observation_ids"].tolist(), [5, 6])
        self.assertFalse(features["language"]["valid_mask"][0, 1])
        self.assertEqual(features["language"]["values"][0, 1], 0.0)
        self.assertEqual(features["language"]["available_at"].tolist(), [0.5, 1.0])

    def test_causal_hdf5_reader_requires_audited_availability_binding(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "features.csd"
            with h5py.File(path, "w") as hdf:
                hdf.create_dataset("features", data=[[1.0]])
                hdf.create_dataset("intervals", data=[[0.0, 1.0]])

            with self.assertRaises(MosiDataError):
                MosiAdapter().read_hdf5_feature_bindings(
                    path,
                    {"language": {"values": "features", "event_intervals": "intervals"}},
                    availability_mode="causal",
                )

    def test_absolute_intervals_convert_to_segment_relative_and_crossing_rows_are_excluded(self):
        segment = {
            "sequence_id": "video1-seg2",
            "video_id": "video1",
            "segment_start": 10.0,
            "segment_end": 12.0,
            "sentiment": -0.5,
        }
        features = {
            "language": feature(
                [[1.0], [2.0], [3.0]],
                [9.5, 10.0, 11.0],
                [10.5, 10.5, 11.5],
                [10.5, 10.5, 11.5],
                ids=[1, 2, 3],
            )
        }

        prepared = MosiAdapter().prepare_segment(
            segment,
            features,
            {"language": 1, "audio": 2},
            timestamps_relative=False,
            availability_mode="offline",
        )

        self.assertEqual(prepared.decision["decision_time"], 2.0)
        self.assertEqual(prepared.sources["language"]["observation_ids"].tolist(), [2, 3])
        np.testing.assert_array_equal(prepared.sources["language"]["event_start"], [0.0, 1.0])
        self.assertEqual(prepared.sources["language"]["available_at"].tolist(), [2.0, 2.0])
        self.assertEqual(prepared.sources["audio"]["values"].shape, (0, 2))
        self.assertEqual(prepared.audit["sources"]["language"]["crossing_intervals"], 1)
        self.assertEqual(prepared.target["target_class"], 0)
        self.assertEqual(prepared.audit["availability_mode"], "offline")

    def test_sentiment_zero_is_excluded_and_positive_is_class_one(self):
        base = {"sequence_id": "s1", "video_id": "v1", "segment_start": 0.0, "segment_end": 1.0}
        source_data = {"language": feature([[1.0]], [0.0], [0.5], [0.5])}
        adapter = MosiAdapter()

        zero = adapter.prepare_segment(
            {**base, "sentiment": 0.0}, source_data, {"language": 1}, timestamps_relative=True
        )
        positive = adapter.prepare_segment(
            {**base, "sentiment": 0.2}, source_data, {"language": 1}, timestamps_relative=True
        )

        self.assertFalse(zero.target["score_eligible"])
        self.assertEqual(zero.target["exclusion_reason"], "zero_sentiment")
        self.assertEqual(positive.target["target_class"], 1)

    def test_causal_mode_requires_audited_feature_dependency_context(self):
        segment = {
            "sequence_id": "s1",
            "video_id": "v1",
            "segment_start": 0.0,
            "segment_end": 1.0,
            "sentiment": -1.0,
        }
        sources = {"language": feature([[1.0]], [0.0], [0.5], [0.5])}
        with self.assertRaises(MosiDataError):
            MosiAdapter().prepare_segment(
                segment,
                sources,
                {"language": 1},
                timestamps_relative=True,
                availability_mode="causal",
            )

        prepared = MosiAdapter().prepare_segment(
            segment,
            sources,
            {"language": 1},
            timestamps_relative=True,
            availability_mode="causal",
            extraction_context={"language": "audited causal feature"},
        )
        self.assertEqual(prepared.sources["language"]["available_at"].tolist(), [0.5])


class MosiPreparationTests(unittest.TestCase):
    def test_development_split_is_deterministic_and_preserves_official_train_test(self):
        folds = {
            "train": ("train1",),
            "validation": ("dev1", "dev2", "dev3", "dev4"),
            "test": ("test1",),
        }
        sentiments = {
            f"{video}[{index}]": value
            for videos in folds.values()
            for video in videos
            for index, value in enumerate((-1.0, 1.0, 0.0))
        }
        membership, report = choose_membership(sentiments, folds, 2)
        repeated, _ = choose_membership(dict(reversed(list(sentiments.items()))), folds, 2)
        pd.testing.assert_frame_equal(membership, repeated)
        self.assertEqual(report["calibration_videos"], ["dev1", "dev2"])
        self.assertTrue(report["official_train_test_preserved"])
        self.assertEqual(membership.groupby("group_id").split.nunique().max(), 1)
        self.assertTrue((membership[membership.group_id == "train1"].split == "train").all())
        self.assertTrue((membership[membership.group_id == "test1"].split == "test").all())
        with self.assertRaises(MosiDataError):
            choose_membership(sentiments | {"unlisted[0]": 1.0}, folds, 2)
        with self.assertRaises(MosiDataError):
            choose_membership(sentiments, folds, 4)

    def test_word_lookup_preserves_times_and_masks_unknown_and_silence(self):
        words = np.asarray([[b"Known"], [b"missing"], [b"sp"]])
        intervals = np.asarray([[10.0, 10.4], [10.4, 10.7], [10.7, 11.0]])
        source_data, stats = embed_words(words, intervals, {"known": np.asarray([0.0, 2.0])}, 2)
        self.assertEqual(stats, {"rows": 3, "matched": 1, "silence": 1, "unknown": 1})
        self.assertEqual(
            source_data["valid_mask"].tolist(), [[True, True], [False, False], [False, False]]
        )
        self.assertEqual(source_data["values"][0].tolist(), [0.0, 2.0])
        np.testing.assert_array_equal(source_data["event_start"], intervals[:, 0])
        np.testing.assert_array_equal(source_data["event_end"], intervals[:, 1])

    def test_fold_reader_rejects_executable_source(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "folds.py"
            path.write_text("import os\n", encoding="utf-8")
            with self.assertRaises(MosiDataError):
                read_official_folds(path)

    def test_native_offline_preparation_writes_three_sources_and_separate_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folds = {"train": ["train1"], "validation": ["dev1", "dev2"], "test": ["test1"]}
            fold_path = root / "folds.py"
            fold_path.write_text(
                f"standard_train_fold={folds['train']!r}\n"
                f"standard_valid_fold={folds['validation']!r}\n"
                f"standard_test_fold={folds['test']!r}\n",
                encoding="utf-8",
            )
            glove_path = root / "glove.gz"
            with gzip.open(glove_path, "wt", encoding="utf-8") as stream:
                stream.write("1 300\nknown " + " ".join(["1.0"] * 300) + "\n")
            raw_path = root / "mosi.hdf5"
            with h5py.File(raw_path, "w") as hdf:
                for videos in folds.values():
                    for video in videos:
                        for index, sentiment in enumerate((-1.0, 1.0, 0.0)):
                            key = f"{video}[{index}]"
                            label = hdf.require_group(f"Opinion Segment Labels/{key}")
                            label.create_dataset("features", data=[[sentiment]])
                            label.create_dataset("intervals", data=[[10.0, 12.0]])
                            words = hdf.require_group(f"words/{key}")
                            words.create_dataset(
                                "features", data=np.asarray([[b"known"], [b"unknown"]])
                            )
                            words.create_dataset("intervals", data=[[10.0, 11.0], [11.0, 12.0]])
                            for name, dimension in (("COVAREP", 74), ("FACET_4.2", 35)):
                                item = hdf.require_group(f"{name}/{key}")
                                values = np.ones((2, dimension), dtype=np.float32)
                                values[0, 0] = np.nan
                                item.create_dataset("features", data=values)
                                item.create_dataset("intervals", data=[[10.0, 10.5], [10.5, 12.0]])
            hashes = {
                name: file_hash(path)
                for name, path in {"raw": raw_path, "glove": glove_path, "folds": fold_path}.items()
            }
            prepared = prepare_mosi_dataset(
                raw_path,
                glove_path,
                fold_path,
                root / "prepared",
                "fixture",
                expected_hashes=hashes,
                provenance={"provider": "synthetic-test-only"},
                calibration_videos=1,
            )
            manifest = load_manifest(prepared)
            self.assertEqual(manifest["source_dims"], SOURCE_DIMS)
            self.assertEqual(manifest["availability_mode"], "offline")
            self.assertEqual(validate_prepared_splits(prepared / "manifest.json")["n_classes"], 2)
            sequence = load_sequence(prepared, "mosi_train1_0000")
            self.assertEqual(sequence["language"]["observation_ids"].tolist(), [0])
            np.testing.assert_array_equal(sequence["language"]["event_start"], [0.0])
            np.testing.assert_array_equal(sequence["language"]["available_at"], [2.0])
            self.assertFalse(sequence["audio"]["valid_mask"][0, 0])
            targets = pd.read_csv(prepared / "targets.csv")
            self.assertEqual(len(targets), 12)
            self.assertEqual(int(targets.score_eligible.sum()), 8)
            self.assertEqual(targets.exclusion_reason.eq("zero_sentiment").sum(), 4)
            partitions = pd.read_csv(prepared / "partitions.csv")
            self.assertEqual(partitions.groupby("group_id").split.nunique().max(), 1)
            with self.assertRaises(MosiDataError):
                prepare_mosi_dataset(
                    raw_path,
                    glove_path,
                    fold_path,
                    root / "prepared",
                    "bad-hash",
                    expected_hashes=hashes | {"raw": "0" * 64},
                    provenance={},
                    calibration_videos=1,
                )


if __name__ == "__main__":
    unittest.main()
