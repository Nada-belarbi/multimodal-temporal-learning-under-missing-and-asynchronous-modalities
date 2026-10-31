import tempfile
import unittest
from pathlib import Path

import numpy as np

from mmt.applications.pamap2 import PAMAP2_COLUMN_COUNT, Pamap2Adapter, Pamap2DataError


def write_fixture(path: Path, rows: int = 4) -> np.ndarray:
    values = np.zeros((rows, PAMAP2_COLUMN_COUNT), dtype=np.float64)
    values[:, 0] = np.arange(rows, dtype=np.float64)
    values[:, 1] = [1, 0, 24, 2][:rows]
    for column in range(PAMAP2_COLUMN_COUNT):
        values[:, column] += column * 100
    values[:, 0] = np.arange(rows, dtype=np.float64)
    values[:, 1] = [1, 0, 24, 2][:rows]
    if rows > 2:
        values[2, 4] = np.nan
    np.savetxt(path, values)
    return values


class Pamap2AdapterTests(unittest.TestCase):
    def test_columns_masks_targets_and_ids_are_independent_of_chunksize(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "Protocol_fixture.dat"
            original = write_fixture(path)
            one_row_chunks = Pamap2Adapter(chunksize=1).read_recording(path, "fixture-sequence")
            large_chunks = Pamap2Adapter(chunksize=100).read_recording(path, "fixture-sequence")

        self.assertEqual(one_row_chunks.sequence_id, "fixture-sequence")
        self.assertEqual(one_row_chunks.sources["hand"]["values"].shape, (4, 6))
        self.assertEqual(one_row_chunks.sources["heart_rate"]["values"].shape, (4, 1))
        np.testing.assert_array_equal(
            one_row_chunks.sources["hand"]["observation_ids"], np.arange(4)
        )
        np.testing.assert_array_equal(
            one_row_chunks.sources["hand"]["observation_ids"],
            large_chunks.sources["hand"]["observation_ids"],
        )
        self.assertEqual(one_row_chunks.sources["hand"]["values"][0, 0], original[0, 4])
        self.assertFalse(one_row_chunks.sources["hand"]["valid_mask"][2, 0])
        self.assertEqual(one_row_chunks.sources["hand"]["values"][2, 0], 0.0)
        self.assertEqual(
            one_row_chunks.sources["hand"]["values"][0, 0],
            large_chunks.sources["hand"]["values"][0, 0],
        )
        np.testing.assert_array_equal(one_row_chunks.target_valid, [True, False, True, True])
        np.testing.assert_array_equal(one_row_chunks.target_class, [0, 0, 11, 1])
        self.assertEqual(one_row_chunks.audit["availability_mode"], "simulated")

    def test_rejects_wrong_column_count_unknown_labels_and_nonmonotonic_time(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.dat"
            np.savetxt(path, np.zeros((2, PAMAP2_COLUMN_COUNT - 1)))
            with self.assertRaises(Pamap2DataError):
                Pamap2Adapter().read_recording(path, "s1")

            values = write_fixture(path, rows=2)
            values[0, 1] = 99
            np.savetxt(path, values)
            with self.assertRaises(Pamap2DataError):
                Pamap2Adapter().read_recording(path, "s1")

            values[0, 1] = 1
            values[:, 0] = [2.0, 1.0]
            np.savetxt(path, values)
            with self.assertRaises(Pamap2DataError):
                Pamap2Adapter().read_recording(path, "s1")


if __name__ == "__main__":
    unittest.main()
