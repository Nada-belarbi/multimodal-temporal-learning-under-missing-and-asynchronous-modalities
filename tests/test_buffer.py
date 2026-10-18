import unittest

from mmt.runtime.buffer import BufferError, InsertResult, ObservationBuffer


def observation(observation_id, event_start, event_end, available_at, value=1.0,
                sequence_id="sequence", source_id="source"):
    return {
        "sequence_id": sequence_id,
        "source_id": source_id,
        "observation_id": observation_id,
        "values": [value],
        "valid_mask": [True],
        "event_start": event_start,
        "event_end": event_end,
        "available_at": available_at,
    }


class ObservationBufferTests(unittest.TestCase):
    def test_copies_and_sorts_out_of_event_order_arrivals(self):
        buffer = ObservationBuffer()
        first = observation(2, 2.0, 2.0, 3.0)
        buffer.insert(first)
        buffer.insert(observation(1, 1.0, 1.0, 4.0))
        first["values"][0] = 99.0

        snapshot = buffer.snapshot("sequence", "source")

        self.assertEqual(snapshot.observation_ids, (1, 2))
        self.assertEqual(snapshot.values, ((1.0,), (1.0,)))

    def test_identical_duplicate_is_ignored_and_conflict_preserves_original(self):
        buffer = ObservationBuffer()
        record = observation(1, 1.0, 1.0, 1.0)
        buffer.insert(record)

        self.assertEqual(buffer.insert(record), InsertResult.DUPLICATE_IGNORED)
        with self.assertRaises(BufferError) as raised:
            buffer.insert(observation(1, 1.0, 1.0, 1.0, value=2.0))
        self.assertEqual(raised.exception.code, "duplicate_conflict")
        self.assertEqual(buffer.snapshot("sequence", "source").values, ((1.0,),))

    def test_floor_evicts_expired_rows_and_rejects_late_history(self):
        buffer = ObservationBuffer()
        buffer.insert(observation(1, 1.0, 1.0, 1.0))
        buffer.advance_retention_floor("sequence", 2.0)

        self.assertEqual(buffer.row_count, 0)
        with self.assertRaises(BufferError) as raised:
            buffer.insert(observation(2, 1.0, 1.0, 2.0))
        self.assertEqual(raised.exception.code, "expired_observation")

    def test_out_of_order_arrival_rejected_without_advancing_state(self):
        buffer = ObservationBuffer()
        buffer.insert(observation(1, 1.0, 1.0, 2.0))

        with self.assertRaises(BufferError) as raised:
            buffer.insert(observation(2, 1.5, 1.5, 1.9))
        self.assertEqual(raised.exception.code, "invalid_arrival_order")
        self.assertEqual(buffer.row_count, 1)

    def test_capacity_overflow_does_not_partially_commit(self):
        buffer = ObservationBuffer(max_observations_per_source=1)
        buffer.insert(observation(1, 1.0, 1.0, 1.0))

        with self.assertRaises(BufferError) as raised:
            buffer.insert(observation(2, 2.0, 2.0, 2.0))
        self.assertEqual(raised.exception.code, "buffer_overflow")
        self.assertEqual(buffer.row_count, 1)


if __name__ == "__main__":
    unittest.main()