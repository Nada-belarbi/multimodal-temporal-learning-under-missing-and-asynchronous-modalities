import unittest

from mmt.runtime.buffer import BufferError
from mmt.runtime.engine import InferenceEngine
from mmt.runtime.replay import replay


def observation(observation_id, available_at, value=1.0, sequence_id="sequence"):
    return {
        "sequence_id": sequence_id, "source_id": "source", "observation_id": observation_id,
        "values": [value], "valid_mask": [True], "event_start": available_at,
        "event_end": available_at, "available_at": available_at,
    }


def decision(decision_id, decision_time, sequence_id="sequence"):
    return {
        "sequence_id": sequence_id, "decision_id": decision_id, "decision_time": decision_time,
        "support_start": 0.0, "support_end": decision_time,
    }


def make_engine(threshold=None):
    return InferenceEngine(
        source_ids=("source",),
        batch_builder=lambda selections: selections,
        predict_fn=lambda _: {"scores": [[0.0, 2.0]], "computable": [True], "score_kind": "logits"},
        n_classes=2,
        threshold=threshold,
    )


class RuntimeEngineTests(unittest.TestCase):
    def test_replay_delivers_arrivals_before_the_same_time_decision(self):
        engine = make_engine()
        predictions = list(replay(engine, [observation(1, 2.0)], [decision("d1", 2.0)]))

        self.assertEqual(predictions[0].selected_ids["source"], (1,))
        self.assertEqual(predictions[0].status, "accepted")

    def test_no_data_does_not_call_the_model_or_fabricate_probabilities(self):
        engine = InferenceEngine(
            source_ids=("source",), batch_builder=lambda selections: selections,
            predict_fn=lambda _: self.fail("model must not run without selected observations"), n_classes=2,
        )

        prediction = engine.predict(decision("d1", 2.0))

        self.assertEqual(prediction.status, "unavailable_no_data")
        self.assertIsNone(prediction.probabilities)

    def test_new_arrival_cannot_change_an_already_emitted_decision(self):
        engine = make_engine()
        engine.update(observation(1, 1.0))
        engine.predict(decision("d1", 1.0))

        with self.assertRaises(BufferError) as raised:
            engine.update(observation(2, 1.0))
        self.assertEqual(raised.exception.code, "invalid_arrival_order")

    def test_invalid_decision_does_not_evict_buffered_observations(self):
        engine = make_engine()
        engine.update(observation(1, 1.0))

        with self.assertRaises(BufferError):
            engine.predict({"sequence_id": "sequence", "decision_id": "bad", "decision_time": 3.0,
                            "support_start": 2.0, "support_end": 4.0})
        self.assertEqual(engine.buffer.row_count, 1)

    def test_threshold_keeps_candidate_scores_on_abstention(self):
        prediction = list(replay(make_engine(threshold=0.9), [observation(1, 1.0)],
                                 [decision("d1", 1.0)]))[0]

        self.assertEqual(prediction.status, "abstained_low_confidence")
        self.assertFalse(prediction.accepted)
        self.assertIsNotNone(prediction.probabilities)

    def test_batch_and_replay_select_identical_ids_and_scores(self):
        records = [observation(1, 1.0), observation(2, 2.0, value=3.0)]
        stateful = list(replay(make_engine(), records, [decision("d1", 2.0)]))[0]
        source = {
            ("sequence", "source"): {
                "observation_ids": [record["observation_id"] for record in records],
                "values": [[record["values"][0]] for record in records],
                "valid_mask": [[True] for _ in records],
                "event_start": [record["event_start"] for record in records],
                "event_end": [record["event_end"] for record in records],
                "available_at": [record["available_at"] for record in records],
            }
        }
        stateless = make_engine().predict_batch([decision("d1", 2.0)], source)[0]

        self.assertEqual(stateless.selected_ids, stateful.selected_ids)
        self.assertEqual(stateless.scores, stateful.scores)
        self.assertEqual(stateless.status, stateful.status)

    def test_reset_advances_prediction_generation(self):
        engine = InferenceEngine(
            source_ids=("source",), batch_builder=lambda selections: selections,
            predict_fn=lambda _: {"scores": [[1.0, 0.0]], "computable": [True], "score_kind": "logits"},
            n_classes=2, session_id="session", run_id="run", scenario_id="clean",
        )
        engine.update(observation(1, 1.0))
        first = engine.predict(decision("d1", 1.0))
        engine.reset("sequence")
        engine.update(observation(1, 1.0))
        second = engine.predict(decision("d1", 1.0))

        self.assertEqual((first.session_id, first.generation, first.run_id, first.scenario_id),
                         ("session", 0, "run", "clean"))
        self.assertEqual(second.generation, 1)

    def test_model_exception_is_not_reported_as_no_data(self):
        engine = InferenceEngine(
            source_ids=("source",), batch_builder=lambda selections: selections,
            predict_fn=lambda _: (_ for _ in ()).throw(ValueError("bad model")), n_classes=2,
        )
        engine.update(observation(1, 1.0))

        with self.assertRaisesRegex(RuntimeError, "model_execution_failed: decision d1"):
            engine.predict(decision("d1", 1.0))

    def test_replay_rejects_unsorted_arrivals(self):
        engine = make_engine()
        arrivals = [observation(1, 2.0), observation(2, 1.0)]

        with self.assertRaises(BufferError):
            list(replay(engine, arrivals, [decision("d1", 2.0), decision("d2", 3.0)]))


if __name__ == "__main__":
    unittest.main()