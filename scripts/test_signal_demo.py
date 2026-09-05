#!/usr/bin/env python3
"""Check the data and measurement boundaries of the connected demonstration."""

import copy
import csv
from pathlib import Path
import tempfile
import unittest

import signal_demo as demo


class DemoContractTests(unittest.TestCase):
    def setUp(self):
        self.config = demo.read_json(demo.DEFAULT_CONFIG)
        self.training = {"schema": "crlp.signal_flight.summary.v1", "training_mode": "counterfactual_utility",
                         "feature_mode": "live_observable", "model_hash": "a" * 64,
                         "parameter_count": 2113, "train_rows": 18, "test_rows": 9,
                         **self.config["training"]}
        self.rollout = [{"schema": "crlp.signal_flight_rollout.summary.v1", "controller": name,
                         "checkpoint_hash": "a" * 64, "episodes": self.config["rollout"]["episodes"],
                         "ticks_per_episode": self.config["rollout"]["ticks"], "seed": self.config["rollout"]["seed"],
                         "wall_probes": True, "avg_utility": 0.2, "damage_episode_rate": 0.25, "death_rate": 0.0}
                        for name in sorted(demo.CONTROLLERS)]

    def metrics(self, training=None, rollout=None):
        return demo.result_metrics(training or self.training, rollout or self.rollout, self.config, "a" * 64)

    def test_negative_measured_result_is_retained(self):
        next(row for row in self.rollout if row["controller"] == "neural_guarded")["avg_utility"] = -0.2
        self.assertEqual(self.metrics()["utility_delta"], -0.4)

    def test_future_information_model_is_rejected(self):
        self.training["feature_mode"] = "replay_outcome"
        with self.assertRaisesRegex(ValueError, "outcome features"):
            self.metrics()

    def test_different_model_checkpoint_is_rejected(self):
        self.rollout[0]["checkpoint_hash"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "different checkpoint"):
            self.metrics()

    def test_missing_or_duplicate_controller_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "four distinct"):
            self.metrics(rollout=self.rollout[:-1])
        self.rollout[0] = copy.deepcopy(self.rollout[1])
        with self.assertRaisesRegex(ValueError, "four distinct"):
            self.metrics()

    def test_changed_evaluation_seed_is_rejected(self):
        self.rollout[0]["seed"] += 1
        with self.assertRaisesRegex(ValueError, "seed differs"):
            self.metrics()

    def test_nonfinite_outcome_is_rejected(self):
        self.rollout[0]["avg_utility"] = float("nan")
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            self.metrics()

    def test_changed_schedule_is_rejected(self):
        self.training["epochs"] += 1
        with self.assertRaisesRegex(ValueError, "schedule differs"):
            self.metrics()

    def test_artifact_change_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.csv"
            path.write_text("original\n")
            receipt = demo.binding("train-csv", path)
            demo.verify_bindings([receipt])
            path.write_text("changed\n")
            with self.assertRaisesRegex(ValueError, "Artifact changed"):
                demo.verify_bindings([receipt])

    def test_replay_group_requires_all_nine_actions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.csv"
            fields = ["schema", "seed", "episode", "tick", "action", "certificate_status", "proof_kind",
                      "utility", "damage_events", "death_events", "hull_loss"]
            def trace(actions):
                with path.open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=fields)
                    writer.writeheader()
                    for action in actions:
                        writer.writerow(dict(zip(fields, ["signal.flight_counterfactual.v1", 4242, 0, 0, action,
                                                         "certified", "deterministic_authoritative_replay", 0.1, 0, 0, 0])))
            trace(range(9))
            self.assertEqual(demo.inspect_trace(path)["rows"], 9)
            trace(range(8))
            with self.assertRaisesRegex(ValueError, "all nine actions"):
                demo.inspect_trace(path)
            trace([0] + list(range(8)))
            with self.assertRaisesRegex(ValueError, "all nine actions"):
                demo.inspect_trace(path)


if __name__ == "__main__":
    unittest.main()
