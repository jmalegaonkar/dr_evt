################################################################################
#         Copyright 2023 Lawrence Livermore National Security, LLC             #
#         See the top-level LICENSE file for details.                          #
#                                                                              #
#         SPDX-License-Identifier: MIT                                         #
################################################################################

"""Tests for start-time prediction over platform snapshots."""

from pathlib import Path
from statistics import median
import tempfile
import unittest

from dr_evt_market import (
    InProcessPlatform,
    PlatformSnapshot,
    Prediction,
    ResourceRelease,
    predict_start,
)
from dr_evt_market.tests.fixtures import CONTENDED_JOBS, EXPECTED_BEGIN_TIMES


def _snapshot(
    *,
    time_s: int = 10,
    free_nodes: int = 5,
    shadow_time_s: float = -1.0,
    releases: tuple[ResourceRelease, ...] = (),
) -> PlatformSnapshot:
    return PlatformSnapshot(
        name="test",
        time_s=time_s,
        total_nodes=20,
        free_nodes=free_nodes,
        in_use_nodes=20 - free_nodes,
        waiting_jobs=0 if shadow_time_s < 0 else 1,
        shadow_time_s=shadow_time_s,
        releases=releases,
    )


class PredictionRuleTests(unittest.TestCase):
    """Exercise every prediction source with hand-built snapshots."""

    def test_free_now(self) -> None:
        """An idle queue with enough capacity starts immediately."""
        prediction = predict_start(_snapshot(free_nodes=8), 8, 20, ())
        self.assertEqual(prediction, Prediction(10, "free_now"))

    def test_after_reported_release(self) -> None:
        """Reported release events accumulate until demand fits."""
        snapshot = _snapshot(
            free_nodes=2,
            releases=(
                ResourceRelease(20.0, 3),
                ResourceRelease(30.0, 2),
            ),
        )
        prediction = predict_start(snapshot, 7, 20, ((15.0, 20),))
        self.assertEqual(prediction, Prediction(30, "after_release"))

    def test_after_release_uses_ledger_fallback(self) -> None:
        """An empty release list falls back to future projected ledger ends."""
        prediction = predict_start(
            _snapshot(free_nodes=2),
            9,
            20,
            ((5.0, 20), (40.0, 3), (25.0, 4)),
        )
        self.assertEqual(prediction, Prediction(40, "after_release"))

    def test_unknown(self) -> None:
        """Insufficient known future capacity has no predicted start."""
        prediction = predict_start(
            _snapshot(free_nodes=2),
            10,
            20,
            ((20.0, 3),),
        )
        self.assertEqual(prediction, Prediction(None, "unknown"))

    def test_backfill(self) -> None:
        """A fitting job ending strictly before the head can backfill."""
        prediction = predict_start(
            _snapshot(time_s=20, free_nodes=5, shadow_time_s=100.0),
            5,
            79,
            (),
        )
        self.assertEqual(prediction, Prediction(20, "backfill"))

    def test_strict_backfill_boundary_is_behind_head(self) -> None:
        """Ending exactly at the reservation is not a safe backfill."""
        prediction = predict_start(
            _snapshot(time_s=20, free_nodes=5, shadow_time_s=100.0),
            5,
            80,
            (),
        )
        self.assertEqual(prediction, Prediction(100, "behind_head"))

    def test_waiting_head_can_start_after_release(self) -> None:
        """A release before the shadow permits a job that can finish in time."""
        snapshot = _snapshot(
            time_s=10,
            free_nodes=10,
            shadow_time_s=100.0,
            releases=(
                ResourceRelease(20.0, 30),
                ResourceRelease(100.0, 60),
            ),
        )
        prediction = predict_start(snapshot, 40, 30, ())
        self.assertEqual(prediction, Prediction(20, "after_release"))

    def test_waiting_head_release_must_finish_before_shadow(self) -> None:
        """A release start that cannot finish before the shadow stays behind."""
        snapshot = _snapshot(
            time_s=10,
            free_nodes=10,
            shadow_time_s=100.0,
            releases=(
                ResourceRelease(20.0, 30),
                ResourceRelease(100.0, 60),
            ),
        )
        prediction = predict_start(snapshot, 40, 90, ())
        self.assertEqual(prediction, Prediction(100, "behind_head"))

    def test_behind_head_when_capacity_is_not_free(self) -> None:
        """A blocked job receives the head reservation as a lower bound."""
        prediction = predict_start(
            _snapshot(time_s=20, free_nodes=4, shadow_time_s=100.0),
            8,
            10,
            (),
        )
        self.assertEqual(prediction, Prediction(100, "behind_head"))


class PredictionIntegrationTests(unittest.TestCase):
    """Compare fixture predictions with observed in-process start times."""

    def test_contended_stream_predictions_bound_observed_starts(self) -> None:
        """Immediate starts are exact and other predictions bound this stream."""
        with tempfile.TemporaryDirectory(
            prefix="dr_evt_market_prediction_"
        ) as directory:
            platform = InProcessPlatform(
                "prediction",
                100,
                Path(directory),
            )
            handles: list[int] = []
            predictions: list[Prediction] = []
            projected: list[tuple[Prediction, int, int]] = []

            for job in CONTENDED_JOBS:
                platform.advance_to(job.submit_s)
                ledger_ends = [
                    (prediction.start_s + limit_s, num_nodes)
                    for prediction, limit_s, num_nodes in projected
                    if prediction.start_s is not None
                ]
                prediction = predict_start(
                    platform.snapshot(),
                    job.num_nodes,
                    job.limit_s,
                    ledger_ends,
                )
                predictions.append(prediction)
                projected.append((prediction, job.limit_s, job.num_nodes))
                handles.extend(platform.submit([job]))
                platform.advance_to(job.submit_s)

            platform.advance_to(1_000)
            observed = [
                timing.begin_s for timing in platform.timings(handles)
            ]

        self.assertEqual(
            observed,
            [float(value) for value in EXPECTED_BEGIN_TIMES],
        )
        errors = []
        for prediction, observed_start in zip(predictions, observed):
            self.assertIsNotNone(prediction.start_s)
            predicted_start = prediction.start_s
            if predicted_start is None:
                continue
            if prediction.source == "free_now":
                self.assertEqual(predicted_start, observed_start)
            else:
                self.assertLessEqual(predicted_start, observed_start)
            errors.append(int(observed_start - predicted_start))

        print(
            "prediction errors (s): "
            f"values={errors}, min={min(errors)}, "
            f"median={median(errors):g}, max={max(errors)}"
        )


if __name__ == "__main__":
    unittest.main()
