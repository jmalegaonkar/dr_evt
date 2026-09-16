################################################################################
#         Copyright 2023 Lawrence Livermore National Security, LLC             #
#         See the top-level LICENSE file for details.                          #
#                                                                              #
#         SPDX-License-Identifier: MIT                                         #
################################################################################

"""Tests for building and submitting market clearing windows."""

from pathlib import Path
import tempfile
import unittest

from dr_evt_market import InProcessPlatform, PlatformSnapshot
from dr_evt_market.mechanisms import (
    Decision,
    JobBid,
    LegBid,
    LegSpec,
    QueuedJob,
    build_observation,
    submit_decisions,
    validate_decisions,
)


def _snapshot(name: str, total_nodes: int, free_nodes: int) -> PlatformSnapshot:
    return PlatformSnapshot(
        name=name,
        time_s=60,
        total_nodes=total_nodes,
        free_nodes=free_nodes,
        in_use_nodes=total_nodes - free_nodes,
        waiting_jobs=0,
        current_utilization=(total_nodes - free_nodes) / total_nodes,
    )


class ClearingTests(unittest.TestCase):
    """Exercise deterministic placement construction and live submission."""

    def test_builds_feasible_candidates_and_costs(self) -> None:
        """One-leg and composite placements match hand-computed values."""
        queue = (
            QueuedJob("single", 0, (LegSpec("0", 20, 3600),)),
            QueuedJob(
                "composite",
                0,
                (
                    LegSpec("cpu", 10, 3600),
                    LegSpec("gpu", 20, 1800),
                ),
            ),
        )
        bids = {
            "single": JobBid.single(
                "single",
                {"alpha": 100.0, "beta": 100.0},
            ),
            "composite": JobBid(
                "composite",
                (
                    LegBid("cpu", {"alpha": 30.0, "beta": 40.0}),
                    LegBid("gpu", {"alpha": 50.0, "beta": 60.0}),
                ),
            ),
        }
        snapshots = {
            "beta": _snapshot("beta", 60, 15),
            "alpha": _snapshot("alpha", 100, 50),
        }

        observation = build_observation(
            60,
            2,
            9,
            queue,
            bids,
            snapshots,
            {"alpha": 2.0, "beta": 4.0},
        )

        single = observation.job("single")
        self.assertEqual(
            [candidate.placement_id for candidate in single.candidates],
            ["alpha"],
        )
        self.assertEqual(
            dict(single.candidates[0].demand_by_platform),
            {"alpha": 20},
        )
        self.assertEqual(single.candidates[0].resource_cost_credits, 40.0)

        composite = observation.job("composite")
        self.assertEqual(
            [candidate.placement_id for candidate in composite.candidates],
            ["alpha+alpha", "beta+alpha"],
        )
        self.assertEqual(
            dict(composite.candidates[0].demand_by_platform),
            {"alpha": 30},
        )
        self.assertEqual(
            dict(composite.candidates[1].demand_by_platform),
            {"alpha": 20, "beta": 10},
        )
        self.assertEqual(
            [item.resource_cost_credits for item in composite.candidates],
            [40.0, 60.0],
        )

    def test_candidate_limit_records_truncated_jobs(self) -> None:
        """Candidate truncation is deterministic and keeps blocked jobs."""
        snapshots = {
            name: _snapshot(name, 10, 10)
            for name in ("gamma", "alpha", "beta")
        }
        wide = QueuedJob(
            "wide",
            0,
            (LegSpec("first", 1, 1), LegSpec("second", 1, 1)),
        )
        blocked = QueuedJob("blocked", 0, (LegSpec("0", 11, 1),))
        bids = {
            "wide": JobBid(
                "wide",
                (
                    LegBid("first", {name: 1.0 for name in snapshots}),
                    LegBid("second", {name: 1.0 for name in snapshots}),
                ),
            ),
            "blocked": JobBid.single(
                "blocked",
                {name: 1.0 for name in snapshots},
            ),
        }

        observation = build_observation(
            60,
            1,
            3,
            (wide, blocked),
            bids,
            snapshots,
            {name: 1.0 for name in snapshots},
            max_candidates=2,
        )

        self.assertEqual(
            [
                candidate.placement_id
                for candidate in observation.job("wide").candidates
            ],
            ["alpha+alpha", "alpha+beta"],
        )
        self.assertEqual(observation.truncated_jobs, ("wide",))
        self.assertEqual(observation.job("blocked").candidates, ())

    def test_live_submission_starts_every_leg_at_window_time(self) -> None:
        """Two live platforms start every feasible submitted leg immediately."""
        with tempfile.TemporaryDirectory(
            prefix="dr_evt_market_clearing_"
        ) as directory:
            root = Path(directory)
            platforms = {
                "alpha": InProcessPlatform("alpha", 100, root / "alpha"),
                "beta": InProcessPlatform("beta", 50, root / "beta"),
            }
            for platform in platforms.values():
                platform.advance_to(60)
            snapshots = {
                name: platform.snapshot()
                for name, platform in platforms.items()
            }
            queue = (
                QueuedJob("single", 5, (LegSpec("0", 60, 30),)),
                QueuedJob(
                    "composite",
                    30,
                    (
                        LegSpec("left", 20, 40),
                        LegSpec("right", 30, 50),
                    ),
                ),
            )
            bids = {
                "single": JobBid.single(
                    "single",
                    {"alpha": 100.0, "beta": 80.0},
                ),
                "composite": JobBid(
                    "composite",
                    (
                        LegBid("left", {"alpha": 60.0, "beta": 70.0}),
                        LegBid("right", {"alpha": 80.0, "beta": 90.0}),
                    ),
                ),
            }
            observation = build_observation(
                60,
                1,
                5,
                queue,
                bids,
                snapshots,
                {"alpha": 1.0, "beta": 2.0},
            )
            decisions = [
                Decision("single", "alpha", 100.0),
                Decision("composite", "beta+alpha", 150.0),
            ]
            accepted, rejected = validate_decisions(observation, decisions)
            self.assertEqual(rejected, [])

            handles = submit_decisions(
                accepted,
                observation,
                platforms,
                60,
            )
            self.assertEqual(
                list(handles),
                [
                    ("single", "0"),
                    ("composite", "left"),
                    ("composite", "right"),
                ],
            )
            for platform in platforms.values():
                platform.advance_to(60)

            routed_platforms = {
                ("single", "0"): "alpha",
                ("composite", "left"): "beta",
                ("composite", "right"): "alpha",
            }
            for key, handle in handles.items():
                timing = platforms[routed_platforms[key]].timings([handle])[0]
                self.assertEqual(timing.begin_s, 60.0)

            for platform in platforms.values():
                platform.finish()


if __name__ == "__main__":
    unittest.main()
