################################################################################
#         Copyright 2023 Lawrence Livermore National Security, LLC             #
#         See the top-level LICENSE file for details.                          #
#                                                                              #
#         SPDX-License-Identifier: MIT                                         #
################################################################################

"""Tests for placements, windows and submission to live platforms."""

from pathlib import Path
import tempfile
import unittest

from dr_evt_market import InProcessPlatform
from dr_evt_market.mechanisms import (
    Decision,
    Job,
    Leg,
    Platform,
    base_cost,
    build_window,
    placements,
    submit,
    validate_decisions,
)

PLATFORMS = {
    "alpha": Platform("alpha", 100, 1.0, {"cpu"}),
    "beta": Platform("beta", 60, 2.0, {"cpu"}),
    "gamma": Platform("gamma", 40, 3.0, {"cpu", "gpu"}),
}
FULL = {name: platform.total_nodes for name, platform in PLATFORMS.items()}


class PlacementTests(unittest.TestCase):
    """Feasibility, cost and value of placements."""

    def test_single_bid_values_the_base_cost_everywhere(self) -> None:
        """Value is the multiplier times the cheapest usable cost."""
        job = Job("s", 0, (Leg("0", 20, 3600),), 1.5)
        self.assertAlmostEqual(base_cost(job, PLATFORMS), 20.0)
        found = {
            p.id: (p.cost_credits, p.value_credits)
            for p in placements(job, PLATFORMS, FULL)
        }
        self.assertEqual(found, {"alpha": (20.0, 30.0)})  # beta costs 40, gamma 60

    def test_hardware_and_per_platform_bids_restrict_placements(self) -> None:
        """A gpu leg goes only to gamma; a mapped bid only where it names."""
        gpu = Job("g", 0, (Leg("0", 10, 3600, {"gpu"}),), 2.0)
        self.assertEqual([p.id for p in placements(gpu, PLATFORMS, FULL)], ["gamma"])
        mapped = Job("m", 0, (Leg("0", 10, 3600),), {"beta": 1.5, "gamma": 1.1})
        found = {p.id: p.value_credits for p in placements(mapped, PLATFORMS, FULL)}
        self.assertEqual(found, {"beta": 30.0, "gamma": 33.0})

    def test_composite_legs_share_capacity(self) -> None:
        """Legs on one platform must fit together; costs add up."""
        job = Job("c", 0, (Leg("a", 30, 3600), Leg("b", 30, 3600)), 2.0)
        found = {p.id: p for p in placements(job, PLATFORMS, FULL)}
        self.assertIn("alpha+alpha", found)
        self.assertIn(
            "beta+beta", found
        )  # 60 nodes fit beta; cost 120 equals value 120
        self.assertNotIn("gamma+gamma", found)  # 60 nodes do not fit gamma's 40
        self.assertEqual(
            (found["alpha+alpha"].cost_credits, found["alpha+alpha"].value_credits),
            (60.0, 120.0),
        )

    def test_free_nodes_narrow_the_window(self) -> None:
        """A window only offers placements that fit in the free nodes now."""
        job = Job("s", 0, (Leg("0", 50, 3600),), 3.0)
        window = build_window(
            0, 0, [job], PLATFORMS, {"alpha": 40, "beta": 60, "gamma": 40}
        )
        self.assertEqual([p.id for p in window.candidates["s"]], ["beta"])


class SubmissionTests(unittest.TestCase):
    """One window through live platforms, every leg starting at the window."""

    def test_live_submission_starts_every_leg_at_window_time(self) -> None:
        """Winners handed to their platforms begin at once."""
        with tempfile.TemporaryDirectory(prefix="dr_evt_market_clearing_") as directory:
            root = Path(directory)
            sessions = {
                name: InProcessPlatform(name, platform.total_nodes, root / name)
                for name, platform in PLATFORMS.items()
            }
            for session in sessions.values():
                session.advance_to(60)
            jobs = [
                Job("single", 5, (Leg("0", 60, 30),), 3.0),
                Job(
                    "composite",
                    30,
                    (Leg("left", 20, 40), Leg("right", 30, 50, {"gpu"})),
                    3.0,
                ),
            ]
            free = {
                name: session.snapshot().free_nodes
                for name, session in sessions.items()
            }
            window = build_window(60, 1, jobs, PLATFORMS, free)
            decisions = [
                Decision(
                    "single",
                    single := next(
                        p for p in window.candidates["single"] if p.id == "alpha"
                    ),
                    single.cost_credits,
                ),
                Decision(
                    "composite",
                    pair := next(
                        p
                        for p in window.candidates["composite"]
                        if p.id == "beta+gamma"
                    ),
                    pair.cost_credits,
                ),
            ]
            accepted, rejected = validate_decisions(window, decisions)
            self.assertEqual(rejected, [])
            handles = submit(accepted, window, sessions, 60)
            self.assertEqual(
                list(handles),
                [("single", "0"), ("composite", "left"), ("composite", "right")],
            )
            for session in sessions.values():
                session.advance_to(60)
            where = {
                ("single", "0"): "alpha",
                ("composite", "left"): "beta",
                ("composite", "right"): "gamma",
            }
            for key, handle in handles.items():
                self.assertEqual(
                    sessions[where[key]].timings([handle])[0].begin_s, 60.0
                )
            for session in sessions.values():
                session.finish()


if __name__ == "__main__":
    unittest.main()
