################################################################################
#         Copyright 2023 Lawrence Livermore National Security, LLC             #
#         See the top-level LICENSE file for details.                          #
#                                                                              #
#         SPDX-License-Identifier: MIT                                         #
################################################################################

"""Tests for the market's data model and decision validation."""

import unittest

from dr_evt_market.mechanisms import (
    Decision,
    Job,
    Leg,
    Mechanism,
    Placement,
    Platform,
    Window,
    demand,
    validate_decisions,
)

PLATFORMS = {
    "alpha": Platform("alpha", 10, 1.0, {"cpu"}),
    "beta": Platform("beta", 10, 2.0, {"cpu", "gpu"}),
}


def _job(job_id: str, nodes: int, bid=2.0, requires=()) -> Job:
    return Job(job_id, 0, (Leg("0", nodes, 3600, frozenset(requires)),), bid)


def _window(free: dict, jobs: list[Job], candidates: dict) -> Window:
    return Window(0, 0, PLATFORMS, free, tuple(jobs), candidates)


class DataModelTests(unittest.TestCase):
    """Check construction rules, feasibility and the value model."""

    def test_construction_errors(self) -> None:
        """Bad names, sizes, bids and empty jobs are refused."""
        with self.assertRaises(ValueError):
            Platform("", 10, 1.0)
        with self.assertRaises(ValueError):
            Leg("0", 0, 60)
        with self.assertRaises(ValueError):
            Job("j", 0, (), 2.0)
        with self.assertRaises(ValueError):
            Job("j", 0, (Leg("0", 1, 60),), -1.0)
        with self.assertRaises(ValueError):
            Job("j", 0, (Leg("0", 1, 60),), {})
        with self.assertRaises(ValueError):
            Placement((), 1.0, 2.0)

    def test_platform_cost_and_leg_fit(self) -> None:
        """Cost is price times nodes times hours; fit needs hardware and nodes."""
        self.assertAlmostEqual(PLATFORMS["beta"].cost(5, 1800), 5.0)
        self.assertTrue(Leg("0", 4, 60, {"gpu"}).fits(PLATFORMS["beta"]))
        self.assertFalse(Leg("0", 4, 60, {"gpu"}).fits(PLATFORMS["alpha"]))
        self.assertFalse(Leg("0", 11, 60).fits(PLATFORMS["alpha"]))

    def test_multiplier_forms(self) -> None:
        """A single bid applies everywhere; a mapping only where it names."""
        single = _job("s", 1, 1.5)
        mapped = _job("m", 1, {"beta": 2.0})
        self.assertEqual(single.multiplier("alpha"), 1.5)
        self.assertEqual(mapped.multiplier("beta"), 2.0)
        self.assertIsNone(mapped.multiplier("alpha"))

    def test_placement_and_demand(self) -> None:
        """The id joins platforms in leg order and demand sums shared legs."""
        job = Job("c", 0, (Leg("a", 3, 60), Leg("b", 4, 60)), 2.0)
        placement = Placement(("alpha", "alpha"), 2.0, 5.0)
        self.assertEqual(placement.id, "alpha+alpha")
        self.assertEqual(demand(job, placement), {"alpha": 7})
        self.assertAlmostEqual(placement.net_credits, 3.0)

    def test_mechanism_is_abstract(self) -> None:
        """The base cannot be instantiated; a subclass with decide can."""
        with self.assertRaises(TypeError):
            Mechanism()

        class Empty(Mechanism):
            name = "empty"

            def decide(self, window):
                return []

        self.assertEqual(Empty().decide(_window({"alpha": 1, "beta": 1}, [], {})), [])


class ValidationTests(unittest.TestCase):
    """Every rejection reason, in decision order, without capacity leaks."""

    def test_every_reason_in_order(self) -> None:
        """Duplicates, unknowns, bad charges and over-capacity are named."""
        good = Placement(("alpha",), 1.0, 4.0)
        big = Placement(("alpha",), 1.0, 4.0)
        jobs = [_job("a", 4), _job("b", 1), _job("c", 7)]
        window = _window(
            {"alpha": 10, "beta": 10},
            jobs,
            {"a": (good,), "b": (good,), "c": (big,)},
        )
        first = Decision("a", good, 2.0)
        decisions = [
            first,
            Decision("a", good, 2.0),
            Decision("x", good, 2.0),
            Decision("b", Placement(("beta",), 1.0, 4.0), 2.0),
            Decision("c", big, 5.0),
        ]
        accepted, rejected = validate_decisions(window, decisions)
        self.assertEqual(accepted, [first])
        self.assertEqual(
            [r.reason for r in rejected],
            ["duplicate_job", "unknown_job", "unknown_placement", "charge_above_value"],
        )

        below = validate_decisions(window, [Decision("b", good, 0.5)])[1]
        self.assertEqual([r.reason for r in below], ["charge_below_cost"])
        over = validate_decisions(window, [first, Decision("c", big, 2.0)])[1]
        self.assertEqual([r.reason for r in over], ["over_capacity"])


if __name__ == "__main__":
    unittest.main()
