################################################################################
#         Copyright 2023 Lawrence Livermore National Security, LLC             #
#         See the top-level LICENSE file for details.                          #
#                                                                              #
#         SPDX-License-Identifier: MIT                                         #
################################################################################

"""Tests for market mechanism types and decision validation."""

import unittest

from dr_evt_market.mechanisms import (
    Decision,
    JobBid,
    JobOffer,
    LegBid,
    LegSpec,
    MarketObservation,
    Mechanism,
    Placement,
    validate_decisions,
)


def _offer(job_id: str, nodes: int) -> JobOffer:
    leg = LegSpec("0", nodes, 60)
    placement = Placement("alpha", {"0": "alpha"}, {"alpha": nodes}, 2.0)
    return JobOffer(job_id, 0, (leg,), (placement,))


class EmptyMechanism(Mechanism):
    """Return no decisions for any observation."""

    name = "empty"

    def decide(self, obs: MarketObservation) -> list[Decision]:
        """Return an empty decision list."""
        del obs
        return []


class MechanismBaseTests(unittest.TestCase):
    """Exercise mechanism value types and ordered decision validation."""

    def test_construction_errors(self) -> None:
        """Invalid values, empty jobs, and bad placement IDs are rejected."""
        with self.assertRaisesRegex(ValueError, "non-negative"):
            LegBid("0", {"alpha": -1.0})
        with self.assertRaisesRegex(ValueError, "at least one leg"):
            JobBid("empty", ())
        with self.assertRaisesRegex(ValueError, "at least one leg"):
            JobOffer("empty", 0, (), ())
        with self.assertRaisesRegex(ValueError, "placement_id"):
            Placement("wrong", {"0": "alpha"}, {"alpha": 1}, 0.0)

    def test_value_of_single_and_composite_jobs(self) -> None:
        """Job values add by leg and reject an unacceptable assignment."""
        single = JobBid.single("single", {"beta": 3.0, "alpha": 5.0})
        single_placement = Placement(
            "alpha",
            {"0": "alpha"},
            {"alpha": 2},
            1.0,
        )
        self.assertEqual(single.value_of(single_placement), 5.0)
        self.assertEqual(
            hash(single.legs[0]),
            hash(LegBid("0", {"alpha": 5.0, "beta": 3.0})),
        )
        with self.assertRaises(TypeError):
            single.legs[0].value_by_platform["alpha"] = 1.0

        composite = JobBid(
            "composite",
            (
                LegBid("cpu", {"alpha": 7.0}),
                LegBid("gpu", {"gamma": 11.0}),
            ),
        )
        accepted = Placement(
            "alpha+gamma",
            {"cpu": "alpha", "gpu": "gamma"},
            {"alpha": 2, "gamma": 1},
            4.0,
        )
        unacceptable = Placement(
            "alpha+beta",
            {"cpu": "alpha", "gpu": "beta"},
            {"alpha": 2, "beta": 1},
            4.0,
        )
        self.assertEqual(composite.value_of(accepted), 18.0)
        self.assertIsNone(composite.value_of(unacceptable))

    def test_mechanism_is_abstract_and_subclass_is_concrete(self) -> None:
        """The base is abstract while a minimal implementation is usable."""
        with self.assertRaises(TypeError):
            Mechanism()
        mechanism = EmptyMechanism()
        self.assertEqual(mechanism.name, "empty")
        self.assertEqual(mechanism.decide(_observation()), [])
        self.assertEqual(
            mechanism.charge_bounds(_observation(), "accepted", "alpha"),
            (2.0, 10.0),
        )

    def test_validation_reports_every_reason_in_decision_order(self) -> None:
        """Invalid decisions keep stable reasons and consume no capacity."""
        observation = _observation()
        first = Decision("accepted", "alpha", 5.0)
        decisions = [
            first,
            Decision("accepted", "alpha", 5.0),
            Decision("missing", "alpha", 5.0),
            Decision("bad-placement", "missing", 5.0),
            Decision("above", "alpha", 11.0),
            Decision("below", "alpha", 1.0),
            Decision("capacity", "alpha", 5.0),
        ]

        accepted, rejected = validate_decisions(observation, decisions)

        self.assertEqual(accepted, [first])
        self.assertEqual(
            [item.reason for item in rejected],
            [
                "duplicate_job",
                "unknown_job",
                "unknown_placement",
                "charge_above_value",
                "charge_below_cost",
                "over_capacity",
            ],
        )


def _observation() -> MarketObservation:
    job_ids_and_nodes = (
        ("accepted", 4),
        ("bad-placement", 1),
        ("above", 1),
        ("below", 1),
        ("capacity", 7),
    )
    jobs = tuple(_offer(job_id, nodes) for job_id, nodes in job_ids_and_nodes)
    bids = {
        job_id: JobBid.single(job_id, {"alpha": 10.0})
        for job_id, _ in job_ids_and_nodes
    }
    return MarketObservation(0, 0, 7, jobs, bids, {"alpha": 10})


if __name__ == "__main__":
    unittest.main()
