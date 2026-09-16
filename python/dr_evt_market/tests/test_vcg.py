################################################################################
#         Copyright 2023 Lawrence Livermore National Security, LLC             #
#         See the top-level LICENSE file for details.                          #
#                                                                              #
#         SPDX-License-Identifier: MIT                                         #
################################################################################

"""Tests for exact and fallback VCG market clearing."""

import random
import unittest

from dr_evt_market import PlatformSnapshot
from dr_evt_market.mechanisms import (
    JobBid,
    LegBid,
    LegSpec,
    MarketObservation,
    QueuedJob,
    Vcg,
    build_observation,
)


def _observation(
    jobs: tuple[QueuedJob, ...],
    bids: dict[str, JobBid],
    capacities: dict[str, int],
    prices: dict[str, float],
) -> MarketObservation:
    snapshots = {
        name: PlatformSnapshot(
            name=name,
            time_s=0,
            total_nodes=nodes,
            free_nodes=nodes,
            in_use_nodes=0,
            waiting_jobs=0,
            current_utilization=0.0,
        )
        for name, nodes in capacities.items()
    }
    return build_observation(0, 0, 17, jobs, bids, snapshots, prices)


def _random_observation() -> MarketObservation:
    generator = random.Random(20260916)
    platforms = ("alpha", "beta", "gamma")
    prices = {"alpha": 1.0, "beta": 2.0, "gamma": 3.0}
    capacities = {name: 1 for name in platforms}
    jobs = []
    bids = {}
    for index in range(10):
        job_id = f"job-{index}"
        jobs.append(QueuedJob(job_id, 0, (LegSpec("0", 1, 3600),)))
        bids[job_id] = JobBid.single(
            job_id,
            {
                platform: prices[platform] + generator.uniform(1.0, 100.0)
                for platform in platforms
            },
        )
    return _observation(tuple(jobs), bids, capacities, prices)


def _brute_force(obs: MarketObservation) -> dict[str, str]:
    best_welfare = -1.0
    best_placements: dict[str, str] = {}

    def search(
        job_index: int,
        remaining: dict[str, int],
        welfare: float,
        placements: dict[str, str],
    ) -> None:
        nonlocal best_welfare, best_placements
        if job_index == len(obs.jobs):
            if welfare > best_welfare:
                best_welfare = welfare
                best_placements = dict(placements)
            return

        offer = obs.jobs[job_index]
        search(job_index + 1, remaining, welfare, placements)
        for candidate in offer.candidates:
            if any(
                nodes > remaining.get(platform, 0)
                for platform, nodes in candidate.demand_by_platform.items()
            ):
                continue
            net_value = obs.net_value(offer.job_id, candidate.placement_id)
            if net_value is None or net_value < 0.0:
                continue
            next_remaining = dict(remaining)
            for platform, nodes in candidate.demand_by_platform.items():
                next_remaining[platform] -= nodes
            placements[offer.job_id] = candidate.placement_id
            search(
                job_index + 1,
                next_remaining,
                welfare + net_value,
                placements,
            )
            del placements[offer.job_id]

    search(0, dict(obs.free_nodes), 0.0, {})
    return best_placements


def _scaled_observation(
    obs: MarketObservation,
    job_id: str,
    scale: float,
    platform_only: str | None,
) -> MarketObservation:
    bids = dict(obs.bids)
    scaled_legs = []
    for leg in bids[job_id].legs:
        values = {
            platform: (
                value * scale
                if platform_only is None or platform == platform_only
                else value
            )
            for platform, value in leg.value_by_platform.items()
        }
        scaled_legs.append(LegBid(leg.leg_id, values))
    bids[job_id] = JobBid(job_id, tuple(scaled_legs))
    return MarketObservation(
        obs.time_s,
        obs.window_index,
        obs.seed,
        obs.jobs,
        bids,
        obs.free_nodes,
        obs.truncated_jobs,
    )


def _true_utility(
    obs: MarketObservation,
    decisions: list,
    job_id: str,
) -> float:
    decision = next(
        (item for item in decisions if item.job_id == job_id),
        None,
    )
    if decision is None:
        return 0.0
    value = obs.value(job_id, decision.placement_id)
    if value is None:
        raise AssertionError("misreport selected an unacceptable placement")
    return value - decision.charge_credits


class VcgTests(unittest.TestCase):
    """Check exact VCG allocation, payments, truthfulness, and fallback."""

    def test_contended_winner_pays_loser_externality(self) -> None:
        """A sole winner pays resource cost plus the losing net value."""
        jobs = tuple(
            QueuedJob(job_id, 0, (LegSpec("0", 60, 60),))
            for job_id in ("first", "second")
        )
        bids = {
            "first": JobBid.single("first", {"alpha": 12.0}),
            "second": JobBid.single("second", {"alpha": 9.0}),
        }
        observation = _observation(
            jobs,
            bids,
            {"alpha": 100},
            {"alpha": 2.0},
        )

        decisions = Vcg().decide(observation)

        self.assertEqual(len(decisions), 1)
        self.assertEqual(decisions[0].job_id, "first")
        self.assertEqual(decisions[0].placement_id, "alpha")
        self.assertAlmostEqual(decisions[0].charge_credits, 9.0)
        self.assertAlmostEqual(decisions[0].score or 0.0, 10.0)

    def test_uncontended_jobs_pay_resource_cost(self) -> None:
        """Jobs that both fit impose no externality beyond resource cost."""
        jobs = tuple(
            QueuedJob(job_id, 0, (LegSpec("0", 40, 90),))
            for job_id in ("first", "second")
        )
        bids = {
            "first": JobBid.single("first", {"alpha": 12.0}),
            "second": JobBid.single("second", {"alpha": 9.0}),
        }
        observation = _observation(
            jobs,
            bids,
            {"alpha": 100},
            {"alpha": 2.0},
        )

        decisions = Vcg().decide(observation)

        self.assertEqual(
            {item.job_id: item.charge_credits for item in decisions},
            {"first": 2.0, "second": 2.0},
        )

    def test_composite_beats_two_lower_value_single_jobs(self) -> None:
        """A higher-value composite wins both platforms as one job."""
        jobs = (
            QueuedJob(
                "composite",
                0,
                (LegSpec("left", 5, 60), LegSpec("right", 5, 60)),
            ),
            QueuedJob("alpha-job", 0, (LegSpec("0", 5, 60),)),
            QueuedJob("beta-job", 0, (LegSpec("0", 5, 60),)),
        )
        bids = {
            "composite": JobBid(
                "composite",
                (
                    LegBid("left", {"alpha": 10.0}),
                    LegBid("right", {"beta": 8.0}),
                ),
            ),
            "alpha-job": JobBid.single("alpha-job", {"alpha": 8.0}),
            "beta-job": JobBid.single("beta-job", {"beta": 9.0}),
        }
        observation = _observation(
            jobs,
            bids,
            {"alpha": 5, "beta": 5},
            {"alpha": 0.0, "beta": 0.0},
        )

        decisions = Vcg().decide(observation)

        self.assertEqual(
            [(item.job_id, item.placement_id) for item in decisions],
            [("composite", "alpha+beta")],
        )

    def test_exact_allocation_matches_brute_force(self) -> None:
        """A seeded ten-job window matches exhaustive allocation."""
        observation = _random_observation()

        decisions = Vcg().decide(observation)
        selected = {
            decision.job_id: decision.placement_id
            for decision in decisions
        }

        self.assertEqual(selected, _brute_force(observation))
        for decision in decisions:
            cost = observation.candidate(
                decision.job_id,
                decision.placement_id,
            ).resource_cost_credits
            value = observation.value(
                decision.job_id,
                decision.placement_id,
            )
            self.assertIsNotNone(value)
            self.assertGreaterEqual(decision.charge_credits, cost)
            self.assertLessEqual(decision.charge_credits, value or 0.0)

    def test_seeded_window_has_no_profitable_misreport(self) -> None:
        """Whole-vector and per-platform scaling cannot improve true utility."""
        observation = _random_observation()
        mechanism = Vcg()
        truthful = mechanism.decide(observation)
        scales = (0.5, 0.8, 1.2, 2.0)

        for offer in observation.jobs:
            truthful_utility = _true_utility(
                observation,
                truthful,
                offer.job_id,
            )
            variants = (None, *sorted(observation.free_nodes))
            for scale in scales:
                for platform in variants:
                    with self.subTest(
                        job=offer.job_id,
                        scale=scale,
                        platform=platform,
                    ):
                        changed = _scaled_observation(
                            observation,
                            offer.job_id,
                            scale,
                            platform,
                        )
                        utility = _true_utility(
                            observation,
                            mechanism.decide(changed),
                            offer.job_id,
                        )
                        self.assertGreaterEqual(
                            truthful_utility + 1.0e-9,
                            utility,
                        )

    def test_large_window_uses_greedy_critical_payment(self) -> None:
        """More than 800 variables use density and bisection payments."""
        jobs = tuple(
            QueuedJob(f"job-{index:04d}", 0, (LegSpec("0", 1, 1),))
            for index in range(801)
        )
        bids = {
            job.job_id: JobBid.single(
                job.job_id,
                {"alpha": float(1_000 - index)},
            )
            for index, job in enumerate(jobs)
        }
        observation = _observation(
            jobs,
            bids,
            {"alpha": 1},
            {"alpha": 0.0},
        )

        decisions = Vcg().decide(observation)

        self.assertEqual(len(decisions), 1)
        self.assertEqual(decisions[0].job_id, "job-0000")
        self.assertAlmostEqual(decisions[0].charge_credits, 999.0, places=6)


if __name__ == "__main__":
    unittest.main()
