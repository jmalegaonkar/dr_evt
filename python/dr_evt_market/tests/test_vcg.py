################################################################################
#         Copyright 2023 Lawrence Livermore National Security, LLC             #
#         See the top-level LICENSE file for details.                          #
#                                                                              #
#         SPDX-License-Identifier: MIT                                         #
################################################################################

"""Tests for VCG: allocation, pivots, truthfulness, agreement with brute force."""

import random
import unittest

from dr_evt_market.mechanisms import Job, Leg, Platform, Vcg, build_window, demand


def _window(platforms: dict, jobs: list[Job]):
    return build_window(
        0, 0, jobs, platforms, {n: p.total_nodes for n, p in platforms.items()}
    )


def _brute_force(window) -> tuple[float, dict]:
    best = (-1.0, {})

    def search(index, remaining, welfare, chosen):
        nonlocal best
        if index == len(window.jobs):
            if welfare > best[0] + 1e-12:
                best = (welfare, dict(chosen))
            return
        job = window.jobs[index]
        search(index + 1, remaining, welfare, chosen)
        for placement in window.candidates[job.job_id]:
            nodes = demand(job, placement)
            if placement.net_credits < 0 or any(
                c > remaining[n] for n, c in nodes.items()
            ):
                continue
            after = dict(remaining)
            for n, c in nodes.items():
                after[n] -= c
            chosen[job.job_id] = placement
            search(index + 1, after, welfare + placement.net_credits, chosen)
            del chosen[job.job_id]

    search(0, dict(window.free_nodes), 0.0, {})
    return best


def _utility(window, decisions, job_id) -> float:
    for decision in decisions:
        if decision.job_id == job_id:
            return decision.placement.value_credits - decision.charge_credits
    return 0.0


def _random_jobs(seed: int, count: int = 6) -> list[Job]:
    rng = random.Random(seed)
    jobs = []
    for index in range(count):
        legs = (
            Leg(
                "0",
                rng.randint(5, 40),
                rng.randint(600, 3600),
                frozenset({"gpu"}) if rng.random() < 0.2 else frozenset(),
            ),
        )
        if rng.random() < 0.3:
            legs = legs + (Leg("1", rng.randint(5, 25), rng.randint(600, 3600)),)
        bid = (
            rng.uniform(1.2, 4.0)
            if rng.random() < 0.5
            else {n: rng.uniform(1.2, 4.0) for n in ("alpha", "beta", "gamma")}
        )
        jobs.append(Job(f"j{index}", 0, legs, bid))
    return jobs


PLATFORMS = {
    "alpha": Platform("alpha", 100, 1.0, {"cpu"}),
    "beta": Platform("beta", 60, 2.0, {"cpu"}),
    "gamma": Platform("gamma", 40, 3.0, {"cpu", "gpu"}),
}


class VcgTests(unittest.TestCase):
    """Small hand cases, then random windows against brute force."""

    def test_contended_winner_pays_the_displaced_net_value(self) -> None:
        """Two jobs, one slot: the winner pays cost plus the loser's net value."""
        platforms = {"alpha": Platform("alpha", 100, 2.0)}
        first = Job("first", 0, (Leg("0", 60, 3600),), 1.5)  # cost 120, value 180
        second = Job("second", 0, (Leg("0", 60, 3600),), 1.25)  # cost 120, value 150
        decisions = Vcg().decide(_window(platforms, [first, second]))
        self.assertEqual([d.job_id for d in decisions], ["first"])
        self.assertAlmostEqual(decisions[0].charge_credits, 120.0 + 30.0)

    def test_uncontended_jobs_pay_cost_only(self) -> None:
        """Two jobs that both fit pay exactly their cost."""
        platforms = {"alpha": Platform("alpha", 100, 2.0)}
        jobs = [
            Job("a", 0, (Leg("0", 40, 3600),), 1.5),
            Job("b", 0, (Leg("0", 40, 3600),), 1.25),
        ]
        charges = {
            d.job_id: d.charge_credits for d in Vcg().decide(_window(platforms, jobs))
        }
        self.assertEqual(charges, {"a": 80.0, "b": 80.0})

    def test_composite_beats_two_singles(self) -> None:
        """A composite with more net value takes both platforms."""
        platforms = {
            "alpha": Platform("alpha", 5, 1.0),
            "beta": Platform("beta", 5, 1.0),
        }
        composite = Job(
            "c", 0, (Leg("l", 5, 60), Leg("r", 5, 60)), {"alpha": 10.0, "beta": 8.0}
        )
        singles = [
            Job("a", 0, (Leg("0", 5, 60),), 8.0),
            Job("b", 0, (Leg("0", 5, 60),), 9.0),
        ]
        decisions = Vcg().decide(_window(platforms, [composite, *singles]))
        self.assertEqual(
            [(d.job_id, d.placement.id) for d in decisions], [("c", "alpha+beta")]
        )

    def test_random_windows_match_brute_force_and_are_ir(self) -> None:
        """Allocation equals exhaustive search; charges stay within bounds."""
        for seed in range(1, 13):
            window = _window(PLATFORMS, _random_jobs(seed))
            decisions = Vcg().decide(window)
            welfare, chosen = _brute_force(window)
            with self.subTest(seed=seed):
                self.assertAlmostEqual(
                    sum(d.placement.net_credits for d in decisions), welfare
                )
                for decision in decisions:
                    self.assertGreaterEqual(
                        decision.charge_credits, decision.placement.cost_credits - 1e-9
                    )
                    self.assertLessEqual(
                        decision.charge_credits, decision.placement.value_credits + 1e-9
                    )

    def test_no_profitable_misreport(self) -> None:
        """Scaling any multiplier never raises a job's true utility."""
        jobs = _random_jobs(7)
        window = _window(PLATFORMS, jobs)
        mechanism = Vcg()
        truthful = mechanism.decide(window)
        for job in jobs:
            true_utility = _utility(window, truthful, job.job_id)
            for scale in (0.5, 0.8, 1.2, 2.0):
                if isinstance(job.bid, dict):
                    variants = [
                        {
                            n: v * (scale if n == only or only is None else 1.0)
                            for n, v in job.bid.items()
                        }
                        for only in (None, *job.bid)
                    ]
                else:
                    variants = [job.bid * scale]
                for bid in variants:
                    lied = [
                        (
                            Job(j.job_id, j.submit_s, j.legs, bid)
                            if j.job_id == job.job_id
                            else j
                        )
                        for j in jobs
                    ]
                    lied_window = _window(PLATFORMS, lied)
                    # utility at true values, looked up in the truthful window
                    decisions = mechanism.decide(lied_window)
                    utility = 0.0
                    for decision in decisions:
                        if decision.job_id == job.job_id:
                            true_value = (
                                next(
                                    p.value_credits
                                    for p in window.candidates[job.job_id]
                                    if p.id == decision.placement.id
                                )
                                if any(
                                    p.id == decision.placement.id
                                    for p in window.candidates[job.job_id]
                                )
                                else 0.0
                            )
                            utility = true_value - decision.charge_credits
                    with self.subTest(job=job.job_id, scale=scale, bid=bid):
                        self.assertLessEqual(utility, true_utility + 1e-9)


if __name__ == "__main__":
    unittest.main()
