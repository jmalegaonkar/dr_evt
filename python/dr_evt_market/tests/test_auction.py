################################################################################
#         Copyright 2023 Lawrence Livermore National Security, LLC             #
#         See the top-level LICENSE file for details.                          #
#                                                                              #
#         SPDX-License-Identifier: MIT                                         #
################################################################################

"""Tests for candidates and the VCG auction."""

import itertools
import random
import tempfile
import unittest
from dataclasses import dataclass, replace

from dr_evt_market import Decision, Job, Vcg, base_cost, candidates, federation


@dataclass(frozen=True)
class _Platform:
    name: str
    exposed_nodes: int
    price: float
    hardware: frozenset[str]

    def fits(self, job) -> bool:
        return job.requires <= self.hardware and job.num_nodes <= self.exposed_nodes

    def cost(self, job) -> float:
        return self.price * job.num_nodes * job.limit_s / 3600


def _instances(count=30):
    rng = random.Random(4815)
    platforms = {
        "a": _Platform("a", 4, 1.0, frozenset({"cpu"})),
        "b": _Platform("b", 5, 1.7, frozenset({"cpu", "gpu"})),
        "c": _Platform("c", 6, 2.6, frozenset({"cpu", "gpu"})),
    }
    for case in range(count):
        free = {
            name: rng.randint(2, platform.exposed_nodes)
            for name, platform in platforms.items()
        }
        jobs = []
        scalar_workloads = set()
        for index in range(rng.randint(1, 5)):
            if index % 2:
                bid = {
                    name: rng.uniform(0.5, 4.5)
                    for name in platforms
                    if rng.random() < 0.75
                }
                bid = bid or {"a": rng.uniform(0.5, 4.5)}
            else:
                bid = rng.uniform(0.5, 4.5)
            requires = frozenset({"gpu"}) if (case + index) % 3 == 0 else frozenset()
            nodes = rng.randint(1, 3)
            limit_s = rng.randint(1, 5) * 60
            if not isinstance(bid, dict):
                while nodes * limit_s in scalar_workloads:
                    limit_s += 1
                scalar_workloads.add(nodes * limit_s)
            jobs.append(
                Job(
                    f"j{case}-{index}",
                    0,
                    nodes,
                    limit_s,
                    bid,
                    requires,
                )
            )
        yield jobs, platforms, free


def _brute(jobs, platforms, free_nodes, excluded=None):
    offers = [candidates(job, platforms, free_nodes) for job in jobs]
    indexes = {}
    index = 0
    for job_index, choices in enumerate(offers):
        for name in choices:
            indexes[job_index, name] = index
            index += 1
    options = [
        ((None,) if i == excluded else (None, *offer)) for i, offer in enumerate(offers)
    ]
    best_score, best_choice, best_welfare = float("-inf"), None, 0.0
    for choice in itertools.product(*options):
        used = {
            name: sum(
                job.num_nodes for job, selected in zip(jobs, choice) if selected == name
            )
            for name in platforms
        }
        if any(used[name] > free_nodes[name] for name in platforms):
            continue
        welfare = sum(
            offers[i][name][1] - offers[i][name][0]
            for i, name in enumerate(choice)
            if name is not None
        )
        penalty = (
            sum(indexes[i, name] for i, name in enumerate(choice) if name is not None)
            * 1.0e-9
        )
        score = welfare - penalty
        if score > best_score:
            best_score, best_choice, best_welfare = score, choice, welfare
    chosen = {i: name for i, name in enumerate(best_choice) if name is not None}
    return chosen, best_welfare, offers


def _value(job, platform, platforms):
    multiplier = job.multiplier(platform)
    cost = platforms[platform].cost(job)
    return (
        multiplier * cost
        if isinstance(job.bid, dict)
        else multiplier * base_cost(job, platforms)
    )


class AuctionTests(unittest.TestCase):
    """Check candidate construction, allocation and Clarke payments."""

    def test_candidates_support_both_bid_forms(self) -> None:
        """Candidates respect fit, capacity, value and mapped omissions."""
        with tempfile.TemporaryDirectory() as directory:
            platforms = federation(directory, share=0.1)
            free = {
                name: platform.exposed_nodes for name, platform in platforms.items()
            }
            scalar = Job("scalar", 0, 4, 360, 2.0, {"gpu"})
            offers = candidates(scalar, platforms, free)
            self.assertAlmostEqual(base_cost(scalar, platforms), 0.8)
            self.assertEqual(list(offers), ["corona", "lassen"])
            self.assertEqual(offers["corona"], (0.8, 1.6))
            mapped = Job(
                "mapped",
                0,
                2,
                360,
                {"lassen": 0.9, "tioga": 2.0, "tuolumne": 1.5},
                {"gpu"},
            )
            self.assertEqual(
                list(candidates(mapped, platforms, free)), ["tioga", "tuolumne"]
            )
            free["tioga"] = 1
            self.assertEqual(list(candidates(mapped, platforms, free)), ["tuolumne"])

    def test_vcg_matches_brute_force(self) -> None:
        """The optimizer and pivot charges match exhaustive search."""
        mechanism = Vcg()
        for case, (jobs, platforms, free) in enumerate(_instances()):
            with self.subTest(case=case):
                actual = mechanism.decide(jobs, platforms, free)
                chosen, welfare, offers = _brute(jobs, platforms, free)
                expected_pairs = [(jobs[i].job_id, name) for i, name in chosen.items()]
                self.assertEqual(
                    [(item.job_id, item.platform) for item in actual], expected_pairs
                )
                actual_welfare = sum(
                    offers[i][decision.platform][1] - offers[i][decision.platform][0]
                    for decision in actual
                    for i, job in enumerate(jobs)
                    if job.job_id == decision.job_id
                )
                self.assertAlmostEqual(actual_welfare, welfare)
                for decision in actual:
                    job_index = next(
                        i for i, job in enumerate(jobs) if job.job_id == decision.job_id
                    )
                    cost, value = offers[job_index][decision.platform]
                    _, without, _ = _brute(jobs, platforms, free, excluded=job_index)
                    pivot = max(without - (welfare - (value - cost)), 0.0)
                    self.assertAlmostEqual(decision.charge, min(value, cost + pivot))

    def test_truthful_bid_resists_scaled_misreports(self) -> None:
        """Scaling either bid form cannot improve a job's true utility."""
        mechanism = Vcg()
        for case, (jobs, platforms, free) in enumerate(_instances(10)):
            truthful = {
                item.job_id: item for item in mechanism.decide(jobs, platforms, free)
            }
            for index, job in enumerate(jobs):
                decision = truthful.get(job.job_id)
                truth = (
                    0.0
                    if decision is None
                    else (_value(job, decision.platform, platforms) - decision.charge)
                )
                for scale in (0.5, 0.8, 1.25, 2.0):
                    bid = (
                        {name: value * scale for name, value in job.bid.items()}
                        if isinstance(job.bid, dict)
                        else job.bid * scale
                    )
                    reports = list(jobs)
                    reports[index] = replace(job, bid=bid)
                    outcome = {
                        item.job_id: item
                        for item in mechanism.decide(reports, platforms, free)
                    }.get(job.job_id)
                    utility = (
                        0.0
                        if outcome is None
                        else (_value(job, outcome.platform, platforms) - outcome.charge)
                    )
                    with self.subTest(case=case, job=job.job_id, scale=scale):
                        self.assertLessEqual(utility, truth + 1.0e-9)

    def test_winner_without_displacement_pays_cost(self) -> None:
        """A sole winner has no pivot premium."""
        platform = _Platform("only", 10, 2.0, frozenset({"gpu"}))
        platforms = {platform.name: platform}
        job = Job("solo", 0, 2, 360, 3.0, {"gpu"})
        decisions = Vcg().decide([job], platforms, {"only": 10})
        self.assertEqual(len(decisions), 1)
        self.assertEqual(decisions[0].job_id, job.job_id)
        self.assertAlmostEqual(decisions[0].charge, platform.cost(job))


if __name__ == "__main__":
    unittest.main()
