################################################################################
#         Copyright 2023 Lawrence Livermore National Security, LLC             #
#         See the top-level LICENSE file for details.                          #
#                                                                              #
#         SPDX-License-Identifier: MIT                                         #
################################################################################

"""Tests for candidates, offers and the auctions."""

import itertools
import random
import tempfile
import unittest
from dataclasses import dataclass, field, replace
from unittest import mock

from federated_market import (
    Decision,
    FirstFit,
    FirstPrice,
    Job,
    Vcg,
    candidates,
    federation,
    offers,
)

_TIE_BREAK = 1.0e-9


@dataclass(frozen=True)
class _Platform:
    name: str
    exposed_nodes: int
    price_per_node_hour: float
    hardware: frozenset[str]
    speed: dict[str, float] = field(default_factory=lambda: {"cpu": 1.0, "gpu": 1.0})

    def fits(self, job) -> bool:
        return job.requires <= self.hardware and job.num_nodes <= self.exposed_nodes

    def job_speed(self, job) -> float:
        hardware = "gpu" if "gpu" in job.requires else "cpu"
        return self.speed[hardware]

    def cost(self, job) -> float:
        return (
            self.price_per_node_hour
            * job.num_nodes
            * job.limit_s
            / self.job_speed(job)
            / 3600
        )


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
                    name: rng.uniform(1.0, 4.0)
                    for name in platforms
                    if rng.random() < 0.75
                }
                bid = bid or {"a": rng.uniform(1.0, 4.0)}
            else:
                bid = rng.uniform(1.0, 4.0)
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
    offered = [offers(job, platforms, free_nodes) for job in jobs]
    indexes = {}
    index = 0
    for job_index, choices in enumerate(offered):
        for name in choices:
            indexes[job_index, name] = index
            index += 1
    # The solve takes only bids at or above the price; the others wait for leftovers.
    options = [
        (
            (None,)
            if i == excluded
            else (
                None,
                *(name for name, (cost, value) in offer.items() if value >= cost),
            )
        )
        for i, offer in enumerate(offered)
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
            offered[i][name][1] - offered[i][name][0]
            for i, name in enumerate(choice)
            if name is not None
        )
        penalty = (
            sum(indexes[i, name] for i, name in enumerate(choice) if name is not None)
            * _TIE_BREAK
        )
        score = welfare - penalty
        if score > best_score:
            best_score, best_choice, best_welfare = score, choice, welfare
    chosen = {i: name for i, name in enumerate(best_choice) if name is not None}
    return chosen, best_welfare, offered


def _leftovers(jobs, platforms, free_nodes, chosen):
    """Fill the nodes the solve leaves, in batch order, at each job's best offer."""
    left = dict(free_nodes)
    for i, name in chosen.items():
        left[name] -= jobs[i].num_nodes
    filled = {}
    for i, job in enumerate(jobs):
        found = offers(job, platforms, left)
        if i in chosen or not found:
            continue
        name = max(found, key=lambda name: found[name][1] - found[name][0])
        left[name] -= job.num_nodes
        filled[i] = name
    return filled


def _under_the_price(job, platforms) -> bool:
    return any(
        job.price(name) is not None and job.price(name) < platform.price_per_node_hour
        for name, platform in platforms.items()
    )


def _value(job, platform, platforms):
    selected = platforms[platform]
    speed = selected.job_speed(job) if isinstance(job.bid, dict) else 1.0
    return job.price(platform) * job.num_nodes * job.limit_s / speed / 3600


class AuctionTests(unittest.TestCase):
    """Check candidate construction, allocation and Clarke payments."""

    def test_first_fit_ignores_bids_and_charges_posted_cost(self) -> None:
        """First-fit chooses by posted cost and never by the bid."""
        platforms = {
            "slow": _Platform("slow", 2, 1.0, frozenset({"cpu"}), {"cpu": 0.5}),
            "fast": _Platform("fast", 2, 1.5, frozenset({"cpu"}), {"cpu": 2.0}),
        }
        job = Job("job", 0, 1, 3600, 0.0)
        low = FirstFit().decide([job], platforms, {"slow": 2, "fast": 2})
        high = FirstFit().decide(
            [replace(job, bid=100.0)], platforms, {"slow": 2, "fast": 2}
        )
        self.assertEqual(low, high)
        self.assertEqual(low[0].platform, "fast")
        self.assertEqual(low[0].charge, platforms["fast"].cost(job))

    def test_first_fit_respects_capacity(self) -> None:
        """First-fit uses each platform's remaining nodes."""
        platforms = {
            "cheap": _Platform("cheap", 2, 1.0, frozenset({"cpu"})),
            "dear": _Platform("dear", 2, 2.0, frozenset({"cpu"})),
        }
        jobs = [Job(f"j{index}", 0, 2, 60, 0.0) for index in range(3)]
        decisions = FirstFit().decide(jobs, platforms, {"cheap": 2, "dear": 2})
        self.assertEqual(
            [(item.job_id, item.platform) for item in decisions],
            [("j0", "cheap"), ("j1", "dear")],
        )

    def test_offers_support_both_bid_forms(self) -> None:
        """Candidates are where a job fits now; offers are those it bid on."""
        with tempfile.TemporaryDirectory() as directory:
            platforms = federation(directory, share=0.1)
            free = {
                name: platform.exposed_nodes for name, platform in platforms.items()
            }
            scalar = Job("scalar", 0, 4, 360, 3.0, {"gpu"})
            low = Job("low", 0, 4, 360, 1.0, {"gpu"})
            # Four nodes do not fit the three-node slices of Matrix and Tioga.
            self.assertEqual(
                candidates(scalar, platforms, free), ["corona", "tuolumne"]
            )
            self.assertEqual(candidates(low, platforms, free), ["corona", "tuolumne"])
            # A bid under Corona's price is an offer too; a bid of zero is none.
            under = offers(low, platforms, free)
            self.assertEqual(list(under), ["corona", "tuolumne"])
            self.assertLess(under["corona"][1], under["corona"][0])
            self.assertEqual(offers(replace(low, bid=0.0), platforms, free), {})
            found = offers(scalar, platforms, free)
            self.assertEqual(list(found), ["corona", "tuolumne"])
            self.assertEqual(found["corona"], (0.6, 1.2))
            self.assertAlmostEqual(found["tuolumne"][0], 0.076 / 3.313)
            self.assertEqual(found["tuolumne"][1], 1.2)
            mapped = Job(
                "mapped",
                0,
                2,
                360,
                {"matrix": 1.6, "tioga": 2.8, "tuolumne": 0.2},
                {"gpu"},
            )
            self.assertEqual(
                list(offers(mapped, platforms, free)),
                ["matrix", "tioga", "tuolumne"],
            )
            free["matrix"] = 1
            self.assertEqual(
                list(offers(mapped, platforms, free)), ["tioga", "tuolumne"]
            )

    def test_single_bid_value_is_independent_of_platform_speed(self) -> None:
        """A scalar bid values the same reference work on every platform."""
        platforms = {
            "slow": _Platform(
                "slow",
                2,
                1.0,
                frozenset({"cpu"}),
                {"cpu": 1.0},
            ),
            "fast": _Platform(
                "fast",
                2,
                1.0,
                frozenset({"cpu"}),
                {"cpu": 2.0},
            ),
        }
        job = Job("scalar", 0, 1, 3600, 3.0, requested_s=7200)
        found = offers(job, platforms, {"slow": 2, "fast": 2})
        self.assertEqual(found["slow"][1], 3.0)
        self.assertEqual(found["fast"][1], 3.0)
        self.assertEqual(found["slow"][0], 2.0 * found["fast"][0])

    def test_vcg_matches_brute_force(self) -> None:
        """The optimizer, pivot charges and leftover fill match exhaustive search."""
        mechanism = Vcg()
        filled_under_the_price = 0
        for case, (jobs, platforms, free) in enumerate(_instances()):
            with self.subTest(case=case):
                actual = mechanism.decide(jobs, platforms, free)
                chosen, welfare, offered = _brute(jobs, platforms, free)
                filled = _leftovers(jobs, platforms, free, chosen)
                expected = []
                for i, job in enumerate(jobs):
                    if i in chosen:
                        name = chosen[i]
                        cost, value = offered[i][name]
                        _, without, _ = _brute(jobs, platforms, free, excluded=i)
                        pivot = max(without - (welfare - (value - cost)), 0.0)
                        expected.append((job.job_id, name, min(value, cost + pivot)))
                    elif i in filled:
                        cost, value = offered[i][filled[i]]
                        filled_under_the_price += value < cost
                        expected.append((job.job_id, filled[i], min(cost, value)))
                self.assertEqual(
                    [(item.job_id, item.platform) for item in actual],
                    [(job_id, name) for job_id, name, _ in expected],
                )
                for decision, (_, _, charge) in zip(actual, expected):
                    self.assertAlmostEqual(decision.charge, charge)
        self.assertGreater(filled_under_the_price, 0)

    def test_truthful_bid_resists_scaled_misreports(self) -> None:
        """Among bids at or above the price, scaling cannot improve true utility."""
        mechanism = Vcg()
        for case, (jobs, dear, free) in enumerate(_instances(10)):
            platforms = {
                name: replace(item, price_per_node_hour=item.price_per_node_hour / 4)
                for name, item in dear.items()
            }
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
                    if _under_the_price(job, platforms) or _under_the_price(
                        reports[index], platforms
                    ):
                        continue
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

    def test_bidding_under_the_price_pays_on_idle_nodes(self) -> None:
        """A bid under the price wins idle nodes and pays itself, below the cost."""
        platform = _Platform("only", 10, 2.0, frozenset({"gpu"}))
        platforms = {platform.name: platform}
        job = Job("solo", 0, 2, 360, 3.0, {"gpu"})
        truthful = Vcg().decide([job], platforms, {"only": 10})
        shaded = Vcg().decide([replace(job, bid=1.0)], platforms, {"only": 10})
        self.assertAlmostEqual(truthful[0].charge, platform.cost(job))
        self.assertEqual(shaded[0].platform, "only")
        self.assertAlmostEqual(shaded[0].charge, 1.0 * 2 * 360 / 3600)
        self.assertLess(shaded[0].charge, truthful[0].charge)

    def test_three_jobs_on_two_machines(self) -> None:
        """The guide's example: a tie, a joint choice and bids under the price."""
        platforms = {
            "A": _Platform("A", 4, 1.0, frozenset({"cpu"})),
            "B": _Platform("B", 6, 2.0, frozenset({"cpu"})),
        }
        jobs = [
            Job("J1", 0, 4, 3600, {"A": 3.0, "B": 4.0}),
            Job("J2", 0, 4, 3600, {"A": 2.75, "B": 1.5}),
            Job("J3", 0, 2, 3600, {"A": 1.5, "B": 1.5}),
        ]
        # VCG sends J1 to B so J2 can have A; pay what you bid breaks J1's tie by
        # name; FirstFit takes the cheapest machine and charges its cost.
        expected = [
            (Vcg(), [("J1", "B", 8.0), ("J2", "A", 5.0), ("J3", "B", 3.0)]),
            (FirstPrice(), [("J1", "A", 12.0), ("J2", "B", 6.0), ("J3", "B", 3.0)]),
            (FirstFit(), [("J1", "A", 4.0), ("J2", "B", 8.0), ("J3", "B", 4.0)]),
        ]
        for mechanism, outcome in expected:
            with self.subTest(mechanism=mechanism.name):
                decisions = mechanism.decide(jobs, platforms, {"A": 4, "B": 6})
                self.assertEqual(
                    [(item.job_id, item.platform) for item in decisions],
                    [(job_id, name) for job_id, name, _ in outcome],
                )
                for decision, (_, _, charge) in zip(decisions, outcome):
                    self.assertAlmostEqual(decision.charge, charge)

    def test_winner_without_displacement_pays_cost(self) -> None:
        """A sole winner has no pivot premium."""
        platform = _Platform("only", 10, 2.0, frozenset({"gpu"}))
        platforms = {platform.name: platform}
        job = Job("solo", 0, 2, 360, 3.0, {"gpu"})
        decisions = Vcg().decide([job], platforms, {"only": 10})
        self.assertEqual(len(decisions), 1)
        self.assertEqual(decisions[0].job_id, job.job_id)
        self.assertAlmostEqual(decisions[0].charge, platform.cost(job))

    def test_posted_price_bids_take_leftover_nodes(self) -> None:
        """Jobs bidding exactly the posted price run on nodes winners leave free."""
        platform = _Platform("only", 10, 2.0, frozenset({"gpu"}))
        platforms = {platform.name: platform}
        tier = Job("tier", 0, 1, 360, 2.5, {"gpu"})
        stickers = [Job(f"sticker{i}", 0, 1, 360, 2.0, {"gpu"}) for i in range(5)]
        jobs = [tier, *stickers]

        roomy = Vcg().decide(jobs, platforms, {"only": 10})
        self.assertEqual([item.job_id for item in roomy], [job.job_id for job in jobs])
        for decision, job in zip(roomy[1:], stickers):
            self.assertAlmostEqual(decision.charge, platform.cost(job))

        crowded = Vcg().decide(jobs, platforms, {"only": 3})
        self.assertEqual(
            [item.job_id for item in crowded], ["tier", "sticker0", "sticker1"]
        )

    def test_a_job_the_solve_misses_still_runs_at_cost(self) -> None:
        """A job left out by the solver takes its best leftover platform at cost."""
        platforms = {
            "cheap": _Platform("cheap", 4, 1.0, frozenset({"gpu"})),
            "dear": _Platform("dear", 4, 2.0, frozenset({"gpu"})),
        }
        job = Job("missed", 0, 2, 360, 3.0, {"gpu"})
        with mock.patch(
            "federated_market.mechanism.vcg._solve", return_value=({}, 0.0)
        ):
            decisions = Vcg().decide([job], platforms, {"cheap": 4, "dear": 4})
        self.assertEqual(
            decisions, [Decision("missed", "cheap", platforms["cheap"].cost(job))]
        )

    def test_first_price_is_feasible_and_bounded_by_vcg(self) -> None:
        """Greedy winners fit and pay their values; above the price, VCG does better."""
        first_price = FirstPrice()
        for case, (jobs, platforms, free) in enumerate(_instances()):
            with self.subTest(case=case):
                offered = [offers(job, platforms, free) for job in jobs]
                job_indexes = {job.job_id: index for index, job in enumerate(jobs)}
                decisions = first_price.decide(jobs, platforms, free)
                indexes = [job_indexes[decision.job_id] for decision in decisions]
                self.assertEqual(indexes, sorted(indexes))
                self.assertEqual(len(indexes), len(set(indexes)))

                used = {name: 0 for name in platforms}
                welfare = 0.0
                for decision in decisions:
                    index = job_indexes[decision.job_id]
                    job = jobs[index]
                    cost, value = offered[index][decision.platform]
                    used[decision.platform] += job.num_nodes
                    welfare += max(value - cost, 0.0)
                    self.assertAlmostEqual(decision.charge, value)
                self.assertTrue(all(used[name] <= free[name] for name in platforms))

                vcg = Vcg().decide(jobs, platforms, free)
                vcg_welfare = sum(
                    max(
                        offered[job_indexes[item.job_id]][item.platform][1]
                        - offered[job_indexes[item.job_id]][item.platform][0],
                        0.0,
                    )
                    for item in vcg
                )
                self.assertLessEqual(welfare, vcg_welfare + 1.0e-9)


if __name__ == "__main__":
    unittest.main()
