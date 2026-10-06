################################################################################
#         Copyright 2023 Lawrence Livermore National Security, LLC             #
#         See the top-level LICENSE file for details.                          #
#                                                                              #
#         SPDX-License-Identifier: MIT                                         #
################################################################################

"""Tests for regret by item-wise grid and guided refinement."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

from dr_evt_market import (
    FirstFit,
    FirstPrice,
    Job,
    RegretFormer,
    Vcg,
    federation,
    grid_regret,
    read_jobs,
    refined_regret,
)

_DATA = Path(__file__).with_name("data")
_TORCH = importlib.util.find_spec("torch") is not None


def _first_window(directory):
    """Return the fixture's first window: its jobs, platforms and free nodes."""
    jobs = [job for job in read_jobs(_DATA / "jobs.csv") if job.submit_s == 0]
    platforms = federation(Path(directory), share=0.1)
    free = {name: platform.exposed_nodes for name, platform in platforms.items()}
    return jobs, platforms, free


class GridRegretTests(unittest.TestCase):
    """Measure the grid lower bound on mechanisms with known incentives."""

    def test_vcg_regret_is_the_cost_saved_under_the_price(self) -> None:
        """Alone, a job pays its cost under VCG, or the least bid on the grid."""
        job = Job("shade", 0, 2, 360, 3.0, {"gpu"})
        with tempfile.TemporaryDirectory() as directory:
            platforms = federation(Path(directory), share=0.1, names=("corona",))
            regret = grid_regret(Vcg(), [job], platforms, {"corona": 12})
        # The grid's least positive price is 4 x 3.0 / 100 = 0.12.
        self.assertAlmostEqual(regret["shade"], (1.5 - 0.12) * 2 * 360 / 3600)

    def test_reports_at_or_above_the_price_leave_vcg_without_regret(self) -> None:
        """Without bids under the price, VCG has no regret and shading stops there."""
        job = Job("shade", 0, 2, 360, 3.0, {"gpu"})
        with tempfile.TemporaryDirectory() as directory:
            platforms = federation(Path(directory), share=0.1, names=("corona",))
            free = {"corona": 12}
            vcg = grid_regret(Vcg(), [job], platforms, free, under_price=False)
            paid = grid_regret(FirstPrice(), [job], platforms, free, under_price=False)
        self.assertAlmostEqual(vcg["shade"], 0.0)
        self.assertAlmostEqual(paid["shade"], (3.0 - 1.5) * 2 * 360 / 3600)

    def test_first_fit_has_no_regret(self) -> None:
        """FirstFit reads no bid, so no report changes what a job gets or pays."""
        with tempfile.TemporaryDirectory() as directory:
            jobs, platforms, free = _first_window(directory)
            regret = grid_regret(FirstFit(), jobs, platforms, free, points=5)
        self.assertEqual(set(regret.values()), {0.0})

    def test_pay_what_you_bid_regret_is_the_shaded_surplus(self) -> None:
        """Alone, a job keeps its value less the least bid on the grid."""
        job = Job("shade", 0, 2, 360, 3.0, {"gpu"})
        with tempfile.TemporaryDirectory() as directory:
            platforms = federation(Path(directory), share=0.1, names=("corona",))
            regret = grid_regret(FirstPrice(), [job], platforms, {"corona": 12})
        self.assertAlmostEqual(regret["shade"], (3.0 - 0.12) * 2 * 360 / 3600)


@unittest.skipUnless(_TORCH, "RegretFormer needs torch")
class RefinedRegretTests(unittest.TestCase):
    """Check the refinement on RegretFormer's network."""

    def test_batched_grid_matches_the_grid_through_decide(self) -> None:
        """The grid computed in batches through the network equals grid_regret."""
        with tempfile.TemporaryDirectory() as directory:
            jobs, platforms, free = _first_window(directory)
            mechanism = RegretFormer(seed=4)
            expected = grid_regret(mechanism, jobs, platforms, free, points=9)
            estimates = refined_regret(
                mechanism,
                jobs,
                platforms,
                free,
                points=9,
                starts=2,
                steps=2,
                gradient_steps=2,
            )
        for job_id, value in expected.items():
            self.assertAlmostEqual(estimates[job_id].grid, value)

    def test_refinement_only_adds_to_the_grid(self) -> None:
        """Refined regret is never below the grid's, and a seed fixes it."""
        options = {"points": 9, "starts": 4, "steps": 10, "gradient_steps": 10}
        with tempfile.TemporaryDirectory() as directory:
            jobs, platforms, free = _first_window(directory)
            first = refined_regret(
                RegretFormer(seed=5), jobs, platforms, free, **options
            )
            again = refined_regret(
                RegretFormer(seed=5), jobs, platforms, free, **options
            )
        self.assertEqual(first, again)
        for estimate in first.values():
            self.assertGreaterEqual(estimate.grid, 0.0)
            self.assertGreaterEqual(estimate.refined, estimate.grid)
            self.assertGreaterEqual(estimate.gradient, 0.0)


if __name__ == "__main__":
    unittest.main()
