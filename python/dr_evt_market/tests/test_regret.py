################################################################################
#         Copyright 2023 Lawrence Livermore National Security, LLC             #
#         See the top-level LICENSE file for details.                          #
#                                                                              #
#         SPDX-License-Identifier: MIT                                         #
################################################################################

"""Tests for regret estimation by grid search and guided ascent."""

import unittest

try:
    import torch

    from dr_evt_market.learned.net import RegretFormerNet
    from dr_evt_market.learned.regret import (
        grid_regret,
        guided_refinement_regret,
        summarize_regret,
    )
    from dr_evt_market.learned.windows import WindowBatch, structure_from_window
except ModuleNotFoundError:
    torch = None

from dr_evt_market.mechanisms import Job, Leg, Platform, build_window

PLATFORMS = {"alpha": Platform("alpha", 4, 1.0), "beta": Platform("beta", 4, 1.5)}


def _window():
    jobs = [
        Job("first", 0, (Leg("0", 2, 3600),), {"alpha": 4.0, "beta": 3.0}),
        Job("second", 0, (Leg("0", 2, 3600),), 3.0),
    ]
    return build_window(0, 0, jobs, PLATFORMS, {"alpha": 4, "beta": 4})


if torch is not None:

    class FirstCandidateNet(torch.nn.Module):
        """Give every job its first candidate and charge nothing above cost."""

        def forward(self, values, slot_mask, job_mask):
            """Return fixed allocations and zero payment fractions."""
            del values
            probabilities = torch.zeros(*slot_mask.shape, dtype=torch.float32)
            probabilities[..., 0] = job_mask.to(torch.float32)
            return probabilities, torch.zeros_like(job_mask, dtype=torch.float32)


@unittest.skipIf(torch is None, "torch is not installed")
class RegretTests(unittest.TestCase):
    """Bounds, the zero-regret toy, and the summary."""

    def setUp(self) -> None:
        """Build one two-job batch and a small random network."""
        torch.manual_seed(29)
        self.batch = WindowBatch((structure_from_window(_window()),), "cpu")
        self.net = RegretFormerNet(hid=16, hid_att=8, n_layers=1, n_heads=2)

    def test_grid_is_nonnegative_and_below_refinement(self) -> None:
        """Ascent can only find more regret than the grid."""
        grid, _ = grid_regret(self.net, self.batch, self.batch.dimension_values)
        refined = guided_refinement_regret(
            self.net, self.batch, self.batch.dimension_values, steps=3, n_perturb=1
        )
        self.assertTrue(bool(torch.all(grid >= 0.0)))
        self.assertTrue(bool(torch.all(refined + 1e-6 >= grid)))

    def test_fixed_allocation_at_cost_has_zero_regret(self) -> None:
        """A report cannot help when the outcome ignores it."""
        regret, _ = grid_regret(
            FirstCandidateNet(), self.batch, self.batch.dimension_values
        )
        torch.testing.assert_close(regret, torch.zeros_like(regret))

    def test_summary_keys(self) -> None:
        """The summary reports the five statistics."""
        regret, _ = grid_regret(self.net, self.batch, self.batch.dimension_values)
        summary = summarize_regret(
            regret, self.batch, self.batch.dimension_values, self.net
        )
        self.assertEqual(
            set(summary),
            {
                "mean_regret",
                "median_regret",
                "p95_regret",
                "max_regret",
                "mean_regret_over_mean_truthful_utility",
            },
        )


if __name__ == "__main__":
    unittest.main()
