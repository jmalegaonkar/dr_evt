################################################################################
#         Copyright 2023 Lawrence Livermore National Security, LLC             #
#         See the top-level LICENSE file for details.                          #
#                                                                              #
#         SPDX-License-Identifier: MIT                                         #
################################################################################

"""Tests for deployed grid and guided RegretFormer regret estimates."""

import unittest

try:
    import torch

    from dr_evt_market.learned.net import RegretFormerNet
    from dr_evt_market.learned.regret import (
        grid_regret,
        guided_refinement_regret,
        summarize_regret,
    )
    from dr_evt_market.learned.windows import (
        WindowBatch,
        structure_from_observation,
    )
except ModuleNotFoundError:
    torch = None

from dr_evt_market.mechanisms import (
    JobBid,
    JobOffer,
    LegBid,
    LegSpec,
    MarketObservation,
    Placement,
)


def _observation() -> MarketObservation:
    jobs = (
        JobOffer(
            "first",
            0,
            (LegSpec("0", 2, 3600),),
            (
                Placement("alpha", {"0": "alpha"}, {"alpha": 2}, 2.0),
                Placement("beta", {"0": "beta"}, {"beta": 2}, 3.0),
            ),
        ),
        JobOffer(
            "second",
            0,
            (LegSpec("0", 2, 3600),),
            (
                Placement("alpha", {"0": "alpha"}, {"alpha": 2}, 2.0),
                Placement("beta", {"0": "beta"}, {"beta": 2}, 3.0),
            ),
        ),
    )
    bids = {
        "first": JobBid(
            "first",
            (LegBid("0", {"alpha": 8.0, "beta": 7.0}),),
        ),
        "second": JobBid(
            "second",
            (LegBid("0", {"alpha": 6.0, "beta": 9.0}),),
        ),
    }
    return MarketObservation(
        0,
        0,
        5,
        jobs,
        bids,
        {"alpha": 4, "beta": 4},
    )


if torch is not None:
    class FirstCandidateNet(torch.nn.Module):
        """Give every job its first candidate with a zero payment fraction."""

        def forward(self, values, slot_mask, job_mask):
            """Return fixed allocations and payment fractions."""
            del values
            probabilities = torch.zeros(
                *slot_mask.shape,
                dtype=torch.float32,
                device=slot_mask.device,
            )
            probabilities[..., 0] = job_mask.to(torch.float32)
            payments = torch.zeros_like(job_mask, dtype=torch.float32)
            return probabilities, payments


@unittest.skipIf(torch is None, "torch is not installed")
class RegretTests(unittest.TestCase):
    """Check deployed regret bounds and summary statistics."""

    def setUp(self) -> None:
        """Create one deterministic two-job learned window."""
        torch.manual_seed(29)
        structure = structure_from_observation(_observation())
        self.batch = WindowBatch((structure,), "cpu")
        self.net = RegretFormerNet(hid=16, hid_att=8, n_layers=1, n_heads=2)

    def test_grid_is_nonnegative_and_bounded_by_refinement(self) -> None:
        """Guided refinement never lowers the deployed grid bound."""
        grid, _ = grid_regret(
            self.net,
            self.batch,
            self.batch.dimension_values,
            grid=(0.5, 1.0, 1.5),
        )
        refined = guided_refinement_regret(
            self.net,
            self.batch,
            self.batch.dimension_values,
            grid=(0.5, 1.0, 1.5),
            steps=3,
            n_perturb=1,
            seed=31,
        )

        self.assertTrue(bool(torch.all(grid >= 0.0)))
        self.assertTrue(bool(torch.all(refined >= grid)))
        self.assertEqual(grid.shape, self.batch.job_mask.shape)

    def test_constant_first_candidate_network_has_zero_regret(self) -> None:
        """A report-independent allocation and payment has zero regret."""
        regret, _ = grid_regret(
            FirstCandidateNet(),
            self.batch,
            self.batch.dimension_values,
            grid=(0.5, 1.0, 1.5),
        )

        torch.testing.assert_close(regret, torch.zeros_like(regret))

    def test_summary_contains_required_statistics(self) -> None:
        """The regret summary reports central, tail, and normalized values."""
        regret, _ = grid_regret(
            self.net,
            self.batch,
            self.batch.dimension_values,
            grid=(0.5, 1.0, 1.5),
        )

        summary = summarize_regret(
            regret,
            self.batch,
            self.batch.dimension_values,
            self.net,
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
