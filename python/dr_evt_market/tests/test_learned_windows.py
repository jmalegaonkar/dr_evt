################################################################################
#         Copyright 2023 Lawrence Livermore National Security, LLC             #
#         See the top-level LICENSE file for details.                          #
#                                                                              #
#         SPDX-License-Identifier: MIT                                         #
################################################################################

"""Tests for RegretFormer window structures and padded tensors."""

import math
import unittest

try:
    import numpy as np
    import torch

    from dr_evt_market.learned.windows import (
        ValueSamplingSpec,
        WindowBatch,
        sample_values,
        structure_from_observation,
    )
except ModuleNotFoundError:
    np = None
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
    single = JobOffer(
        "single",
        0,
        (LegSpec("only", 2, 3600),),
        (Placement("alpha", {"only": "alpha"}, {"alpha": 2}, 4.0),),
    )
    composite = JobOffer(
        "composite",
        0,
        (
            LegSpec("left", 1, 1800),
            LegSpec("right", 2, 7200),
        ),
        (
            Placement(
                "alpha+beta",
                {"left": "alpha", "right": "beta"},
                {"alpha": 1, "beta": 2},
                5.0,
            ),
            Placement(
                "beta+alpha",
                {"left": "beta", "right": "alpha"},
                {"alpha": 2, "beta": 1},
                6.0,
            ),
        ),
    )
    bids = {
        "single": JobBid(
            "single",
            (LegBid("only", {"alpha": 12.0}),),
        ),
        "composite": JobBid(
            "composite",
            (
                LegBid("left", {"alpha": 3.0, "beta": 5.0}),
                LegBid("right", {"alpha": 7.0, "beta": 11.0}),
            ),
        ),
    }
    return MarketObservation(
        0,
        0,
        17,
        (single, composite),
        bids,
        {"alpha": 8, "beta": 4},
    )


@unittest.skipIf(torch is None, "torch is not installed")
class LearnedWindowTests(unittest.TestCase):
    """Check hand-computed channels, masks, demands, and report scaling."""

    def test_window_tensor_matches_hand_computed_values(self) -> None:
        """The padded tensor contains the specified six input channels."""
        structure = structure_from_observation(_observation())
        batch = WindowBatch((structure,), "cpu")
        inputs = batch.input_tensor(batch.true_values)

        self.assertEqual(structure.platforms, ("alpha", "beta"))
        self.assertEqual(batch.job_mask.tolist(), [[True, True]])
        self.assertEqual(
            batch.candidate_mask.tolist(),
            [[[True, False], [True, True]]],
        )
        self.assertEqual(
            batch.demands.tolist(),
            [[[[2.0, 0.0], [0.0, 0.0]], [[1.0, 2.0], [2.0, 1.0]]]],
        )
        expected_single = (
            3.0,
            1.0,
            math.log1p(2.0),
            1.0,
            0.8,
            1.0,
        )
        expected_composite = (
            14.0 / 5.0,
            1.0,
            math.log1p(3.0),
            2.0,
            2.0 / 3.0,
            2.0,
        )
        torch.testing.assert_close(
            inputs[0, 0, 0],
            torch.tensor(expected_single),
        )
        torch.testing.assert_close(
            inputs[0, 1, 0],
            torch.tensor(expected_composite),
        )
        torch.testing.assert_close(inputs[0, 0, 1], torch.zeros(6))

    def test_one_dimension_changes_only_affected_candidates(self) -> None:
        """A leg-platform factor changes precisely its mapped candidates."""
        structure = structure_from_observation(_observation())
        batch = WindowBatch((structure,), "cpu")
        factors = torch.ones_like(batch.dimension_mask, dtype=torch.float32)
        keys = structure.dimension_keys[1]
        factors[0, 1, keys.index(("composite", "left", "alpha"))] = 2.0

        changed = batch.values_from_factors(batch.dimension_values, factors)

        torch.testing.assert_close(
            changed[0, 1],
            torch.tensor([17.0, 12.0]),
        )
        torch.testing.assert_close(changed[0, 0], batch.true_values[0, 0])

    def test_two_leg_candidate_value_sums_scaled_legs(self) -> None:
        """Each composite candidate value is the sum of its scaled leg bids."""
        structure = structure_from_observation(_observation())
        batch = WindowBatch((structure,), "cpu")
        factors = torch.ones_like(batch.dimension_mask, dtype=torch.float32)
        keys = structure.dimension_keys[1]
        factors[0, 1, keys.index(("composite", "left", "beta"))] = 0.5
        factors[0, 1, keys.index(("composite", "right", "alpha"))] = 3.0

        changed = batch.values_from_factors(batch.dimension_values, factors)

        self.assertAlmostEqual(float(changed[0, 1, 0]), 14.0)
        self.assertAlmostEqual(float(changed[0, 1, 1]), 23.5)

    def test_sampled_dimension_scaling_moves_mapped_candidates(self) -> None:
        """A sampled dimension contributes only to candidates that use it."""
        structure = structure_from_observation(_observation())
        batch = WindowBatch((structure,), "cpu")
        sampled = sample_values(
            (structure,),
            np.random.default_rng(20260916),
            ValueSamplingSpec(),
        )
        dimensions = torch.zeros_like(batch.dimension_values)
        for job_index, values in enumerate(sampled[0]):
            dimensions[0, job_index, : len(values)] = torch.tensor(values)
        baseline = batch.candidate_values(dimensions)
        factors = torch.ones_like(dimensions)
        keys = structure.dimension_keys[1]
        dimension = keys.index(("composite", "left", "alpha"))
        factors[0, 1, dimension] = 2.0

        changed = batch.values_from_factors(dimensions, factors)
        movement = changed[0, 1] - baseline[0, 1]
        expected = (
            dimensions[0, 1, dimension]
            * batch.dimension_use[0, 1, :, dimension]
        )

        torch.testing.assert_close(movement, expected)


if __name__ == "__main__":
    unittest.main()
