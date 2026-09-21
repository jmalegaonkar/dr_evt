################################################################################
#         Copyright 2023 Lawrence Livermore National Security, LLC             #
#         See the top-level LICENSE file for details.                          #
#                                                                              #
#         SPDX-License-Identifier: MIT                                         #
################################################################################

"""Tests for the window tensor: channels, masks and multiplier reports."""

import math
import unittest

try:
    import numpy as np
    import torch

    from dr_evt_market.learned.windows import (
        ValueSamplingSpec,
        WindowBatch,
        sample_values,
        structure_from_window,
    )
except ModuleNotFoundError:
    torch = None

from dr_evt_market.mechanisms import Job, Leg, Platform, build_window

PLATFORMS = {
    "alpha": Platform("alpha", 8, 1.0, {"cpu"}),
    "beta": Platform("beta", 4, 2.0, {"cpu"}),
}


def _window():
    single = Job("single", 0, (Leg("0", 2, 3600),), 3.0)  # base cost 2, value 6
    composite = Job(
        "composite",
        0,
        (Leg("l", 1, 1800), Leg("r", 2, 7200)),
        {"alpha": 2.0, "beta": 1.5},
    )
    return build_window(0, 0, [single, composite], PLATFORMS, {"alpha": 8, "beta": 4})


@unittest.skipIf(torch is None, "torch is not installed")
class LearnedWindowTests(unittest.TestCase):
    """Hand-checked channels and the linear value model."""

    def test_tensor_matches_hand_values(self) -> None:
        """Masks, demands and the six channels match the window."""
        structure = structure_from_window(_window())
        batch = WindowBatch((structure,), "cpu")
        self.assertEqual(structure.dimension_keys, (("*",), ("alpha", "beta")))
        self.assertEqual(batch.job_mask.tolist(), [[True, True]])
        self.assertEqual(batch.candidate_mask[0, 0].tolist()[:2], [True, True])
        inputs = batch.input_tensor(batch.true_values)
        # single job on alpha: value 6 and cost 2 over scale 2, log1p(2 nodes),
        # 1 hour, 8 free of 10, one leg
        torch.testing.assert_close(
            inputs[0, 0, 0], torch.tensor([3.0, 1.0, math.log1p(2.0), 1.0, 0.8, 1.0])
        )

    def test_candidate_values_are_linear_in_multipliers(self) -> None:
        """Scaling a multiplier moves exactly the candidates that use it."""
        structure = structure_from_window(_window())
        batch = WindowBatch((structure,), "cpu")
        factors = torch.ones_like(batch.dimension_mask, dtype=torch.float32)
        keys = structure.dimension_keys[1]
        factors[0, 1, keys.index("beta")] = 2.0
        changed = batch.values_from_factors(batch.dimension_values, factors)
        ids = structure.candidate_ids[1]
        for index, candidate_id in enumerate(ids):
            legs_on_beta = candidate_id.split("+").count("beta")
            before = float(batch.true_values[0, 1, index])
            after = float(changed[0, 1, index])
            if legs_on_beta == 0:
                self.assertAlmostEqual(after, before, places=4)
            else:
                self.assertGreater(after, before)
        torch.testing.assert_close(changed[0, 0], batch.true_values[0, 0])

    def test_sampled_multipliers_have_one_per_dimension(self) -> None:
        """Sampling gives one positive multiplier per report dimension."""
        structure = structure_from_window(_window())
        sampled = sample_values(
            (structure,), np.random.default_rng(3), ValueSamplingSpec()
        )[0]
        self.assertEqual([len(values) for values in sampled], [1, 2])
        self.assertTrue(all(float(v) > 0 for values in sampled for v in values))


if __name__ == "__main__":
    unittest.main()
