################################################################################
#         Copyright 2023 Lawrence Livermore National Security, LLC             #
#         See the top-level LICENSE file for details.                          #
#                                                                              #
#         SPDX-License-Identifier: MIT                                         #
################################################################################

"""Tests for the deployed RegretFormer mechanism."""

from pathlib import Path
import tempfile
import unittest

try:
    import torch

    from dr_evt_market.mechanisms import RegretFormer
except ModuleNotFoundError:
    torch = None

from dr_evt_market import read_jobs, read_platforms
from dr_evt_market.mechanisms import build_window, validate_decisions

_DATA = Path(__file__).with_name("data")


def _window():
    platforms = {p.name: p for p in read_platforms(_DATA / "market_platforms.csv")}
    jobs = read_jobs(_DATA / "market_jobs.csv")
    return build_window(
        0, 0, jobs, platforms, {n: p.total_nodes for n, p in platforms.items()}
    )


@unittest.skipIf(torch is None, "torch is not installed")
class RegretFormerTests(unittest.TestCase):
    """Feasibility, determinism and checkpoints."""

    def test_random_network_gives_valid_bounded_decisions(self) -> None:
        """Deployment passes validation with charges within cost and value."""
        window = _window()
        decisions = RegretFormer(None, seed=7).decide(window)
        accepted, rejected = validate_decisions(window, decisions)
        self.assertEqual(rejected, [])
        self.assertEqual(accepted, decisions)
        self.assertTrue(decisions)
        for decision in decisions:
            self.assertGreaterEqual(
                decision.charge_credits, decision.placement.cost_credits
            )
            self.assertLessEqual(
                decision.charge_credits, decision.placement.value_credits
            )

    def test_same_seed_same_decisions(self) -> None:
        """Two networks with one seed decide identically."""
        window = _window()
        self.assertEqual(
            RegretFormer(None, seed=19).decide(window),
            RegretFormer(None, seed=19).decide(window),
        )

    def test_checkpoint_round_trip(self) -> None:
        """A saved checkpoint restores the same mechanism."""
        window = _window()
        original = RegretFormer(None, hid=16, hid_att=8, n_layers=1, n_heads=2, seed=23)
        with tempfile.TemporaryDirectory(
            prefix="dr_evt_market_regretformer_"
        ) as directory:
            path = Path(directory) / "model.pt"
            torch.save(
                {
                    "state_dict": original.net.state_dict(),
                    "in_channels": original.in_channels,
                    "hid": 16,
                    "hid_att": 8,
                    "n_layers": 1,
                    "n_heads": 2,
                    "trained_on": "fixture",
                    "created": "2026-09-17",
                },
                path,
            )
            restored = RegretFormer(path, seed=99)
        self.assertEqual(restored.decide(window), original.decide(window))
        self.assertEqual(restored.trained_on, "fixture")


if __name__ == "__main__":
    unittest.main()
