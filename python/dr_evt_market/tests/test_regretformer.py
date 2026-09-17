################################################################################
#         Copyright 2023 Lawrence Livermore National Security, LLC             #
#         See the top-level LICENSE file for details.                          #
#                                                                              #
#         SPDX-License-Identifier: MIT                                         #
################################################################################

"""Tests for deployed RegretFormer decisions and checkpoints."""

from pathlib import Path
import tempfile
import unittest

try:
    import torch

    from dr_evt_market.mechanisms import RegretFormer
except ModuleNotFoundError:
    torch = None

from dr_evt_market import PlatformSnapshot, read_jobs, read_platforms
from dr_evt_market.mechanisms import build_observation, validate_decisions

_DATA_DIR = Path(__file__).with_name("data")


def _observation():
    platforms = read_platforms(_DATA_DIR / "market_platforms.csv")
    jobs, bids = read_jobs(_DATA_DIR / "market_jobs.csv", platforms)
    snapshots = {
        platform.system_id: PlatformSnapshot(
            name=platform.system_id,
            time_s=0,
            total_nodes=platform.total_nodes,
            free_nodes=platform.total_nodes,
            in_use_nodes=0,
            waiting_jobs=0,
            current_utilization=0.0,
        )
        for platform in platforms
    }
    prices = {
        platform.system_id: platform.price_per_node_hour
        for platform in platforms
    }
    return build_observation(0, 0, 11, jobs, bids, snapshots, prices)


@unittest.skipIf(torch is None, "torch is not installed")
class RegretFormerTests(unittest.TestCase):
    """Check feasibility, determinism, and checkpoint loading."""

    def test_random_network_emits_valid_individually_rational_decisions(self) -> None:
        """Random deployment passes clearing without rejected decisions."""
        observation = _observation()
        decisions = RegretFormer(None, seed=7).decide(observation)

        accepted, rejected = validate_decisions(observation, decisions)

        self.assertEqual(rejected, [])
        self.assertEqual(accepted, decisions)
        self.assertTrue(decisions)
        for decision in decisions:
            cost = observation.candidate(
                decision.job_id,
                decision.placement_id,
            ).resource_cost_credits
            value = observation.value(
                decision.job_id,
                decision.placement_id,
            )
            self.assertGreaterEqual(decision.charge_credits, cost)
            self.assertLessEqual(decision.charge_credits, value or 0.0)

    def test_same_seed_produces_same_decisions(self) -> None:
        """Two random networks with the same seed deploy identically."""
        observation = _observation()

        first = RegretFormer(None, seed=19).decide(observation)
        second = RegretFormer(None, seed=19).decide(observation)

        self.assertEqual(first, second)

    def test_checkpoint_round_trip_preserves_decisions(self) -> None:
        """Checkpoint metadata and weights reconstruct the same mechanism."""
        observation = _observation()
        original = RegretFormer(
            None,
            hid=16,
            hid_att=8,
            n_layers=1,
            n_heads=2,
            seed=23,
        )
        with tempfile.TemporaryDirectory(
            prefix="dr_evt_market_regretformer_"
        ) as directory:
            checkpoint = Path(directory) / "model.pt"
            torch.save(
                {
                    "state_dict": original.net.state_dict(),
                    "in_channels": original.in_channels,
                    "hid": original.hid,
                    "hid_att": original.hid_att,
                    "n_layers": original.n_layers,
                    "n_heads": original.n_heads,
                    "trained_on": "fixture federation",
                    "created": "2026-09-16",
                },
                checkpoint,
            )

            restored = RegretFormer(checkpoint, seed=99)

        self.assertEqual(
            restored.decide(observation),
            original.decide(observation),
        )
        self.assertEqual(restored.trained_on, "fixture federation")
        self.assertEqual(restored.created, "2026-09-16")


if __name__ == "__main__":
    unittest.main()
