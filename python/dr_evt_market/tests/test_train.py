################################################################################
#         Copyright 2023 Lawrence Livermore National Security, LLC             #
#         See the top-level LICENSE file for details.                          #
#                                                                              #
#         SPDX-License-Identifier: MIT                                         #
################################################################################

"""Smoke tests for synthetic RegretFormer training and checkpoints."""

from pathlib import Path
import tempfile
import unittest

try:
    import torch

    from dr_evt_market.learned.synthetic import synthetic_structures
    from dr_evt_market.learned.train import TrainConfig, Trainer
    from dr_evt_market.mechanisms import RegretFormer
except ModuleNotFoundError:
    torch = None

from dr_evt_market import PlatformSnapshot, read_jobs, read_platforms
from dr_evt_market.mechanisms import build_observation, validate_decisions

_DATA_DIR = Path(__file__).with_name("data")


def _fixture_observation():
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
    return build_observation(0, 0, 13, jobs, bids, snapshots, prices)


@unittest.skipIf(torch is None, "torch is not installed")
class TrainerTests(unittest.TestCase):
    """Exercise a short deterministic training and loading cycle."""

    def test_two_epoch_training_saves_a_deployable_checkpoint(self) -> None:
        """Twenty synthetic windows train, evaluate, save, and reload."""
        structures = synthetic_structures(20, seed=37)
        config = TrainConfig(
            epochs=2,
            batch_size=5,
            misreport_steps=1,
            regret_jobs_per_batch=1,
            hid=8,
            hid_att=4,
            n_layers=1,
            n_heads=1,
            evaluation_grid=(0.5, 1.0, 1.5),
            evaluation_steps=1,
            evaluation_perturbations=0,
            seed=37,
        )
        trainer = Trainer(config, structures)

        result = trainer.train()

        self.assertEqual(len(result["epochs"]), 2)
        self.assertEqual(
            set(result["epochs"][0]),
            {
                "epoch",
                "objective",
                "relaxed_regret",
                "capacity_penalty",
                "dual",
                "regret_target",
                "loss",
            },
        )
        self.assertEqual(
            set(result["evaluation"]),
            {
                "vcg_welfare_coverage",
                "mean_regret",
                "max_regret",
                "mean_regret_over_mean_truthful_utility",
            },
        )

        with tempfile.TemporaryDirectory(
            prefix="dr_evt_market_train_"
        ) as directory:
            checkpoint = trainer.save(Path(directory) / "trained.pt")
            mechanism = RegretFormer(checkpoint)

        decisions = mechanism.decide(_fixture_observation())
        accepted, rejected = validate_decisions(
            _fixture_observation(),
            decisions,
        )
        self.assertEqual(accepted, decisions)
        self.assertEqual(rejected, [])


if __name__ == "__main__":
    unittest.main()
