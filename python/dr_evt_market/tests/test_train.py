################################################################################
#         Copyright 2023 Lawrence Livermore National Security, LLC             #
#         See the top-level LICENSE file for details.                          #
#                                                                              #
#         SPDX-License-Identifier: MIT                                         #
################################################################################

"""Tests for training on synthetic and harvested windows."""

from pathlib import Path
import tempfile
import unittest

try:
    import torch

    from dr_evt_market.learned.harvest import harvest_structures
    from dr_evt_market.learned.synthetic import synthetic_structures
    from dr_evt_market.learned.train import TrainConfig, Trainer
    from dr_evt_market.mechanisms import RegretFormer
except ModuleNotFoundError:
    torch = None

from dr_evt_market import (
    Controller,
    InProcessPlatform,
    read_jobs,
    read_platforms,
    write_outputs,
)
from dr_evt_market.mechanisms import Vcg, build_window, validate_decisions

_DATA = Path(__file__).with_name("data")


@unittest.skipIf(torch is None, "torch is not installed")
class TrainerTests(unittest.TestCase):
    """A short training cycle and the harvest path."""

    def test_two_epochs_train_save_and_reload(self) -> None:
        """Synthetic windows train, evaluate, save and deploy."""
        trainer = Trainer(
            TrainConfig(
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
            ),
            synthetic_structures(20, seed=37),
        )
        result = trainer.train()
        self.assertEqual(len(result["epochs"]), 2)
        self.assertIn("vcg_welfare_coverage", result["evaluation"])
        platforms = {p.name: p for p in read_platforms(_DATA / "market_platforms.csv")}
        window = build_window(
            0,
            0,
            read_jobs(_DATA / "market_jobs.csv"),
            platforms,
            {n: p.total_nodes for n, p in platforms.items()},
        )
        with tempfile.TemporaryDirectory(prefix="dr_evt_market_train_") as directory:
            mechanism = RegretFormer(trainer.save(Path(directory) / "trained.pt"))
        decisions = mechanism.decide(window)
        self.assertEqual(validate_decisions(window, decisions), (decisions, []))

    def test_logged_windows_can_be_harvested(self) -> None:
        """A run with a log directory yields structures for training."""
        platforms = {p.name: p for p in read_platforms(_DATA / "market_platforms.csv")}
        with tempfile.TemporaryDirectory(prefix="dr_evt_market_harvest_") as directory:
            root = Path(directory)
            sessions = {
                n: InProcessPlatform(n, p.total_nodes, root / n)
                for n, p in platforms.items()
            }
            report = Controller(
                sessions,
                platforms,
                Vcg(),
                read_jobs(_DATA / "market_jobs.csv"),
                window_s=60,
                log_dir=root / "out",
            ).run()
            write_outputs(report, root / "out")
            structures = harvest_structures(root / "out")
        self.assertEqual(len(structures), sum(1 for w in report.windows if w.queued))


if __name__ == "__main__":
    unittest.main()
