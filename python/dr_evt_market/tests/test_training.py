################################################################################
#         Copyright 2023 Lawrence Livermore National Security, LLC             #
#         See the top-level LICENSE file for details.                          #
#                                                                              #
#         SPDX-License-Identifier: MIT                                         #
################################################################################

"""Tests for recording market windows and training RegretFormer."""

import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from dr_evt_market import (
    Vcg,
    federation,
    read_jobs,
    record_windows,
    run,
    train_regretformer,
)

_DATA = Path(__file__).with_name("data")
_ROOT = Path(__file__).resolve().parents[3]
_TORCH = importlib.util.find_spec("torch") is not None


def _command(*arguments):
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(_ROOT / "install/lib/python"), str(_ROOT / "python"))
    )
    return subprocess.run(
        [sys.executable, "-m", "dr_evt_market", *arguments],
        capture_output=True,
        text=True,
        env=environment,
    )


class RecordWindowsTests(unittest.TestCase):
    """Record the windows a market run auctions."""

    def test_every_window_is_recorded(self) -> None:
        """Recording keeps each window's batch and free nodes, in order."""
        jobs = read_jobs(_DATA / "jobs.csv")
        with tempfile.TemporaryDirectory() as directory:
            platforms = federation(Path(directory) / "recorded", share=0.1)
            windows = record_windows(jobs, platforms)
            result = run(jobs, federation(Path(directory) / "run", share=0.1), Vcg())
        self.assertEqual(len(windows), 6)
        self.assertEqual(
            [job.job_id for job in windows[0][0]],
            ["j000001", "j000002", "j000003", "j000004", "j000005"],
        )
        self.assertEqual(
            windows[0][1], {name: platforms[name].exposed_nodes for name in platforms}
        )
        batched = {job.job_id for batch, _ in windows for job in batch}
        self.assertEqual(batched, {row.job_id for row in result.routed})


@unittest.skipUnless(_TORCH, "RegretFormer needs torch")
class TrainingTests(unittest.TestCase):
    """Train briefly and deploy the result."""

    def test_short_training_is_deterministic_and_deploys(self) -> None:
        """A seed fixes the trained network, which then passes the market's checks."""
        import torch

        jobs = read_jobs(_DATA / "jobs.csv")
        options = {"steps": 3, "batch": 4, "regret_jobs": 2, "points": 5, "seed": 1}
        with tempfile.TemporaryDirectory() as directory:
            platforms = federation(Path(directory) / "recorded", share=0.1)
            windows = record_windows(jobs, platforms)
            first, history = train_regretformer(windows, platforms, **options)
            second, _ = train_regretformer(windows, platforms, **options)
            result = run(jobs, federation(Path(directory) / "run", share=0.1), first)
        self.assertEqual(
            set(history), {"objective", "regret", "multiplier", "overbooked", "idle"}
        )
        self.assertTrue(all(len(values) == 3 for values in history.values()))
        weights = second.net.state_dict()
        for name, tensor in first.net.state_dict().items():
            self.assertTrue(torch.equal(tensor, weights[name]), name)
        self.assertEqual(len(result.routed) + len(result.waiting), len(jobs))

    def test_command_line_trains_a_network_that_runs(self) -> None:
        """The train command writes a checkpoint that the run command uses."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jobs = str(_DATA / "jobs.csv")
            network = str(root / "network.pt")
            trained = _command(
                "train",
                "--jobs",
                jobs,
                "--out",
                network,
                "--share",
                "0.1",
                "--steps",
                "2",
            )
            self.assertEqual(trained.returncode, 0, trained.stderr)
            self.assertIn("windows=6", trained.stdout.splitlines())
            ran = _command(
                "run",
                "--jobs",
                jobs,
                "--out",
                str(root / "run"),
                "--share",
                "0.1",
                "--mechanism",
                "regretformer",
                "--checkpoint",
                network,
            )
        self.assertEqual(ran.returncode, 0, ran.stderr)
        self.assertIn("routed=18", ran.stdout.splitlines())


if __name__ == "__main__":
    unittest.main()
