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

from federated_market import (
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
        (str(_ROOT / "install/lib/python"), str(_ROOT / "experimental"))
    )
    return subprocess.run(
        [sys.executable, "-m", "federated_market", *arguments],
        capture_output=True,
        text=True,
        env=environment,
    )


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
        """Harvested windows train a checkpoint that the run command uses."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jobs = str(_DATA / "jobs.csv")
            windows = str(root / "windows.jsonl.gz")
            network = str(root / "network.pt")
            harvested = _command(
                "harvest",
                "--jobs",
                jobs,
                "--out",
                windows,
                "--share",
                "0.1",
                "--mechanism",
                "vcg",
                "--mechanism",
                "firstfit",
            )
            self.assertEqual(harvested.returncode, 0, harvested.stderr)
            self.assertIn("windows:vcg=6", harvested.stdout.splitlines())
            trained = _command(
                "train", "--windows", windows, "--out", network, "--steps", "2"
            )
            self.assertEqual(trained.returncode, 0, trained.stderr)
            self.assertIn(
                f"windows={harvested.stdout.split()[0].split('=')[1]}",
                trained.stdout.splitlines(),
            )
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
        self.assertIn("routed=19", ran.stdout.splitlines())


if __name__ == "__main__":
    unittest.main()
