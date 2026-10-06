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
    FirstFit,
    Vcg,
    federation,
    harvest,
    read_jobs,
    read_windows,
    record_windows,
    run,
    train_regretformer,
    write_windows,
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


class HarvestTests(unittest.TestCase):
    """Harvest windows under several mechanisms, and save and load them."""

    def test_windows_survive_a_file(self) -> None:
        """Every harvested window comes back with its tags, free nodes and jobs."""
        jobs = read_jobs(_DATA / "jobs.csv")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            windows = harvest(
                {"fixture": jobs}, {"vcg": Vcg, "firstfit": FirstFit}, share=0.1
            )
            platforms = federation(root / "facts", share=0.1)
            write_windows(root / "windows.jsonl.gz", windows, platforms)
            loaded_platforms, loaded = read_windows(
                [root / "windows.jsonl.gz"], root / "read"
            )
        self.assertEqual(
            {window["mechanism"] for window in windows}, {"vcg", "firstfit"}
        )
        self.assertEqual(loaded, windows)
        self.assertEqual(
            {name: item.exposed_nodes for name, item in loaded_platforms.items()},
            {name: item.exposed_nodes for name, item in platforms.items()},
        )

    def test_platform_facts_need_no_simulation(self) -> None:
        """A federation's prices, sizes and costs load without dr_evt."""
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; sys.modules['dr_evt'] = None\n"
                "from dr_evt_market import Job, federation\n"
                "platforms = federation('unused', share=0.1)\n"
                "job = Job('a', 0, 2, 3600, 1.0)\n"
                "print(platforms['tuolumne'].exposed_nodes,"
                " round(platforms['dane'].cost(job), 4))",
            ],
            capture_output=True,
            text=True,
            env={
                **os.environ,
                "PYTHONPATH": os.pathsep.join(
                    (str(_ROOT / "install/lib/python"), str(_ROOT / "python"))
                ),
            },
        )
        self.assertEqual(completed.stdout.split(), ["115", "0.4181"], completed.stderr)


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
