################################################################################
#         Copyright 2023 Lawrence Livermore National Security, LLC             #
#         See the top-level LICENSE file for details.                          #
#                                                                              #
#         SPDX-License-Identifier: MIT                                         #
################################################################################

"""Tests for the RegretFormer mechanism."""

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from dr_evt_market import RegretFormer, federation, offers, read_jobs, run

_DATA = Path(__file__).with_name("data")
_ROOT = Path(__file__).resolve().parents[3]
_TORCH = importlib.util.find_spec("torch") is not None


def _command(*arguments):
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(_ROOT / "install/lib/python"), str(_ROOT / "python"))
    )
    return subprocess.run(
        [sys.executable, *arguments], capture_output=True, text=True, env=environment
    )


class RegretFormerWithoutTorchTests(unittest.TestCase):
    """Check what holds whether or not torch is installed."""

    def test_package_imports_without_torch(self) -> None:
        """The package and its mechanism table load with torch blocked."""
        completed = _command(
            "-c",
            "import sys; sys.modules['torch'] = None; import dr_evt_market; "
            "print(dr_evt_market.MECHANISMS['regretformer'].name)",
        )
        self.assertEqual(completed.stdout.strip(), "regretformer", completed.stderr)

    def test_command_line_needs_a_checkpoint(self) -> None:
        """Running RegretFormer without a checkpoint is an error."""
        with tempfile.TemporaryDirectory() as directory:
            completed = _command(
                "-m",
                "dr_evt_market",
                "run",
                "--jobs",
                str(_DATA / "jobs.csv"),
                "--out",
                directory,
                "--mechanism",
                "regretformer",
            )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("--checkpoint", completed.stderr)


@unittest.skipUnless(_TORCH, "RegretFormer needs torch")
class RegretFormerTests(unittest.TestCase):
    """Run the network through the market."""

    def test_untrained_networks_route_every_admitted_job(self) -> None:
        """Whatever its weights, the network routes every job that can run."""
        jobs = read_jobs(_DATA / "jobs.csv")
        for seed in range(6):
            with self.subTest(seed=seed), tempfile.TemporaryDirectory() as directory:
                platforms = federation(Path(directory), share=0.1)
                result = run(jobs, platforms, RegretFormer(seed=seed))
                self.assertEqual(len(result.routed), 18)
                self.assertEqual(
                    {row.reason for row in result.waiting},
                    {"oversize", "unaffordable"},
                )
                for row in result.routed:
                    self.assertLessEqual(row.cost, row.charge)
                    self.assertLessEqual(row.charge, row.value)

    def test_a_platform_with_no_nodes_leaves_the_network_finite(self) -> None:
        """A platform with no nodes scales by one and receives no job."""
        jobs = [job for job in read_jobs(_DATA / "jobs.csv") if job.submit_s == 0]
        with tempfile.TemporaryDirectory() as directory:
            platforms = federation(Path(directory), share={"corona": 0.0, "dane": 0.1})
            free = {name: item.exposed_nodes for name, item in platforms.items()}
            decisions = RegretFormer(seed=0).decide(jobs, platforms, free)
        self.assertTrue(decisions)
        self.assertNotIn("corona", {decision.platform for decision in decisions})

    def test_a_job_stays_queued_only_without_room(self) -> None:
        """After a contended window, no queued job fits the nodes that are left."""
        jobs = [job for job in read_jobs(_DATA / "jobs.csv") if job.submit_s == 0]
        by_id = {job.job_id: job for job in jobs}
        with tempfile.TemporaryDirectory() as directory:
            platforms = federation(Path(directory), share=0.1)
            free = {name: item.exposed_nodes for name, item in platforms.items()}
            for seed in range(6):
                decisions = RegretFormer(seed=seed).decide(jobs, platforms, free)
                left = dict(free)
                for decision in decisions:
                    left[decision.platform] -= by_id[decision.job_id].num_nodes
                placed = {decision.job_id for decision in decisions}
                with self.subTest(seed=seed):
                    self.assertLess(len(placed), len(jobs))
                    for job in jobs:
                        if job.job_id not in placed:
                            self.assertFalse(offers(job, platforms, left))

    def test_seed_and_checkpoint_fix_the_decisions(self) -> None:
        """One seed decides the same twice, and so does its saved network."""
        jobs = [job for job in read_jobs(_DATA / "jobs.csv") if job.submit_s == 0]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            platforms = federation(root / "platforms", share=0.1)
            free = {name: item.exposed_nodes for name, item in platforms.items()}
            mechanism = RegretFormer(seed=4)
            decisions = mechanism.decide(jobs, platforms, free)
            mechanism.save(root / "network.pt", note="untrained")
            again = RegretFormer(seed=4).decide(jobs, platforms, free)
            loaded = RegretFormer(root / "network.pt").decide(jobs, platforms, free)
        self.assertTrue(decisions)
        self.assertEqual(again, decisions)
        self.assertEqual(loaded, decisions)

    def test_command_line_runs_a_checkpoint(self) -> None:
        """The run command loads a saved network and routes the fixture."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            RegretFormer(seed=4).save(root / "network.pt")
            completed = _command(
                "-m",
                "dr_evt_market",
                "run",
                "--jobs",
                str(_DATA / "jobs.csv"),
                "--out",
                str(root / "run"),
                "--share",
                "0.1",
                "--mechanism",
                "regretformer",
                "--checkpoint",
                str(root / "network.pt"),
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            summary = json.loads((root / "run" / "summary.json").read_text())
        self.assertEqual(summary["configuration"]["mechanism"], "regretformer")
        self.assertEqual(summary["routed"] + summary["waiting"], 20)


if __name__ == "__main__":
    unittest.main()
