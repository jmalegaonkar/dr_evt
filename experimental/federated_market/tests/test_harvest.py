################################################################################
#         Copyright 2023 Lawrence Livermore National Security, LLC             #
#         See the top-level LICENSE file for details.                          #
#                                                                              #
#         SPDX-License-Identifier: MIT                                         #
################################################################################

"""Tests for recording market windows and the files that carry them to training."""

import tempfile
import unittest
from pathlib import Path

from federated_market import (
    FirstFit,
    Vcg,
    federation,
    harvest,
    read_jobs,
    read_windows,
    record_windows,
    run,
    write_windows,
)

_DATA = Path(__file__).with_name("data")


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


if __name__ == "__main__":
    unittest.main()
