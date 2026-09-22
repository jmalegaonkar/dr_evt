################################################################################
#         Copyright 2023 Lawrence Livermore National Security, LLC             #
#         See the top-level LICENSE file for details.                          #
#                                                                              #
#         SPDX-License-Identifier: MIT                                         #
################################################################################

"""Tests for the market loop, its outputs and the command line."""

import collections
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from dr_evt_market import (
    Decision,
    FirstPrice,
    MarketError,
    Mechanism,
    Vcg,
    federation,
    read_jobs,
    run,
    write_outputs,
)

_DATA = Path(__file__).with_name("data")
_ROOT = Path(__file__).resolve().parents[3]


class _BadMechanism(Mechanism):
    """Return an outside-batch or over-capacity decision."""

    name = "bad"

    def __init__(self, mode: str) -> None:
        self.mode = mode

    def decide(self, jobs, platforms, free_nodes) -> list[Decision]:
        """Return the selected invalid decision set."""
        if self.mode == "outside":
            return [Decision("outside", next(iter(platforms)), 0.0)]
        return [
            Decision(job.job_id, "corona", platforms["corona"].cost(job))
            for job in jobs[:2]
        ]


class MarketTests(unittest.TestCase):
    """Exercise routing, output stability, prefixes and guarantees."""

    def _fixture(self, root: Path, *, prefix: int = 32):
        jobs = read_jobs(_DATA / "jobs.csv")
        platforms = federation(root / "platforms", share=0.1)
        result = run(jobs, platforms, Vcg(), window_s=60, prefix=prefix)
        return jobs, result

    def test_fixture_routes_at_windows_and_rejects_two(self) -> None:
        """The fixture routes eighteen jobs and rejects only intake failures."""
        with tempfile.TemporaryDirectory() as directory:
            jobs, result = self._fixture(Path(directory))
        by_id = {job.job_id: job for job in jobs}
        self.assertEqual(len(result.routed), 18)
        self.assertEqual(
            [(row.job_id, row.reason, row.time_s) for row in result.rejected],
            [
                ("j000014", "oversize", 180),
                ("j000015", "unaffordable", 180),
            ],
        )
        for row in result.routed:
            self.assertEqual(row.begin_s, row.window_s)
            self.assertEqual(row.end_s, row.begin_s + by_id[row.job_id].limit_s)
        self.assertTrue(
            any(row.begin_s > row.submit_s for row in result.routed),
            "the contended fixture must make at least one job wait",
        )

    def test_routed_output_has_pinned_hash(self) -> None:
        """The fixture's routed ledger has its pinned digest."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, result = self._fixture(root)
            outputs = write_outputs(result, root / "out")
        # This changes only when the fixture or the model changes.
        self.assertEqual(
            outputs["sha256"],
            "3cca4e4cce7106a213753c9fff762c62a354b9e680b66195fef59465f51c4ae9",
        )

    def test_routed_output_is_byte_identical(self) -> None:
        """Two independent fixture runs produce the same routed ledger."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, first = self._fixture(root / "first")
            _, second = self._fixture(root / "second")
            first_paths = write_outputs(first, root / "first-out")
            second_paths = write_outputs(second, root / "second-out")
            self.assertEqual(
                Path(first_paths["routed"]).read_bytes(),
                Path(second_paths["routed"]).read_bytes(),
            )

    def test_prefix_two_limits_each_window(self) -> None:
        """A prefix of two drains the stream with at most two winners per window."""
        with tempfile.TemporaryDirectory() as directory:
            _, result = self._fixture(Path(directory), prefix=2)
        per_window = collections.Counter(row.window for row in result.routed)
        self.assertEqual(len(result.routed), 18)
        self.assertTrue(all(count <= 2 for count in per_window.values()))

    def test_invalid_mechanism_decisions_raise(self) -> None:
        """Outside-batch and over-capacity decisions violate the guarantee."""
        jobs = read_jobs(_DATA / "jobs.csv")
        for mode in ("outside", "capacity"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                platforms = federation(Path(directory) / "platforms", share=0.1)
                with self.assertRaises(MarketError):
                    run(jobs, platforms, _BadMechanism(mode))

    def test_first_price_revenue_is_routed_value(self) -> None:
        """Pay what you bid gives the routed surplus to the center."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jobs = read_jobs(_DATA / "jobs.csv")
            platforms = federation(root / "platforms", share=0.1)
            result = run(jobs, platforms, FirstPrice())
            paths = write_outputs(result, root / "out")
            summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))
        self.assertEqual(len(result.routed), 18)
        self.assertEqual(result.configuration["mechanism"], "firstprice")
        self.assertAlmostEqual(
            summary["revenue"], sum(row.value for row in result.routed)
        )

    def test_command_line_runs_and_prepares(self) -> None:
        """The command line runs the fixture and prepares two trace sources."""
        environment = os.environ.copy()
        environment["PYTHONPATH"] = os.pathsep.join(
            (str(_ROOT / "install/lib/python"), str(_ROOT / "python"))
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "dr_evt_market",
                    "run",
                    "--jobs",
                    str(_DATA / "jobs.csv"),
                    "--out",
                    str(root / "run"),
                    "--share",
                    "0.1",
                    "--platforms",
                    "corona,lassen,tioga,tuolumne",
                ],
                check=True,
                capture_output=True,
                text=True,
                env=environment,
            )
            self.assertIn("windows=9", completed.stdout.splitlines())
            self.assertIn("routed=18", completed.stdout.splitlines())
            self.assertIn("rejected=2", completed.stdout.splitlines())
            self.assertIn(
                "routed_sha256="
                "3cca4e4cce7106a213753c9fff762c62a354b9e680b66195fef59465f51c4ae9",
                completed.stdout.splitlines(),
            )
            first_price = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "dr_evt_market",
                    "run",
                    "--jobs",
                    str(_DATA / "jobs.csv"),
                    "--out",
                    str(root / "first-price"),
                    "--share",
                    "0.1",
                    "--mechanism",
                    "firstprice",
                ],
                check=True,
                capture_output=True,
                text=True,
                env=environment,
            )
            self.assertIn("routed=18", first_price.stdout.splitlines())

            prepared_path = root / "prepared.csv"
            prepared = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "dr_evt_market",
                    "prepare",
                    "--trace",
                    f"corona={_DATA / 'trace.csv'}",
                    "--trace",
                    f"tioga={_DATA / 'trace.csv'}",
                    "--out",
                    str(prepared_path),
                    "--start",
                    "1000",
                    "--hours",
                    "0.05",
                    "--per-platform",
                    "corona,lassen",
                ],
                check=True,
                capture_output=True,
                text=True,
                env=environment,
            )
            self.assertIn("kept=10", prepared.stdout.splitlines())
            jobs = read_jobs(prepared_path)
            self.assertEqual(len(jobs), 10)
            self.assertEqual({job.source for job in jobs}, {"corona", "tioga"})


if __name__ == "__main__":
    unittest.main()
