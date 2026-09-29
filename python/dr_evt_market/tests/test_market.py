################################################################################
#         Copyright 2023 Lawrence Livermore National Security, LLC             #
#         See the top-level LICENSE file for details.                          #
#                                                                              #
#         SPDX-License-Identifier: MIT                                         #
################################################################################

"""Tests for the market loop, its outputs and the command line."""

import collections
import csv
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from dr_evt_market.cli import _parser
from dr_evt_market import (
    Decision,
    FirstFit,
    FirstPrice,
    Job,
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


class _Decline(Mechanism):
    """Place nothing, which the market forbids while a job fits."""

    name = "decline"

    def decide(self, jobs, platforms, free_nodes) -> list[Decision]:
        """Return no decisions."""
        return []


class MarketTests(unittest.TestCase):
    """Exercise routing, output stability, prefixes and guarantees."""

    def _fixture(self, root: Path, *, prefix: int = 32):
        jobs = read_jobs(_DATA / "jobs.csv")
        platforms = federation(root / "platforms", share=0.1)
        result = run(jobs, platforms, Vcg(), window_s=60, prefix=prefix)
        return jobs, result

    def test_command_line_parses_a_share_per_platform(self) -> None:
        """The share option accepts a comma-separated platform mapping."""
        arguments = _parser().parse_args(
            [
                "run",
                "--jobs",
                "jobs.csv",
                "--out",
                "results",
                "--share",
                "corona=0.2,dane=0.5,matrix=0.5,tioga=0.5,tuolumne=0.05",
            ]
        )
        self.assertEqual(
            arguments.share,
            {
                "corona": 0.2,
                "dane": 0.5,
                "matrix": 0.5,
                "tioga": 0.5,
                "tuolumne": 0.05,
            },
        )

    def test_fixture_routes_at_windows_and_keeps_two_waiting(self) -> None:
        """The fixture routes eighteen jobs; the two that cannot run still wait."""
        with tempfile.TemporaryDirectory() as directory:
            jobs, result = self._fixture(Path(directory))
        by_id = {job.job_id: job for job in jobs}
        self.assertEqual(len(result.routed), 18)
        self.assertEqual(
            [(row.job_id, row.reason, row.submit_s) for row in result.waiting],
            [
                ("j000014", "oversize", 180),
                ("j000015", "unaffordable", 180),
            ],
        )
        self.assertEqual(
            [row.job_id for row in result.routed if row.window == 0],
            ["j000001", "j000002", "j000003", "j000005"],
        )
        self.assertEqual(
            {row.platform for row in result.routed},
            {"corona", "dane", "matrix", "tioga", "tuolumne"},
        )
        for row in result.routed:
            self.assertEqual(row.begin_s, row.window_s)
            job = by_id[row.job_id]
            hardware = "gpu" if "gpu" in job.requires else "cpu"
            speed = result.configuration["platforms"][row.platform]["speed"][hardware]
            self.assertEqual(
                row.end_s,
                row.begin_s + max(1, math.ceil(job.limit_s / speed)),
            )
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
            "8f63322a780326174b55220d7d0695cf35017ec7ab658112e433495acc204f8b",
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

    def test_service_has_one_row_per_routed_source(self) -> None:
        """The service table covers every routed source and the aggregate."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jobs, result = self._fixture(root)
            paths = write_outputs(result, root / "out")
            with Path(paths["service"]).open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))
        routed = {row.job_id for row in result.routed}
        communities = sorted({job.source for job in jobs if job.job_id in routed})
        self.assertEqual([row["community"] for row in rows], [*communities, "all"])
        self.assertEqual(set(summary["service"]), {*communities, "all"})

    def test_all_service_matches_the_routed_mean(self) -> None:
        """The all row averages every routed job rather than communities."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jobs, result = self._fixture(root)
            paths = write_outputs(result, root / "out")
            summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))
        by_id = {job.job_id: job for job in jobs}
        waits = [row.begin_s - row.submit_s for row in result.routed]
        weights = [
            by_id[row.job_id].num_nodes * (row.end_s - row.begin_s) / 3600
            for row in result.routed
        ]
        service = summary["service"]["all"]
        self.assertEqual(service["count"], len(result.routed))
        self.assertAlmostEqual(service["mean_wait_s"], sum(waits) / len(waits))
        self.assertAlmostEqual(
            service["node_hour_weighted_mean_wait_s"],
            sum(wait * weight for wait, weight in zip(waits, weights)) / sum(weights),
        )

    def test_service_uses_bounded_slowdown_with_a_ten_second_floor(self) -> None:
        """One short job has a hand-checked bounded slowdown."""
        job = Job("short", 7, 2, 5, 2.0, {"gpu"})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            platforms = federation(root / "platforms", share=0.1, names=("corona",))
            result = run([job], platforms, Vcg(), window_s=60)
            paths = write_outputs(result, root / "out")
            summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))
        self.assertEqual(result.routed[0].begin_s - job.submit_s, 53)
        self.assertEqual(result.routed[0].end_s - result.routed[0].begin_s, 5)
        self.assertAlmostEqual(summary["service"][""]["mean_bounded_slowdown"], 5.8)

    def test_prefix_two_limits_each_window(self) -> None:
        """A prefix of two drains the stream with at most two winners per window."""
        with tempfile.TemporaryDirectory() as directory:
            _, result = self._fixture(Path(directory), prefix=2)
        per_window = collections.Counter(row.window for row in result.routed)
        self.assertEqual(len(result.routed), 18)
        self.assertTrue(all(count <= 2 for count in per_window.values()))

    def test_wide_head_job_does_not_block_prefix(self) -> None:
        """A temporarily wide head job does not hide a placeable job."""
        jobs = [
            Job("running", 0, 10, 120, 2.0, {"gpu"}),
            Job("wide", 0, 3, 60, 2.0, {"gpu"}),
            Job("narrow", 0, 2, 60, 2.0, {"gpu"}),
        ]
        with tempfile.TemporaryDirectory() as directory:
            platforms = federation(
                Path(directory) / "platforms", share=0.1, names=("corona",)
            )
            result = run(jobs, platforms, Vcg(), window_s=60, prefix=1)
        self.assertEqual(
            {row.job_id: row.begin_s for row in result.routed},
            {"running": 0, "wide": 120, "narrow": 60},
        )

    def test_invalid_mechanism_decisions_raise(self) -> None:
        """Outside-batch and over-capacity decisions violate the guarantee."""
        jobs = read_jobs(_DATA / "jobs.csv")
        for mode in ("outside", "capacity"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                platforms = federation(Path(directory) / "platforms", share=0.1)
                with self.assertRaises(MarketError):
                    run(jobs, platforms, _BadMechanism(mode))

    def test_cpu_only_federation_reports_gpu_hardware_waiting(self) -> None:
        """A GPU job waits for hardware in a CPU-only federation."""
        jobs = [Job("gpu", 0, 1, 60, 1.0, {"gpu"})]
        with tempfile.TemporaryDirectory() as directory:
            platforms = federation(Path(directory), share=0.1, names=("dane",))
            result = run(jobs, platforms, Vcg())
        self.assertEqual(
            [(row.job_id, row.reason) for row in result.waiting],
            [("gpu", "hardware")],
        )
        self.assertFalse(result.routed)

    def test_leaving_a_job_waiting_on_free_nodes_raises(self) -> None:
        """A mechanism must place every batch job that fits the nodes left over."""
        jobs = [Job("fits", 0, 1, 60, 3.0, {"gpu"})]
        with tempfile.TemporaryDirectory() as directory:
            platforms = federation(Path(directory) / "platforms", share=0.1)
            with self.assertRaisesRegex(MarketError, "fits: left waiting"):
                run(jobs, platforms, _Decline())

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

    def test_first_fit_routes_every_fixture_job_that_vcg_routes(self) -> None:
        """The no-market baseline covers every fixture winner under VCG."""
        jobs = read_jobs(_DATA / "jobs.csv")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            vcg = run(
                jobs,
                federation(root / "vcg", share=0.1),
                Vcg(),
            )
            first_fit = run(
                jobs,
                federation(root / "firstfit", share=0.1),
                FirstFit(),
            )
        vcg_jobs = {row.job_id for row in vcg.routed}
        first_fit_jobs = {row.job_id for row in first_fit.routed}
        self.assertLessEqual(vcg_jobs, first_fit_jobs)
        self.assertIn("j000015", first_fit_jobs)

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
                    "corona,dane,matrix,tioga,tuolumne",
                ],
                check=True,
                capture_output=True,
                text=True,
                env=environment,
            )
            self.assertIn("windows=6", completed.stdout.splitlines())
            self.assertIn("routed=18", completed.stdout.splitlines())
            self.assertIn("waiting=2", completed.stdout.splitlines())
            self.assertIn(
                "routed_sha256="
                "8f63322a780326174b55220d7d0695cf35017ec7ab658112e433495acc204f8b",
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
                    "corona,matrix",
                    "--gpu-fraction",
                    "0",
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
            self.assertTrue(all(not job.requires for job in jobs))


if __name__ == "__main__":
    unittest.main()
