################################################################################
#         Copyright 2023 Lawrence Livermore National Security, LLC             #
#         See the top-level LICENSE file for details.                          #
#                                                                              #
#         SPDX-License-Identifier: MIT                                         #
################################################################################

"""Tests for the job stream and trace preparation."""

import csv
import tempfile
import unittest
from pathlib import Path

from dr_evt_market import Job, prepare, read_jobs, write_jobs


_DATA = Path(__file__).with_name("data")


class JobTests(unittest.TestCase):
    """Exercise job validation, CSV input and trace preparation."""

    def test_read_jobs_fixture(self) -> None:
        """The jobs fixture preserves order, tags and oversize demand."""
        jobs = read_jobs(_DATA / "jobs.csv")
        self.assertEqual(len(jobs), 20)
        self.assertEqual(jobs[0].job_id, "j000001")
        self.assertEqual(jobs[3].requires, frozenset({"gpu"}))
        self.assertEqual(jobs[14].num_nodes, 100)
        self.assertFalse(hasattr(jobs[0], "user"))

    def test_job_validation(self) -> None:
        """Invalid scalar fields fail and requirements become immutable."""
        base = dict(job_id="j1", submit_s=0, num_nodes=1, limit_s=1, bid=1.0)
        invalid = (
            {"job_id": ""}, {"submit_s": -1}, {"submit_s": True},
            {"num_nodes": 0}, {"num_nodes": True}, {"limit_s": 0},
            {"limit_s": True}, {"bid": -1.0}, {"bid": float("nan")},
            {"runtime_s": 0}, {"runtime_s": True},
        )
        for changes in invalid:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                Job(**(base | changes))
        job = Job(**base, requires={"gpu"})
        self.assertEqual(job.requires, frozenset({"gpu"}))

    def test_prepare_interval_shift_and_summary(self) -> None:
        """Preparation keeps one interval, shifts it and counts every drop."""
        jobs, summary, rows = prepare(
            _DATA / "trace.csv", start=1000, hours=0.05, seed=3,
            requires="gpu nvidia",
        )
        self.assertEqual(summary, {
            "read": 12, "kept": 6, "no_nodes": 1, "no_limit": 1,
            "bad_runtime": 1, "outside_interval": 3,
        })
        self.assertEqual([job.submit_s for job in jobs], [0, 35, 45, 85, 115, 174])
        self.assertTrue(all(job.requires == {"gpu", "nvidia"} for job in jobs))
        self.assertIsNone(jobs[-1].runtime_s)
        self.assertEqual(len(rows), len(jobs))

    def test_prepare_is_deterministic_and_round_trips(self) -> None:
        """A seed fixes output bytes, and written rows read back as the jobs."""
        jobs, _, rows = prepare(
            _DATA / "trace.csv", start=1000, hours=0.05, seed=4
        )
        _, _, same_rows = prepare(
            _DATA / "trace.csv", start=1000, hours=0.05, seed=4
        )
        _, _, other_rows = prepare(
            _DATA / "trace.csv", start=1000, hours=0.05, seed=5
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, second, other = root / "a.csv", root / "b.csv", root / "c.csv"
            write_jobs(rows, first)
            write_jobs(same_rows, second)
            write_jobs(other_rows, other)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertNotEqual(first.read_bytes(), other.read_bytes())
            self.assertEqual(read_jobs(first), jobs)

    def test_persona_and_multiplier_are_hash_seeded(self) -> None:
        """A fixed seed, user and job ID produce a pinned value multiplier."""
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "trace.csv"
            trace.write_text(
                "job_submit_time,num_nodes,time_limit,user\n"
                "100,2,60,fixed-user\n",
                encoding="utf-8",
            )
            jobs, _, rows = prepare(trace, seed=7)
        self.assertEqual(rows[0]["persona"], "value")
        self.assertEqual(jobs[0].bid, 2.4199)

    def test_lassen_positions_supply_runtime(self) -> None:
        """Lassen input uses its documented fixed positions and numeric times."""
        header = [f"column_{index}" for index in range(1, 34)]

        def row(nodes: int, begin: int, end: int, submit: int, limit: int):
            values = [""] * 33
            for index, value in ((10, nodes), (22, begin), (23, end),
                                 (28, submit), (31, limit)):
                values[index] = value
            return values

        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "lassen.csv"
            with trace.open("w", newline="", encoding="utf-8") as stream:
                csv.writer(stream, lineterminator="\n").writerows([
                    header, row(4, 105, 145, 100, 60),
                    row(8, 220, 270, 200, 90),
                ])
            jobs, summary, _ = prepare(trace, trace_format="lassen")
        self.assertEqual([job.runtime_s for job in jobs], [40, 50])
        self.assertEqual([job.submit_s for job in jobs], [0, 100])
        self.assertEqual(summary["kept"], 2)


if __name__ == "__main__":
    unittest.main()
