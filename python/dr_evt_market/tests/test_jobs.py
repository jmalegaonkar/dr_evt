################################################################################
#         Copyright 2023 Lawrence Livermore National Security, LLC             #
#         See the top-level LICENSE file for details.                          #
#                                                                              #
#         SPDX-License-Identifier: MIT                                         #
################################################################################

"""Tests for the job stream and trace preparation."""

import tempfile
import unittest
from pathlib import Path

from dr_evt_market import Job, prepare, read_jobs, write_jobs

_DATA = Path(__file__).with_name("data")


class JobTests(unittest.TestCase):
    """Exercise job input and merged trace preparation."""

    def test_read_jobs_fixture(self) -> None:
        """The fixture preserves order, tags and oversize demand."""
        jobs = read_jobs(_DATA / "jobs.csv")
        self.assertEqual(len(jobs), 20)
        self.assertEqual(jobs[0].job_id, "j000001")
        self.assertEqual(jobs[3].requires, frozenset({"gpu", "nvidia"}))
        self.assertEqual(jobs[13].num_nodes, 120)
        self.assertEqual(
            (jobs[0].source, jobs[0].user, jobs[0].persona),
            ("tioga", "1", "tier"),
        )

    def test_job_freezes_requirements_and_duplicate_ids_fail(self) -> None:
        """Job freezes tags, and the reader rejects duplicate identifiers."""
        job = Job("j1", -1, 0, 0, -1.0, {"gpu"}, 0)
        self.assertEqual(job.requires, frozenset({"gpu"}))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "jobs.csv"
            path.write_text(
                "job_id,job_submit_time,num_nodes,time_limit,bid\n"
                "same,0,1,1,1\n"
                "same,1,1,1,1\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                read_jobs(path)

    def test_prepare_lc_interval_and_summary(self) -> None:
        """LC preparation filters one interval and counts each drop reason."""
        jobs, summary = prepare(
            {"tioga": _DATA / "trace.csv"},
            home_prices={"tioga": 6.0},
            start=1000,
            hours=0.05,
            seed=3,
        )
        self.assertEqual(
            summary,
            {
                "read": 12,
                "kept": 5,
                "no_nodes": 1,
                "no_limit": 1,
                "no_start": 2,
                "bad_runtime": 1,
                "outside_interval": 2,
                "kept:tioga": 5,
            },
        )
        self.assertEqual([job.submit_s for job in jobs], [0, 35, 85, 115, 174])
        self.assertEqual(jobs[0].limit_s, 40)
        self.assertEqual(jobs[0].requested_s, 120)
        self.assertEqual(jobs[0].source, "tioga")
        self.assertTrue(all(job.requires == {"gpu"} for job in jobs))

    def test_sources_merge_on_one_origin(self) -> None:
        """Sources share one clock and contribute to persona identity."""
        traces = {"beta": _DATA / "trace.csv", "alpha": _DATA / "trace.csv"}
        jobs, summary = prepare(
            traces,
            home_prices={"alpha": 2.0, "beta": 3.0},
            start=1000,
            hours=0.05,
            seed=0,
        )
        self.assertEqual(summary["kept:alpha"], 5)
        self.assertEqual(summary["kept:beta"], 5)
        self.assertEqual([job.submit_s for job in jobs[:2]], [0, 0])
        self.assertEqual([job.source for job in jobs[:2]], ["alpha", "beta"])
        self.assertEqual([job.user for job in jobs[:2]], ["2", "2"])
        self.assertEqual([job.persona for job in jobs[:2]], ["value", "tier"])

    def test_prepare_is_deterministic_and_round_trips(self) -> None:
        """A seed fixes output bytes, and written rows read back as jobs."""
        traces = {"tioga": _DATA / "trace.csv"}
        options = {"home_prices": {"tioga": 6.0}, "start": 1000, "hours": 0.05}
        jobs, _ = prepare(traces, seed=4, **options)
        same_jobs, _ = prepare(traces, seed=4, **options)
        other_jobs, _ = prepare(traces, seed=5, **options)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, second, other = root / "a.csv", root / "b.csv", root / "c.csv"
            write_jobs(jobs, first)
            write_jobs(same_jobs, second)
            write_jobs(other_jobs, other)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertNotEqual(first.read_bytes(), other.read_bytes())
            self.assertEqual(read_jobs(first), jobs)

    def test_persona_and_price_are_hash_seeded(self) -> None:
        """A fixed seed, source, user and job ID give a pinned price."""
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "simple.csv"
            trace.write_text(
                "job_submit_time,num_nodes,time_limit,user\n" "100,2,60,fixed-user\n",
                encoding="utf-8",
            )
            jobs, _ = prepare(
                {"tioga": trace},
                home_prices={"tioga": 6.0},
                trace_format="simple",
                seed=8,
            )
        self.assertEqual(jobs[0].persona, "value")
        self.assertEqual(jobs[0].bid, 27.7068)

    def test_simple_times_are_floored_sorted_and_shifted(self) -> None:
        """Simple fractional times use the earliest floored submit as zero."""
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "simple.csv"
            trace.write_text(
                "job_submit_time,num_nodes,time_limit,actual_run_time\n"
                "12.9,2,5.9,3.9\n"
                "10.8,1,4.2,2.7\n",
                encoding="utf-8",
            )
            jobs, summary = prepare(
                {"simple": trace},
                home_prices={"simple": 2.0},
                trace_format="simple",
            )
        self.assertEqual([job.submit_s for job in jobs], [0, 2])
        self.assertEqual([job.limit_s for job in jobs], [2, 3])
        self.assertEqual([job.runtime_s for job in jobs], [2, 3])
        self.assertEqual([job.requested_s for job in jobs], [4, 5])
        self.assertEqual(summary["kept:simple"], 2)
        self.assertEqual(
            sum(
                summary[key]
                for key in (
                    "no_nodes",
                    "no_limit",
                    "no_start",
                    "bad_runtime",
                    "outside_interval",
                )
            ),
            0,
        )

    def test_per_platform_fixture_bids_and_precedence(self) -> None:
        """Mapped bids select exact platforms and override a scalar bid."""
        jobs = read_jobs(_DATA / "jobs.csv")
        self.assertIsInstance(jobs[0].bid, float)
        self.assertEqual(jobs[0].price("anything"), 2.2)
        self.assertEqual(jobs[3].bid, {"lassen": 3.2, "tuolumne": 8.5})
        self.assertIsNone(jobs[3].price("corona"))
        self.assertEqual(jobs[9].price("corona"), 20.0)
        self.assertEqual(jobs[14].bid, {"corona": 1.5})

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "both.csv"
            path.write_text(
                "job_id,job_submit_time,num_nodes,time_limit,bid,bid:corona\n"
                "mixed,0,1,1,9.0,2.5\n",
                encoding="utf-8",
            )
            self.assertEqual(read_jobs(path)[0].bid, {"corona": 2.5})

    def test_prepare_per_platform_bids_round_trip(self) -> None:
        """Prepared mapped bids are deterministic and retain dynamic columns."""
        traces = {"tioga": _DATA / "trace.csv"}
        jobs, _ = prepare(
            traces,
            home_prices={"tioga": 6.0},
            start=1000,
            hours=0.05,
            seed=4,
            per_platform=("corona", "lassen"),
        )
        again, _ = prepare(
            traces,
            home_prices={"tioga": 6.0},
            start=1000,
            hours=0.05,
            seed=4,
            per_platform=("corona", "lassen"),
        )
        self.assertTrue(all(list(job.bid) == ["corona", "lassen"] for job in jobs))
        self.assertEqual(jobs[0].bid, {"corona": 10.8524, "lassen": 11.1589})
        self.assertEqual(jobs, again)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mapped.csv"
            write_jobs(jobs, path)
            header = path.read_text(encoding="utf-8").splitlines()[0]
            self.assertIn("bid,bid:corona,bid:lassen,requires", header)
            self.assertEqual(read_jobs(path), jobs)

    def test_limit_can_follow_runtime_or_request(self) -> None:
        """Preparation carries the request and selects either limit source."""
        traces = {"tioga": _DATA / "trace.csv"}
        options = {
            "home_prices": {"tioga": 6.0},
            "start": 1000,
            "hours": 0.05,
        }
        runtime_jobs, _ = prepare(traces, **options)
        request_jobs, _ = prepare(traces, limit_from="request", **options)
        self.assertEqual(runtime_jobs[0].limit_s, 40)
        self.assertEqual(request_jobs[0].limit_s, 120)
        self.assertEqual(runtime_jobs[0].requested_s, request_jobs[0].requested_s)
        self.assertEqual(runtime_jobs[0].requested_s, 120)


if __name__ == "__main__":
    unittest.main()
