################################################################################
#         Copyright 2023 Lawrence Livermore National Security, LLC             #
#         See the top-level LICENSE file for details.                          #
#                                                                              #
#         SPDX-License-Identifier: MIT                                         #
################################################################################

"""Tests for the job stream and trace preparation."""

import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from dr_evt_market import Job, prepare, read_jobs, write_jobs
from dr_evt_market.cli import main

_DATA = Path(__file__).with_name("data")
_REFERENCE_PRICE = 1.234


class JobTests(unittest.TestCase):
    """Exercise job input and merged trace preparation."""

    def test_read_jobs_fixture(self) -> None:
        """The fixture preserves order, tags and oversize demand."""
        jobs = read_jobs(_DATA / "jobs.csv")
        self.assertEqual(len(jobs), 20)
        self.assertEqual(jobs[0].job_id, "j000001")
        self.assertEqual(jobs[3].requires, frozenset({"gpu"}))
        self.assertEqual(jobs[13].num_nodes, 155)
        self.assertEqual(
            (jobs[0].source, jobs[0].user, jobs[0].persona),
            ("corona", "1", "tier"),
        )
        self.assertEqual(
            {job.source for job in jobs},
            {"corona", "dane", "matrix", "tioga", "tuolumne"},
        )
        self.assertEqual(sum(isinstance(job.bid, dict) for job in jobs), 3)

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
            reference_price=_REFERENCE_PRICE,
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
        self.assertEqual(
            [job.requires for job in jobs],
            [
                frozenset(),
                frozenset({"gpu"}),
                frozenset({"gpu"}),
                frozenset({"gpu"}),
                frozenset(),
            ],
        )

    def test_sources_merge_on_one_origin(self) -> None:
        """Sources share one clock and contribute to persona identity."""
        traces = {"beta": _DATA / "trace.csv", "alpha": _DATA / "trace.csv"}
        jobs, summary = prepare(
            traces,
            reference_price=_REFERENCE_PRICE,
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
        options = {
            "reference_price": _REFERENCE_PRICE,
            "start": 1000,
            "hours": 0.05,
        }
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
        """A fixed seed, source, user and trace row give a pinned price."""
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "simple.csv"
            trace.write_text(
                "job_submit_time,num_nodes,time_limit,user\n" "100,2,60,fixed-user\n",
                encoding="utf-8",
            )
            jobs, _ = prepare(
                {"tioga": trace},
                reference_price=_REFERENCE_PRICE,
                trace_format="simple",
                seed=8,
            )
        self.assertEqual(jobs[0].persona, "value")
        self.assertEqual(jobs[0].bid, 4.7308)

    def test_gpu_fraction_and_requires_override(self) -> None:
        """GPU draws are per job, and an explicit requirement overrides them."""
        traces = {"tioga": _DATA / "trace.csv"}
        options = {
            "reference_price": _REFERENCE_PRICE,
            "start": 1000,
            "hours": 0.05,
            "seed": 4,
        }
        cpu_jobs, _ = prepare(traces, gpu_fraction=0.0, **options)
        gpu_jobs, _ = prepare(traces, gpu_fraction=1.0, **options)
        overridden, _ = prepare(traces, gpu_fraction=0.0, requires="gpu", **options)
        self.assertTrue(all(not job.requires for job in cpu_jobs))
        self.assertTrue(all(job.requires == {"gpu"} for job in gpu_jobs))
        self.assertTrue(all(job.requires == {"gpu"} for job in overridden))
        with self.assertRaises(ValueError):
            prepare(traces, gpu_fraction=1.1, **options)

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
                reference_price=_REFERENCE_PRICE,
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
        self.assertEqual(jobs[0].price("anything"), 0.3)
        self.assertEqual(jobs[3].bid, {"tuolumne": 0.5})
        self.assertIsNone(jobs[3].price("corona"))
        self.assertEqual(jobs[9].price("matrix"), 16.0)
        self.assertEqual(jobs[14].bid, {"corona": 1.4})

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
            reference_price=_REFERENCE_PRICE,
            start=1000,
            hours=0.05,
            seed=4,
            per_platform=("corona", "matrix"),
        )
        again, _ = prepare(
            traces,
            reference_price=_REFERENCE_PRICE,
            start=1000,
            hours=0.05,
            seed=4,
            per_platform=("corona", "matrix"),
        )
        self.assertTrue(all(list(job.bid) == ["corona", "matrix"] for job in jobs))
        self.assertEqual(jobs[0].bid, {"corona": 5.9795, "matrix": 6.1484})
        self.assertEqual(jobs, again)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mapped.csv"
            write_jobs(jobs, path)
            header = path.read_text(encoding="utf-8").splitlines()[0]
            self.assertIn("bid,bid:corona,bid:matrix,requires", header)
            self.assertEqual(read_jobs(path), jobs)

    def test_limit_can_follow_runtime_or_request(self) -> None:
        """Preparation carries the request and selects either limit source."""
        traces = {"tioga": _DATA / "trace.csv"}
        options = {
            "reference_price": _REFERENCE_PRICE,
            "start": 1000,
            "hours": 0.05,
        }
        runtime_jobs, _ = prepare(traces, **options)
        request_jobs, _ = prepare(traces, limit_from="request", **options)
        self.assertEqual(runtime_jobs[0].limit_s, 40)
        self.assertEqual(request_jobs[0].limit_s, 120)
        self.assertEqual(runtime_jobs[0].requested_s, request_jobs[0].requested_s)
        self.assertEqual(runtime_jobs[0].requested_s, 120)

    def test_home_per_platform_bid_uses_persona_price(self) -> None:
        """A home entry keeps the persona price while other entries vary."""
        traces = {"tioga": _DATA / "trace.csv"}
        options = {
            "anchor": "home",
            "home_prices": {"tioga": 2.7},
            "start": 1000,
            "hours": 0.05,
            "seed": 4,
        }
        scalar, _ = prepare(traces, **options)
        mapped, _ = prepare(traces, per_platform=("corona", "tioga"), **options)
        self.assertEqual(
            [job.bid for job in scalar],
            [job.bid["tioga"] for job in mapped],
        )
        self.assertTrue(any(job.bid["corona"] != job.bid["tioga"] for job in mapped))

    def test_job_draws_follow_trace_row_identity(self) -> None:
        """Changing the interval does not change a retained row's draws."""
        with tempfile.TemporaryDirectory() as directory:
            for id_header, id_values in (
                ("job_id,", "before,target"),
                ("", ","),
            ):
                with self.subTest(trace_id=bool(id_header)):
                    before_id, target_id = id_values.split(",")
                    trace = Path(directory) / f"simple-{bool(id_header)}.csv"
                    trace.write_text(
                        f"{id_header}job_submit_time,num_nodes,time_limit,user\n"
                        f"{before_id + ',' if id_header else ''}100,1,60,"
                        "fixed-user\n"
                        f"{target_id + ',' if id_header else ''}200,1,60,"
                        "fixed-user\n",
                        encoding="utf-8",
                    )
                    options = {
                        "reference_price": _REFERENCE_PRICE,
                        "trace_format": "simple",
                        "seed": 8,
                    }
                    whole, _ = prepare({"tioga": trace}, start=100, **options)
                    sliced, _ = prepare({"tioga": trace}, start=200, **options)
                    self.assertEqual(whole[1].job_id, "j000002")
                    self.assertEqual(sliced[0].job_id, "j000001")
                    self.assertEqual(
                        (
                            whole[1].persona,
                            whole[1].bid,
                            whole[1].requires,
                        ),
                        (
                            sliced[0].persona,
                            sliced[0].bid,
                            sliced[0].requires,
                        ),
                    )

    def test_mean_and_home_anchors(self) -> None:
        """Mean is the default, while home anchoring needs a home price."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace = root / "simple.csv"
            trace.write_text(
                "job_id,job_submit_time,num_nodes,time_limit,user\n"
                "stable,100,2,60,fixed-user\n",
                encoding="utf-8",
            )
            direct, _ = prepare(
                {"community": trace},
                reference_price=_REFERENCE_PRICE,
                trace_format="simple",
                seed=8,
            )
            mean_path = root / "mean.csv"
            with redirect_stdout(StringIO()):
                status = main(
                    [
                        "prepare",
                        "--trace",
                        f"community={trace}",
                        "--out",
                        str(mean_path),
                        "--format",
                        "simple",
                        "--seed",
                        "8",
                    ]
                )
            self.assertEqual(status, 0)
            self.assertEqual(read_jobs(mean_path), direct)

            home_path = root / "home.csv"
            with redirect_stdout(StringIO()):
                status = main(
                    [
                        "prepare",
                        "--trace",
                        f"tioga={trace}",
                        "--out",
                        str(home_path),
                        "--format",
                        "simple",
                        "--seed",
                        "8",
                        "--anchor",
                        "home",
                    ]
                )
            self.assertEqual(status, 0)
            home, _ = prepare(
                {"tioga": trace},
                anchor="home",
                home_prices={"tioga": 2.7},
                trace_format="simple",
                seed=8,
            )
            self.assertEqual(read_jobs(home_path), home)
            with self.assertRaises(ValueError):
                prepare(
                    {"tioga": trace},
                    anchor="home",
                    trace_format="simple",
                )


if __name__ == "__main__":
    unittest.main()
