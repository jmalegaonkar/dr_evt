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
from types import SimpleNamespace

from federated_market import (
    DEFAULT_FEDERATION,
    PLATFORMS,
    PROFILES,
    Job,
    prepare,
    read_jobs,
    write_jobs,
)
from federated_market.cli import main

_DATA = Path(__file__).with_name("data")


def _simple(directory, rows, header="job_submit_time,num_nodes,time_limit,user"):
    trace = Path(directory) / "simple.csv"
    trace.write_text(header + "\n" + "".join(rows), encoding="utf-8")
    return trace


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
        """LC preparation filters one interval and counts drops and personas."""
        jobs, summary = prepare(
            {"tioga": _DATA / "trace.csv"},
            platforms=PLATFORMS,
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
                "persona:sticker": 1,
                "persona:tier": 2,
                "persona:value": 2,
                "persona:whale": 0,
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
                frozenset(),
                frozenset({"gpu"}),
                frozenset({"gpu"}),
                frozenset({"gpu"}),
            ],
        )

    def test_sources_merge_on_one_origin(self) -> None:
        """Sources share one clock and contribute to persona identity."""
        traces = {"beta": _DATA / "trace.csv", "alpha": _DATA / "trace.csv"}
        jobs, summary = prepare(
            traces,
            platforms=PLATFORMS,
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
        options = {"platforms": PLATFORMS, "start": 1000, "hours": 0.05}
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

    def test_persona_and_bids_are_hash_seeded(self) -> None:
        """A fixed seed, source, user and trace row give pinned bids."""
        with tempfile.TemporaryDirectory() as directory:
            trace = _simple(directory, ["100,2,60,fixed-user\n"])
            options = {"platforms": PLATFORMS, "trace_format": "simple", "seed": 8}
            single, _ = prepare({"tioga": trace}, bids="single", **options)
            multi, _ = prepare({"tioga": trace}, **options)
        self.assertEqual([single[0].persona, multi[0].persona], ["value", "value"])
        self.assertEqual(single[0].bid, 1.3533)
        self.assertEqual(
            multi[0].bid,
            {"corona": 1.6245, "matrix": 2.0781, "tioga": 3.4878, "tuolumne": 0.1305},
        )

    def test_single_bid_prices_work_at_the_mean_level(self) -> None:
        """A single bid is the mean price of a unit of work times the multiple."""
        jobs, _ = prepare(
            {"tioga": _DATA / "trace.csv"},
            platforms=PLATFORMS,
            bids="single",
            start=1000,
            hours=0.05,
            seed=4,
        )
        for job in jobs:
            hardware = "gpu" if "gpu" in job.requires else "cpu"
            usable = [item for item in PLATFORMS if job.requires <= item.hardware]
            level = sum(
                item.price_per_node_hour / item.speed[hardware] for item in usable
            ) / len(usable)
            with self.subTest(job=job.job_id):
                self.assertIsInstance(job.bid, float)
                if job.persona == "sticker":
                    self.assertEqual(job.bid, round(level, 4))
        self.assertEqual([job.bid for job in jobs][1:3], [1.6641, 0.5934])

    def test_gpu_fraction_and_requires_override(self) -> None:
        """GPU draws are per job, and an override moves no other draw."""
        traces = {"tioga": _DATA / "trace.csv"}
        options = {"platforms": PLATFORMS, "start": 1000, "hours": 0.05, "seed": 4}
        cpu_jobs, _ = prepare(traces, gpu_fraction=0.0, **options)
        gpu_jobs, _ = prepare(traces, gpu_fraction=1.0, **options)
        overridden, _ = prepare(traces, gpu_fraction=0.0, requires="gpu", **options)
        self.assertTrue(all(not job.requires for job in cpu_jobs))
        self.assertTrue(all(job.requires == {"gpu"} for job in gpu_jobs))
        self.assertEqual(overridden, gpu_jobs)
        with self.assertRaises(ValueError):
            prepare(traces, gpu_fraction=1.1, **options)

    def test_hardware_does_not_depend_on_the_user(self) -> None:
        """Changing every user changes personas but not a single hardware draw."""
        with tempfile.TemporaryDirectory() as directory:
            rows = [f"job{row},{row},1,60,{{}}{row}\n" for row in range(40)]
            header = "job_id,job_submit_time,num_nodes,time_limit,user"
            first = _simple(directory, [row.format("a") for row in rows], header)
            jobs, _ = prepare(
                {"tioga": first}, platforms=PLATFORMS, trace_format="simple"
            )
            second = _simple(directory, [row.format("b") for row in rows], header)
            others, _ = prepare(
                {"tioga": second}, platforms=PLATFORMS, trace_format="simple"
            )
        self.assertEqual(
            [job.requires for job in jobs], [job.requires for job in others]
        )
        self.assertNotEqual(
            [job.persona for job in jobs], [job.persona for job in others]
        )

    def test_simple_times_are_floored_sorted_and_shifted(self) -> None:
        """Simple fractional times use the earliest floored submit as zero."""
        with tempfile.TemporaryDirectory() as directory:
            trace = _simple(
                directory,
                ["12.9,2,5.9,3.9\n", "10.8,1,4.2,2.7\n"],
                "job_submit_time,num_nodes,time_limit,actual_run_time",
            )
            jobs, summary = prepare(
                {"simple": trace}, platforms=PLATFORMS, trace_format="simple"
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
                "mixed,0,1,1,9.0,2.5\n"
                "none,0,1,1,,\n",
                encoding="utf-8",
            )
            self.assertEqual(
                [job.bid for job in read_jobs(path)], [{"corona": 2.5}, {}]
            )

    def test_multi_bids_cover_each_platform_with_the_hardware(self) -> None:
        """Every platform with a job's hardware gets a bid, and they round-trip."""
        traces = {"tioga": _DATA / "trace.csv"}
        options = {
            "platforms": PLATFORMS,
            "bids": "multi",
            "start": 1000,
            "hours": 0.05,
            "seed": 4,
        }
        jobs, _ = prepare(traces, **options)
        again, _ = prepare(traces, **options)
        for job in jobs:
            expected = [
                profile.name
                for profile in PLATFORMS
                if job.requires <= profile.hardware
            ]
            self.assertEqual(list(job.bid), expected)
        self.assertEqual(
            jobs[1].bid,
            {
                "corona": 1.7873,
                "dane": 0.5701,
                "matrix": 5.0557,
                "tioga": 4.7905,
                "tuolumne": 0.5039,
            },
        )
        self.assertEqual(jobs, again)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mapped.csv"
            write_jobs(jobs, path)
            header = path.read_text(encoding="utf-8").splitlines()[0]
            self.assertIn(
                "bid,bid:corona,bid:dane,bid:matrix,bid:tioga,bid:tuolumne,requires",
                header,
            )
            self.assertEqual(read_jobs(path), jobs)

    def test_bids_follow_the_posted_prices(self) -> None:
        """Doubling every posted price doubles single bids and each multi bid."""

        def profiles(scale):
            return [
                SimpleNamespace(
                    name=name,
                    price_per_node_hour=scale * price,
                    hardware={"cpu"},
                    speed={"cpu": speed},
                )
                for name, price, speed in (("low", 0.5, 2.0), ("high", 2.0, 1.0))
            ]

        traces = {"tioga": _DATA / "trace.csv"}
        options = {"start": 1000, "hours": 0.05, "seed": 4, "gpu_fraction": 0.0}
        for bids in ("single", "multi"):
            jobs, _ = prepare(traces, platforms=profiles(1), bids=bids, **options)
            again, _ = prepare(traces, platforms=profiles(2), bids=bids, **options)
            for job, other in zip(jobs, again):
                with self.subTest(bids=bids, job=job.job_id):
                    if bids == "single":
                        self.assertAlmostEqual(other.bid, 2 * job.bid, places=3)
                        continue
                    for name in ("low", "high"):
                        self.assertAlmostEqual(
                            other.bid[name], 2 * job.bid[name], places=3
                        )

    def test_multi_bids_can_differ_by_platform_or_agree(self) -> None:
        """One user can be a whale on one platform and under the price on another."""
        with tempfile.TemporaryDirectory() as directory:
            trace = _simple(directory, [f"{row},1,60,-\n" for row in range(300)])
            jobs, _ = prepare(
                {"simple": trace},
                platforms=PLATFORMS,
                bids="multi",
                trace_format="simple",
            )
        prices = {item.name: item.price_per_node_hour for item in PLATFORMS}
        ratios = [[bid / prices[name] for name, bid in job.bid.items()] for job in jobs]
        self.assertTrue(any(max(row) >= 5 and min(row) < 1 for row in ratios))
        self.assertTrue(any(all(0.5 <= ratio < 2 for ratio in row) for row in ratios))

    def test_preferences_belong_to_the_user_and_platform(self) -> None:
        """A bid on one platform does not depend on the other platforms listed."""
        traces = {"tioga": _DATA / "trace.csv"}
        options = {"bids": "multi", "start": 1000, "hours": 0.05, "seed": 4}
        names = [("corona", "tioga"), ("tioga", "corona"), ("tioga",)]
        tioga = [
            [
                job.bid["tioga"]
                for job in prepare(
                    traces, platforms=[PROFILES[name] for name in listed], **options
                )[0]
            ]
            for listed in names
        ]
        self.assertEqual(tioga[0], tioga[1])
        self.assertEqual(tioga[0], tioga[2])

    def test_limit_is_the_run_time_or_else_the_request(self) -> None:
        """The limit is the run time, or the request when the trace has none."""
        with tempfile.TemporaryDirectory() as directory:
            trace = _simple(
                directory,
                ["0,1,120,40\n", "1,1,60,\n"],
                "job_submit_time,num_nodes,time_limit,actual_run_time",
            )
            jobs, _ = prepare(
                {"simple": trace}, platforms=PLATFORMS, trace_format="simple"
            )
        self.assertEqual([job.limit_s for job in jobs], [40, 60])
        self.assertEqual([job.runtime_s for job in jobs], [40, None])
        self.assertEqual([job.requested_s for job in jobs], [120, 60])

    def test_a_row_without_a_user_is_its_own_user(self) -> None:
        """A row without a user draws its persona and hardware independently."""
        with tempfile.TemporaryDirectory() as directory:
            trace = _simple(directory, [f"{row},1,60,-\n" for row in range(400)])
            jobs, _ = prepare(
                {"simple": trace}, platforms=PLATFORMS, trace_format="simple"
            )
        self.assertEqual({job.user for job in jobs}, {""})
        for persona in ("sticker", "whale"):
            with self.subTest(persona=persona):
                self.assertEqual(
                    {bool(job.requires) for job in jobs if job.persona == persona},
                    {False, True},
                )

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
                        "platforms": PLATFORMS,
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

    def test_command_line_bids_on_the_default_federation(self) -> None:
        """The command line bids on the five default platforms unless told."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace = _simple(
                directory,
                ["stable,100,2,60,fixed-user\n"],
                "job_id,job_submit_time,num_nodes,time_limit,user",
            )
            direct, _ = prepare(
                {"community": trace},
                platforms=[PROFILES[name] for name in DEFAULT_FEDERATION],
                trace_format="simple",
                seed=8,
            )
            path = root / "jobs.csv"
            with redirect_stdout(StringIO()):
                status = main(
                    [
                        "prepare",
                        "--trace",
                        f"community={trace}",
                        "--out",
                        str(path),
                        "--format",
                        "simple",
                        "--seed",
                        "8",
                    ]
                )
            self.assertEqual(status, 0)
            self.assertEqual(read_jobs(path), direct)


if __name__ == "__main__":
    unittest.main()
