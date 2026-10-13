################################################################################
#         Copyright 2023 Lawrence Livermore National Security, LLC             #
#         See the top-level LICENSE file for details.                          #
#                                                                              #
#         SPDX-License-Identifier: MIT                                         #
################################################################################

"""Tests for the workload model: fitting day types and generating days from them."""

import math
import tempfile
import unittest
from collections import defaultdict
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from federated_market import (
    PLATFORMS,
    fit_profiles,
    generate,
    read_jobs,
    read_profiles,
    write_profiles,
)
from federated_market.cli import main

_DATA = Path(__file__).with_name("data")
_HEADER = "job_id,job_submit_time,num_nodes,time_limit,actual_run_time,user\n"
_DAY = 86400


def _trace(directory):
    """Three days. Day 0 holds one 1-node job of 60 s at 10:00; day 1 an array of three
    4-node jobs of 600 s, a second apart, at 9:00; day 2 two such arrays, at 9:00 and
    13:00, and a 1-node job of 60 s at 20:00. Users play no part."""
    rows = ["d0,36000,1,90,60,alice\n"]
    for day, hours in ((1, (9,)), (2, (9, 13))):
        for hour in hours:
            at = day * _DAY + hour * 3600
            rows += [f"d{day}{hour}{k},{at + k},4,700,600,bob\n" for k in range(3)]
    rows.append(f"d2x,{2 * _DAY + 20 * 3600},1,90,60,alice\n")
    path = Path(directory) / "cluster.csv"
    path.write_text(_HEADER + "".join(rows), encoding="utf-8")
    return path


def _example():
    return read_profiles(_DATA / "workload.json")


class WorkloadTests(unittest.TestCase):
    """Fit known arrays and day types; generate days that follow a profile."""

    def test_fit_types_the_days_and_finds_the_arrays(self) -> None:
        """Days go into thirds by work; arrays keep their hour, nodes, size and span."""
        with tempfile.TemporaryDirectory() as directory:
            profile = fit_profiles(
                {"cluster": _trace(directory)}, trace_format="simple", start=0, hours=72
            )["cluster"]
        self.assertEqual(profile["jobs"], 11)
        self.assertEqual(profile["run_time_max_s"], 600)
        short, long = [round(math.log(60), 3)], [round(math.log(600), 3)]
        types = {
            kind: [sorted(arrays) for arrays in outlines]
            for kind, outlines in profile["types"].items()
        }
        self.assertEqual(
            types,
            {
                "light": [[[10, 1, 1, 0, short]]],
                "medium": [[[9, 4, 3, 2, long]]],
                "busy": [
                    [[9, 4, 3, 2, long], [13, 4, 3, 2, long], [20, 1, 1, 0, short]]
                ],
            },
        )

    def test_generation_follows_the_seed(self) -> None:
        """A seed fixes the jobs; another seed draws another workload."""
        first, _ = generate(_example(), platforms=PLATFORMS, days=3, seed=3)
        again, _ = generate(_example(), platforms=PLATFORMS, days=3, seed=3)
        other, _ = generate(_example(), platforms=PLATFORMS, days=3, seed=4)
        self.assertEqual(first, again)
        self.assertNotEqual(first, other)

    def test_day_types_and_volume_set_the_arrays(self) -> None:
        """A day refills one of its type's outlines; volume copies its arrays."""
        counts = {}
        for key, volume in (("once", 1.0), ("twice", 2.0)):
            _, summary = generate(
                _example(), platforms=PLATFORMS, days=["medium"] * 4, volume=volume
            )
            counts[key] = summary["arrays:example"]
        self.assertEqual(counts, {"once": 12, "twice": 24})
        _, summary = generate(_example(), platforms=PLATFORMS, days=30)
        drawn = [
            summary[f"days:example:{kind}"] for kind in ("light", "medium", "busy")
        ]
        self.assertEqual(sum(drawn), 30)
        self.assertTrue(all(drawn))

    def test_a_node_range_keeps_the_days(self) -> None:
        """Refills come only from the range; the days and their arrays stay."""
        free, free_summary = generate(_example(), platforms=PLATFORMS, days=12, seed=7)
        small, small_summary = generate(
            _example(), platforms=PLATFORMS, days=12, seed=7, nodes=(1, 4)
        )
        self.assertGreater(max(job.num_nodes for job in free), 4)
        self.assertLessEqual(max(job.num_nodes for job in small), 4)
        for key in free_summary:
            if key.startswith(("days:", "arrays:")):
                self.assertEqual(free_summary[key], small_summary[key])

    def test_arrays_arrive_together_and_jobs_carry_their_fields(self) -> None:
        """An array's jobs arrive within its span; a job's limit is its run time."""
        jobs, summary = generate(
            _example(), platforms=PLATFORMS, days=["medium", "busy"], seed=1
        )
        self.assertEqual(summary["jobs"], len(jobs))
        self.assertEqual(
            [job.submit_s for job in jobs], sorted(job.submit_s for job in jobs)
        )
        self.assertGreater(jobs[-1].submit_s, _DAY)
        arrivals = defaultdict(list)
        for job in jobs:
            arrivals[job.user].append(job.submit_s)
            self.assertEqual(job.limit_s, job.runtime_s)
            self.assertIsNone(job.requested_s)
            self.assertEqual(job.source, "example")
            self.assertLessEqual(job.limit_s, 86400)
            hardware = {p.name for p in PLATFORMS if job.requires <= p.hardware}
            self.assertEqual(set(job.bid), hardware)
        # The widest span in the example is 5 s; its 64-job arrays arrive at once.
        self.assertLessEqual(max(max(t) - min(t) for t in arrivals.values()), 5)

    def test_profiles_round_trip_and_command_line(self) -> None:
        """Profiles survive a file; the commands fit, generate and target a load."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profiles = _example()
            write_profiles(profiles, root / "copy.json")
            self.assertEqual(read_profiles(root / "copy.json"), profiles)
            trace = _trace(directory)
            with redirect_stdout(StringIO()) as printed:
                self.assertEqual(
                    main(
                        [
                            "fit",
                            "--trace",
                            f"cluster={trace}",
                            "--format",
                            "simple",
                            "--start",
                            "0",
                            "--hours",
                            "72",
                            "--out",
                            str(root / "fitted.json"),
                        ]
                    ),
                    0,
                )
            self.assertIn("days:cluster:busy=1", printed.getvalue())
            with redirect_stdout(StringIO()) as printed:
                code = main(
                    [
                        "generate",
                        "--profile",
                        str(_DATA / "workload.json"),
                        "--out",
                        str(root / "jobs.csv"),
                        "--days",
                        "light,busy",
                        "--load",
                        "0.5",
                        "--share",
                        "0.1",
                        "--nodes",
                        "1,8",
                    ]
                )
            self.assertEqual(code, 0)
            lines = dict(line.split("=", 1) for line in printed.getvalue().split())
            self.assertGreater(float(lines["volume"]), 0)
            self.assertEqual(lines["days:example:busy"], "1")
            jobs = read_jobs(root / "jobs.csv")
            self.assertEqual(len(jobs), int(lines["jobs"]))
            self.assertLessEqual(max(job.num_nodes for job in jobs), 8)


if __name__ == "__main__":
    unittest.main()
