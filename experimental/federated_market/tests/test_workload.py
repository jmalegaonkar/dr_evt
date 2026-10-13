################################################################################
#         Copyright 2023 Lawrence Livermore National Security, LLC             #
#         See the top-level LICENSE file for details.                          #
#                                                                              #
#         SPDX-License-Identifier: MIT                                         #
################################################################################

"""Tests for the workload model: fitting day types and generating days from them."""

import tempfile
import unittest
from collections import defaultdict
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from federated_market import (
    PLATFORMS,
    fit_profile,
    generate,
    read_jobs,
    read_profile,
    write_profile,
)
from federated_market.cli import main

_DATA = Path(__file__).with_name("data")
_HEADER = "job_id,job_submit_time,num_nodes,time_limit,actual_run_time,user\n"
_DAY = 86400


def _traces(directory):
    """Three days of two traces. Alpha has one 1-node job of 60 s at 10:00 on day 0,
    three 4-node jobs of 600 s a second apart at 9:00 on day 1, and a 1-node job of
    60 s at 20:00 on day 2; beta has two 2-node jobs of 300 s at 13:00 and 13:00:30 on
    day 2."""
    alpha = [
        "a0,36000,1,90,60,x\n",
        *[f"a1{k},{_DAY + 32400 + k},4,700,600,x\n" for k in range(3)],
        f"a2,{2 * _DAY + 72000},1,90,60,x\n",
    ]
    beta = [f"b{k},{2 * _DAY + 46800 + 30 * k},2,400,300,y\n" for k in range(2)]
    paths = {}
    for name, rows in (("alpha", alpha), ("beta", beta)):
        paths[name] = Path(directory) / f"{name}.csv"
        paths[name].write_text(_HEADER + "".join(rows), encoding="utf-8")
    return paths


def _example():
    return read_profile(_DATA / "workload.json")


class WorkloadTests(unittest.TestCase):
    """Fit a merged stream into day types; generate days that replay its streams."""

    def test_fit_merges_the_traces_and_types_the_days(self) -> None:
        """Days go into thirds by the merged work; each keeps its stream by second."""
        with tempfile.TemporaryDirectory() as directory:
            profile = fit_profile(
                _traces(directory), trace_format="simple", start=0, hours=72
            )
        self.assertEqual(profile["sources"], ["alpha", "beta"])
        self.assertEqual(profile["jobs"], 7)
        self.assertEqual(
            profile["types"],
            {
                "light": [[[0, 36000, 1, 60]]],
                "medium": [[[1, 46800, 2, 300], [1, 46830, 2, 300], [0, 72000, 1, 60]]],
                "busy": [[[0, 32400, 4, 600], [0, 32401, 4, 600], [0, 32402, 4, 600]]],
            },
        )

    def test_generation_follows_the_seed(self) -> None:
        """A seed fixes the jobs; another seed draws another workload."""
        first, _ = generate(_example(), platforms=PLATFORMS, days=3, seed=3)
        again, _ = generate(_example(), platforms=PLATFORMS, days=3, seed=3)
        other, _ = generate(_example(), platforms=PLATFORMS, days=3, seed=4)
        self.assertEqual(first, again)
        self.assertNotEqual(first, other)

    def test_day_types_and_volume_set_the_streams(self) -> None:
        """A day replays a stream of its type; volume overlays streams."""
        counts = {}
        for key, volume in (("once", 1.0), ("twice", 2.0)):
            jobs, _ = generate(
                _example(), platforms=PLATFORMS, days=["medium"] * 4, volume=volume
            )
            counts[key] = len(jobs)
        self.assertEqual(counts, {"once": 80, "twice": 160})
        _, summary = generate(_example(), platforms=PLATFORMS, days=30)
        drawn = [summary[f"days:{kind}"] for kind in ("light", "medium", "busy")]
        self.assertEqual(sum(drawn), 30)
        self.assertTrue(all(drawn))

    def test_a_node_range_keeps_the_arrivals(self) -> None:
        """Refills come only from the range, and every job keeps its arrival."""
        free, _ = generate(_example(), platforms=PLATFORMS, days=12, seed=7)
        small, _ = generate(
            _example(), platforms=PLATFORMS, days=12, seed=7, nodes=(1, 4)
        )
        self.assertGreater(max(job.num_nodes for job in free), 4)
        self.assertLessEqual(max(job.num_nodes for job in small), 4)
        self.assertEqual(
            [(job.submit_s, job.source) for job in free],
            [(job.submit_s, job.source) for job in small],
        )

    def test_bursts_keep_their_spacing_and_jobs_their_fields(self) -> None:
        """A burst shifts as one, so its jobs stay a second apart."""
        jobs, summary = generate(
            _example(), platforms=PLATFORMS, days=["busy", "busy"], seed=1
        )
        self.assertEqual(summary["jobs"], len(jobs))
        self.assertEqual(summary["jobs:alpha"] + summary["jobs:beta"], len(jobs))
        bursts = defaultdict(list)
        for job in jobs:
            bursts[job.user].append(job.submit_s)
            self.assertEqual(job.limit_s, job.runtime_s)
            self.assertIn(job.source, {"alpha", "beta"})
            hardware = {p.name for p in PLATFORMS if job.requires <= p.hardware}
            self.assertEqual(set(job.bid), hardware)
        widest = max(bursts.values(), key=len)
        self.assertEqual(len(widest), 64)
        self.assertEqual(sorted(widest), list(range(min(widest), min(widest) + 64)))
        self.assertGreater(max(job.submit_s for job in jobs), _DAY)

    def test_profile_round_trip_and_command_line(self) -> None:
        """A profile survives a file; the commands fit, generate and target a load."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile = _example()
            write_profile(profile, root / "copy.json")
            self.assertEqual(read_profile(root / "copy.json"), profile)
            traces = _traces(directory)
            with redirect_stdout(StringIO()) as printed:
                self.assertEqual(
                    main(
                        [
                            "fit",
                            "--trace",
                            f"alpha={traces['alpha']}",
                            "--trace",
                            f"beta={traces['beta']}",
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
            self.assertIn("days:busy=1", printed.getvalue())
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
            self.assertEqual(lines["days:busy"], "1")
            jobs = read_jobs(root / "jobs.csv")
            self.assertEqual(len(jobs), int(lines["jobs"]))
            self.assertLessEqual(max(job.num_nodes for job in jobs), 8)


if __name__ == "__main__":
    unittest.main()
