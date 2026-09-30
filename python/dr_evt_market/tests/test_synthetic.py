################################################################################
#         Copyright 2023 Lawrence Livermore National Security, LLC             #
#         See the top-level LICENSE file for details.                          #
#                                                                              #
#         SPDX-License-Identifier: MIT                                         #
################################################################################

"""Tests for synthetic days: real groups of jobs, picked at random, at their times."""

import tempfile
import unittest
from collections import Counter
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from dr_evt_market import PLATFORMS, prepare, read_jobs
from dr_evt_market.cli import main
from dr_evt_market.jobs.synthetic import synthesize
from dr_evt_market.jobs.traces import read_simple

_HEADER = "job_id,job_submit_time,num_nodes,time_limit,actual_run_time,user\n"
_DAY = 86400


def _trace(directory, name, rows):
    path = Path(directory) / f"{name}.csv"
    path.write_text(_HEADER + "".join(rows), encoding="utf-8")
    return path


def _grouped(directory):
    """Three days in which two users take turns submitting arrays of four jobs."""
    rows = []
    for day in range(3):
        for index in range(40):
            user, nodes = ("wide", 4) if index % 2 else ("narrow", 1)
            at = day * _DAY + 900 * index
            rows += [
                f"{user}-{day}-{index}-{slot},{at + slot},{nodes},60,50,{user}\n"
                for slot in range(4)
            ]
    return {"cluster": _trace(directory, "cluster", rows)}


def _options(**extra):
    return {
        "platforms": PLATFORMS,
        "trace_format": "simple",
        "start": 0,
        "hours": 72,
        "synthetic": True,
        **extra,
    }


class SyntheticTests(unittest.TestCase):
    """Exercise group sampling, times of day, seeds and the command line."""

    def test_each_group_keeps_its_time_of_day(self) -> None:
        """A drawn group lands on the day at the times of day it really had."""
        with tempfile.TemporaryDirectory() as directory:
            early, late = [], []
            for day in range(3):
                for index in range(10):
                    at = day * _DAY + 3600 + 300 * index
                    early += [
                        f"e{day}-{index}-{s},{at + s},1,60,50,e\n" for s in range(3)
                    ]
                    when = day * _DAY + 18000 + 300 * index
                    late.append(f"l{day}-{index},{when},1,60,50,l{index}\n")
            kept = read_simple("early", _trace(directory, "early", early))
            kept += read_simple("late", _trace(directory, "late", late))
        kept.sort(key=lambda row: (row.submit, row.source, row.order))
        real = {row.identity: row.submit for row in kept}
        rows, found = synthesize(kept, 0, 3 * _DAY, seed=5)
        self.assertEqual(found, {"early": 10 * 3, "late": 10 * 3})
        self.assertEqual({row.source for row in rows}, {"early", "late"})
        for row in rows:
            self.assertEqual(row.submit, real[row.identity] % _DAY)

    def test_groups_arrive_whole_and_at_most_once(self) -> None:
        """A day takes whole groups, each once, about as many as a day had."""
        with tempfile.TemporaryDirectory() as directory:
            kept = read_simple("cluster", _grouped(directory)["cluster"])
        kept.sort(key=lambda row: (row.submit, row.source, row.order))
        counts = []
        for seed in range(20):
            rows, found = synthesize(kept, 0, 3 * _DAY, seed=seed)
            identities = [row.identity for row in rows]
            per_group = Counter(identity.rsplit("-", 1)[0] for identity in identities)
            with self.subTest(seed=seed):
                self.assertEqual(len(identities), len(set(identities)))
                self.assertEqual(set(per_group.values()), {4})
            counts.append(len(per_group))
        self.assertEqual(found, {"cluster": 3 * 40})
        self.assertLess(abs(sum(counts) / len(counts) - 40), 5)

    def test_synthetic_days_follow_the_seed(self) -> None:
        """A seed fixes a synthetic day, and another seed draws another day."""
        with tempfile.TemporaryDirectory() as directory:
            traces = _grouped(directory)
            first, _ = prepare(traces, **_options(seed=1))
            again, _ = prepare(traces, **_options(seed=1))
            other, _ = prepare(traces, **_options(seed=2))
        self.assertEqual(first, again)
        self.assertNotEqual(
            [job.submit_s for job in first], [job.submit_s for job in other]
        )

    def test_command_line_prepares_a_synthetic_day(self) -> None:
        """The command line writes a synthetic day and counts its groups."""
        with tempfile.TemporaryDirectory() as directory:
            traces = _grouped(directory)
            path = Path(directory) / "day.csv"
            output = StringIO()
            with redirect_stdout(output):
                status = main(
                    [
                        "prepare",
                        "--trace",
                        f"cluster={traces['cluster']}",
                        "--format",
                        "simple",
                        "--out",
                        str(path),
                        "--start",
                        "0",
                        "--hours",
                        "72",
                        "--synthetic",
                    ]
                )
            self.assertEqual(status, 0)
            self.assertIn("groups:cluster=120", output.getvalue().splitlines())
            self.assertGreater(len(read_jobs(path)), 0)


if __name__ == "__main__":
    unittest.main()
