################################################################################
#         Copyright 2023 Lawrence Livermore National Security, LLC             #
#         See the top-level LICENSE file for details.                          #
#                                                                              #
#         SPDX-License-Identifier: MIT                                         #
################################################################################

"""End-to-end tests: input files, the loop, the outputs, both transports."""

import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from dr_evt_market import (
    Controller,
    GrpcPlatform,
    InProcessPlatform,
    ServerProcess,
    read_jobs,
    read_platforms,
    write_outputs,
)
from dr_evt_market.mechanisms import Vcg

_DATA = Path(__file__).with_name("data")
# Pinned on the fixture files as committed; regenerate only when they change.
_ROUTED_SHA256 = "2328e1fe136e040566d52fde52f00b6115f73b3e7109476418bf363bfb41e6ba"


class InputTests(unittest.TestCase):
    """The two file formats."""

    def test_platforms_carry_hardware(self) -> None:
        """Hardware tags and the blank address are read."""
        platforms = {p.name: p for p in read_platforms(_DATA / "market_platforms.csv")}
        self.assertEqual(platforms["gamma"].hardware, frozenset({"cpu", "gpu"}))
        self.assertIsNone(platforms["alpha"].address)

    def test_jobs_single_and_per_platform_bids(self) -> None:
        """A bid column gives one multiplier; bid:<platform> columns give a mapping."""
        jobs = {job.job_id: job for job in read_jobs(_DATA / "market_jobs.csv")}
        self.assertEqual(jobs["s03"].legs[0].requires, frozenset({"gpu"}))
        self.assertEqual(jobs["c01"].bid, 3.0)
        self.assertEqual([leg.leg_id for leg in jobs["c01"].legs], ["left", "right"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "jobs.csv"
            path.write_text(
                "job_id,job_submit_time,num_nodes,time_limit,bid:alpha,bid:beta\n"
                "a,0,4,60,1.5,\n",
                encoding="utf-8",
            )
            self.assertEqual(read_jobs(path)[0].bid, {"alpha": 1.5})
            path.write_text("job_id,job_submit_time,num_nodes,time_limit\na,0,4,60\n")
            with self.assertRaisesRegex(ValueError, "bid"):
                read_jobs(path)


class ControllerTests(unittest.TestCase):
    """The fixture through the loop, in process and with one gRPC platform."""

    def setUp(self) -> None:
        """Read the fixture and make an output root."""
        self._tmp = tempfile.TemporaryDirectory(prefix="dr_evt_market_controller_")
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.platforms = {
            p.name: p for p in read_platforms(_DATA / "market_platforms.csv")
        }
        self.jobs = read_jobs(_DATA / "market_jobs.csv")

    def _sessions(self, root: Path, grpc_alpha=None) -> dict:
        return {
            name: (
                GrpcPlatform(
                    name, p.total_nodes, grpc_alpha[0], grpc_alpha[1], session_name=name
                )
                if grpc_alpha is not None and name == "alpha"
                else InProcessPlatform(name, p.total_nodes, root / name)
            )
            for name, p in self.platforms.items()
        }

    def _run(self, sessions: dict, out: Path):
        report = Controller(
            sessions, self.platforms, Vcg(), self.jobs, window_s=60, seed=11
        ).run()
        return report, write_outputs(report, out)

    def test_pipeline_routes_and_writes_pinned_outputs(self) -> None:
        """Every leg starts at its window, charges are bounded, files are pinned."""
        report, paths = self._run(self._sessions(self.root / "p"), self.root / "out")
        self.assertTrue(report.routed)
        self.assertTrue(any(len(w.queued) > len(w.placed) for w in report.windows))
        for leg in report.routed:
            self.assertEqual(leg.begin_s, leg.window_time_s)
            self.assertGreaterEqual(leg.charge_credits, leg.cost_credits - 1e-9)
            self.assertLessEqual(leg.charge_credits, leg.value_credits + 1e-9)
            self.assertAlmostEqual(
                leg.premium_credits, leg.charge_credits - leg.cost_credits
            )
        composite = [leg for leg in report.routed if leg.job_id == "c01"]
        self.assertEqual(len(composite), 2)
        self.assertEqual(len({leg.begin_s for leg in composite}), 1)
        self.assertEqual(composite[1].platform, "gamma")  # the gpu leg
        gpu_only = [leg for leg in report.routed if leg.job_id in ("s03", "s10")]
        self.assertTrue(gpu_only and all(leg.platform == "gamma" for leg in gpu_only))
        with Path(paths["rejected"]).open(newline="", encoding="utf-8") as f:
            rejected = list(csv.DictReader(f))
        self.assertEqual(
            rejected, [{"job_id": "s12", "reason": "unaffordable", "time_s": "600"}]
        )
        digest = hashlib.sha256(Path(paths["routed"]).read_bytes()).hexdigest()
        self.assertEqual(digest, _ROUTED_SHA256)
        manifest = json.loads(Path(paths["manifest"]).read_text(encoding="utf-8"))
        self.assertEqual(manifest["sha256"]["routed.csv"], digest)
        self.assertEqual(
            manifest["configuration"]["platforms"]["gamma"]["hardware"], ["cpu", "gpu"]
        )

    def test_one_grpc_platform_gives_identical_output(self) -> None:
        """Serving alpha over gRPC leaves routed.csv byte-identical."""
        _, local = self._run(
            self._sessions(self.root / "local"), self.root / "local-out"
        )
        server_dir = self.root / "server"
        with ServerProcess(None, server_dir) as server:
            sessions = self._sessions(self.root / "mixed", (server.address, server_dir))
            _, mixed = self._run(sessions, self.root / "mixed-out")
        self.assertEqual(
            Path(local["routed"]).read_bytes(), Path(mixed["routed"]).read_bytes()
        )


if __name__ == "__main__":
    unittest.main()
