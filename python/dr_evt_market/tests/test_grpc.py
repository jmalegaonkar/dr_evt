################################################################################
#         Copyright 2023 Lawrence Livermore National Security, LLC             #
#         See the top-level LICENSE file for details.                          #
#                                                                              #
#         SPDX-License-Identifier: MIT                                         #
################################################################################

"""Tests for the gRPC DR_EVT platform adapter and server helper."""

from pathlib import Path
import tempfile
import unittest

from dr_evt_market import (
    ConfigurationError,
    GrpcPlatform,
    ServerProcess,
    SubmitRequest,
)
from dr_evt_market.tests.fixtures import (
    CONTENDED_JOBS,
    EXPECTED_BEGIN_TIMES,
    cli_schedule,
)


class GrpcPlatformTests(unittest.TestCase):
    """Exercise one adapter against a managed local server process."""

    def setUp(self) -> None:
        """Start one isolated server for each test."""
        self._temporary_directory = tempfile.TemporaryDirectory(
            prefix="dr_evt_market_grpc_"
        )
        self.addCleanup(self._temporary_directory.cleanup)
        self.work_dir = Path(self._temporary_directory.name)
        self.server = ServerProcess(None, self.work_dir)
        self.server.start()
        self.addCleanup(self.server.stop)

    def make_platform(self, name: str) -> GrpcPlatform:
        """Construct a 100-node adapter connected to the test server."""
        return GrpcPlatform(
            name,
            100,
            self.server.address,
            self.work_dir,
            session_name=name,
        )

    def test_lifecycle_matches_expected_and_cli(self) -> None:
        """Remote scheduling and final files match the independent CLI."""
        platform = self.make_platform("lifecycle")
        first_handle = platform.submit([CONTENDED_JOBS[0]])[0]

        queued = platform.snapshot()
        self.assertEqual(queued.waiting_jobs, 1)
        self.assertEqual(queued.free_nodes, 100)
        self.assertIsNone(queued.current_utilization)
        self.assertIsNone(queued.resource_area)
        self.assertIsNone(queued.prediction_horizon_s)
        self.assertFalse(platform.timings([first_handle])[0].scheduled)

        platform.advance_to(0)
        self.assertEqual(platform.snapshot().in_use_nodes, 20)

        handles = [first_handle]
        for job in CONTENDED_JOBS[1:]:
            platform.advance_to(job.submit_s)
            handles.extend(platform.submit([job]))
            platform.advance_to(job.submit_s)
        report = platform.finish()

        self.assertEqual(
            [timing.begin_s for timing in report.timings],
            [float(value) for value in EXPECTED_BEGIN_TIMES],
        )
        cli_rows = cli_schedule(self.work_dir / "oracle", CONTENDED_JOBS)
        for timing, row in zip(report.timings, cli_rows):
            self.assertEqual(timing.submit_s, float(row["job_submit_time"]))
            self.assertEqual(timing.begin_s, float(row["begin_time"]))
            self.assertEqual(timing.end_s, float(row["end_time"]))

        simulated_path = Path(report.simulated_trace_path or "")
        resource_path = Path(report.resource_trace_path or "")
        self.assertTrue(simulated_path.is_file())
        self.assertTrue(resource_path.is_file())
        self.assertEqual(
            simulated_path.read_text(encoding="utf-8").splitlines()[0],
            "job_submit_time,begin_time,end_time,num_nodes,exit_status,time_limit",
        )
        self.assertEqual(
            resource_path.read_text(encoding="utf-8").splitlines()[0],
            "time,free_nodes,allocated_nodes",
        )

    def test_unknown_handle_and_bad_queue_are_typed(self) -> None:
        """Known remote and local failures use their contract exceptions."""
        platform = self.make_platform("errors")

        with self.assertRaisesRegex(KeyError, "99"):
            platform.timings([99])
        with self.assertRaisesRegex(ConfigurationError, "q_id"):
            platform.submit([
                SubmitRequest("bad-queue", 0, 1, 1, q_id="pbatch")
            ])
        self.assertEqual(platform.snapshot().waiting_jobs, 0)
        platform.finish()


if __name__ == "__main__":
    unittest.main()
