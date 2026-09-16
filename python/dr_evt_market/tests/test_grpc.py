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
    ClockViolation,
    ConfigurationError,
    GrpcPlatform,
    InfrastructureFailure,
    ServerProcess,
    StructuralRejection,
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
        """Remote scheduling and final files match the batch-mode CLI."""
        platform = self.make_platform("lifecycle")
        first_handle = platform.submit([CONTENDED_JOBS[0]])[0]

        queued = platform.snapshot()
        self.assertEqual(queued.waiting_jobs, 1)
        self.assertEqual(queued.free_nodes, 100)
        self.assertEqual(queued.current_utilization, 0.0)
        self.assertFalse(platform.timings([first_handle])[0].scheduled)

        platform.advance_to(0)
        running = platform.snapshot()
        self.assertEqual(running.in_use_nodes, 20)
        self.assertEqual(running.current_utilization, 0.2)

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

    def test_unknown_handle_is_typed(self) -> None:
        """An unknown remote handle raises a keyed contract exception."""
        platform = self.make_platform("errors")

        with self.assertRaisesRegex(KeyError, "99"):
            platform.timings([99])
        platform.finish()

    def test_guards_do_not_mutate_scheduler_state(self) -> None:
        """Every invalid request fails before the gRPC append call."""
        platform = self.make_platform("guards")
        platform.advance_to(10)
        invalid_batches = [
            (
                ClockViolation,
                [SubmitRequest("fractional", 10.5, 1, 1)],
            ),
            (
                ClockViolation,
                [SubmitRequest("past", 9, 1, 1)],
            ),
            (
                ClockViolation,
                [
                    SubmitRequest("later", 20, 1, 1),
                    SubmitRequest("earlier", 15, 1, 1),
                ],
            ),
            (
                ConfigurationError,
                [SubmitRequest("queue", 10, 1, 1, q_id="pbatch")],
            ),
            (
                StructuralRejection,
                [SubmitRequest("oversize", 10, 101, 1)],
            ),
            (
                StructuralRejection,
                [SubmitRequest("zero-nodes", 10, 0, 1)],
            ),
            (
                StructuralRejection,
                [SubmitRequest("zero-limit", 10, 1, 0)],
            ),
        ]

        for error_type, batch in invalid_batches:
            with self.subTest(key=batch[0].key):
                before = platform.snapshot().waiting_jobs
                with self.assertRaises(error_type):
                    platform.submit(batch)
                self.assertEqual(platform.snapshot().waiting_jobs, before)

    def test_finish_closes_adapter_and_caches_report(self) -> None:
        """Finishing closes operations and returns the cached report again."""
        platform = self.make_platform("closed")
        platform.submit([SubmitRequest("job", 0, 1, 1)])
        report = platform.finish()

        self.assertIs(platform.finish(), report)
        operations = {
            "now": platform.now,
            "submit": lambda: platform.submit([
                SubmitRequest("late", 0, 1, 1)
            ]),
            "advance_to": lambda: platform.advance_to(0),
            "snapshot": platform.snapshot,
            "timings": lambda: platform.timings([]),
        }
        for name, operation in operations.items():
            with self.subTest(operation=name):
                with self.assertRaises(InfrastructureFailure):
                    operation()


if __name__ == "__main__":
    unittest.main()
