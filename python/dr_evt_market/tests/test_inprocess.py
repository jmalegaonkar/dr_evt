################################################################################
#         Copyright 2023 Lawrence Livermore National Security, LLC             #
#         See the top-level LICENSE file for details.                          #
#                                                                              #
#         SPDX-License-Identifier: MIT                                         #
################################################################################

"""Tests for the in-process DR_EVT platform adapter."""

from pathlib import Path
import tempfile
import unittest

from dr_evt_market import (
    ClockViolation,
    ConfigurationError,
    InProcessPlatform,
    StructuralRejection,
    SubmitRequest,
)
from dr_evt_market.tests.fixtures import (
    CONTENDED_JOBS,
    EXPECTED_BEGIN_TIMES,
    cli_schedule,
)


class InProcessPlatformTests(unittest.TestCase):
    """Exercise validation, scheduling, snapshots, and final output."""

    def setUp(self) -> None:
        """Create one isolated directory for each test."""
        self._temporary_directory = tempfile.TemporaryDirectory(
            prefix="dr_evt_market_inprocess_"
        )
        self.addCleanup(self._temporary_directory.cleanup)
        self.work_dir = Path(self._temporary_directory.name)

    def make_platform(
        self,
        name: str,
        *,
        use_custom_scheduler: bool = True,
    ) -> InProcessPlatform:
        """Construct a 100-node adapter in a test-specific directory."""
        return InProcessPlatform(
            name,
            100,
            self.work_dir / name,
            use_custom_scheduler=use_custom_scheduler,
        )

    @staticmethod
    def drive_stream(platform: InProcessPlatform) -> tuple:
        """Submit and evaluate the shared contended stream."""
        handles = []
        for job in CONTENDED_JOBS:
            platform.advance_to(job.submit_s)
            handles.extend(platform.submit([job]))
            platform.advance_to(job.submit_s)
        platform.advance_to(1_000)
        return tuple(platform.timings(handles))

    def test_lifecycle_matches_expected_and_cli(self) -> None:
        """Append, advance, and timing records match the independent CLI."""
        platform = self.make_platform("lifecycle")
        first_handle = platform.submit([CONTENDED_JOBS[0]])[0]

        queued = platform.snapshot()
        self.assertEqual(queued.waiting_jobs, 1)
        self.assertEqual(queued.free_nodes, 100)
        self.assertFalse(platform.timings([first_handle])[0].scheduled)

        platform.advance_to(0)
        self.assertEqual(platform.snapshot().in_use_nodes, 20)
        self.assertEqual(platform.timings([first_handle])[0].begin_s, 0.0)

        handles = [first_handle]
        for job in CONTENDED_JOBS[1:]:
            platform.advance_to(job.submit_s)
            handles.extend(platform.submit([job]))
            platform.advance_to(job.submit_s)
        platform.advance_to(1_000)
        timings = platform.timings(handles)

        self.assertEqual(
            [timing.begin_s for timing in timings],
            [float(value) for value in EXPECTED_BEGIN_TIMES],
        )
        cli_rows = cli_schedule(self.work_dir / "oracle", CONTENDED_JOBS)
        self.assertEqual(len(cli_rows), len(timings))
        for timing, row in zip(timings, cli_rows):
            self.assertEqual(timing.submit_s, float(row["job_submit_time"]))
            self.assertEqual(timing.begin_s, float(row["begin_time"]))
            self.assertEqual(timing.end_s, float(row["end_time"]))
            self.assertEqual(timing.num_nodes, int(row["num_nodes"]))
            self.assertEqual(timing.limit_s, int(float(row["time_limit"])))

    def test_guards_do_not_mutate_scheduler_state(self) -> None:
        """Every invalid request fails before the DR_EVT append call."""
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

        with self.assertRaises(ClockViolation):
            platform.advance_to(9)
        with self.assertRaises(ClockViolation):
            platform.advance_to(10.5)
        with self.assertRaises(ConfigurationError):
            InProcessPlatform("bad", 100, self.work_dir, backfill="unknown")

    def test_snapshot_field_availability(self) -> None:
        """Both schedulers expose utilization and only custom adds metrics."""
        custom = self.make_platform("custom")
        custom.submit([SubmitRequest("job", 0, 30, 10)])
        custom.advance_to(0)
        custom_snapshot = custom.snapshot()
        self.assertEqual(custom_snapshot.current_utilization, 0.3)
        self.assertIsNotNone(custom_snapshot.resource_area)
        self.assertIsNotNone(custom_snapshot.prediction_horizon_s)

        plain = self.make_platform("plain", use_custom_scheduler=False)
        plain.submit([SubmitRequest("job", 0, 30, 10)])
        plain.advance_to(0)
        plain_snapshot = plain.snapshot()
        self.assertEqual(plain_snapshot.current_utilization, 0.3)
        self.assertIsNone(plain_snapshot.resource_area)
        self.assertIsNone(plain_snapshot.prediction_horizon_s)

    def test_finish_writes_both_trace_files(self) -> None:
        """Finishing drains jobs and materializes schedule and resource CSVs."""
        platform = self.make_platform("finish")
        handles = platform.submit([
            SubmitRequest("first", 0, 30, 10),
            SubmitRequest("second", 0, 20, 20),
        ])
        platform.advance_to(0)
        report = platform.finish()

        self.assertEqual([timing.handle for timing in report.timings], handles)
        self.assertEqual(report.statistics["jobs_completed"], 2.0)
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

    def test_two_adapters_are_deterministic(self) -> None:
        """Independent adapters produce identical records for one stream."""
        first = self.drive_stream(self.make_platform("first"))
        second = self.drive_stream(self.make_platform("second"))
        self.assertEqual(first, second)

    def test_unknown_timing_handle_names_the_handle(self) -> None:
        """Unknown DR_EVT handles become clear Python KeyErrors."""
        platform = self.make_platform("unknown")
        with self.assertRaisesRegex(KeyError, "99"):
            platform.timings([99])


if __name__ == "__main__":
    unittest.main()
