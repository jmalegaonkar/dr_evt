################################################################################
#         Copyright 2023 Lawrence Livermore National Security, LLC             #
#         See the top-level LICENSE file for details.                          #
#                                                                              #
#         SPDX-License-Identifier: MIT                                         #
################################################################################

"""Tests for the pure platform adapter contract."""

import unittest
from dataclasses import FrozenInstanceError

from dr_evt_market import (
    ClockViolation,
    ConfigurationError,
    InfrastructureFailure,
    JobTiming,
    PlatformReport,
    PlatformSnapshot,
    StructuralRejection,
    SubmitRequest,
    validate,
)


class ContractDataclassTests(unittest.TestCase):
    """Verify that contract values are immutable snapshots."""

    def test_dataclasses_are_frozen(self) -> None:
        """Every public contract dataclass rejects field assignment."""
        timing = JobTiming(
            handle=2,
            key="job-2",
            submit_s=1.0,
            begin_s=3.0,
            end_s=8.0,
            limit_s=5,
            actual_run_s=5.0,
            num_nodes=4,
            scheduled=True,
        )
        values = [
            (SubmitRequest("job-1", 0, 2, 5), "key", "changed"),
            (timing, "begin_s", 4.0),
            (
                PlatformSnapshot(
                    name="cluster",
                    time_s=3,
                    total_nodes=16,
                    free_nodes=12,
                    in_use_nodes=4,
                    waiting_jobs=0,
                ),
                "time_s",
                4,
            ),
            (
                PlatformReport(
                    name="cluster",
                    timings=(timing,),
                    statistics={"jobs_completed": 1.0},
                    simulated_trace_path=None,
                    resource_trace_path=None,
                ),
                "name",
                "changed",
            ),
        ]

        for value, attribute, replacement in values:
            with self.subTest(type=type(value).__name__):
                with self.assertRaises(FrozenInstanceError):
                    setattr(value, attribute, replacement)


class ContractErrorTests(unittest.TestCase):
    """Verify the public error hierarchy and request validation."""

    def test_error_bases(self) -> None:
        """Client errors are ValueErrors and infrastructure errors are not."""
        for error_type in (
            ClockViolation,
            ConfigurationError,
            StructuralRejection,
        ):
            with self.subTest(error=error_type.__name__):
                self.assertTrue(issubclass(error_type, ValueError))
        self.assertTrue(issubclass(InfrastructureFailure, RuntimeError))

    def test_validate_refuses_fractional_submit_time(self) -> None:
        """Fractional submission timestamps fail before reaching DR_EVT."""
        request = SubmitRequest("job", 1.5, 2, 10)

        with self.assertRaisesRegex(ClockViolation, "integer"):
            validate(request)

    def test_validate_accepts_a_well_formed_request(self) -> None:
        """A request matching the fixed contract passes shape validation."""
        validate(SubmitRequest("job", 1, 2, 10, q_id="1"))


if __name__ == "__main__":
    unittest.main()
