################################################################################
#         Copyright 2023 Lawrence Livermore National Security, LLC             #
#         See the top-level LICENSE file for details.                          #
#                                                                              #
#         SPDX-License-Identifier: MIT                                         #
################################################################################

"""Cross-adapter parity tests for the market platform contract."""

import csv
from pathlib import Path
import tempfile
import unittest

from dr_evt_market import (
    GrpcPlatform,
    InProcessPlatform,
    PlatformReport,
    ServerProcess,
)
from dr_evt_market.tests.fixtures import CONTENDED_JOBS


def _drive(platform: InProcessPlatform | GrpcPlatform) -> PlatformReport:
    for job in CONTENDED_JOBS:
        platform.advance_to(job.submit_s)
        platform.submit([job])
        platform.advance_to(job.submit_s)
    return platform.finish()


def _trace_rows(path: str | None) -> list[dict[str, str]]:
    if path is None:
        raise AssertionError("adapter report omitted its simulated trace path")
    with Path(path).open(newline="", encoding="utf-8") as trace_file:
        return list(csv.DictReader(trace_file))


class AdapterParityTests(unittest.TestCase):
    """Compare plain in-process and gRPC adapters on one contended stream."""

    def test_timing_records_and_simulated_trace_rows_match(self) -> None:
        """Both transport paths produce the same schedule and trace rows."""
        with tempfile.TemporaryDirectory(
            prefix="dr_evt_market_parity_"
        ) as directory:
            root = Path(directory)
            server_work = root / "server"
            with ServerProcess(None, server_work) as server:
                in_process = InProcessPlatform(
                    "in-process",
                    100,
                    root / "in-process",
                )
                remote = GrpcPlatform(
                    "grpc",
                    100,
                    server.address,
                    server_work,
                    session_name="parity",
                )
                in_process_report = _drive(in_process)
                remote_report = _drive(remote)

            self.assertEqual(
                in_process_report.timings,
                remote_report.timings,
            )
            self.assertEqual(
                _trace_rows(in_process_report.simulated_trace_path),
                _trace_rows(remote_report.simulated_trace_path),
            )


if __name__ == "__main__":
    unittest.main()
