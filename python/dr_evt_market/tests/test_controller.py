################################################################################
#         Copyright 2023 Lawrence Livermore National Security, LLC             #
#         See the top-level LICENSE file for details.                          #
#                                                                              #
#         SPDX-License-Identifier: MIT                                         #
################################################################################

"""End-to-end tests for market input, routing, and deterministic output."""

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

_DATA_DIR = Path(__file__).with_name("data")
_ROUTED_SHA256 = (
    "1c1f53716b1fcf6e0cbca38f0c66884b6681f24e09ca4fec10f308c8f97ccf1d"
)


class ControllerTests(unittest.TestCase):
    """Exercise the complete market pipeline over both platform transports."""

    def setUp(self) -> None:
        """Create an isolated output root and read the shared input files."""
        self._temporary_directory = tempfile.TemporaryDirectory(
            prefix="dr_evt_market_controller_"
        )
        self.addCleanup(self._temporary_directory.cleanup)
        self.root = Path(self._temporary_directory.name)
        self.specs = read_platforms(_DATA_DIR / "market_platforms.csv")
        self.jobs, self.bids = read_jobs(
            _DATA_DIR / "market_jobs.csv",
            self.specs,
        )
        self.prices = {
            spec.system_id: spec.price_per_node_hour
            for spec in self.specs
        }

    def _inprocess_platforms(self, root: Path) -> dict:
        return {
            spec.system_id: InProcessPlatform(
                spec.system_id,
                spec.total_nodes,
                root / spec.system_id,
            )
            for spec in self.specs
        }

    def _run(self, platforms: dict, output_dir: Path) -> tuple:
        report = Controller(
            platforms,
            self.prices,
            Vcg(),
            self.jobs,
            self.bids,
            window_s=60,
            seed=11,
        ).run()
        paths = write_outputs(report, output_dir)
        return report, paths

    def test_vcg_pipeline_writes_verified_outputs(self) -> None:
        """The fixture routes immediately and writes pinned deterministic CSVs."""
        report, paths = self._run(
            self._inprocess_platforms(self.root / "platforms"),
            self.root / "outputs",
        )

        self.assertTrue(report.routed)
        self.assertTrue(
            any(len(window.queued) > len(window.placed) for window in report.windows)
        )
        for route in report.routed:
            self.assertEqual(route.begin_s, route.window_time_s)
            self.assertGreaterEqual(
                route.charge_credits,
                route.resource_cost_credits,
            )
            self.assertLessEqual(route.charge_credits, route.value_credits)

        composite = [route for route in report.routed if route.job_id == "c01"]
        self.assertEqual(len(composite), 2)
        self.assertEqual(
            {route.begin_s for route in composite},
            {composite[0].window_time_s},
        )

        rejected_path = Path(paths["rejected"])
        with rejected_path.open(newline="", encoding="utf-8") as input_file:
            rejected = list(csv.DictReader(input_file))
        self.assertIn(
            {"job_id": "s12", "reason": "unaffordable", "time_s": "600"},
            rejected,
        )

        routed_path = Path(paths["routed"])
        digest = hashlib.sha256(routed_path.read_bytes()).hexdigest()
        self.assertEqual(digest, _ROUTED_SHA256)
        manifest = json.loads(
            Path(paths["manifest"]).read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["sha256"]["routed.csv"], digest)

    def test_one_grpc_platform_matches_inprocess_output(self) -> None:
        """Replacing one platform with gRPC leaves routed.csv byte-identical."""
        _, local_paths = self._run(
            self._inprocess_platforms(self.root / "local-platforms"),
            self.root / "local-output",
        )

        server_dir = self.root / "grpc-server"
        with ServerProcess(None, server_dir) as server:
            mixed_platforms = {
                spec.system_id: (
                    GrpcPlatform(
                        spec.system_id,
                        spec.total_nodes,
                        server.address,
                        server_dir,
                        session_name=spec.system_id,
                    )
                    if spec.system_id == "alpha"
                    else InProcessPlatform(
                        spec.system_id,
                        spec.total_nodes,
                        self.root / "mixed-platforms" / spec.system_id,
                    )
                )
                for spec in self.specs
            }
            _, mixed_paths = self._run(
                mixed_platforms,
                self.root / "mixed-output",
            )

        self.assertEqual(
            Path(local_paths["routed"]).read_bytes(),
            Path(mixed_paths["routed"]).read_bytes(),
        )


if __name__ == "__main__":
    unittest.main()
