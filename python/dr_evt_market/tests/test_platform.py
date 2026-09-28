################################################################################
#         Copyright 2023 Lawrence Livermore National Security, LLC             #
#         See the top-level LICENSE file for details.                          #
#                                                                              #
#         SPDX-License-Identifier: MIT                                         #
################################################################################

"""Tests for the platform class and the five machines."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from dr_evt_market import (
    Corona,
    Dane,
    DEFAULT_FEDERATION,
    Matrix,
    PLATFORMS,
    Tioga,
    Tuolumne,
    federation,
)


def _job(num_nodes: int, limit_s: int, *requires: str) -> SimpleNamespace:
    """Build the minimal job shape accepted by a platform."""
    return SimpleNamespace(
        num_nodes=num_nodes,
        limit_s=limit_s,
        requires=frozenset(requires),
    )


class PlatformTests(unittest.TestCase):
    """Exercise platform profiles and their streaming simulations."""

    def test_profiles_expose_a_share_and_validate_it(self) -> None:
        """Each profile sizes its simulation from a valid share."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for profile in PLATFORMS:
                with self.subTest(profile=profile.name):
                    platform = profile(root / profile.name, share=0.05)
                    expected = max(1, round(profile.total_nodes * 0.05))
                    self.assertEqual(platform.exposed_nodes, expected)

            with self.assertRaises(ValueError):
                Dane(root / "zero", share=0)
            with self.assertRaises(ValueError):
                Dane(root / "large", share=1.5)

    def test_jobs_start_together_and_release_nodes(self) -> None:
        """Jobs that fit together start immediately and release at their limit."""
        with tempfile.TemporaryDirectory() as directory:
            platform = Tuolumne(directory, share=0.1)
            platform.advance_to(0)
            self.assertEqual(platform.free_nodes(), platform.exposed_nodes)

            jobs = [_job(20, 10, "gpu"), _job(30, 10, "cpu")]
            self.assertEqual(len(platform.submit(jobs, 0)), 2)
            platform.advance_to(0)
            self.assertEqual(platform.waiting(), 0)
            self.assertEqual(platform.free_nodes(), platform.exposed_nodes - 50)

            platform.advance_to(10)
            self.assertEqual(platform.free_nodes(), platform.exposed_nodes)

    def test_waiting_job_starts_when_nodes_return(self) -> None:
        """A queued job starts as soon as a running job releases enough nodes."""
        with tempfile.TemporaryDirectory() as directory:
            platform = Matrix(directory, share=0.1)
            jobs = [_job(2, 10, "cpu"), _job(2, 20, "gpu")]
            platform.submit(jobs, 0)
            platform.advance_to(0)
            self.assertEqual(platform.waiting(), 1)
            self.assertEqual(platform.free_nodes(), platform.exposed_nodes - 2)

            platform.advance_to(10)
            self.assertEqual(platform.waiting(), 0)
            self.assertEqual(platform.free_nodes(), platform.exposed_nodes - 2)

    def test_fits_and_cost_use_capacity_hardware_and_price(self) -> None:
        """Fit checks tags and exposed nodes, while cost uses the profile price."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corona = Corona(root / "corona", share=0.1)
            dane = Dane(root / "dane", share=0.1)
            matrix = Matrix(root / "matrix", share=0.1)
            gpu_job = _job(2, 3600, "gpu")

            self.assertTrue(corona.fits(gpu_job))
            self.assertFalse(dane.fits(gpu_job))
            self.assertTrue(matrix.fits(gpu_job))
            self.assertFalse(matrix.fits(_job(matrix.exposed_nodes + 1, 1)))
            self.assertEqual(matrix.cost(gpu_job), 3.2)
            self.assertEqual(
                [profile.hardware for profile in PLATFORMS],
                [
                    frozenset({"cpu", "gpu"}),
                    frozenset({"cpu"}),
                    frozenset({"cpu", "gpu"}),
                    frozenset({"cpu", "gpu"}),
                    frozenset({"cpu", "gpu"}),
                ],
            )

    def test_federation_preserves_profile_order(self) -> None:
        """The federation contains one instance of each named profile in order."""
        with tempfile.TemporaryDirectory() as directory:
            platforms = federation(directory, share=0.1)
            self.assertEqual(tuple(platforms), DEFAULT_FEDERATION)
            self.assertEqual(
                [type(platform) for platform in platforms.values()],
                [Corona, Dane, Matrix, Tioga, Tuolumne],
            )
            self.assertEqual(
                {name: value.exposed_nodes for name, value in platforms.items()},
                {
                    "corona": 12,
                    "dane": 154,
                    "matrix": 3,
                    "tioga": 3,
                    "tuolumne": 115,
                },
            )
            self.assertEqual(
                {name: value.price_per_node_hour for name, value in platforms.items()},
                {
                    "corona": 1.5,
                    "dane": 0.18,
                    "matrix": 1.6,
                    "tioga": 2.7,
                    "tuolumne": 0.19,
                },
            )
            selected = federation(directory, share=0.05, names=("dane", "corona"))
            self.assertEqual(list(selected), ["dane", "corona"])
            self.assertEqual(
                [type(value) for value in selected.values()], [Dane, Corona]
            )
            with self.assertRaises(ValueError):
                federation(directory, names=("unknown",))


if __name__ == "__main__":
    unittest.main()
