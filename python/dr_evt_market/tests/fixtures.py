################################################################################
#         Copyright 2023 Lawrence Livermore National Security, LLC             #
#         See the top-level LICENSE file for details.                          #
#                                                                              #
#         SPDX-License-Identifier: MIT                                         #
################################################################################

"""Shared schedules and batch-mode CLI oracle helpers for adapter tests."""

from collections.abc import Sequence
import csv
import os
from pathlib import Path
import subprocess

from dr_evt_market import ConfigurationError, InfrastructureFailure, SubmitRequest

CONTENDED_JOBS = tuple(
    SubmitRequest(
        key=f"job-{index}",
        submit_s=index * 10,
        num_nodes=num_nodes,
        limit_s=limit_s,
    )
    for index, (num_nodes, limit_s) in enumerate(zip(
        (20, 30, 15, 40, 25, 10, 35, 60, 20, 45),
        (200, 150, 300, 100, 250, 80, 180, 220, 90, 160),
    ))
)

# From DR_EVT's python/examples/sample_trace.csv under EASY and FCFS, walked
# by hand in learn/01_repo_tour.ipynb.
EXPECTED_BEGIN_TIMES = (0, 10, 20, 160, 160, 50, 260, 410, 260, 630)


def _find_simulator() -> Path:
    repo_root = Path(__file__).resolve().parents[3]
    configured_prefix = os.environ.get("CMAKE_INSTALL_PREFIX")
    if configured_prefix:
        candidates = [Path(configured_prefix) / "bin" / "simulator"]
    else:
        candidates = [
            repo_root / "install" / "bin" / "simulator",
            repo_root / "build" / "simulator",
        ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise ConfigurationError(
        "simulator binary not found: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def cli_schedule(
    work_dir: str | Path,
    jobs: Sequence[SubmitRequest],
) -> list[dict[str, str]]:
    """Run the batch-mode simulator CLI and return its schedule rows."""
    directory = Path(work_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    input_path = directory / "cli.input.csv"
    output_path = directory / "cli.simulated.csv"
    with input_path.open("w", newline="", encoding="utf-8") as input_file:
        writer = csv.writer(input_file)
        writer.writerow(("job_submit_time", "num_nodes", "time_limit"))
        for job in jobs:
            writer.writerow((job.submit_s, job.num_nodes, job.limit_s))

    command = [
        str(_find_simulator()),
        str(input_path),
        "--total_nodes",
        "100",
        "--trace_format",
        "simple",
        "--timestamp_format",
        "epoch",
        "--run_time_mode",
        "limit",
        "--backfill_policy",
        "easy",
        "--priority_policy",
        "fcfs",
        "--msec_output",
        "--outfile",
        str(output_path),
    ]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise InfrastructureFailure(f"simulator CLI failed: {detail}")

    with output_path.open(newline="", encoding="utf-8") as output_file:
        return list(csv.DictReader(output_file))
