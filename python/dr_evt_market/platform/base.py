################################################################################
#         Copyright 2023 Lawrence Livermore National Security, LLC             #
#         See the top-level LICENSE file for details.                          #
#                                                                              #
#         SPDX-License-Identifier: MIT                                         #
################################################################################

"""One real machine exposing a share of itself as a DR_EVT simulation."""

import math
from numbers import Real
from pathlib import Path

# The market submits only jobs that start at once, so dr_evt's queue statistics are
# empty by construction; these are the ones that say something.
_STATISTIC_FIELDS = ("jobs_completed", "utilization", "makespan")


class Platform:
    """One real machine, exposing a share of itself as a dr_evt simulation."""

    name: str
    total_nodes: int
    price_per_node_hour: float
    hardware: frozenset[str]
    speed: dict[str, float]

    def __init__(self, work_dir: str | Path, share: float = 1.0) -> None:
        """Create the platform's header file and streaming simulation."""
        if (
            isinstance(share, bool)
            or not isinstance(share, Real)
            or not 0 <= share <= 1
        ):
            raise ValueError("share must be a number in [0, 1]")

        import dr_evt

        self.share = float(share)
        self.exposed_nodes = round(self.total_nodes * self.share)
        self._work_dir = Path(work_dir).resolve()
        self._work_dir.mkdir(parents=True, exist_ok=True)
        self._header_path = self._work_dir / f"{self.name}.csv"
        self._header_path.write_text(
            "job_submit_time,num_nodes,q_id,time_limit\n", encoding="utf-8"
        )

        params = dr_evt.SimParams()
        params.infile = str(self._header_path)
        params.total_nodes = self.exposed_nodes
        params.trace_format = "simple"
        params.timestamp_format = "epoch"
        params.run_time_mode = dr_evt.RunTimeMode.LIMIT
        params.backfill_policy = dr_evt.BackfillPolicy.EASY
        params.priority_policy = dr_evt.PriorityPolicy.FCFS
        self._params = params
        self._simulation = dr_evt.Simulation(params)
        self._dr_evt = dr_evt
        self._last_time_s = 0

    def fits(self, job) -> bool:
        """Return whether a job's hardware and node demand fit this platform."""
        return job.requires <= self.hardware and job.num_nodes <= self.exposed_nodes

    def job_speed(self, job) -> float:
        """Return this platform's speed for a job's hardware."""
        hardware = "gpu" if "gpu" in job.requires else "cpu"
        return self.speed[hardware]

    def run_time(self, job) -> int:
        """Return the integer time for which this job holds nodes."""
        return max(1, math.ceil(job.limit_s / self.job_speed(job)))

    def cost(self, job) -> float:
        """Return the platform cost of the job's priced reservation."""
        return (
            self.price_per_node_hour
            * job.num_nodes
            * job.limit_s
            / self.job_speed(job)
            / 3600
        )

    def advance_to(self, time_s: int) -> None:
        """Advance the simulation to a nondecreasing integer time."""
        if isinstance(time_s, bool) or not isinstance(time_s, int):
            raise ValueError("advance time must be an integer")
        if time_s < self._last_time_s:
            raise ValueError("advance time cannot move backwards")
        self._simulation.advance_to(time_s)
        self._last_time_s = time_s

    def free_nodes(self) -> int:
        """Return the number of nodes currently available."""
        return int(self._simulation.get_available_nodes())

    def waiting(self) -> int:
        """Return the number of jobs waiting for scheduler placement."""
        return int(self._simulation.get_active_job_count())

    def submit(self, jobs, time_s: int) -> None:
        """Submit jobs at one time, in their given order."""
        requests = [
            self._dr_evt.JobAppendRequest(
                time_s, job.num_nodes, "1", self.run_time(job)
            )
            for job in jobs
        ]
        if requests:
            self._simulation.append_jobs(requests)

    def statistics(self) -> dict[str, float]:
        """Return the simulation's completed jobs, utilization and makespan."""
        statistics = self._simulation.get_statistics()
        return {field: float(getattr(statistics, field)) for field in _STATISTIC_FIELDS}
