"""Two measured facts about the custom FCFS scheduler and its prediction horizon.

1. With a cost function returning the arrival index and a selector that picks the
   lowest cost, the custom scheduler reproduces standard EASY exactly.
2. get_prediction_horizon() estimates when the whole waiting queue drains; it is not a
   per-job start prediction (a small job appended at the tail can backfill much earlier).
"""
import os
import sys
import tempfile
from pathlib import Path

INSTALL = os.environ.get("DR_EVT_INSTALL", str(Path(__file__).resolve().parents[2] / "install"))
sys.path.insert(0, INSTALL + "/lib/python")
import dr_evt  # noqa: E402

QUEUE = "pbatch" if getattr(dr_evt, "legacy_queue_input", True) else "1"
print(f"dr_evt from {INSTALL}, queue value {QUEUE!r}")

header_only = Path(tempfile.mkdtemp()) / "header_only.csv"
header_only.write_text("job_submit_time,num_nodes,time_limit\n")
JOBS = [(0, 20, 200), (10, 30, 150), (20, 15, 300), (30, 40, 100), (40, 25, 250),
        (50, 10, 80), (60, 35, 180), (70, 60, 220), (80, 20, 90), (90, 45, 160)]


def make_params(num_max_candidates=None):
    params = dr_evt.SimParams()
    params.infile = str(header_only)
    params.total_nodes = 100
    params.trace_format = "simple"
    params.timestamp_format = "epoch"
    params.run_time_mode = dr_evt.RunTimeMode.LIMIT
    params.backfill_policy = dr_evt.BackfillPolicy.EASY
    params.priority_policy = dr_evt.PriorityPolicy.FCFS
    if num_max_candidates is not None:
        params.num_max_candidates = num_max_candidates
    return params


def begin_times(sim):
    ids = []
    for submit, nodes, limit in JOBS:
        ids.append(sim.append_job(float(submit), nodes, QUEUE, float(limit)))
        sim.advance_to(float(submit))
    sim.advance_to(1e6)
    return [int(t.begin_time) for t in sim.get_job_timings(ids)]


def arrival_order_cost(job_id, submit_time, run_time, nodes):
    return int(job_id)              # job_cost_t is an int


def lowest_cost(candidates):
    return min(candidates, key=lambda c: c[1])[0]


keep = []                            # SimParams must outlive their Simulation
standard_params = make_params(); keep.append(standard_params)
standard = begin_times(dr_evt.Simulation(standard_params))
print("standard EASY begin times:", standard)
for k in (1, 4, 64):
    params = make_params(k); keep.append(params)
    custom = begin_times(dr_evt.Simulation(params, arrival_order_cost, lowest_cost))
    print(f"custom scheduler, FCFS-order selector, num_max_candidates={k}: identical to EASY: {custom == standard}")

print("\nprediction horizon versus actual starts")
params = make_params(64); keep.append(params)
sim = dr_evt.Simulation(params, arrival_order_cost, lowest_cost)
ids = []
for submit, nodes, limit in JOBS:
    ids.append(sim.append_job(float(submit), nodes, QUEUE, float(limit)))
    sim.advance_to(float(submit))
window = sim.get_backfill_window()
print(f"at t=90: waiting={sim.get_active_job_count()} free={window.available_nodes} shadow={window.shadow_time}")
for utilization in (1.0, 0.8):
    horizon = sim.get_prediction_horizon(utilization)
    print(f"  prediction_horizon(utilization={utilization}) = {horizon:.1f}, queue predicted to drain by t={window.shadow_time + horizon:.1f}")
tail = sim.append_job(90.0, 30, QUEUE, 100.0)
sim.advance_to(90.0)
sim.advance_to(1e6)
timings = sim.get_job_timings(ids + [tail])
print(f"  actual: last queued job began at {int(timings[9].begin_time)}, "
      f"the 30-node job appended at the tail began at {int(timings[-1].begin_time)} (it backfilled)")
