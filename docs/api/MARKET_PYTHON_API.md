# Market Python API

The `dr_evt_market` package provides one contract for driving DR_EVT either
through the in-process Python extension or through a gRPC server.

The package itself imports without loading `dr_evt` or gRPC. Those dependencies
are loaded only when their corresponding adapter is constructed.

## Installation

From a source checkout, install the package and its gRPC extra with:

```bash
python3 -m pip install -e "python[grpc]"
```

The in-process adapter also requires the built `dr_evt` extension on
`PYTHONPATH`. The gRPC adapter requires a server built with
`DR_EVT_ENABLE_GRPC=ON`.

## Example

```python
from dr_evt_market import InProcessPlatform, SubmitRequest

platform = InProcessPlatform("alpha", 100, "run/alpha")
handle = platform.submit([
    SubmitRequest("job-1", submit_s=0, num_nodes=20, limit_s=60)
])[0]

# Submission enqueues the job. Advancement evaluates it.
platform.advance_to(0)
snapshot = platform.snapshot()
timing = platform.timings([handle])[0]
report = platform.finish()
```

Submission times and advancement targets must be integer seconds. Call
`snapshot()` after `advance_to()` at the same time; a snapshot between submit
and advance describes a queue the scheduler has not evaluated yet.

## Contract types

| Type | Purpose |
|---|---|
| `SubmitRequest` | Client key, submission time, node demand, time limit, and numeric queue ID. |
| `JobTiming` | Submitted job handle and key with observed or projected timing fields. |
| `PlatformSnapshot` | Capacity, queue, and live metric state. |
| `PlatformReport` | Final timings, statistics, and output trace paths. |
| `PlatformSession` | Protocol implemented by both adapters. |

All contract dataclasses are frozen. `SubmitRequest.q_id` defaults to `"1"`
and must be a digit string from `"1"` through `"10"`.

Both adapters implement these methods:

| Method | Behavior |
|---|---|
| `now()` | Return the current integer simulation time. |
| `submit(jobs)` | Validate and append one non-decreasing batch without advancing. |
| `advance_to(time_s)` | Process all scheduler events at or before `time_s`. |
| `snapshot()` | Return current capacity, queue, and live metrics. |
| `timings(handles)` | Return timing records in request order. |
| `finish()` | Drain work, write both traces, and return a cached final report. |

The adapters reject malformed requests before appending any part of a batch:

- `ClockViolation` covers fractional, backward, or out-of-order time;
- `ConfigurationError` covers invalid policy, queue, name, or binary settings;
- `StructuralRejection` covers nonpositive demand or limits and jobs larger
  than the platform; and
- `InfrastructureFailure` covers process, stream, file, and server failures.

An unknown or unavailable timing handle raises `KeyError` naming the handle.

## In-process adapter

`InProcessPlatform(name, total_nodes, work_dir, *, backfill="easy",
use_custom_scheduler=True)` owns a `dr_evt.SimParams` and `Simulation` for its
whole lifetime. It runs LIMIT mode with EASY and FCFS scheduling.

The default custom scheduler preserves FCFS order and exposes `resource_area`
in snapshots. Set `use_custom_scheduler=False` to use DR_EVT's standard
scheduler; that field is then `None`. Both modes populate instantaneous
`current_utilization`.

## gRPC adapter and local server

`GrpcPlatform(name, total_nodes, address, work_dir_on_server, *, session_name)`
creates one streaming server session. `session_name` must be filename safe.
`work_dir_on_server` must be the server's working directory and must be visible
to the client when it needs to read returned output paths.

`ServerProcess(binary, work_dir, address=None)` manages a local
`dr_evt_server` as a context manager. Pass `None` for `binary` to use
`CMAKE_INSTALL_PREFIX` when set, or otherwise search `install/bin` and then
`build`. When no address is given it selects a free localhost port and waits
for the gRPC channel before returning. Server output is written to
`work_dir/server.log`.

gRPC snapshots populate `current_utilization`, but leave `resource_area` as
`None`. The wire statistics resource area is committed scheduled-job area,
while the custom in-process snapshot field is live consumed area, so they are
not interchangeable.

`SessionClient` is the lower-level correlated request wrapper. Most callers
should use `GrpcPlatform` instead.

## Testing

After building and installing DR_EVT, run all market package tests with:

```bash
./tests/run_market_tests.sh
```

The suite covers both adapters, typed validation, output files, exact parity
between plain in-process and gRPC scheduling.
