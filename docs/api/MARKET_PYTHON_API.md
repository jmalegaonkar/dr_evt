# Market Python API

The `dr_evt_market` package provides one contract for driving DR_EVT either
through the in-process Python extension or through a gRPC server.

The package itself imports without loading `dr_evt` or gRPC. Those dependencies
are loaded only when their corresponding adapter is constructed.

## Installation

From a source checkout, install the package editable from the `python/`
directory with its gRPC and mechanism extras:

```bash
python3 -m pip install -e "python[grpc,mechanisms]"
```

The in-process adapter also requires the built `dr_evt` extension on
`PYTHONPATH`. The gRPC adapter requires a server built with
`DR_EVT_ENABLE_GRPC=ON`.

Building a wheel from `python/` also invokes the extension's CMake build.

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
| `SubmitRequest` | Client key, submit time, node demand, limit, and queue ID. |
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
| `finish()` | Drain work, write traces, close the adapter, and cache its report. |

The adapters reject malformed requests before appending any part of a batch:

- `ClockViolation` covers fractional, backward, or out-of-order time;
- `ConfigurationError` covers invalid policy, queue, name, or binary settings;
- `StructuralRejection` covers nonpositive demand or limits and jobs larger
  than the platform; and
- `InfrastructureFailure` covers process, stream, file, and server failures.

An unknown or unavailable timing handle raises `KeyError` naming the handle.
After `finish()`, every method except `finish()` raises
`InfrastructureFailure`.

## In-process adapter

`InProcessPlatform(name, total_nodes, work_dir, *, backfill="easy")` owns a
`dr_evt.SimParams` and standard `Simulation` for its whole lifetime. It runs
LIMIT mode with EASY backfill and FCFS priority. The `backfill` argument accepts
only `"easy"` today. Snapshots populate instantaneous `current_utilization`.

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

gRPC snapshots populate instantaneous `current_utilization`.

`SessionClient` is the lower-level correlated request wrapper. Most callers
should use `GrpcPlatform` instead.

## Running the market

The platform file gives each platform a node count and public hourly node
price. A blank `address` selects the in-process adapter; a nonblank address
selects gRPC.

```text
system_id,total_nodes,price_per_node_hour,address
alpha,100,1.0,
beta,60,2.0,127.0.0.1:50062
```

The jobs file has one row per leg. Bid columns name acceptable platforms and
give the private value in credits. Blank bids make that platform unacceptable.
Rows sharing a job ID form a composite job in file order.

```text
job_id,job_submit_time,num_nodes,time_limit,leg_id,bid:alpha,bid:beta
job-1,0,20,60,0,10,8
job-2,30,10,90,left,7,9
job-2,30,15,90,right,8,10
```

At each fixed window boundary, the controller advances every platform, admits
arrivals, reads free capacity, asks the mechanism for decisions, validates and
submits accepted placements, and verifies that every routed leg started at the
boundary. Unplaced jobs wait for the next window. The run then drains every
platform.

`Mechanism` is the abstract interface for allocation and charging policies.
`Vcg` maximizes exact net welfare with Clarke pivot charges for windows of at
most 800 candidate variables and uses its documented bounded greedy fallback
for larger windows.

Run the checked-in examples with:

```bash
python -m dr_evt_market run \
  --jobs python/examples/market/market_jobs.csv \
  --platforms python/examples/market/market_platforms.csv \
  --out market-run --window 60 --mechanism vcg --seed 0
```

The output directory receives `routed.csv`, `windows.csv`, `rejected.csv`, and
`run.json`. The command prints window, routed, and rejected counts followed by
the SHA-256 hashes of `routed.csv` and `windows.csv`.

For each nonblank address, the CLI connects to a gRPC server and uses
`OUT/servers/<system_id>` as the server-visible work directory. Add
`--start-servers` to launch one local `ServerProcess` at each listed address.
The optional `--server-binary PATH` selects the executable; otherwise the
normal `ServerProcess` search rules apply.

## Testing

After building and installing DR_EVT, run all market package tests with:

```bash
./tests/run_market_tests.sh
```

The suite covers both adapters, typed validation, output files, exact parity
between plain in-process and gRPC scheduling.
