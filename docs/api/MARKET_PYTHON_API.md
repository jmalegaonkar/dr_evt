# Market Python API

`dr_evt_market` is a market over DR_EVT clusters. A jobs file with bids goes to
the client, which is the auction; the auction places each job on a platform;
the platform's own DR_EVT scheduler runs it. The package has three layers: the
platform contract and its two adapters, the market's data model with its
mechanisms, and the controller that runs windows over a job stream.

The package imports without `dr_evt`, gRPC, NumPy, SciPy or torch. Each of
those is imported by the adapter, mechanism or module that needs it.

## Installation

From a source checkout, install the package editable from the `python/`
directory with the extras you need:

```bash
python3 -m pip install -e "python[grpc,mechanisms]"
```

`grpc` covers the gRPC adapter, `mechanisms` covers VCG (NumPy and SciPy),
`learned` covers RegretFormer (torch). Building a wheel from `python/` also
invokes the extension's CMake build, so install editable only.

## The platform contract

Everything the market needs from a cluster goes through six methods and five
frozen records in `dr_evt_market.platforms.base`. Times are integer seconds on
the way in and floats on the way out.

| Type | Purpose |
|---|---|
| `SubmitRequest` | One leg for a platform: key, submit time, nodes, limit, queue id. |
| `JobTiming` | What the platform did with one leg: handle, key, submit, begin, end, limit, actual run time, nodes, scheduled. |
| `PlatformSnapshot` | Free, in-use and waiting counts and utilization at one time. |
| `PlatformReport` | Every timing, DR_EVT's statistics and the two output paths. |
| `PlatformSession` | Protocol implemented by both adapters. |

| Method | Behavior |
|---|---|
| `now()` | The platform clock. |
| `submit(jobs)` | Validate one batch, append it in one call, return a handle per leg; appending only queues. |
| `advance_to(time_s)` | Process every event up to and including that time. |
| `snapshot()` | The snapshot at the current time; call it after `advance_to()`. |
| `timings(handles)` | Timing records in request order. |
| `finish()` | Drain, write both traces, close the adapter, return the cached report. |

The adapters refuse what DR_EVT would accept silently, with four error types:
`ClockViolation` (fractional, backward or out-of-order time),
`StructuralRejection` (a leg that can never fit, or a non-positive size or
limit), `ConfigurationError` (a bad queue id, policy, name or address), and
`InfrastructureFailure` (a process, stream or file failure, or any call after
`finish()`). An unknown timing handle raises `KeyError` naming the handle.

`InProcessPlatform(name, total_nodes, work_dir, *, backfill="easy")` owns one
`dr_evt.Simulation` in limit run time mode with EASY backfill and FCFS
priority; `backfill` accepts only `"easy"` today.
`GrpcPlatform(name, total_nodes, address, work_dir_on_server, *, session_name)`
is the same contract over a DR_EVT server session; `work_dir_on_server` must be
a directory both sides can reach, since the client writes the header file there.
`ServerProcess(binary, work_dir, address=None)` starts a local `dr_evt_server`
(pass `None` for `binary` to search the install prefix), writes its output to
`work_dir/server.log`, and stops it on exit.

## The market's data model

Eight records in `dr_evt_market.mechanisms` describe the market.

| Type | Meaning |
|---|---|
| `Platform` | A cluster as the market sees it: `name`, `total_nodes`, `price_per_node_hour`, `hardware` tags, optional `address`. `cost(num_nodes, limit_s)` is the posted cost. |
| `Leg` | Part of a job: `leg_id`, `num_nodes`, `limit_s`, `requires` tags. `fits(platform)` needs the tags and the nodes. |
| `Job` | `job_id`, `submit_s`, `legs`, `bid`. |
| `Placement` | One way to run a job: a platform per leg, `cost_credits`, `value_credits`; `id` joins the platforms with `+`. |
| `Window` | One clearing: `time_s`, `index`, `platforms`, `free_nodes`, `jobs`, `candidates` per job. |
| `Decision` | `job_id`, `placement`, `charge_credits`. |
| `Rejection` | `job_id`, `reason`, `time_s`, for jobs and for decisions. |
| `Mechanism` | Abstract base: `name` and `decide(window)`. |

The value model: a job's base cost is its cost on the cheapest placement it can
use. A `bid` of `1.5` means the job would pay up to one and a half times its
base cost wherever it runs; value is the same on every placement, so the job
prefers cheaper platforms unless it says otherwise. A `bid` that is a mapping
from platform name to multiplier values each platform separately: the value of
a placement is the sum over legs of the leg's cost on its platform times that
platform's multiplier, and platforms without a multiplier are not used. A
placement whose value does not cover its cost is never offered. Whatever the
mechanism, a winner pays the placement's cost plus a premium, and the premium
never exceeds value minus cost.

Functions: `placements(job, platforms, free_nodes)` lists every placement whose
legs have their hardware and fit, together on shared platforms, in the free
nodes; `base_cost(job, platforms)` gives the cheapest structural placement's
cost; `build_window(time_s, index, queue, platforms, free_nodes)` offers every
queued job its placements; `demand(job, placement)` gives the nodes per
platform; `validate_decisions(window, decisions)` keeps the decisions the window
can honour in decision order and names each refusal (`duplicate_job`,
`unknown_job`, `unknown_placement`, `charge_above_value`, `charge_below_cost`,
`over_capacity`), a refused decision consuming no capacity;
`submit(decisions, window, sessions, time_s)` sends one batch per platform and
returns the handle of every leg.

`Vcg` maximizes total net value exactly, one binary per (job, candidate), solved
to a zero gap, and charges each winner its cost plus the Clarke pivot, the net
value the other jobs lose because of it, found by solving the window again
without that job.

## Running the market

`platforms.csv` follows DR_EVT's `sync_systems.csv`: `system_id`,
`total_nodes`, `price_per_node_hour`, an optional `hardware` column of
space-separated tags, and an optional `address` (blank for in process).

```text
system_id,total_nodes,price_per_node_hour,hardware,address
alpha,100,1.0,cpu,
gamma,40,3.0,cpu gpu,127.0.0.1:50062
```

`jobs.csv` is a DR_EVT trace, `job_submit_time`, `num_nodes`, `time_limit`,
with a `job_id`, an optional `leg_id` (one row per leg, in order, as in DR_EVT's
composite format), an optional `requires` column of hardware tags, and the bid:
either one `bid` column, or one `bid:<platform>` column per platform, blank
meaning the platform is not bid on.

```text
job_id,job_submit_time,num_nodes,time_limit,leg_id,requires,bid
job-1,0,20,60,0,,1.5
job-2,30,10,90,left,,2.0
job-2,30,15,90,right,gpu,2.0
```

`Controller(sessions, platforms, mechanism, jobs, *, window_s, seed=0,
log_dir=None)` clears the market at fixed boundaries. At each boundary it
advances every platform, admits arrivals, reads free nodes, builds the window,
asks the mechanism, validates, submits, advances again, and confirms from the
timing records that every routed leg began at the boundary; a leg that did not
raises `RoutingError`. Intake rejects jobs that fit nowhere (`oversize`) or whose
value is below cost on every placement (`unaffordable`); a queue that nothing
places while every platform is idle is rejected as `unplaceable`. The market
never uses start times to decide: winners are handed to the platforms they won,
and the timing records are read back as the outcome. `RunReport` holds
`routed` (`RoutedLeg`: job, leg, platform, window, handle, cost, value,
premium, charge, submit, begin, end), `windows` (`WindowRecord`), `rejected`
(`Rejection`), the platforms' reports and the configuration. `write_outputs`
writes `routed.csv`, `windows.csv`, `rejected.csv` and `run.json` with the
platform statistics and the SHA-256 of the two CSVs. With `log_dir` set, one
`window_NNNNNN.json` per window is written for training.

Run the checked-in examples with:

```bash
python -m dr_evt_market run \
  --jobs python/examples/market/market_jobs.csv \
  --platforms python/examples/market/market_platforms.csv \
  --out market-run --window 60 --mechanism vcg --seed 0
```

The command prints the window, routed and rejected counts followed by the two
hashes. For each nonblank address it connects to a gRPC server and uses
`OUT/servers/<system_id>` as the server-visible work directory; add
`--start-servers` to launch a local `ServerProcess` at each listed address and
`--server-binary PATH` to choose the executable. Add `--log-windows` to write
the window files beside `windows.csv`.

## Learned mechanism

`RegretFormer` is available lazily from `dr_evt_market.mechanisms`; importing
the package does not import torch. `RegretFormer(None, seed=N)` builds a
deterministic random network; a checkpoint path loads a trained one. The
network sees a padded (job, candidate) grid. Its private channel is the
candidate's value over the job's cheapest candidate cost; five public channels
follow: normalized cost, `log1p` of nodes, the longest leg limit in hours, the
smallest `free / (free + demand)` across used platforms, and the number of legs.
A job's report is its multipliers, one for a single bid or one per platform,
and every candidate's value is a fixed linear function of them, so a misreport
is always a report the job could have made.

Deployment sorts candidate cells by allocation probability, accepts feasible
placements greedily with at most one per job, and charges the greater of cost
and the learned fraction of value, capped at value. Checkpoints are
dictionaries written by `torch.save` with `state_dict`, `in_channels`, `hid`,
`hid_att`, `n_layers`, `n_heads`, `trained_on` and `created`.

`TrainConfig` and `Trainer` support welfare or revenue objectives, a capacity
penalty, inner misreport ascent, and a single dual variable for an annealed
regret budget. Windows come from `synthetic_structures` (seeded platforms, jobs
and multipliers) or from a run made with `--log-windows`:

```bash
python -m dr_evt_market train --windows synthetic --out regretformer.pt --epochs 20
python -m dr_evt_market run --jobs jobs.csv --platforms platforms.csv \
  --out market-run --window 60 --log-windows
python -m dr_evt_market train --windows market-run --out regretformer.pt --epochs 20
```

`grid_regret` scales each report dimension separately and then the whole
report over a fixed grid; `guided_refinement_regret` starts gradient ascent
from the grid argmax, perturbed and random reports, and rescores every refined
report through the deployed pipeline. Following You et al. (2026), this is a
lower bound on true regret, not a proof of incentive compatibility.

## Testing

After building and installing DR_EVT, run the market tests with:

```bash
./tests/run_market_tests.sh
```

The suite covers both adapters, parity between them, the data model and
validation, placements and submission, VCG against exhaustive search and
misreports, the controller on both transports with pinned outputs, and, when
torch is installed, the window tensor, deployment, regret and training.
