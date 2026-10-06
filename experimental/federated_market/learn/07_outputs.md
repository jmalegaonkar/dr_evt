# Step 7: outputs and the command line

A run ends with four files: a ledger of the winners, the jobs set aside, service by
community, and a summary. This page gives each file field by field, with the fixture's
run as the example, and the four commands that prepare traces, run a market, harvest
its windows and train RegretFormer. The overview of all steps is `master_overview.md`.

Code: `write_outputs`, `_service` and the records `RoutedJob`, `Waiting` and `_Service`
in `market.py`, and `cli.py`.

## 7.1 What a run writes

`run` returns a `Result`: the winners, the jobs set aside, each platform's statistics,
the resolved configuration, and each job's community and node count.
`write_outputs(result, out_dir)` writes four files and returns their paths and the
SHA-256 digest of the ledger.

| file | one row per | columns |
|---|---|---|
| `routed.csv` | winner, in the order placed | `job_id`, `platform`, `window`, `cost`, `value`, `premium`, `charge`, `submit_s`, `begin_s`, `end_s` |
| `waiting.csv` | job set aside at intake | `job_id`, `reason`, `submit_s` |
| `service.csv` | community, then `all` | `community`, `count`, `mean_wait_s`, `node_hour_weighted_mean_wait_s`, `mean_bounded_slowdown` |
| `summary.json` | run | `configuration`, `service`, `statistics`, `welfare`, `revenue`, `routed`, `waiting`, `sha256` |

The CSV columns are the fields of `RoutedJob`, `Waiting` and `_Service`, so the files
cannot drift from the code. Money is in the unit of the posted prices, and times are
seconds on the jobs file's clock.

## 7.2 The ledger, `routed.csv`

Rows come window by window, and within a window in batch order.

- `window` is the window's number, `t // window_s`, and `begin_s` its time `t`: every
  winner starts in the window it wins (step 3).
- `end_s` is `begin_s` plus the hold time on that platform,
  `max(1, ceil(limit / speed))`.
- `cost` is the posted price times the node-hours the job uses there, and `value` its
  bid times those node-hours (step 4).
- `charge` is what the mechanism set, and `premium` is `charge - cost`. It is zero for a
  winner that displaces nobody, and negative for a job that bid under the price and pays
  its whole bid.

Seven of the fixture's nineteen rows, under VCG at share 0.1:

| job_id | platform | window | cost | value | premium | charge | submit_s | begin_s | end_s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| j000001 | tuolumne | 0 | 0.6781 | 1.5000 | 0.5146 | 1.1927 | 0 | 0 | 129 |
| j000002 | dane | 0 | 0.6969 | 0.9333 | 0.0930 | 0.7898 | 0 | 0 | 175 |
| j000003 | dane | 0 | 0.2236 | 0.2781 | 0 | 0.2236 | 0 | 0 | 64 |
| j000005 | tuolumne | 0 | 0.0024 | 0.0079 | 0 | 0.0024 | 0 | 0 | 16 |
| j000010 | corona | 2 | 0.0500 | 0.5000 | 0 | 0.0500 | 90 | 120 | 160 |
| j000004 | tuolumne | 3 | 0.3154 | 0.8301 | 0 | 0.3154 | 0 | 180 | 235 |
| j000015 | corona | 3 | 0.0833 | 0.0778 | -0.0056 | 0.0778 | 180 | 180 | 230 |

j000001 and j000002 pay pivots (step 4). j000004 waits from 0 to 180 for Tuolumne's
nodes, and j000010 waits for the window after its arrival at 90. j000015 bid under
Corona's price and pays its whole bid. The file holds full precision; this table rounds.

## 7.3 Jobs set aside, `waiting.csv`

A job that could never run on the federation waits outside the auction for the whole run
(step 5), with its submit time and one of three reasons: `hardware` when no platform has
its hardware, `oversize` when it fits no slice, and `no_bid` when it bid on no platform
that could run it. Every other job runs: the loop ends only when the queue is empty. The
fixture sets aside one job:

| job_id | reason | submit_s |
|---|---|---:|
| j000014 | oversize | 180 |

## 7.4 Service by community, `service.csv`

A routed job's community is its `source`, the trace it came from, or the empty string
when the jobs file has none. For a set $J$ of routed jobs, with wait
$w_j = \text{begin}_j - \text{submit}_j$ and run $r_j = \text{end}_j - \text{begin}_j$:

$$
\text{mean wait} = \frac{1}{|J|} \sum_{j \in J} w_j, \qquad
\text{node-hour weighted mean wait} = \frac{\sum_j h_j\, w_j}{\sum_j h_j},
\quad h_j = \frac{n_j\, r_j}{3600},
$$

$$
\text{mean bounded slowdown} = \frac{1}{|J|} \sum_{j \in J}
\max\!\left(1, \frac{w_j + r_j}{\max(r_j, 10)}\right).
$$

Rows are sorted by community, and the `all` row covers every routed job, not the mean of
the communities. Jobs set aside never enter these measures. The fixture's:

| community | count | mean wait (s) | node-hour weighted mean wait (s) | mean bounded slowdown |
|---|---:|---:|---:|---:|
| corona | 4 | 7.5 | 0.79 | 1.23 |
| dane | 4 | 0.0 | 0.00 | 1.00 |
| matrix | 4 | 7.5 | 1.35 | 1.10 |
| tioga | 3 | 70.0 | 175.83 | 2.86 |
| tuolumne | 4 | 7.5 | 7.86 | 1.19 |
| all | 19 | 15.8 | 25.99 | 1.40 |

The communities are the fixture's sources, not where the jobs ran: Tioga's three jobs
all ran on Tuolumne, and j000004, which waited 180 s, holds 1.68 of their 1.72
node-hours.

## 7.5 The summary, `summary.json`

- `configuration`: the mechanism's name, `window_s`, `prefix`, and for each platform its
  total and exposed nodes, posted price, hardware and speeds.
- `service`: the rows of `service.csv`, keyed by community.
- `statistics`: what each platform's dr_evt simulation measured. `jobs_completed`;
  `makespan`, the platform's latest completion time, counted from 0; and `utilization`,
  the node-seconds allocated over the exposed nodes times that makespan.
- `welfare`: the total of `value - cost` over the winners, and `revenue`: the total of
  the charges, what the users pay.
- `routed` and `waiting`: the two counts, and `sha256`: the digest of `routed.csv`.

The fixture's run has welfare 5.3332, revenue 3.0718, 19 routed and 1 waiting. Corona's
statistics read 2 jobs completed, a makespan of 230 s and a utilization of 0.1159: 320
node-seconds over 12 nodes times 230 s.

## 7.6 Determinism

The same jobs file, federation and mechanism give a byte-identical `routed.csv`, so the
digest identifies a run. `test_routed_output_has_pinned_hash` pins the fixture's; it
changes only when the fixture or the model changes. RegretFormer is deterministic too:
an untrained network is fixed by its seed, and a trained one by its checkpoint.

## 7.7 The command line

`python -m federated_market` has four commands. Each prints `key=value` lines, and an
error ends it with status 2 and an `error:` line.

**`prepare`** turns traces into a jobs file (step 1) and prints the preparation summary:
rows read and kept, the drop counts, kept jobs per source, with `--synthetic` the groups
and synthetic jobs per source, and jobs per persona.

| option | default | meaning |
|---|---|---|
| `--trace NAME=PATH` | required, repeatable | a trace and its source label |
| `--out` | required | the jobs file to write |
| `--format` | `lc` | `lc` or `simple` |
| `--start`, `--hours` | the whole trace | the interval, in the trace's clock |
| `--seed` | 0 | the seed for hardware, bids and synthetic days |
| `--gpu-fraction` | 0.5 | the chance that a job requires GPUs |
| `--requires` | none | one hardware requirement for every job |
| `--platforms` | all five | the platforms bid on |
| `--bids` | `multi` | `multi`, a bid on each platform, or `single` |
| `--synthetic` | off | draw one day from the interval instead |

**`run`** routes a jobs file through one mechanism and writes the four files to `--out`,
and nothing else: dr_evt's input files go to a temporary directory. It prints the
number of windows through the last placement, the routed and waiting counts, and the
ledger's digest.

| option | default | meaning |
|---|---|---|
| `--jobs`, `--out` | required | the jobs file and the output directory |
| `--share` | 1.0 | one share, or `name=share,...` with 1.0 for the others |
| `--prefix` | 32 | jobs auctioned per window |
| `--window` | 60 | the window length in seconds |
| `--mechanism` | `vcg` | `vcg`, `firstprice`, `firstfit` or `regretformer` |
| `--checkpoint` | none | the saved network, required for `regretformer` |
| `--platforms` | all five | the federation |

On the fixture, `run --jobs tests/data/jobs.csv --out results --share 0.1` prints

```text
windows=6
routed=19
waiting=1
routed_sha256=960f2328047bb667fe055682833afaaab500fd2e8a8249d3ad0af68575ac0344
```

**`harvest`** runs each `--jobs` file under each `--mechanism` (repeatable; `vcg` when
none is given, and `regretformer` with `--checkpoint`), every run from idle platforms,
and writes every window with jobs queued to `--out`: a gzipped JSON lines file whose
first line holds the federation's facts and each further line one window, with its jobs
file, mechanism, free nodes and batch. It takes `--share`, `--prefix`, `--window` and
`--platforms` as `run` does, and prints the windows in all, per mechanism, with one job
and with a full batch.

**`train`** reads one or more `--windows` files from one federation, trains RegretFormer
on them (step 6) and saves the network to `--out`. Its options are `--objective`
(`revenue` or `welfare`), `--steps` (2000), `--seed` (0), `--under-price` to count
misreports under the price in the regret, and `--device` (`cpu`, or a GPU such as
`cuda`). It needs torch but not dr_evt, and prints the windows read and the mean
objective, regret and multiplier over the last 100 steps.

## 7.8 Modelling choices

These choices are open to revision:

- **Revenue is what users pay**, the total of the charges; welfare subtracts the posted
  costs, so a job placed under the price lowers it.
- **Service counts routed jobs only.** Jobs set aside never wait in the queue, so they
  appear in `waiting.csv` and in no measure.
- **Each platform's utilization runs to its own makespan**, so platforms whose last jobs
  end at different times are measured over different spans.
