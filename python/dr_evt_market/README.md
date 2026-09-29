# Federation market

`dr_evt_market` runs a job trace through a federation of five real machine profiles.
Each platform exposes a configurable share of its nodes, each market window auctions
a prefix of the waiting queue, winners go to the platforms they won, and `dr_evt`
runs the resulting streams.

## Platforms

`Platform` owns one in-process simulation. Its `share` sets `exposed_nodes` to the
larger of one node and the rounded share of the real machine. Hardware tags determine
whether a job fits. Speed is relative to Quartz at 1.0 and selects the platform run
time and priced reservation for that job's hardware.

| Profile | Nodes | Price per node-hour | CPU speed | GPU speed | Hardware |
|---|---:|---:|---:|---:|---|
| Corona | 121 | 1.5 | 1.000 | 1.000 | CPU, GPU |
| Dane | 1544 | 0.18 | 0.861 | | CPU |
| Matrix | 30 | 1.6 | 2.574 | 3.695 | CPU, GPU |
| Tioga | 32 | 2.7 | 1.594 | 7.042 | CPU, GPU |
| Tuolumne | 1152 | 0.19 | 1.401 | 3.313 | CPU, GPU |

These five profiles form the default federation. Hardware is intentionally limited to
`cpu` and `gpu`: Dane is CPU-only, and every other profile accepts both.

## Jobs and bids

A `Job` records its identifier, submission time, node count, execution limit, bid,
hardware requirements, historical run time, requested limit, source, user, and
persona. Required jobs-file columns are `job_id`, `job_submit_time`, `num_nodes`,
`time_limit`, and `bid`. Optional columns are `bid:<platform>`, `requires`, `runtime`,
`requested`, `source`, `user`, and `persona`.

Unknown columns are ignored. A scalar `bid` is a maximum price per reference node-hour
on every platform. A `bid:<platform>` is a maximum price per node-hour on that
platform. If any mapped bid cell is filled, only those named platforms are bid on and
the scalar cell is ignored.

Prepared rows form one anonymous stream. `source` remains a label for analysis; it
does not select a platform or determine a job's hardware. The traces do not identify
GPU work, so each job requires `gpu` with probability 0.5 by default. The other jobs
have no hardware requirement and can use a CPU. `--gpu-fraction` changes that
probability, while `--requires` applies one requirement to the whole output file.

For platform `p`, let `speed(p)` select its GPU speed for a GPU job and its CPU speed
otherwise. The platform holds the nodes for `max(1, ceil(limit / speed(p)))` seconds.
Its cost and the two bid forms' values are:

```text
cost(p)         = posted_price(p) * nodes * limit / speed(p) / 3600
scalar_value(p) = bid              * nodes * limit            / 3600
mapped_value(p) = bid(p)           * nodes * limit / speed(p) / 3600
```

By default, trace preparation anchors every synthetic private price on the mean posted
price of the five-profile default federation. A seed, source, and pseudonymous user
select a persistent persona. The trace's job identifier, or its source and original
file order when it has no identifier, controls each job's random draws. Generated job
numbers therefore do not change those draws when the selected interval changes.
Per-platform bids also apply a persistent user preference for each machine and scale
the offered hourly price by that machine's speed for the job.

| Persona | Share | Price per node-hour |
|---|---:|---|
| sticker | 45% | Reference price |
| tier | 35% | Reference price, 2x when urgent, or 4x when urgent and heavy |
| value | 15% | Reference price times a lognormal multiple with median 3.0 and sigma 0.5 |
| whale | 5% | Ten times the reference price |

An urgent tier job occurs with probability 20 percent. A tier user is a heavy premium
user with probability 20 percent. By default, preparation uses historical run time as
the execution limit. A job is priced, released and valued on that run time, so users
pay for what they use; the original requested limit is carried in `requested` but
otherwise unused. With `--anchor home`, each source must name a profile and its posted
price becomes the reference. A per-platform home entry uses no preference factor,
while its speed still converts the persona price into that platform's hourly price.

## Mechanism and loop

`candidates` keeps platforms that match the hardware, have enough free nodes, were
bid on, and have a reported value that covers cost. `Vcg` maximizes total reported
value minus platform cost subject to one platform per job and node capacity. Its
Clarke pivot is the second price generalized to several platforms with capacity. A job
whose reported value exactly covers cost adds nothing to that total; it takes nodes
the winners leave free, in queue order, and pays the posted cost.

VCG lets users keep the savings between their values and the charges above posted
cost. Pay what you bid assigns offers greedily and gives that surplus to the center.
Both mechanisms report welfare and revenue through the same market outputs.

`FirstFit` is the no-market baseline over the same federation slices and market loop.
It ignores bids and visits batch jobs in arrival order, placing each on the available
platform with the lowest posted cost for that job, including speed, and charging that
cost. Comparisons therefore change the allocation rule without changing capacity or
the workload.

`RegretFormer` is a learned mechanism: the network of Ivanov et al. (NeurIPS 2022) over
a grid of jobs by platforms. For each job it gives a probability for every candidate
platform and for waiting, and a payment fraction. The market takes the most probable
cells first and places a job on the first of its platforms that still has the nodes.
Waiting only lowers a job's place in that order: a job stays in the queue only when
none of its platforms has room left, and is never turned away. A winner pays its cost
plus its payment fraction of the difference between its value and that cost. It needs
`torch`; `RegretFormer(path)` loads a saved network and `RegretFormer(seed=0)` builds
an untrained one.

`grid_regret(mechanism, jobs, platforms, free_nodes)` measures, for one window, how
much each job could gain by misreporting while the others report truthfully, on the
decisions the market applies. It moves one price at a time over a grid that includes
the posted prices, where a job's candidates change, then scales the whole report: the
item-wise grid of You et al. (2026). The result is a lower bound on the regret. VCG
reads zero; pay what you bid reads up to each winner's surplus over the posted price.
`refined_regret(regretformer, ...)` adds their guided gradient refinement for the
learned mechanism, and reports beside it the gradient-only estimate that RegretFormer's
own protocol gives.

`record_windows(jobs, platforms)` runs the market, under VCG unless another mechanism
is given, and returns every window's batch and free nodes. `train_regretformer(windows,
platforms)` trains the network on them with RegretFormer's recipe: it maximizes the
center's premiums (or welfare, with `objective="welfare"`) less a multiplier times the
regret, and raises the multiplier while the regret exceeds a budget that shrinks from 1
to 0.1 percent of the jobs' available surplus. Misreports come from the item-wise grid,
not gradient ascent. A penalty keeps the relaxed allocation within the free nodes and
stops it from holding back jobs that fit, since the market places those anyway. The
regret that training reports is measured on the relaxed network and can be far below
that of the mechanism the market applies: judge a trained network by `refined_regret`
on windows it was not trained on.

At each fixed window, the market advances every platform, admits arrivals, auctions
the first `prefix` queued jobs that have a candidate at the current free nodes, submits
winners, and advances again; jobs without a current candidate are skipped for the
window without losing their queue position. A mechanism must place every batch job
that still fits the nodes left over, so a job waits only when no platform has room for
it; anything else raises `MarketError`. The market's guarantee is that every accepted
winner starts in its market window. Per-job begin and end are recorded by the market
from that guarantee, not read from `dr_evt`, and streamed jobs run their speed-adjusted
limit.

## Outputs

`run` writes `routed.csv`, `waiting.csv`, `service.csv`, and `summary.json`. The routed
ledger holds the chosen platform, cost, value, premium, charge, and timing. A routed
job's community is its `source`, or the empty string when absent. `service.csv` has one
row per community plus `all`: count, mean wait (`begin_s - submit_s`), node-hour-weighted
mean wait with weight `num_nodes * (end_s - begin_s) / 3600`, and mean bounded slowdown
`max(1, (wait + run) / max(run, 10))`, where `run = end_s - begin_s`. The summary repeats
these measures under `service`, along with the resolved configuration, platform
statistics, welfare, revenue, counts, and the SHA-256 digest of `routed.csv`.

The market turns no job away. A job that no platform could run at its price, even with
every node free, waits outside the auction: no platform has its `hardware`, it is
`oversize` for every share, or it is `unaffordable` wherever it fits. At fixed prices
and shares it waits until the run ends, and `waiting.csv` lists it with that reason
and its submission time.

## Command line

Run a prepared jobs file:

```bash
python -m dr_evt_market run --jobs jobs.csv --out results --share 0.1 \
  --prefix 32 --window 60 \
  --platforms corona,dane,matrix,tioga,tuolumne
```

`--share` accepts one fraction for every platform or a comma-separated mapping such as
`corona=0.2,dane=0.5,matrix=0.5,tioga=0.5,tuolumne=0.05`; omitted names use `1.0`.

`--mechanism` is `vcg` (the default), `firstprice` for pay what you bid, `firstfit` for
the bid-blind no-market baseline, or `regretformer` with `--checkpoint PATH` for a saved
network. To train one, record the windows of a VCG run on the same jobs and federation
and train on them:

```bash
python -m dr_evt_market train --jobs jobs.csv --out network.pt --share 0.2 \
  --objective revenue --steps 2000
```

Prepare one merged interval from LC traces:

```bash
python -m dr_evt_market prepare \
  --trace corona=/path/to/corona.csv --trace tioga=/path/to/tioga.csv \
  --out jobs.csv --start 0 --hours 24 --seed 0 --anchor mean \
  --gpu-fraction 0.5 \
  --per-platform corona,dane,matrix,tioga,tuolumne --limit-from runtime
```

Use `--format simple` for the simple trace format. Under the default mean anchor, trace
sources are community labels and need not name profiles. Use `--anchor home` to anchor
each job on its source profile instead.

## Tests and notebook

From the repository root, run:

```bash
PYTHON_EXECUTABLE=python3 ./tests/run_market_tests.sh
```

The worked analysis notebook is `dr_evt_market/learn/market.ipynb`.
