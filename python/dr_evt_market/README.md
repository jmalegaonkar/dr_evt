# Federation market

`dr_evt_market` runs a job trace through a federation of five real machine profiles.
Each platform exposes a configurable share of its nodes, each market window auctions
a prefix of the waiting queue, winners go to the platforms they won, and `dr_evt`
runs the resulting streams.

## Platforms

`Platform` owns one in-process simulation. Its `share` sets `exposed_nodes` to the
rounded share of the real machine; a share of zero exposes no nodes, and the market
routes around that platform. Hardware tags determine whether a job fits. Speed is
relative to Quartz at 1.0 and selects the platform run time and priced reservation for
that job's hardware. The market submits only jobs that start at once, so dr_evt never
holds a job in its queue: it keeps each platform's clock, capacity and utilization.

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

Unknown columns are ignored. A single bid, the `bid` column, is a maximum price per
reference node-hour on every platform. A multi bid gives a maximum price per node-hour
on each platform in `bid:<platform>` columns. If any `bid:<platform>` cell is filled,
only those platforms are bid on and the `bid` cell is ignored. A row with no bid in any
cell bids on no platform.

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
single_value(p) = bid              * nodes * limit            / 3600
multi_value(p)  = bid(p)           * nodes * limit / speed(p) / 3600
```

Trace preparation writes multi bids, a bid on each platform, or single bids with
`--bids single`. A seed, source, and pseudonymous user select a persistent persona; a
row without a user is its own user. Each persona bids a multiple of a price:

| Persona | Share of users | Multiple |
|---|---:|---|
| sticker | 45% | 1 |
| tier | 35% | 1, 2 when urgent, or 4 when urgent and heavy |
| value | 15% | A lognormal multiple per job with median 3.0 and sigma 0.5 |
| whale | 5% | 10 |

An urgent tier job occurs with probability 20 percent. A tier user is a heavy premium
user with probability 20 percent. A single bid is the price level of a unit of work, the
mean over the platforms that have the job's hardware of the posted price divided by the
speed, times the user's multiple. A multi bid gives each such platform the posted price
times a multiple times the user's persistent preference for that platform,
`exp(N(0, 0.3))`. Each platform takes the user's own persona half the time and draws its
own otherwise, so a user can be a whale on one platform and bid under the price on
another:

```text
single: bid    = mean(posted_price(p) / speed(p)) * multiple
multi:  bid(p) = posted_price(p) * multiple(persona on p) * preference(user, p)
```

The trace's job identifier, or its source and original file order when it has no
identifier, keys the job's own draws, which keep fixed places: the hardware, the tier
urgency, then the value multiple. Generated job numbers therefore do not change those
draws when the selected interval changes, and neither the persona rule nor `--requires`
moves them. A platform's draws are keyed by user and platform, so they do not depend on
the other platforms listed. A single bid's surplus, value minus cost, is largest where
the job costs least; a multi bid's, `(multiple * preference - 1) * cost(p)`, grows with
the platform's cost. The preparation summary counts the kept jobs of each persona.

Preparation uses the historical run time as the execution limit, or the requested limit
when a trace has no run time. A job is priced, released and valued on that run time, so
users pay for what they use; the requested limit is carried in `requested` but otherwise
unused.

## Mechanism and loop

A platform is a candidate for a job when the job can be placed there now: it has the
job's hardware and enough free nodes. `candidates` lists them whatever the job bids.
`offers` keeps the candidates the job bid on, at any price: a bid under the posted
price can win too, and then pays in full. A bid of zero or less is no bid. `Vcg`
maximizes total reported value minus platform cost over the offers at or above the
price, subject to one platform per job and node capacity. Its Clarke pivot is the second
price generalized to several platforms with capacity. A job left out of that total,
because it bids the price or under it, takes nodes the winners leave free, in queue
order, and pays the posted cost or its whole bid, whichever is less.

VCG lets users keep the savings between their values and the charges above posted
cost. Pay what you bid takes offers greedily by net value, ties going to the earlier job
and then the first platform by name, and gives that surplus to the center. Both
mechanisms report welfare and revenue through the same market outputs. Since a bid under
the price can win, neither is truthful: a job alone on a platform pays less by bidding
under the price.

`FirstFit` is the no-market baseline over the same federation slices and market loop. It
ignores bids and visits batch jobs in arrival order, placing each on the available
platform with the lowest posted cost for that job, including speed, and charging that
cost. Comparisons therefore change the allocation rule without changing capacity or the
workload. Since it reads no bid, it charges the cost even where a job bid less, or did
not bid at all, so a job can pay more than its value there: compare `FirstFit` on
service, not on welfare.

`RegretFormer` is a learned mechanism: the network of Ivanov et al. (NeurIPS 2022) over
a grid of jobs by platforms. For each job it gives a probability for every platform it
has an offer on and for waiting, and a payment fraction. The market takes the most
probable cells first and places a job on the first of its platforms that still has the
nodes. Waiting only lowers a job's place in that order: a job stays in the queue only
when none of its platforms has room left, and is never turned away. A winner pays its
cost plus its payment fraction of the difference between its value and that cost, or
its whole bid when it bid under the price. It
needs `torch`; `RegretFormer(path)` loads a saved network and `RegretFormer(seed=0)`
builds an untrained one.

`grid_regret(mechanism, jobs, platforms, free_nodes)` measures, for one window, how
much each job could gain by misreporting while the others report truthfully, on the
decisions the market applies. It moves one price at a time over a grid that includes
the posted prices, where the charge rule changes, then scales the whole report: the
item-wise grid of You et al. (2026). The result is a lower bound on the regret. Neither
auction reads zero: alone on a platform, a job keeps nearly all of its cost under VCG,
and of its value under pay what you bid, by bidding the grid's least positive price.
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

At each fixed window, the market advances every platform, admits arrivals, auctions the
first `prefix` queued jobs that can be placed now, submits winners, and advances again;
a job that cannot be placed anywhere now is skipped for the window without losing its
queue position. Who takes part therefore depends only on public facts, never on bids. A
mechanism must place every batch job that has an offer on the nodes left over, so a job
waits only when no platform it bid on has room; anything else raises `MarketError`. The
market's guarantee is that every accepted winner starts in its market window. Per-job
begin and end are recorded by the market from that guarantee, not read from `dr_evt`,
and streamed jobs run their speed-adjusted limit.

## Outputs

`run` writes `routed.csv`, `waiting.csv`, `service.csv`, and `summary.json`. The routed
ledger holds the chosen platform, cost, value, premium (the charge minus the cost,
negative for a job that bid under the price), charge, and timing. A routed job's
community is its `source`, or the empty string when absent. `service.csv` has one row
per community plus `all`: count, mean wait (`begin_s - submit_s`), node-hour-weighted
mean wait with weight `num_nodes * (end_s - begin_s) / 3600`, and mean bounded slowdown
`max(1, (wait + run) / max(run, 10))`, where `run = end_s - begin_s`. The summary
repeats these measures under `service`, along with the resolved configuration, each
platform's completed jobs, utilization and makespan from dr_evt, welfare, revenue,
counts, and the SHA-256 digest of `routed.csv`.

The market turns no job away. A job that no platform could run, even with every node
free, waits outside the auction: no platform has its `hardware`, it is `oversize` for
every share, or it bid on no platform that fits it (`no_bid`). At fixed shares it waits
until the run ends, and `waiting.csv` lists it with that reason and its submission
time.

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
  --out jobs.csv --hours 24 --seed 0 --gpu-fraction 0.5
```

The interval starts at the earliest submission unless `--start` gives a time in the
traces' own clock, which is epoch seconds for LC traces. Bids are multi bids, one on
each of the five default platforms that has the job's hardware; `--bids single` writes
one bid for all of them instead, and `--platforms` names other platforms. Use
`--format simple` for the simple trace format.
Trace sources are community labels and need not name profiles.

To draw a synthetic day from an interval instead of replaying it:

```bash
python -m dr_evt_market prepare \
  --trace corona=/path/to/corona.csv --trace tioga=/path/to/tioga.csv \
  --out day.csv --start START --hours 72 --synthetic --seed 3
```

Each trace's jobs are split into groups one user submitted less than a minute apart. For
each trace, the day takes a Poisson number of its groups, with the trace's mean per day
in the interval, drawn at random and at most once; each group arrives at its real time
of day, counted from `--start`. The seed fixes the day.

## Tests and notebook

From the repository root, run:

```bash
PYTHON_EXECUTABLE=python3 ./tests/run_market_tests.sh
```

The worked analysis notebook is `dr_evt_market/learn/market.ipynb`.
