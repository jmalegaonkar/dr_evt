# Federation market

`dr_evt_market` runs a job trace through a federation of four real machine profiles.
Each platform exposes a configurable share of its nodes, each market window auctions
a prefix of the waiting queue, winners go to the platforms they won, and `dr_evt`
runs the resulting streams.

## Platforms

`Platform` owns one in-process simulation. Its `share` sets `exposed_nodes` to the
larger of one node and the rounded share of the real machine. Hardware tags determine
whether a job fits, and the posted price determines its cost per node-hour.

| Profile | Nodes | Price per node-hour | Hardware |
|---|---:|---:|---|
| Corona | 121 | 2.0 | CPU, GPU, AMD |
| Lassen | 795 | 3.0 | CPU, GPU, NVIDIA |
| Tioga | 32 | 6.0 | CPU, GPU, AMD |
| Tuolumne | 1152 | 8.0 | CPU, GPU, AMD |

These four profiles form the default federation. Dane is also available as a 1544
node CPU profile at 1.0 per node-hour.

## Jobs and bids

A `Job` records its identifier, submission time, node count, execution limit, bid,
hardware requirements, historical run time, requested limit, source, user, and
persona. Required jobs-file columns are `job_id`, `job_submit_time`, `num_nodes`,
`time_limit`, and `bid`. Optional columns are `bid:<platform>`, `requires`, `runtime`,
`requested`, `source`, `user`, and `persona`.

Unknown columns are ignored. A scalar `bid` is a maximum price per node-hour on every
platform. If any `bid:<platform>` cell is filled, only those named platforms are bid
on and the scalar cell is ignored.

For platform `p`, the public cost and reported value are:

```text
cost(p)  = posted_price(p) * nodes * limit / 3600
value(p) = bid_price(p)    * nodes * limit / 3600
```

Trace preparation anchors each synthetic private price on the posted price of the
trace's home machine. A seed, source, and pseudonymous user select a persistent
persona; per-platform bids also apply a persistent user preference for each machine.

| Persona | Share | Price per node-hour |
|---|---:|---|
| sticker | 45% | Home machine's posted price |
| tier | 35% | Home price, 2x when urgent, or 4x when urgent and heavy |
| value | 15% | Home price times a lognormal multiple with median 3.0 and sigma 0.5 |
| whale | 5% | Ten times the home price |

An urgent tier job occurs with probability 20 percent. A tier user is a heavy premium
user with probability 20 percent. By default, preparation uses historical run time as
the execution limit and carries the original requested limit in `requested`.

## Mechanism and loop

`candidates` keeps platforms that match the hardware, have enough free nodes, were
bid on, and have a posted price no greater than the bid. `Vcg` maximizes total reported
value minus platform cost subject to one platform per job and node capacity. Its
Clarke pivot is the second price generalized to several platforms with capacity. A job
that bids exactly the posted price adds nothing to that total; it takes nodes the
winners leave free, in queue order, and pays the posted price.

VCG lets users keep the savings between their values and the charges above posted
cost. Pay what you bid assigns offers greedily and gives that surplus to the center.
Both mechanisms report welfare and revenue through the same market outputs.

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
the first `prefix` queued jobs, submits winners, and advances again. A mechanism must
place every batch job that still fits the nodes left over, so a job waits only when no
platform has room for it; anything else raises `MarketError`. The market's guarantee is
that every accepted winner starts in its market window. Per-job begin and end are
recorded by the market from that guarantee, not read from `dr_evt`, and streamed jobs
run their limit.

## Outputs

`run` writes `routed.csv`, `waiting.csv`, and `summary.json`. The routed ledger holds
the chosen platform, cost, value, premium, charge, and timing. The summary contains
the resolved configuration, platform statistics, welfare, revenue, counts, and the
SHA-256 digest of `routed.csv`.

The market turns no job away. A job that no platform could run at its price, even with
every node free, waits outside the auction: no platform has its `hardware`, it is
`oversize` for every share, or it is `unaffordable` wherever it fits. At fixed prices
and shares it waits until the run ends, and `waiting.csv` lists it with that reason
and its submission time.

## Command line

Run a prepared jobs file:

```bash
python -m dr_evt_market run --jobs jobs.csv --out results --share 0.1 \
  --prefix 32 --window 60 --platforms corona,lassen,tioga,tuolumne
```

`--mechanism` is `vcg` (the default), `firstprice` for pay what you bid, or
`regretformer` with `--checkpoint PATH` for a saved network. To train one, record the
windows of a VCG run on the same jobs and federation and train on them:

```bash
python -m dr_evt_market train --jobs jobs.csv --out network.pt --share 0.2 \
  --objective revenue --steps 2000
```

Prepare one merged interval from LC traces:

```bash
python -m dr_evt_market prepare \
  --trace corona=/path/to/corona.csv --trace tioga=/path/to/tioga.csv \
  --out jobs.csv --start 0 --hours 24 --seed 0 --requires gpu \
  --per-platform corona,lassen,tioga,tuolumne --limit-from runtime
```

Use `--format simple` for the simple trace format. Each trace source must name a known
profile so its posted price can anchor the generated bids.

## Tests and notebook

From the repository root, run:

```bash
PYTHON_EXECUTABLE=python3 ./tests/run_market_tests.sh
```

The worked analysis notebook is `dr_evt_market/learn/market.ipynb`.
