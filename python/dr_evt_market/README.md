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
Clarke pivot is the second price generalized to several platforms with capacity.

VCG lets users keep the savings between their values and the charges above posted
cost. Pay what you bid assigns offers greedily and gives that surplus to the center.
Both mechanisms report welfare and revenue through the same market outputs.

At each fixed window, the market advances every platform, admits arrivals, auctions
the first `prefix` queued jobs, submits winners, and advances again. Its guarantee is
that every accepted winner starts in its market window. Per-job begin and end are
recorded by the market from that guarantee, not read from `dr_evt`, and streamed jobs
run their limit.

## Outputs

`run` writes `routed.csv`, `rejected.csv`, and `summary.json`. The routed ledger holds
the chosen platform, cost, value, premium, charge, and timing. The summary contains
the resolved configuration, platform statistics, welfare, revenue, counts, and the
SHA-256 digest of `routed.csv`.

## Command line

Run a prepared jobs file:

```bash
python -m dr_evt_market run --jobs jobs.csv --out results --share 0.1 \
  --prefix 32 --window 60 --platforms corona,lassen,tioga,tuolumne
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
