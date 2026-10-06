# dr_evt_market

A market that routes HPC jobs across a federation of LLNL machines, built on dr_evt.
Each machine shows the federation a share of its nodes, simulated by its own dr_evt
instance. Every 60 seconds the market auctions the jobs waiting in its queue, starts the
winners on the machines they won, and records what each paid and how long it waited.
The jobs come from real LC traces; the traces carry no bids, so a persona rule draws
them.

## Running it

The market needs dr_evt's Python bindings, built into `install/lib/python`. Preparing
traces needs NumPy, VCG needs SciPy, and RegretFormer needs torch; the package itself
imports without any of them. From the repository root:

```bash
export PYTHONPATH=install/lib/python:python
python -m dr_evt_market prepare --trace corona=corona.csv --trace tioga=tioga.csv \
  --out jobs.csv --hours 24
python -m dr_evt_market run --jobs jobs.csv --out results --share 0.1
```

`prepare` turns traces into a jobs file with a bid on each platform; with `--synthetic`
it draws a new day from the interval instead. `run` routes the jobs through one
mechanism, `vcg`, `firstprice`, `firstfit` or `regretformer`, and writes `routed.csv`,
`waiting.csv`, `service.csv` and `summary.json`.

To train RegretFormer, harvest windows from market runs, then train on them wherever
torch runs; training needs neither dr_evt nor the traces:

```bash
python -m dr_evt_market harvest --jobs jobs.csv --out windows.jsonl.gz --share 0.1 \
  --mechanism vcg --mechanism firstprice --mechanism firstfit
python -m dr_evt_market train --windows windows.jsonl.gz --out network.pt --device cuda
python -m dr_evt_market run --jobs jobs.csv --out results --share 0.1 \
  --mechanism regretformer --checkpoint network.pt
```

`./tests/run_market_tests.sh` runs the suite from the repository root; set
`PYTHON_EXECUTABLE` to choose the interpreter.

## What is where

| path | what it holds |
|---|---|
| `jobs/` | the job record and jobs file, the trace readers, the persona bids, preparation and synthetic days |
| `platforms.py` | a platform over one dr_evt simulation, the five machines, and `federation` |
| `mechanism/` | the mechanism contract, VCG, pay what you bid, FirstFit, and RegretFormer with its training and regret estimates |
| `market.py` | the window loop, its checks, and the output files |
| `harvest.py` | windows recorded from market runs, and the files that carry them to training |
| `cli.py` | the four commands |
| `tests/` | the suite, with a small trace and jobs file in `tests/data/` |
| `learn/` | the guides and the fixture notebook |

## Reading on

`learn/master_overview.md` walks through the market end to end, and one page per step
goes into the detail, from `learn/01_jobs.md` to `learn/07_outputs.md`.
`learn/market.ipynb` runs the fixture from bids to outputs.
