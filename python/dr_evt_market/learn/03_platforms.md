# Step 3: platforms

A platform is one real machine: its size, its posted price, its hardware and its speed,
and a dr_evt simulation of the share of it that the federation sees. This page covers
the five machines, the share, what a job costs and how long it holds nodes on each, and
how a platform drives its simulation. The overview of all steps is `master_overview.md`.

Code: `platform/base.py` (`Platform`) and `platform/profiles.py` (the five machines and
`federation`).

## 3.1 The five machines

Each machine is a small class with five facts: its name, nodes, posted price per
node-hour, hardware and speed per kind of hardware. The base class does the rest.

| machine | nodes | price | hardware | CPU speed | GPU speed | cost of a unit of work, CPU | GPU |
|---|---:|---:|---|---:|---:|---:|---:|
| Corona | 121 | 1.50 | cpu, gpu | 1.000 | 1.000 | 1.500 | 1.500 |
| Dane | 1,544 | 0.18 | cpu | 0.861 | | 0.209 | |
| Matrix | 30 | 1.60 | cpu, gpu | 2.574 | 3.695 | 0.622 | 0.433 |
| Tioga | 32 | 2.70 | cpu, gpu | 1.594 | 7.042 | 1.694 | 0.383 |
| Tuolumne | 1,152 | 0.19 | cpu, gpu | 1.401 | 3.313 | 0.136 | 0.057 |

- **A unit of work** is a node-hour at speed 1.0, the speed of Quartz, which is not in
  the federation. The cost of a unit of work is the posted price divided by the speed.
- **Speeds** come from a matrix of run times relative to Quartz: single-rank runs of
  eight proxy applications. A machine's speed is one over its median relative run
  time. Corona has no runs in the matrix, so its speed is taken as 1.0.
- **Node counts** for Corona and Tuolumne are LLNL's published numbers. Tioga keeps 32
  nodes until its size is confirmed; its page lists 24.

## 3.2 The share

A platform exposes `round(nodes x share)` nodes to the federation. The share is one
number for every machine, or one per machine (`--share tuolumne=0.1,dane=0.1`, where
omitted machines take 1.0). A share of zero exposes no nodes, and the market routes
around that machine.

| share | Corona | Dane | Matrix | Tioga | Tuolumne |
|---|---:|---:|---:|---:|---:|
| 0.5 | 60 | 772 | 15 | 16 | 576 |
| 0.2 | 24 | 309 | 6 | 6 | 230 |

Python's `round` sends halves to the even number, so Corona at 0.5 exposes 60 nodes, not
61.

The market is the only submitter. A slice serves every trace's jobs, and the rest of the
machine is outside the model; squeezing the traced load into slices is what creates
contention.

## 3.3 A job on a platform

- **Where it fits.** The machine must have the job's hardware (a GPU job cannot use
  Dane) and expose at least the job's nodes.
- **How long it holds nodes.** `max(1, ceil(limit / speed))` seconds: the job's limit is
  work at speed 1.0, and the machine's speed for the job's hardware shortens it.
- **What it costs.** `price x nodes x limit / speed / 3600`: the posted price times the
  node-hours the job uses there, without the rounding up to a whole second.

A 4-node job with one hour of work, held seconds and cost:

| | Corona | Dane | Matrix | Tioga | Tuolumne |
|---|---|---|---|---|---|
| GPU job | 3,600 s, 6.00 | no GPUs | 975 s, 1.73 | 512 s, 1.53 | 1,087 s, 0.23 |
| CPU job | 3,600 s, 6.00 | 4,182 s, 0.84 | 1,399 s, 2.49 | 2,259 s, 6.78 | 2,570 s, 0.54 |

Tuolumne is the cheapest place for both kinds of job: a 26th of Corona's cost for GPU
work, and below the CPU-only Dane for CPU work.

## 3.4 How a platform drives dr_evt

- **Setup.** Each platform owns one dr_evt simulation, started from an input file with a
  header and no jobs, with the slice as its total nodes, LIMIT run times, EASY backfill
  and FCFS order.
- **Submitting.** Winners go in with `append_jobs`, one `JobAppendRequest(window time,
  nodes, "1", hold time)` each. In LIMIT mode each job ends exactly when its hold time
  runs out.
- **Advancing.** `advance_to(t)` processes everything up to `t`: a job that ends at `t`
  frees its nodes at `t`. `get_available_nodes` gives the free nodes.
- **The guarantee.** After each window the market advances every platform again and
  asks `get_active_job_count`, the number of jobs waiting in dr_evt's queue. Running
  jobs do not count, and they may run for many windows. The count must be zero: every
  winner started in its window. Anything else raises `MarketError`.

Because the market submits only jobs that fit the free nodes, dr_evt never holds a job
in its queue, and its EASY and FCFS settings never act. dr_evt keeps each platform's
clock, capacity and utilization, and the market records each job's begin (its window)
and end (begin plus hold time). This is deliberate: the auction sells nodes that are
free now, and dr_evt's Python bindings report counts, not when a queued job starts.

`statistics` returns what dr_evt measures that the market does not: each platform's
completed jobs, utilization and makespan. Its queue statistics would be zero by
construction.

## 3.5 What follows

- **Where the work goes depends on the bid form.** A single bid is worth the same on
  every machine, so a mechanism that maximizes total surplus places a job where it costs
  least: Tuolumne, the cheapest for both CPU and GPU work and the largest slice, takes
  the work first, and the other machines take the overflow. A multi bid's surplus on a
  machine grows with that machine's price, so a job keen on a dear machine can prefer it
  (step 2).
- **Contention is a matter of shares.** Between communities it appears only when the
  slices are small; a small share for Tuolumne alone is the most direct lever.

## 3.6 Modelling choices

These choices are open to revision:

- **Speed from proxy applications.** A machine has one speed per kind of hardware, the
  median over single-rank proxy runs; Corona's is assumed.
- **CPU work on GPU machines.** A job without a GPU requirement may run on any
  machine's CPUs, including those of the GPU machines.
- **Fixed posted prices.** Each machine has one price per node-hour, whatever its load.
- **One submitter.** The market is the only source of jobs for a slice; the machine's
  own local work is not modeled.
