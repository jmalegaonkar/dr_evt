# dr_evt_market: master overview

`dr_evt_market` replays real job traces through a federation of five LLNL machines.
dr_evt simulates each machine's exposed share, and a central market decides, every
window, where each waiting job runs and what it pays.

```text
LC traces --prepare--> jobs.csv --run--> every 60 s: admit arrivals
(clean, synthesize                       -> take up to 32 jobs that can be placed now
 bids and hardware)                      -> the mechanism picks platforms and charges
                                         -> the market checks the decisions
                                         -> winners start now on dr_evt
                                         -> outputs, with service by community
```

- **Jobs** are real submissions: arrival time, node count, run time and user. The
  traces carry no bids, so a seeded persona rule synthesizes each job's willingness to
  pay.
- **Platforms** are Corona, Dane, Matrix, Tioga and Tuolumne. Each has a posted price
  per node-hour, a speed relative to Quartz, and a `share` of its nodes exposed to the
  federation.
- **Mechanisms** decide who runs where and what they pay: VCG (the truthful auction),
  pay what you bid, `FirstFit` (the bid-blind baseline) and RegretFormer (a learned
  mechanism).
- **The market never turns a job away.** A job waits in the queue until a platform it
  can win has room for it, and every winner starts in the window it wins.

Each step has a detailed page: `01_jobs.md` to `04_offers.md` so far.

## 1. Jobs: sourcing and cleaning

Code: `jobs/traces.py` (`read_lc`, `read_simple`) and `jobs/prepare.py` (`prepare`,
`_drop_reason`). Details: `01_jobs.md`.

- **Reading.** A trace is either the pseudonymized LC format or dr_evt's simple
  format. Each row keeps its submit time, node count, limit, run time, user, whether
  it ran, and a stable identity:
  - the limit is the user's limit when it is numeric, else the system limit;
  - the run time is `t_inactive - t_run`;
  - the identity is the trace's job id, or the row's position in its file.
- **Interval.** All sources are merged and sorted by submit time, source and file
  order. One contiguous interval `[start, start + hours)` is kept; `start` defaults to
  the earliest submission, and is read in the trace's own clock.
- **Cleaning.** A row is dropped, and counted by reason, when it has no nodes, a limit
  under one second, never ran, or a run time under one second. Oversize jobs are kept.
- **Clock and ids.** Times are shifted so that the first kept job arrives at 0, and
  jobs are numbered `j000001`, `j000002`, and so on.
- **Hardware.** The traces do not say which jobs use GPUs, so each job requires `gpu`
  with probability 0.5: the first draw of a generator seeded by the job's identity.
  `--requires` sets one requirement for the whole file instead.
- **Limit.** A job's limit is its recorded run time, or its request when the trace has
  none, so it is priced and released on its actual use; the request is carried in
  `requested`. The recorded run time is taken as the job's work in reference seconds,
  that is, on a machine of speed 1.0.
- **Source.** The trace a job comes from is a community label for analysis. It does
  not choose a platform or the job's hardware.
- **Synthetic days.** With `--synthetic`, the interval becomes a pool instead of the day
  itself. For each source, a Poisson number of real groups (arrays, or single jobs),
  with the source's mean per day, is drawn at random, and each group arrives at its real
  time of day. Each seed gives a new day.

## 2. Bids: the synthetic willingness to pay

Code: `jobs/bids.py` (`terms`, `single_bid`, `multi_bid`) and the job loop of
`jobs/prepare.py`. Details: `02_bids.md`.

- **Units.** A bid is the most a user will pay per node-hour, in the unit of the
  posted prices. What a job pays is set by the mechanism, and the market checks that
  it lies between the job's cost and its value.
- **Persona.** Each user draws one persona, keyed by the seed, source and user, so a
  user keeps it across jobs. A row without a user is its own user.

  | persona | share of users | multiple |
  |---|---:|---|
  | sticker | 45% | 1 |
  | tier | 35% | 1; 2 if urgent (20% of jobs); 4 if also a heavy user (20%) |
  | value | 15% | lognormal per job, median 3, sigma 0.5 |
  | whale | 5% | 10 |

- **Single bid** (the default). One price per reference node-hour, valid on every
  machine: the mean posted price of a unit of work over the machines that can run the
  job, times the persona's multiple. A sticker bids 0.832 for a CPU job and 0.5934 for
  a GPU job.
- **Multi bid** (`--bids multi`). A price on each machine that has the job's hardware:
  its posted price times a multiple times the user's preference for that machine,
  `exp(N(0, 0.3))`. Each machine takes the user's own persona half the time and draws
  its own otherwise, so one user can be a whale on one machine, near the price on
  another and under it on a third, or alike on all of them.
- **Draws.** The job's own draws (hardware, urgency, value multiple) come from its
  row's generator in fixed places, so a job keeps its bid whatever interval is
  selected, and the persona rule never moves its hardware. A machine's draws are keyed
  by user and machine. `--platforms` chooses the machines, all five by default.

## 3. Platforms

Code: `platform/profiles.py` and `platform/base.py` (`Platform`).

The last two columns are the posted price divided by the speed: the cost of one
reference node-hour of work on that machine.

| machine | nodes | price per node-hour | CPU speed | GPU speed | cost per CPU work | cost per GPU work |
|---|---:|---:|---:|---:|---:|---:|
| Corona | 121 | 1.50 | 1.000 | 1.000 | 1.500 | 1.500 |
| Dane | 1544 | 0.18 | 0.861 | | 0.209 | |
| Matrix | 30 | 1.60 | 2.574 | 3.695 | 0.622 | 0.433 |
| Tioga | 32 | 2.70 | 1.594 | 7.042 | 1.694 | 0.383 |
| Tuolumne | 1152 | 0.19 | 1.401 | 3.313 | 0.136 | 0.057 |

- **Share.** A platform exposes `round(nodes x share)` nodes to the federation, possibly
  none, and the share can differ per machine. It owns one dr_evt simulation of that
  slice, in LIMIT mode with EASY backfilling and FCFS order. Details: `03_platforms.md`.
- **Speed.** On platform `p` a job holds its nodes for `ceil(limit / speed)` seconds
  and costs `posted price x nodes x limit / speed / 3600`: the posted price times the
  node-hours it uses there. Dane has no GPUs.

## 4. Value, cost, candidates and offers

Code: `mechanism/base.py` (`job_value`, `candidates`, `offers`). Details:
`04_offers.md`.

- **Value.** A single bid is worth `bid x nodes x limit / 3600` on every machine: the
  job gets the same work done anywhere. A multi bid is worth
  `bid(p) x nodes x limit / speed(p) / 3600` on machine `p`.
- **Candidate.** A platform is a candidate for a job when the job can be placed there
  now: its hardware fits and its free nodes cover the job. Bids play no part.
- **Offer.** A candidate is an offer when the job bid on it and value covers cost:
  `bid >= posted price(p) / speed(p)` for a single bid, `bid(p) >= posted price(p)` for
  a multi bid. A job can win only among its offers. A single-bid sticker covers the
  machines whose work costs less than the average, and not Corona or Tioga's CPUs.

## 5. The market loop

Code: `market.py` (`run`, `_check_decisions`, `_blocked_by`).

1. **Intake.** A job that could never run on this federation at its price waits outside
   the auction, with its reason: no platform has its `hardware`, it is `oversize` for
   every share, or it is `unaffordable` wherever it fits. It ends the run in
   `waiting.csv`.
2. **Each window,** at `t = 0, 60, 120, ...`: advance every platform to `t`, admit the
   jobs that have arrived, read the free nodes, and form the batch: the first 32 queued
   jobs that can be placed now, whatever they bid. The others keep their place in the
   queue. A batch job without an offer takes part and loses.
3. **Decision.** The mechanism decides, and the market checks that every job is decided
   once, belongs to the batch and gets a platform it has an offer on; that its charge
   lies between its cost and the mechanism's cap; that no platform is over capacity;
   and that no batch job is left waiting while it has an offer on the nodes left over.
4. **Placement.** Winners are submitted to dr_evt with their speed-adjusted run time.
   After another advance, no winner may still wait in dr_evt's queue (jobs that are
   running are fine): every winner starts at the window time. The market records
   `begin = t` and `end = t + run time`, and the other jobs stay queued for the next
   window.

## 6. Mechanisms

- **VCG** (`mechanism/vcg.py`). A MILP maximizes the total of value minus cost, with at
  most one platform per job and node capacity per platform. Each winner pays its cost
  plus its Clarke pivot, the welfare its presence takes from the others. A job the solve
  leaves out that still fits takes the leftover nodes at cost.
- **Pay what you bid** (`mechanism/firstprice.py`). Offers are taken greedily by net
  value, and each winner pays its full value.
- **FirstFit** (`mechanism/firstfit.py`). Bids are ignored: in arrival order, each job
  goes to the platform with room and the lowest speed-adjusted posted cost, and pays
  that cost.
- **RegretFormer** (`mechanism/regretformer.py`, `mechanism/learned.py`). A network
  gives each job a probability for every platform it has an offer on and for waiting,
  and a payment fraction. The market places jobs greedily by probability and charges
  `cost + fraction x (value - cost)`. Training is in `mechanism/training.py`, and regret
  estimation by item-wise grid and guided refinement in `mechanism/regret.py`.

## 7. Outputs and commands

Code: `market.py` (`write_outputs`) and `cli.py`.

- **`routed.csv`**: each routed job's platform, window, cost, value, premium, charge,
  and submit, begin and end times.
- **`waiting.csv`**: the jobs that never ran, with the reason.
- **`service.csv`**: for each community (the job's source) and for `all`: the count,
  the mean wait, the mean wait weighted by node-hours, and the mean bounded slowdown
  `max(1, (wait + run) / max(run, 10))`.
- **`summary.json`**: the configuration, each platform's completed jobs, utilization
  and makespan from dr_evt, welfare (the total of value minus cost), revenue (the total
  of charges), the counts, and the SHA-256 digest of `routed.csv`.

The commands are `python -m dr_evt_market prepare` (traces to a jobs file), `run` (a
jobs file through one mechanism) and `train` (RegretFormer on the windows of a VCG run).
