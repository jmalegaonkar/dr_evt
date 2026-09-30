# Step 1: jobs, from a trace to `jobs.csv`

Step 1 turns raw traces into the job stream that the market runs. It reads each trace,
merges the sources on one clock, keeps one interval, drops the rows that cannot be
simulated, and writes one row per job. The bids that the same pass attaches are step 2.
The overview of all steps is `master_overview.md`.

Code: `jobs/traces.py` (`read_lc`, `read_simple`), `jobs/prepare.py` (`prepare`,
`_drop_reason`) and `jobs/job.py` (`Job`, `read_jobs`, `write_jobs`). The command is
`python -m dr_evt_market prepare`.

## 1.1 What goes in

Two formats:

- **LC**, the pseudonymized LC traces. The reader uses `user.name`, `job.id`,
  `t_submit`, `t_run`, `t_inactive`, `job.node.count`, `time_limit` and
  `user_time_limit`. It ignores the other columns, such as `event.outcome`, `job.queue`
  and `job.name`.
- **simple**, dr_evt's own format: `job_submit_time`, `num_nodes` and `time_limit`, and
  optionally `actual_run_time`, `user` and `job_id`. Every row counts as having run.

Each trace is named on the command line (`--trace corona=PATH`). The name becomes the
jobs' `source`: a community label for analysis. It does not choose a platform or the
job's hardware.

## 1.2 How each field is parsed

| field | LC rule | edge cases |
|---|---|---|
| submit | `floor(t_submit)` | floored before the interval check |
| nodes | `int(float(job.node.count))` | blank or `-` means no nodes |
| ran | `t_run` is not `-` | simple rows always ran |
| run time | `floor(t_inactive) - floor(t_run)` | each end is floored: 1125.2 to 1175.8 gives 50 |
| request | `user_time_limit` if numeric, else `time_limit`, floored | a blank or `-` user limit falls back |
| identity | `job:<job.id>`, or `row:<position in the file>` | keys the job's draws |
| user | `user.name` | blank or `-` means no user |

A simple trace reads `job_submit_time`, `num_nodes`, `time_limit`, `job_id` and `user`
the same way. Its run time is `actual_run_time`, floored, and a blank cell means the row
has no run time.

## 1.3 Merge, interval and clock

- **Merge.** All sources go into one list, sorted by submit time, then source name,
  then file position, so ties are deterministic.
- **Interval.** One interval `[start, start + hours x 3600)` is kept, its end excluded.
  `start` defaults to the earliest submission of any row, including rows dropped later.
  It is read in the traces' own clock, which is epoch seconds for LC traces. Without
  `hours`, the interval runs to the end of the traces.
- **Clock.** After cleaning, times are shifted so that the first kept job arrives at 0.

## 1.4 Cleaning

A row outside the interval is counted as `outside_interval` and not examined further.
Inside it, the first matching reason drops the row:

1. `no_nodes`: no node count, or fewer than one node;
2. `no_limit`: a request under 1 s;
3. `no_start`: the job never ran;
4. `bad_runtime`: a run time under 1 s, including an end before the start.

The summary that `prepare` returns, and the command prints, counts each reason, the
jobs kept from each source, and the kept jobs of each persona (step 2).

Not cleaned:

- oversize jobs. They are kept, and the market lists a job that no platform's share can
  hold as waiting, with reason `oversize` (step 5);
- jobs that failed, timed out or were cancelled after starting. They ran, so they are
  kept with the time they used;
- jobs still running when the interval ends. They keep their full run time;
- repeated submissions. There is no de-duplication.

## 1.5 Building each job

- **Id.** `j000001`, `j000002`, and so on, in merged order.
- **Submit.** `submit_s` is the shifted submit time.
- **Limit.** `limit_s` is the recorded run time, or the request when the trace has no
  run time. The market holds, prices and values a job on its limit (steps 3 and 4), so
  a job pays for the time it used. `runtime_s` and `requested_s` carry both originals.
- **Hardware.** A job requires `gpu` with probability `gpu_fraction` (`--gpu-fraction`,
  0.5 by default). Otherwise it has no requirement and runs on CPUs. `--requires` sets
  one requirement for every job instead.
- **Labels.** `source` is the trace's name, `user` is the trace's user (empty when it
  has none), and `persona` comes from the bid rule (step 2).

Every random draw comes from a generator seeded by a SHA-256 hash of the seed, the
source and a key; step 2 (`02_bids.md`) lists them all. The hardware is the first draw
of the row's generator, keyed by the row's identity, and it is made even when
`--requires` overrides it, so no other draw moves. No key contains the job number, so a
row keeps its draws whatever interval is selected. The source is part of every key, so
the same user name in two traces is two users.

`write_jobs` writes the jobs file with fixed columns, plus one `bid:<platform>` column
per platform when bids are per platform. `read_jobs` reads it back in row order and
rejects a repeated `job_id`.

## 1.6 The fixture trace, row by row

`tests/data/trace.csv` holds twelve made-up LC rows. Prepared as source `tioga` with
`start=1000`, `hours=0.05` and seed 0, the interval is `[1000, 1180)`. The table is in
merged order, and a row's file position counts data rows from 0.

| file position | name | what matters | submit / request / run | outcome |
|---:|---|---|---|---|
| 1 | before | submitted at 990.7 | 990 / 45 / 30 | `outside_interval` |
| 3 | bravo | user limit blank, system limit 120.9 | 1005 / 120 / 40 | **j000001** at 0 s, GPU |
| 5 | missing-nodes | nodes `-` | 1010 / 30 / 20 | `no_nodes` |
| 6 | never-one | `t_run` is `-` | 1020 / 300 / none | `no_start` |
| 4 | never-two | `t_run` is `-` | 1030 / 300 / none | `no_start` |
| 0 | alpha | user limit 60.9 | 1040 / 60 / 40 | **j000002** at 35 s, GPU |
| 8 | no-limit | user limit 0 | 1050 / 0 / 20 | `no_limit` |
| 9 | bad-runtime | ended before it started | 1060 / 30 / -1 | `bad_runtime` |
| 7 | foxtrot | | 1090 / 80 / 70 | **j000003** at 85 s, GPU |
| 2 | echo | ran from 1125.2 to 1175.8 | 1120 / 90 / 50 | **j000004** at 115 s, CPU |
| 10 | golf | submitted at 1179.9 | 1179 / 20 / 15 | **j000005** at 174 s, GPU |
| 11 | boundary | submitted at 1180.1 | 1180 / 50 / 20 | `outside_interval` |

The summary reads 12 rows and keeps 5, all from `tioga`: `no_nodes` 1, `no_limit` 1,
`no_start` 2, `bad_runtime` 1, `outside_interval` 2. Golf is inside because its submit
time is floored before the check, and boundary is outside because the end is excluded.
The kept jobs are 1 sticker, 3 tier and 1 value job.

From the repository's `python/` directory,

```bash
python -m dr_evt_market prepare --trace tioga=dr_evt_market/tests/data/trace.csv \
  --out jobs.csv --start 1000 --hours 0.05 --seed 0
```

writes

```text
job_id,job_submit_time,num_nodes,time_limit,bid,requires,runtime,requested,source,user,persona
j000001,0,6,40,0.5934,gpu,40,120,tioga,2,tier
j000002,35,4,40,0.5934,gpu,40,60,tioga,1,sticker
j000003,85,3,70,4.0325,gpu,70,80,tioga,6,value
j000004,115,8,50,0.832,,50,90,tioga,5,tier
j000005,174,1,15,0.5934,gpu,15,20,tioga,10,tier
```

Each limit is the run time: j000001 asked for 120 s and ran 40 s. The bids are step 2
(`02_bids.md`): single bids by default, and one `bid:<platform>` column per platform
with `--bids multi`.

## 1.7 Synthetic days

With `synthetic=True` (`--synthetic`), `prepare` returns one day drawn from the interval
instead of the interval itself: the jobs are real, but the day's mix is new, and each
seed draws another day. The interval is the pool, typically a busy stretch of whole
days.

1. **Groups.** Each source's cleaned jobs are split into groups: jobs one user submitted
   less than a minute apart. Most groups are a single job; the rest are arrays, often
   many copies of one job.
2. **How many.** For each source, the number of groups in the day is drawn from a
   Poisson distribution whose mean is the source's groups per day in the interval.
3. **Which and when.** That many groups are drawn at random from the source, each at
   most once, and each arrives at the time of day it really arrived, counted from the
   interval's start. A start at midnight gives clock times.
4. **Then as usual.** The sources are merged by time, and ids, hardware and bids follow
   as for a real interval.

Groups matter because arrays arrive as waves, and the waves are what queue: drawing
single jobs at a Poisson rate spreads each array over the day. The times of day carry
each source's daily pattern, so no curve has to be fitted.

- The seed fixes the day: each source draws from a generator keyed by the seed and the
  source.
- A job appears at most once in a day, so it keeps its identity, and with it its
  hardware and bid draws.
- The summary adds, for each source, `groups:<source>` in the interval and
  `synthetic:<source>` jobs in the day; `kept` still counts the interval's jobs.

## 1.8 Modelling choices

These choices are open to revision:

- **Run time as reference work.** A recorded run time is taken as work on a machine of
  speed 1.0, the Quartz reference that the platform speeds are measured against,
  whatever machine the job ran on.
- **Every job that started is load.** Failed, timed-out and cancelled-after-start jobs
  count, with the time they used.
- **Hardware is drawn.** The traces do not say which jobs used GPUs, so hardware is
  drawn per job, with the same probability for every source.
- **The request is only carried.** A job that ran past its request runs its recorded
  time.
- **Synthetic days.** Groups are jobs one user submitted within a minute of each other.
  Their number per day is Poisson, for each source on its own, and each keeps its time
  of day.
