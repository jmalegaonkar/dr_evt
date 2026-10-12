# Step 5: the market loop

The loop turns a stream of jobs into windows: every window it admits the jobs that have
arrived, auctions the ones that can be placed now, and hands the winners to their
platforms. This page follows the loop and then the fixture's whole run, window by
window. Step 4 (`04_offers.md`) looks inside one window; the overview of all steps is
`master_overview.md`.

Code: `run` and `_auction` in `market.py`, with the window length (`window_s`, 30 s by
default), the number of jobs an auction takes (`prefix`, 32), the most jobs a window
reaches among those that can be placed (`max_prefix`, 256), and aging (`aging` and
`aging_cap`, off by default).

## 5.1 The loop

1. **Order.** Jobs are sorted by submit time; jobs submitted together keep their input
   order.
2. **Intake.** Every job is checked once against the federation with every node free
   (`_blocked_by`). A job that could never run, or that bid on no machine that could
   run it, waits outside the auction for the whole run, with its reason: `hardware`,
   `oversize` or `no_bid`.
3. **Each window**, at `t = 0, 30, 60, ...`:
   1. Advance every platform to `t`, so jobs that ended release their nodes.
   2. Admit the arrivals with `submit <= t` to the end of the queue.
   3. Read each platform's free nodes.
   4. Form the batch: the first `prefix` queued jobs that can be placed now. A job with
      no candidate is skipped for the window and keeps its place. With aging, each
      job's bids first rise by its credit (section 5.3).
   5. Let the mechanism decide, and check its decisions (step 4). While the winners
      leave free nodes that a later job bid on, auction the next `prefix` jobs that fit
      those nodes, among the first `max_prefix` that could be placed when the window
      opened. Earlier winners keep theirs.
   6. Submit the winners to their platforms and advance again: no winner may still
      wait in dr_evt's queue, or the run stops with `MarketError`.
   7. Record each winner: its window, begin `t`, end `t` plus its hold time, cost,
      value, premium, charge and subsidy.
   8. Remove the winners from the queue. The others keep their places.
4. **Stop and drain.** The loop ends when every arrival has been admitted and the
   queue is empty. Every platform is then advanced to the last end, and the statistics
   are read.

The loop always ends: a job that could never run was set aside at intake, and a batch
job that fits the nodes left over must be placed, so on an idle federation every queued
job wins.

## 5.2 The fixture's run, window by window

The fixture `tests/data/jobs.csv` under VCG at share 0.1. At intake, j000014 is set
aside as `oversize` from its arrival at 180 s.

| window | arrivals | queue | free on Dane, Tuolumne | winners | still queued |
|---|---|---|---|---|---|
| 0, t = 0 | j000001 to j000005 | j000001 to j000005 | 154, 115 | j000001, j000005 on Tuolumne; j000002, j000003 on Dane | j000004 |
| 1, t = 30 | none | j000004 | 4, 15 | none | j000004 |
| 2, t = 60 | j000006 to j000009 | j000004, j000006 to j000009 | 4, 15 | j000006, j000009 on Tuolumne; j000007 on Matrix; j000008 on Tioga | j000004 |
| 3, t = 90 | j000010 | j000004, j000010 | 74, 2 | j000010 on Corona | j000004 |
| 4, t = 120 | j000011, j000012 | j000004, j000011, j000012 | 74, 15 | j000011 on Dane; j000012 on Tuolumne | j000004 |
| 5, t = 150 | j000013 | j000004, j000013 | 24, 115 | both on Tuolumne | none |
| 6, t = 180 | j000015 | j000015 | 104, 2 | j000015 on Corona | none |
| 7, t = 210 | j000016 | j000016 | 154, 112 | j000016 on Tuolumne | none |
| 8, t = 240 | j000017, j000018 | j000017, j000018 | 154, 101 | both on Tuolumne | none |
| 9, t = 270 | j000019 | j000019 | 154, 112 | j000019 on Tuolumne | none |
| 10, t = 300 | j000020 | j000020 | 154, 115 | j000020 on Tuolumne | none |

- **j000004** needs 110 GPU nodes and can win only Tuolumne. It heads the queue from
  the start, but from 30 s to 120 s Tuolumne has at most 15 nodes free, so it has no
  candidate and is left out of the batch while the jobs behind it are auctioned.
  j000001 ends at 129 s, and j000004 wins Tuolumne at 150 s, after a wait of 150 s.
- **j000015** bids 1.4 on Corona, under its price of 1.5, and nowhere else. At 180 s no
  bid that covers the price wants Corona, so VCG's leftover fill places it there, and it
  pays its whole bid: 0.0778 against a cost of 0.0833.
- **No wait for the window.** Every fixture job arrives at a multiple of 30 s, on a
  window, so the only wait is j000004's, for nodes.
- **The end.** After window 10 no arrival is left and the queue is empty. The last job
  ends at 318 s, and the drain advances every platform to 330 s.

## 5.3 Aging

With `aging` (`run --aging`), a job that keeps waiting bids more. Every whole 10 minutes
since its submission adds `aging` to its value on each platform it bid on, in the posted
prices' unit, up to `aging_cap` in all (`--aging-cap`, 1.0 by default). The credit
depends only on how long the job has waited, a public fact, never on its bid or its
size. Aging is off by default.

- **What the mechanism sees.** The raised bids, in every window where the job can be
  placed. The credit is the same on every platform, so it changes who wins wherever jobs
  compete but never which platform a job prefers. It never creates an offer: a
  platform the job did not bid on gets no credit.
- **Who pays.** The federation. The platform receives the whole charge, and the job
  pays the charge less its credit on that platform, never below zero. The ledger
  records the federation's part as `subsidy` (step 7), and `value` stays what the job
  bid, so the credit never counts as welfare.
- **What it reaches.** A job that lost an auction gains on the jobs that beat it. A job
  that cannot be placed at all, such as a wide one on a crowded platform, gains nothing
  while it waits, but brings its credit to the window when room appears.
- **Why the same credit for every job.** Waiting, not size, earns it. A credit that grew
  with a job's size would let a large job that waited take freed nodes from many small
  ones.
- **Incentives.** Within a window the credit is a public boost: a job is treated exactly
  as a job that bid its credit more, so no mechanism's incentives change there. Across
  windows they do: waiting costs a job nothing in this model, so a job could bid low now
  and collect credit for later. The cap bounds that gain.

Nothing in the fixture waits 10 minutes, so aging changes nothing there.
`test_aging_lets_a_long_wait_outbid_a_newcomer` shows it at work: a job that waited 15
minutes earns a credit of 0.15, outbids a newcomer whose surplus is 0.1, and pays 0.25
of a 0.4 charge.

## 5.4 What follows

- **Most waiting is the window.** A job that arrives when its platforms have room still
  waits for the next window, half a window on average. With a lightly loaded federation
  that is most of the wait.
- **Later jobs get only what is left.** A large array that arrives at once starts
  together when the machines have room: after the first auction, the next jobs are
  auctioned for the nodes left, `prefix` at a time. When the winners fill the free
  nodes, nothing is refilled, and a higher bid behind the batch waits for the next
  window.
- **A job without a candidate does not block the queue.** It keeps its place, and the
  jobs behind it are auctioned in the meantime.

## 5.5 Modelling choices

These choices are open to revision:

- **A fixed clock.** The market clears every `window_s` seconds, not at each arrival.
- **Refills, not a larger auction.** An auction takes at most `prefix` jobs, in arrival
  order, and jobs past them win only nodes the earlier auctions leave, so a later job
  never displaces an earlier winner, a wide job included. The first auction's outcome
  decides which nodes are left, so a bid can change which later jobs win.
- **Empty machines at the start.** A run starts with every platform idle: jobs that
  were running before the first arrival are not carried in.
