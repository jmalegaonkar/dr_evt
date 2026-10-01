# Step 5: the market loop

The loop turns a stream of jobs into windows: every window it admits the jobs that have
arrived, auctions the ones that can be placed now, and hands the winners to their
platforms. This page follows the loop and then the fixture's whole run, window by
window. Step 4 (`04_offers.md`) looks inside one window; the overview of all steps is
`master_overview.md`.

Code: `run` in `market.py`, with the window length (`window_s`, 60 s by default) and
the prefix (32 by default).

## 5.1 The loop

1. **Order.** Jobs are sorted by submit time; jobs submitted together keep their input
   order.
2. **Intake.** Every job is checked once against the federation with every node free
   (`_blocked_by`). A job that could never run at its price waits outside the auction
   for the whole run, with its reason: `hardware`, `oversize` or `unaffordable`.
3. **Each window**, at `t = 0, 60, 120, ...`:
   1. Advance every platform to `t`, so jobs that ended release their nodes.
   2. Admit the arrivals with `submit <= t` to the end of the queue.
   3. Read each platform's free nodes.
   4. Form the batch: the first 32 queued jobs that can be placed now. A job with no
      candidate is skipped for the window and keeps its place.
   5. Let the mechanism decide, and check its decisions (step 4).
   6. Submit the winners to their platforms and advance again: no winner may still
      wait in dr_evt's queue, or the run stops with `MarketError`.
   7. Record each winner: its window, begin `t`, end `t` plus its hold time, cost,
      value, premium and charge.
   8. Remove the winners from the queue. The others keep their places.
4. **Stop and drain.** The loop ends when every arrival has been admitted and the
   queue is empty. Every platform is then advanced to the last end, and the statistics
   are read.

The loop always ends: a job that could never run was set aside at intake, and a batch
job that fits the nodes left over must be placed, so on an idle federation every queued
job wins.

## 5.2 The fixture's run, window by window

The fixture `tests/data/jobs.csv` under VCG at share 0.1. At intake, j000014 is set
aside as `oversize` and j000015 as `unaffordable`, both from their arrival at 180 s.

| window | arrivals | queue | free on Dane, Tuolumne | winners | still queued |
|---|---|---|---|---|---|
| 0, t = 0 | j000001 to j000005 | j000001 to j000005 | 154, 115 | j000001, j000005 on Tuolumne; j000002, j000003 on Dane | j000004 |
| 1, t = 60 | j000006 to j000009 | j000004, j000006 to j000009 | 4, 15 | j000006, j000009 on Tuolumne; j000007 on Matrix; j000008 on Tioga | j000004 |
| 2, t = 120 | j000010 to j000012 | j000004, j000010 to j000012 | 74, 15 | j000010 on Corona; j000011 on Dane; j000012 on Tuolumne | j000004 |
| 3, t = 180 | j000013 | j000004, j000013 | 104, 115 | j000004, j000013 on Tuolumne | none |
| 4, t = 240 | j000016 to j000018 | j000016 to j000018 | 154, 112 | all three on Tuolumne | none |
| 5, t = 300 | j000019, j000020 | j000019, j000020 | 154, 115 | both on Tuolumne | none |

- **j000004** needs 110 GPU nodes and can win only Tuolumne. It heads the queue from
  the start, but at 60 s and 120 s Tuolumne has 15 nodes free, so it has no candidate
  and is left out of the batch while the jobs behind it are auctioned. j000001 ends at
  129 s, and j000004 wins Tuolumne at 180 s, after a wait of 180 s.
- **The window's own delay.** j000010, j000013, j000016 and j000019 arrive halfway
  between two windows and wait 30 s for the next one. The others arrive on a window and
  wait nothing.
- **The end.** After window 5 no arrival is left and the queue is empty. The last job
  ends at 318 s, and the drain advances every platform to 360 s.

## 5.3 What follows

- **Most waiting is the window.** A job that arrives when its platforms have room still
  waits for the next window, half a window on average. With a lightly loaded federation
  that is most of the wait; the prefix adds to it when a large array arrives at once.
- **A job without a candidate does not block the queue.** It keeps its place, and the
  jobs behind it are auctioned in the meantime.

## 5.4 Modelling choices

These choices are open to revision:

- **A fixed clock.** The market clears every `window_s` seconds, not at each arrival.
- **A prefix.** At most `prefix` jobs are auctioned per window, taken in arrival order.
- **Empty machines at the start.** A run starts with every platform idle: jobs that
  were running before the first arrival are not carried in.
