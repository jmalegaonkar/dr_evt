# Step 4: value, cost, candidates and offers

This step is the contract between a job's bid, a machine's price and a mechanism: what a
job is worth where, what it costs there, where it can be placed, what it can win, and
what a mechanism may decide. Sections 4.4 and 4.5 follow one window of the fixture call
by call, under each of the four mechanisms. The overview of all steps is
`master_overview.md`.

Code: `mechanism/base.py` (`job_value`, `candidates`, `offers`, `Mechanism`,
`Decision`, `TOLERANCE`) and the checks in `market.py` (`_blocked_by`,
`_check_decisions`).

## 4.1 Value and cost

- **Cost on p**: `price(p) x nodes x limit / speed(p) / 3600`, the posted price times
  the node-hours the job uses there (step 3).
- **Value on p**: `bid x nodes x limit / 3600` for a single bid, the same on every
  machine; `bid(p) x nodes x limit / speed(p) / 3600` for a multi bid (step 2).
- **Surplus** is value minus cost.

## 4.2 Candidates and offers

- `candidates(job, platforms, free)` lists the machines where the job can be placed now:
  the machine has its hardware and at least its nodes free. Bids play no part.
- `offers(job, platforms, free)` keeps the candidates the job bid on, at any price, as
  `{machine: (cost, value)}`. A bid of zero or less is no bid.
- An offer whose value is under its cost is a bid **under the price**. It can win too,
  and the job then pays its whole bid. An offer whose value equals its cost, within
  `TOLERANCE` (1e-9), is at the price, not under it.

## 4.3 The mechanism contract

A mechanism implements `decide(jobs, platforms, free)`, returning a
`Decision(job_id, platform, charge)` for each winner. Its `offers` method says which
machines it may give a job and the bounds on the charge, as
`{machine: (cost, highest charge)}`. By default these are the bid offers, with the value
as the highest charge; `FirstFit` returns every candidate at its cost.

The market uses a mechanism's offers in three places:

1. **Intake.** A job with no offer even when every node is free waits outside the
   auction: its reason is `hardware`, `oversize` or `no_bid`, when it bid on no machine
   that could run it (`_blocked_by`).
2. **Checking decisions** (`_check_decisions`). Each job is decided at most once and
   belongs to the batch; its machine is one of its offers; its charge is at most the
   highest charge, and at least the cost, or the highest charge when that is less; no
   machine is over capacity.
3. **Work conservation.** No batch job may be left waiting while it has an offer on the
   nodes left over.

The batch itself uses candidates, not offers, so taking part never depends on bids. The
charge bounds protect both sides: a user never pays more than their value, except under
`FirstFit`, which reads no bid; a machine never gets less than its posted price, except
from a job that bid under it, which pays its whole bid.

## 4.4 One window, call by call

The fixture `tests/data/jobs.csv` at share 0.1 exposes Corona 12 nodes, Dane 154,
Matrix 3, Tioga 3 and Tuolumne 115.

**Intake.** Before the first window, `_blocked_by(job, platforms, full, mechanism)`
checks every job against a federation with every node free:

| job | nodes | bid | reason |
|---|---:|---|---|
| j000014 | 155 | 3.0 | `oversize`: no slice has 155 nodes |

Every other job enters the queue when it arrives. That includes j000015, which bids only
on Corona, 1.4 against a price of 1.5: a bid under the price is still an offer.

**The window at t = 0.** After `advance_to(0)`, `free_nodes()` gives
`{corona: 12, dane: 154, matrix: 3, tioga: 3, tuolumne: 115}`. Five jobs are queued:

| job | nodes, limit | bid | `candidates` | `offers`: cost, value |
|---|---|---|---|---|
| j000001 | 100, 180 s, CPU | 0.30 | dane, tuolumne | dane 1.0453, 1.5000; tuolumne 0.6781, 1.5000 |
| j000002 | 80, 150 s, CPU | 0.28 | dane, tuolumne | dane 0.6969, 0.9333; tuolumne 0.4521, 0.9333 |
| j000003 | 70, 55 s, CPU | 0.26 | dane, tuolumne | dane 0.2236, 0.2781; tuolumne 0.1450, 0.2781 |
| j000004 | 110, 180 s, GPU | `{"tuolumne": 0.5}` | tuolumne | tuolumne 0.3154, 0.8301 |
| j000005 | 3, 50 s, GPU | 0.19 | corona, matrix, tioga, tuolumne | corona 0.0625, 0.0079; matrix 0.0180, 0.0079; tioga 0.0160, 0.0079; tuolumne 0.0024, 0.0079 |

j000005 has an offer on all four machines, but its bid covers the price only on
Tuolumne: on Corona its cost of 0.0625 is above its value, and likewise on Matrix
(0.0180) and Tioga (0.0160). All five have a candidate, so the batch is all five.

**VCG's choice.** `Vcg.decide` first maximizes the total of value minus cost over the
offers that cover the price (one under it would only lower the total), with one machine
per job and the free nodes as capacity. j000001 and j000004
both want Tuolumne, and its 115 nodes hold only one of them:

| allocation | total surplus |
|---|---:|
| j000001 and j000005 on Tuolumne, j000002 and j000003 on Dane | 1.1184 |
| j000004 and j000005 on Tuolumne, j000001 on Dane | 0.9749 |

The first wins. j000004 has no other offer, so it waits.

**The charges.** Each winner pays its cost plus its Clarke pivot: how much the other
jobs' surplus would grow if it were absent.

| winner | machine | others without it | others with it | pivot | charge | value |
|---|---|---:|---:|---:|---:|---:|
| j000001 | Tuolumne | 0.8111 | 0.2965 | 0.5146 | 1.1927 | 1.5000 |
| j000002 | Dane | 0.9749 | 0.8819 | 0.0930 | 0.7898 | 0.9333 |
| j000003 | Dane | 1.0639 | 1.0639 | 0 | 0.2236 | 0.2781 |
| j000005 | Tuolumne | 1.1129 | 1.1129 | 0 | 0.0024 | 0.0079 |

j000001's pivot is almost exactly the surplus j000004 would have had on Tuolumne:
j000001 pays for displacing it. j000003 and j000005 displace nobody and pay their cost.

**The market's checks.** `_check_decisions` confirms each charge lies between cost and
value and that no machine is over capacity. The nodes left over are
`{corona: 12, dane: 4, matrix: 3, tioga: 3, tuolumne: 12}`, and
`offers(j000004, platforms, left)` is empty: j000004 may wait, since no machine it bid
on has room.

**Submit and advance.** `platforms["tuolumne"].submit([j000001, j000005], 0)` and
`platforms["dane"].submit([j000002, j000003], 0)` hand the winners to dr_evt with their
hold times: 129 s and 16 s on Tuolumne (180 / 1.401 and 50 / 3.313, rounded up), 175 s
and 64 s on Dane (150 / 0.861 and 55 / 0.861). After `advance_to(0)`, `waiting()` is 0
on every machine: every winner started at t = 0. `free_nodes()` now reads 12 on
Tuolumne and 4 on Dane.

**What happens to j000004.** It keeps its place in the queue. Tuolumne's nodes come
back when j000001 ends at 129 s, so j000004 wins Tuolumne in the window at 180 s.

## 4.5 The same window under the other mechanisms

The batch, the free nodes and the offers are those of section 4.4; only `decide`
changes.

**Pay what you bid (`FirstPrice`).** `FirstPrice().decide` lists every offer with its
net value, sorts them from the highest, and takes each in turn when its job is not
placed yet and the machine still has the nodes. Two offers with the same net would go to
the earlier job, then to the machine first by name. Each winner pays its value.

| offer | net | outcome |
|---|---:|---|
| j000001 on Tuolumne | 0.8219 | placed; Tuolumne has 15 left |
| j000004 on Tuolumne | 0.5146 | no room |
| j000002 on Tuolumne | 0.4813 | no room |
| j000001 on Dane | 0.4547 | already placed |
| j000002 on Dane | 0.2365 | placed; Dane has 74 left |
| j000003 on Tuolumne | 0.1330 | no room |
| j000003 on Dane | 0.0545 | placed; Dane has 4 left |
| j000005 on Tuolumne | 0.0055 | placed; Tuolumne has 12 left |
| j000005 on Tioga | -0.0081 | already placed |
| j000005 on Matrix | -0.0101 | already placed |
| j000005 on Corona | -0.0546 | already placed |

Offers under the price have negative net values, so they come last and can take only
the nodes left over. A job wins at most once: j000005's three offers under the price
find it placed on Tuolumne. The placement is VCG's, and the winners pay their full
values: 1.5000, 0.9333, 0.2781 and 0.0079.

**`FirstFit`.** `FirstFit().decide` visits the batch in arrival order. For each job its
`offers` are every candidate at its cost, whatever the bid, and it takes the cheapest
that still has the nodes:

| job | `FirstFit().offers(job, platforms, left)` | taken |
|---|---|---|
| j000001 | dane 1.0453, tuolumne 0.6781 | Tuolumne; 15 left |
| j000002 | dane 0.6969 | Dane; 74 left |
| j000003 | dane 0.2236 | Dane; 4 left |
| j000004 | none | waits |
| j000005 | corona 0.0625, matrix 0.0180, tioga 0.0160, tuolumne 0.0024 | Tuolumne; 12 left |

j000005's offers here include Corona, Matrix and Tioga, which its bid does not cover:
`FirstFit` reads no bid. The placement is again VCG's, and each winner pays its cost. In
general a job can land where it bid less than the price, or did not bid at all (its
value there is 0), and pay more than its value, so compare `FirstFit` with the auctions
on service, the waits, and not on welfare.

**`RegretFormer`.** The network here is untrained (`RegretFormer(seed=0)`), so its
choices show the mechanics, not a learned policy. `learned.window` turns the batch into
tensors, and the network gives each job a probability for each of its offers and for
waiting, and a payment fraction:

| job | offers, with their probabilities | wait | payment fraction |
|---|---|---:|---:|
| j000001 | dane 0.14, tuolumne 0.14 | 0.72 | 0.351 |
| j000002 | dane 0.17, tuolumne 0.17 | 0.66 | 0.372 |
| j000003 | dane 0.19, tuolumne 0.18 | 0.63 | 0.402 |
| j000004 | tuolumne 0.28 | 0.72 | 0.359 |
| j000005 | corona 0.18, matrix 0.20, tioga 0.20, tuolumne 0.15 | 0.27 | 0.470 |

The market takes the cells from the most probable, placing a job on the first of its
offers that still has the nodes: j000004 on Tuolumne (5 left), j000005 on Tioga (0
left), j000003 on Dane (84 left) and j000002 on Dane (4 left). j000001 then finds room
on neither machine and waits, which the work-conservation check allows. Each winner pays
`cost + fraction x (value - cost)`: 0.7848 for j000002, 0.2455 for j000003 and 0.5000
for j000004. j000005 bid under Tioga's price, so it pays its whole bid, 0.0079.

**Side by side.**

| mechanism | placed | welfare | revenue |
|---|---|---:|---:|
| VCG | j000001 and j000005 on Tuolumne, j000002 and j000003 on Dane | 1.1184 | 2.2085 |
| pay what you bid | the same | 1.1184 | 2.7193 |
| `FirstFit` | the same | 1.1184 | 1.6009 |
| `RegretFormer`, untrained | j000004 on Tuolumne, j000005 on Tioga, j000002 and j000003 on Dane | 0.7975 | 1.5382 |

The first three place the same jobs and differ only in what the winners pay: the cost
(`FirstFit`), the full value (pay what you bid), or the cost plus the surplus taken from
the others (VCG). The untrained network gives Tuolumne to j000004, whose surplus there
is smaller than j000001's, and sends j000005 to Tioga under the price, though Tuolumne
had room.
