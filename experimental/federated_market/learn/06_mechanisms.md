# Step 6: mechanisms

A mechanism decides, for one window's batch, which job runs where and what each winner
pays. This page gives each of the four mechanisms with its math, follows three jobs on
two platforms through all of them, and covers how RegretFormer is trained and how regret
is measured. Step 4 (`04_offers.md`) covers the contract every mechanism works within,
with one window of the fixture under each mechanism; the overview of all steps is
`master_overview.md`.

Code: `mechanism/vcg.py`, `mechanism/firstprice.py`, `mechanism/firstfit.py`,
`mechanism/regretformer.py` with its network in `mechanism/learned.py`, training in
`mechanism/training.py`, and regret in `mechanism/regret.py`.

## 6.1 What every mechanism keeps

In one window, job $i$ needs $n_i$ nodes and platform $p$ has $F_p$ free. On $p$ the job
costs $c_{ip}$, the posted price times the node-hours it uses there, and is worth
$v_{ip}$, its bid times those node-hours. Its surplus there is
$s_{ip} = v_{ip} - c_{ip}$, negative when it bid under the price. With $x_{ip} = 1$ when
job $i$ runs on $p$, every mechanism keeps

$$
\sum_p x_{ip} \le 1 \ \text{ for every job,} \qquad
\sum_i n_i\, x_{ip} \le F_p \ \text{ for every platform.}
$$

So a job never wins two platforms. When it has offers on several, the mechanism picks
one: that choice and the charge are all a mechanism decides. The market then checks each
decision (step 4). A winner pays at most $v_{ip}$, and at least $\min(c_{ip}, v_{ip})$:
its cost or, when it bid under the price, its whole bid. `FirstFit`, which reads no bid,
charges exactly $c_{ip}$. No batch job may wait while a platform it has an offer on has
room left.

## 6.2 VCG

`Vcg.decide` finds the most surplus that the offers covering the price can make,

$$
W = \max_x \sum_{i,p\,:\,s_{ip} \ge 0} s_{ip}\, x_{ip},
$$

leaving out the bids under the price, which could only lower it. `_solve` writes one
binary variable per offer that covers the price and solves the integer program with
SciPy's `milp`, with no optimality gap and a limit of 10 s per solve
(`Vcg(time_limit_s=...)`). A solve that does not reach the optimum raises.

A winner $i$ on $p$ pays its cost plus its Clarke pivot, what its presence costs the
others:

$$
\text{charge}_i = c_{ip} + W_{-i} - (W - s_{ip}),
$$

where $W_{-i}$ is the most the others make without it, found by solving again without
job $i$: a window with $k$ winners takes $k + 1$ solves. A winner that displaces nobody
pays its cost; one that pushes out a rival pays the surplus the rival would have had, a
second price. The charge never exceeds $v_{ip}$, since $W \ge W_{-i}$.

The jobs the solve leaves out that still fit then take the nodes left over
(`_leftover_offer`), in batch order, each on the platform where its surplus is highest,
and pay $\min(c_{ip}, v_{ip})$. Those are the jobs that bid the price exactly, which add
nothing to $W$, and the jobs that bid under it.

**Ties.** Each chosen offer adds $10^{-9}$ times its position in the batch to the
objective, so among equally good placements the solve prefers the earlier job and, for
one job, the platform listed first.

## 6.3 Pay what you bid

`FirstPrice.decide` lists every offer $(i, p)$ by its surplus $s_{ip}$, from the
highest; equal surpluses go to the earlier job, then to the platform first by name.
Walking down the list, it places $i$ on $p$ when $i$ is not placed yet and $p$ still has
$n_i$ nodes, so a placed job's other offers are skipped. A winner pays $v_{ip}$, its
whole bid. Bids under the price have negative surplus, so they come last and take what
is left, by surplus rather than in batch order.

## 6.4 FirstFit

`FirstFit` reads no bid: its offers are every candidate, at its cost. In batch order,
each job goes to the platform with room where $c_{ip}$ is lowest, and pays $c_{ip}$;
equal costs go to the platform listed first. A job can land where it bid less than the
price, or did not bid at all (its value there is 0), and pay more than its value, so
compare `FirstFit` with the auctions on service, not on welfare.

## 6.5 RegretFormer

**The network** (`learned.Network`) is that of Ivanov et al. (NeurIPS 2022), with the
paper's defaults: hidden size 32, 2 attention heads, 1 layer. For every cell $(i, p)$ of
a window it reads five channels: the value and the cost, divided by the window's mean
posted cost over the cells that fit; the job's nodes and the free nodes, as shares of
the platform's slice; and whether the cell is an offer. Attention runs across platforms
and across jobs. For each job the network returns probabilities over its offers and
waiting, and a payment fraction $f_i$ between 0 and 1.

**Deployment** (`learned.deploy`, `RegretFormer.decide`). The market takes cells from
the most probable and places each job on the first of its offers that still has room.
Waiting only lowers a job's place in that order, so a job stays queued only when none of
its offers has room, as the market requires. A winner pays

$$
\text{charge}_i = m_{ip} + f_i\,(v_{ip} - m_{ip}), \qquad m_{ip} = \min(c_{ip}, v_{ip}),
$$

its cost plus a share of its surplus or, when it bid under the price, its whole bid.
`learned.charge` holds this rule for deployment, training and regret alike.

**Training** (`train_regretformer`, or `python -m federated_market train`). `harvest` runs
the market on many job streams under several mechanisms and keeps every window's batch
and free nodes (step 7). Each step draws 8 of those windows and computes a relaxed
outcome, in which the probabilities stand for the allocation. The loss is

$$
-\,\text{objective} + \lambda \cdot \text{regret}
+ 10\,(\text{overbooked} + \text{idle}),
$$

where:

- the objective is the premiums the winners pay over the posted cost, weighted by the
  probabilities, or the welfare with `objective="welfare"`, divided by the window's
  mean cost;
- regret is the most a job gains by misreporting, over a 9-point item-wise grid of
  reports for up to four jobs per window, counting only reports at or above each posted
  price unless `under_price`;
- overbooked counts the nodes the relaxed allocation uses beyond the free ones, and idle
  the free nodes it leaves idle while a job that fits them waits.

The multiplier $\lambda$ rises while the regret exceeds a budget that shrinks from 1 to
0.1 percent of the jobs' available surplus, and falls while it is below. The defaults
are 2,000 steps at a learning rate of 0.001, on the CPU unless `device` names a GPU.

**Regret** (`mechanism/regret.py`). `grid_regret(mechanism, jobs, platforms, free)`
measures, for one window and any mechanism, how much each job could gain by misreporting
while the others report truthfully, on the decisions the mechanism applies. It moves one
price at a time over 101 levels from zero to four times the job's highest price, and to
that platform's posted price, then scales the whole report over the same range and to
each posted price: the item-wise grid of You et al. (2026). The result is a lower bound.
`refined_regret` adds their guided gradient refinement for RegretFormer, and reports
beside it the gradient-only estimate of RegretFormer's own protocol.

## 6.6 Three jobs on two platforms

Platform A has 4 free nodes at 1.0 a node-hour and B has 6 at 2.0. Three jobs each have
one hour of work at speed 1.0 and a bid on each platform:

| job | nodes | A: bid; cost, value, surplus | B: bid; cost, value, surplus |
|---|---:|---|---|
| J1 | 4 | 3.0; 4, 12, 8 | 4.0; 8, 16, 8 |
| J2 | 4 | 2.75; 4, 11, 7 | 1.5; 8, 6, -2 |
| J3 | 2 | 1.5; 2, 3, 1 | 1.5; 4, 3, -1 |

J1 gains 8 on either platform. J2 gains on A but bid under B's price, and so did J3. A
holds one of the 4-node jobs; B holds one of them and J3. Each mechanism decides
(`test_three_jobs_on_two_machines` pins the first three):

| job | VCG | pay what you bid | `FirstFit` | RegretFormer, untrained |
|---|---|---|---|---|
| J1 | B, pays 8 | A, pays 12 | A, pays 4 | B, pays 11.7857 |
| J2 | A, pays 5 | B, pays 6, under the price | B, pays 8, under the price | waits |
| J3 | B, pays 3, under the price | B, pays 3, under the price | B, pays 4, under the price | A, pays 2.5083 |
| welfare | 14 | 5 | 5 | 9 |
| revenue | 16 | 21 | 16 | 14.294 |

"Under the price" marks a job that bid under the posted price where it runs.

**VCG.** The offers that cover the price make at most $W = 15$: J1 on B (8) and J2 on A
(7). J1 gains 8 on either platform, but J2 gains only on A, so the solve sends J1 to B.

| winner | platform | placed by | cost | value | others without it | others with it | pivot | charge |
|---|---|---|---:|---:|---:|---:|---:|---:|
| J1 | B | the solve | 8 | 16 | 7 | 7 | 0 | 8 |
| J2 | A | the solve | 4 | 11 | 9 | 8 | 1 | 5 |
| J3 | B | the leftover fill | 4 | 3 | | | | 3 |

Without J1 the others make 7, as they do with it: J1's pivot is 0, and it pays its
cost. Without J2, J1 on B and J3 on A would make 9, against 8 with J2: J2's pivot is 1,
and it pays 4 + 1 = 5. B has 2 nodes left, and J3 takes them in the leftover fill, under
B's price: it pays its whole bid, 3.

**Pay what you bid** walks down its list:

| offer, by surplus | surplus | outcome |
|---|---:|---|
| J1 on A | 8 | placed, pays 12; A has 0 left |
| J1 on B | 8 | skipped: already placed |
| J2 on A | 7 | no room |
| J3 on A | 1 | no room |
| J3 on B | -1 | placed, pays 3; B has 4 left |
| J2 on B | -2 | placed, pays 6; B has 0 left |

J1's two offers tie at 8, and the tie goes to A by name. J1 takes A, and its offer on B
is skipped, since it is placed. J2 and J3 find A full, and their offers under the price
then take B: J3's first, then J2's. Everyone runs, but J2 runs where it gains least, and
the welfare is 5 against VCG's 14.

**`FirstFit`** sends J1 to the cheaper A, then J2 and J3 to B, where they pay the cost,
8 and 4, more than their values there, 6 and 3.

**RegretFormer**, untrained (`RegretFormer(seed=0)`), puts J1 on B and J3 on A, which
leaves no platform with 4 nodes for J2. Its choices show the mechanics, not a learned
policy.

## 6.7 Incentives

- **VCG is truthful among bids that cover the price.** When every bid is at or above the
  posted price, no report beats the truth
  (`test_truthful_bid_resists_scaled_misreports`). A bid under the price wins only the
  nodes left over and pays itself, so shading under the price can pay: alone on a
  platform, a job pays its cost when truthful, and less by bidding under the price
  (`test_bidding_under_the_price_pays_on_idle_nodes`).
- **Pay what you bid** charges the bid, so shading pays whenever the job still wins.
- **`FirstFit`** reads no bid, so no report changes what a job gets or pays: its regret
  is zero (`test_first_fit_has_no_regret`), but a winner can pay more than its value.
- **RegretFormer's** regret is whatever its training leaves; `refined_regret` measures
  it.

## 6.8 Modelling choices

These choices are open to revision:

- **Leftover nodes go in batch order under VCG**, each job to its best leftover
  platform, and by surplus under pay what you bid.
- **Welfare counts the posted cost.** A job placed under the price counts as a loss,
  $v_{ip} - c_{ip} < 0$, although the platform's nodes would otherwise sit idle.
- **RegretFormer's revenue is the premiums over the posted cost.** A winner under the
  price earns it a negative premium in training, while the market reports revenue as the
  total of the charges.
- **Training's regret leaves out bids under the price.** A mechanism that places every
  job it can must accept such bids on idle nodes, so shading under the price pays under
  every mechanism, and a regret budget that counts it pushes RegretFormer, in training,
  to keep jobs waiting. `grid_regret` and `refined_regret` count every report by
  default; `under_price=False` measures what training holds the network to.
