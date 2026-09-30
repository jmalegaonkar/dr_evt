# Step 2: bids, from a user to a price

The traces carry no bids, so preparation synthesizes each job's willingness to pay from
a persona rule, in one of two forms: a single bid valid on every platform, or a bid on
each platform. This page covers what a bid means, how each form is set, where the
random draws come from, and how the market reads a bid. Step 1 (`01_jobs.md`) builds the
jobs these bids belong to; the overview of all steps is `master_overview.md`.

Code: `jobs/bids.py` (`generator`, `terms`, `single_bid`, `multi_bid`), the job loop of
`jobs/prepare.py`, `jobs/job.py` (`Job.price`), and `mechanism/base.py` (`job_value`,
`candidates`, `offers`).

## 2.1 What a bid is

A bid is the most the user will pay per node-hour, in the unit of the posted prices.

| form | jobs-file column | unit | value on platform p |
|---|---|---|---|
| single | `bid` | per node-hour at speed 1.0 | `bid x nodes x limit / 3600` |
| multi | `bid:<p>` | per node-hour of p | `bid(p) x nodes x limit / speed(p) / 3600` |

A single bid prices the work, so the job is worth the same on every platform. A multi
bid prices each platform's node-hours separately. The cost of a job on p is
`posted price(p) x nodes x limit / speed(p) / 3600`: the posted price times the
node-hours the job uses there. What a winner pays is set by the mechanism, and the
market checks that it lies between the job's cost and its value.

`prepare` writes single bids unless told `bids="multi"` (`--bids multi`).

## 2.2 Candidates and offers

- A platform is a **candidate** for a job when the job can be placed there now: the
  platform has the job's hardware and enough free nodes. Bids play no part.
- A candidate is an **offer** when the job bid on it and its value there covers its
  cost: `bid >= posted price(p) / speed(p)` for a single bid, and
  `bid(p) >= posted price(p)` for a multi bid.

Every window auctions the first 32 queued jobs that have a candidate, so who takes part
depends only on public facts. A job can win only among its offers: a job whose bid is
under the price wherever it fits takes part, loses, and stays queued.

## 2.3 Personas

Every user draws one persona, and each persona bids a multiple of a price:

| persona | share of users | multiple | mean multiple |
|---|---:|---|---:|
| sticker | 45% | 1 | 1 |
| tier | 35% | 1; 2 if the job is urgent (20% of jobs); 4 if also a heavy user (20%) | 1.28 |
| value | 15% | a lognormal multiple per job, median 3, sigma 0.5 | 3.40 |
| whale | 5% | 10 | 10 |

The persona and the heavy flag belong to the user; urgency and the value multiple
belong to the job.

## 2.4 The single bid

The user looks up what a unit of work costs on each platform that can run the job, the
posted price divided by the speed, and takes the average: the price level. The single
bid is that level times the persona's multiple:

```text
bid = mean over usable platforms of (posted price(p) / speed(p)) x multiple
```

With the five default platforms the level is 0.832 for a CPU job and 0.5934 for a GPU
job, so a sticker bids exactly that. Relative to each posted price, the same bid is
generous where work is cheap and short where it is dear: a sticker's CPU bid is about 6
times Tuolumne's price per unit of work and about half of Corona's.

## 2.5 The multi bid

On each platform p that has the job's hardware, the user takes an attitude of its own
toward that platform's price:

```text
bid(p) = posted price(p) x multiple(persona on p) x preference(user, p)
```

- **The persona on p** is the user's own persona half the time; otherwise the platform
  draws one from the same table. A sticker can therefore be a whale on one platform, and
  a whale can be a sticker on another.
- **The preference** is `exp(N(0, 0.3))`: median 1, two thirds of preferences between
  0.74 and 1.35. On a platform where the multiple is 1 it puts the bid under the posted
  price half the time.

So a user's bids can be well above the price on some platforms, near it on others and
under it on the rest, or alike everywhere: all near the price, all above it, or all
under it. The persona a platform takes and the preference belong to the user and the
platform, so they hold for every job of that user.

## 2.6 Where the draws come from

Every draw comes from a generator seeded by a SHA-256 hash of the seed, the source and
a key:

| generator | key | draws, in order |
|---|---|---|
| the row's | the row's identity | the hardware, the urgency, the value multiple |
| the user's | the user | the persona, the heavy flag |
| a platform's | the user and the platform | own persona or not, a persona, the preference |

- Each draw keeps a fixed place: every generator makes all of its draws, whatever the
  persona or the form, so changing the persona rule or overriding the hardware with
  `--requires` moves no other draw.
- No key contains the job number, so a job keeps its draws whatever interval is
  selected.
- A platform's draws do not depend on which other platforms are listed.
- A row without a user is its own user. Its persona and heavy flag follow in the row's
  generator, after the row's three draws, and its platforms are keyed by the row's
  identity.

## 2.7 What follows

- **A single bid favors cheap work.** Its value is the same everywhere, so the surplus,
  value minus cost, is largest where the job costs least, and it covers the price on
  every platform whose cost per unit of work is under the job's level times its
  multiple.
- **A multi bid follows each platform's price.** Its surplus on p is
  `(multiple x preference - 1) x cost(p)`: for the same attitude, larger on the platform
  that costs more. A mechanism that maximizes total surplus sends a job to the platform
  where it is most eager relative to the price, scaled by that price.
- **Jobs that can win nowhere.** A job whose bid is under the price on every platform
  where it fits cannot win at fixed prices. The market lists it as waiting from its
  arrival, with reason `unaffordable` (step 5). A multi bidder can be in that position
  on its own draws.

## 2.8 The fixture, draw by draw

The five jobs of `01_jobs.md` (the fixture trace as source `tioga`, `start=1000`,
`hours=0.05`, seed 0), bid on the five default platforms:

| job | user | hardware draw | urgency draw | persona draw | persona | multiple |
|---|---|---|---|---|---|---:|
| j000001 | 2 | 0.362: GPU | 0.787: not urgent | 0.551 | tier, not heavy | 1 |
| j000002 | 1 | 0.456: GPU | 0.811 | 0.053 | sticker | 1 |
| j000003 | 6 | 0.203: GPU | 0.266 | 0.941 | value | 6.795 |
| j000004 | 5 | 0.852: CPU | 0.759: not urgent | 0.663 | tier, not heavy | 1 |
| j000005 | 10 | 0.464: GPU | 0.265: not urgent | 0.538 | tier, not heavy | 1 |

A job is GPU when its hardware draw is under 0.5. User 5 escapes the heavy flag by a
hair: its draw is 0.204 and the flag needs less than 0.2.

**Single bids.** Four jobs bid their level: 0.5934 for the GPU jobs j000001, j000002 and
j000005, and 0.832 for the CPU job j000004. The value job j000003 bids
0.5934 x 6.795 = 4.0325.

**Multi bids**, each with the persona it used on that platform (own: the user's
persona; otherwise the platform's draw) and its multiple of the posted price. A bid of
at least 1x the price is an offer whenever the job fits:

| job | Corona 1.5 | Dane 0.18 | Matrix 1.6 | Tioga 2.7 | Tuolumne 0.19 |
|---|---|---|---|---|---|
| j000001 | own tier, 1.28x | | sticker, 0.85x | sticker, 0.97x | sticker, 0.92x |
| j000002 | sticker, 0.56x | | sticker, 1.10x | own sticker, 1.52x | own sticker, 1.55x |
| j000003 | value, 6.31x | | own value, 4.92x | sticker, 1.28x | own value, 6.69x |
| j000004 | own tier, 0.76x | value, 5.50x | own tier, 0.75x | own tier, 1.75x | own tier, 1.50x |
| j000005 | own tier, 0.76x | | own tier, 0.95x | sticker, 1.13x | tier, 1.04x |

- j000001 can win only on Corona: the other three platforms drew a sticker and a
  preference under 1.
- j000004 bids 5.5 times Dane's price, where its platform drew a value persona, and
  under the price on Corona and Matrix.
- j000003 covers every price, and its two forms place it differently. Its 3 GPU nodes
  for 70 s give:

| | Corona | Matrix | Tioga | Tuolumne |
|---|---:|---:|---:|---:|
| cost | 0.0875 | 0.0253 | 0.0224 | 0.0033 |
| single: value | 0.2352 | 0.2352 | 0.2352 | 0.2352 |
| single: surplus | 0.1477 | 0.2100 | 0.2129 | 0.2319 |
| multi: value | 0.5518 | 0.1242 | 0.0287 | 0.0224 |
| multi: surplus | 0.4643 | 0.0990 | 0.0063 | 0.0190 |

With a single bid the job gains most on Tuolumne, the cheapest platform; with its
multi bid it gains most on Corona, the dearest, where it bids 6.31 times the price.

## 2.9 Modelling choices

These choices are open to revision:

- **A single bid prices work at the average level.** The level is the mean cost of a
  unit of work over the platforms that can run the job.
- **A multi bid is an attitude per platform.** Each platform takes the user's persona
  half the time and draws its own otherwise, shaded by a preference.
- **Personas, platform attitudes and preferences persist per user.** A user is keyed by
  the source and the pseudonym, so the same pseudonym in two traces is two users.
- **The parameters are fixed.** The persona shares and multiples, the 20 percent
  urgency and heavy rates, the even split between a user's own persona and a platform's
  draw, and the preference spread of 0.3 are the rule's constants.
