# Market Paper Plan

A plan for a machine learning conference paper built on the federated market in this
repository. It follows the directions chosen in
[Market Research Directions](MARKET_RESEARCH_DIRECTIONS.md): an exact-truthful learned
mechanism as the method, certified evaluation as the backbone, learning bidders as the
robustness study, and the dynamic result as the headline if the measured gap justifies
it. RegretFormer is the approximately truthful comparison arm.

## Working title

Truthful learned placement of composite jobs across heterogeneous clusters, evaluated
with a real scheduler in the loop.

## Claims

1. A learned affine maximizer over the exact allocation MILP is dominant-strategy
   truthful by construction and reaches higher welfare, or lower slowdown, than VCG on
   realistic federated workloads, because its boosts encode aging and platform
   preference that VCG cannot express without breaking incentives.
2. Its coverage of the per-window optimum and of the horizon optimum is measured, not
   assumed, and the gap between the two quantifies what myopic clearing costs.
3. The mechanism holds up against bidders that learn, including no-regret and LLM
   bidders under a credit budget, where approximately truthful learned mechanisms with
   corrected regret estimates do not.
4. Every welfare number comes with realized wait, bounded slowdown and utilization from
   EASY-backfilled platforms, and composites never partially start.

## Method

- Observation: the market observation of this repository, public context per job and
  candidate (sizes, limits, waiting time, resource cost, free nodes per platform) and
  private per-platform values per leg.
- Mechanism family: weighted VCG with per-job weights and per-candidate boosts produced
  by a permutation-equivariant network over public context only. Allocation by the
  exact MILP with the boosted objective; payments by the affine maximizer formula.
- Training: decision-focused learning through the MILP, with a lottery relaxation or a
  Lagrangian decomposition where the exact gradient is unavailable; objectives welfare,
  revenue and bounded slowdown; aging enters as a boost on waiting time.
- Guarantee: bounded boosts imply a welfare bound relative to VCG, stated and checked.

## Baselines

VCG (truthful, myopic); RegretFormer trained on the same windows, with regret estimated
by a fixed grid search per report dimension followed by gradient ascent, scored through
the deployed rounding and payment pipeline; a posted-price mechanism; the bid-blind
routers of the previous system, run unchanged.

## Experiments

1. Coverage against the per-window MILP optimum on the fixture world and on the Lassen
   workload with synthetic per-platform values that carry a turnaround component.
2. Coverage against the horizon optimum, one MILP over the whole run on realized
   capacity, to size the myopic gap and decide whether the dynamic result is a headline.
3. Corrected regret for RegretFormer and zero regret for the affine maximizer by
   construction, plus a misreport probe over per-platform reports for every arm.
4. Learning bidders: no-regret and LLM agents under the credit budget over repeated
   windows; welfare, revenue and manipulation gain per arm.
5. Systems metrics from DR_EVT: wait, bounded slowdown, utilization, per platform and
   pooled; partial starts reported.
6. Sweeps: window length, composite fraction, platform speed heterogeneity through the
   runtime scale, and value dispersion.

## What the repository must provide, in order

1. End to end with VCG (PR 3).
2. RegretFormer as a mechanism subclass with training, grid-plus-ascent regret
   evaluation and checkpoints (PR 4).
3. The runtime model on streamed jobs, so platforms can differ in speed (PR 5).
4. Window logging for training data and the horizon optimum inputs; the value
   generator with a turnaround component; the learned affine maximizer (PR 6 onward).
5. Bidder agents for the robustness study.

## Risks

Training through a MILP is the technical risk of the method; if learned boosts add
little over VCG the paper has no method, so the horizon-gap and aging experiments come
first. The dynamic claim needs values that reward earlier completion, so the value
generator is designed before the data exists. Reviewers will ask why learning beats a
hand-tuned affine maximizer, so a hand-tuned boost family is a baseline from the start.
