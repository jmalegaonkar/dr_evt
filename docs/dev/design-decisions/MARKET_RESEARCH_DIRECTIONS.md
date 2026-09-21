# Market Research Directions

Where the federated market over DR_EVT can go as a research project, written after a
review of the 2025 and 2026 literature on learned mechanism design, dynamic mechanisms,
learning bidders, decision-focused learning and HPC scheduling. This page records the
landscape and the candidate directions; the companion page,
[Market Paper Plan](MARKET_PAPER_PLAN.md), turns the chosen ones into a paper plan.

## What the market in this repository is

Once per window the client is the auction: it receives the queued jobs, each with a
private value per platform for each of its legs, and the free nodes every platform
reports. It decides which jobs go to which platform now and what they pay, submits the
winners into empty queues at the window time, and reads start and end times back from
the platforms. The allocation problem is a capacitated generalized assignment with
all-or-nothing composite jobs, solved exactly by a MILP. VCG on that MILP is the
truthful reference. Every platform is a `Simulation` with EASY backfilling, so every
welfare number comes with realized wait, slowdown and utilization from a real scheduler.

Two properties make the setting unusual for a learning paper. The optimum is computable
per window, so any learned method has a ground-truth coverage number. And the scheduler
underneath is real, so systems claims do not rest on a stylized model.

## What changed in the field

1. Approximate truthfulness measured by regret is no longer accepted at face value.
   You et al. (2026) show that RegretNet, RegretFormer and CITransNet underestimate true
   regret, in some models by hundreds of times. The field has moved to exact incentive
   compatibility by construction: affine maximizers (AMenuNet), menu mechanisms
   (GemNet, TEDI, BundleFlow), and dual certificates that bound what any truthful
   mechanism can achieve.
2. Dynamic mechanisms have an online learning story: dynamic VCG learned in an unknown
   MDP with regret guarantees (2025), and incentive-aware primal-dual allocation under
   long-term constraints with square-root welfare regret (2025). None of it has a real
   scheduler generating the state transitions.
3. Bidders that learn are a subfield: strategically robust learning in repeated auctions
   (2026), auctions designed for algorithmic bidders, and LLM bidders that behave
   truthfully under VCG but not under other rules.
4. Learning-augmented mechanism design uses predictions inside a truthful mechanism with
   consistency when the prediction is right and robustness when it is wrong.
5. Training through a MILP is standard: decision-focused learning with differentiable
   optimization, Lagrangian decomposition for scale, and surrogate losses.
6. LLM-driven mechanism discovery exists and needs only a fitness function and an
   incentive checker, both of which this repository provides.
7. The closest HPC neighbours are RLBackfilling, which improves on EASY with learned
   backfilling but has no incentives, and hand-designed VCG knapsack auctions for HPC
   network allocation. A learned, truthful, capacitated, composite-aware placement
   mechanism does not exist yet.

## Candidate directions

1. Exact-truthful learned affine maximizers over the MILP. Per-job weights and
   per-candidate boosts come from a network over public context only (sizes, limits,
   waiting time, free nodes, prices), so truthfulness holds by construction and the
   allocation stays the exact solver with composites. Trained by decision-focused
   learning toward welfare, revenue or a systems objective such as bounded slowdown.
   Aging becomes a boost on waiting time, which yields truthful anti-starvation.
2. Beyond myopic VCG. Per-window VCG is blind to the next window. With actual runtimes
   below limits, capacity frees between windows and deferring a job can raise welfare.
   Treat the window sequence as an MDP whose transitions come from DR_EVT, learn a
   continuation value, and evaluate against a hindsight optimum over the whole horizon.
3. Certified evaluation. Coverage against the per-window optimum and against the
   horizon optimum, corrected regret for approximate arms, realized service metrics
   from the scheduler, zero partial starts by construction, and the environment itself
   as a reproducible testbed.
4. Robustness to learning bidders. No-regret and LLM bidder agents inside the simulator
   with the credit model as a budget, over repeated windows.
5. In-context mechanism design. One network conditioned on the window, evaluated across
   worlds and platform mixes, with exact incentive compatibility through the affine
   maximizer parameterization.
6. LLM-evolved priority rules as an interpretable alternative to learned boosts.
7. Learning-augmented backfill with predicted runtimes, a systems paper rather than a
   learning one.

RegretFormer, the approximately truthful learned mechanism, is implemented in this
repository as the comparison arm with regret estimated by a fixed grid search followed
by gradient ascent, scored through the deployed pipeline.

## References

- You, Zhuang, Wang, Wang. Bridging the gap between estimated and true regret. 2026.
  <https://arxiv.org/abs/2601.13489>
- Duan et al. A scalable neural network for DSIC affine maximizer auction design.
  <https://arxiv.org/abs/2305.12162>
- Wang et al. GemNet: menu-based strategy-proof multi-bidder auctions. 2024.
  <https://arxiv.org/abs/2406.07428>
- Ma et al. Learning truthful mechanisms without discretization. 2025.
  <https://arxiv.org/abs/2506.22911>
- BundleFlow: deep menus for combinatorial auctions by diffusion-based optimization. 2025.
  <https://arxiv.org/abs/2502.15283>
- Duality for optimal multi-item multi-bidder auction design: revenue certificates. 2026.
  <https://arxiv.org/abs/2606.10112>
- Online learning for dynamic VCG in unknown environments. 2025.
  <https://arxiv.org/abs/2506.19038>
- Efficiency, feasibility and incentive-awareness in constrained online allocation. 2025.
  <https://arxiv.org/abs/2507.09473>
- From no-regret to strategically robust learning in repeated auctions. 2026.
  <https://arxiv.org/abs/2601.03853>
- Strategic bidding in 6G spectrum auctions with large language models. 2026.
  <https://arxiv.org/abs/2604.24156>
- Gkatzelis, Schoepflin, Tan. Learning-augmented clock auctions. SODA 2025.
- Scaling decision-focused learning with Lagrangian decomposition. 2026.
  <https://arxiv.org/abs/2606.08797>
- An interpretable automated mechanism design framework with LLMs. 2025.
  <https://arxiv.org/abs/2502.12203>
- XOR bidding and knapsack formulations for HPC network resource allocation. 2026.
  <https://arxiv.org/abs/2606.00490>
- A reinforcement learning based backfilling strategy for HPC batch jobs. 2024.
  <https://arxiv.org/abs/2404.09264>
- Duan et al. A context-integrated transformer-based neural network for auction design.
  ICML 2022. <https://arxiv.org/abs/2201.12489>
