################################################################################
#         Copyright 2023 Lawrence Livermore National Security, LLC             #
#         See the top-level LICENSE file for details.                          #
#                                                                              #
#         SPDX-License-Identifier: MIT                                         #
################################################################################

"""Training RegretFormer under a regret budget, with grid misreports."""

import math

from .base import Mechanism
from .regret import _misreports

_CHUNK = 2048


class _Recorder(Mechanism):
    """Record each window's batch and free nodes while another mechanism decides."""

    name = "recorder"

    def __init__(self, inner) -> None:
        self.inner = inner
        self.windows = []

    def decide(self, jobs, platforms, free_nodes):
        """Record the window, then return the inner mechanism's decisions."""
        jobs = list(jobs)
        if jobs:
            self.windows.append((jobs, dict(free_nodes)))
        return self.inner.decide(jobs, platforms, free_nodes)


def record_windows(jobs, platforms, *, mechanism=None, window_s=60, prefix=32):
    """Run the market and return every window's batch and free nodes.

    The queue, and so every window, follows the mechanism that decides: VCG unless
    another is given.
    """
    from ..market import run
    from .vcg import Vcg

    recorder = _Recorder(Vcg() if mechanism is None else mechanism)
    run(jobs, platforms, recorder, window_s=window_s, prefix=prefix)
    return recorder.windows


def train_regretformer(
    windows,
    platforms,
    *,
    objective="revenue",
    steps=2000,
    batch=8,
    rate=1.0e-3,
    budget=(0.01, 0.001),
    budget_rate=0.5,
    regret_jobs=4,
    points=9,
    span=4.0,
    capacity=10.0,
    seed=0,
    **shape,
):
    """Train RegretFormer's network on market windows under a regret budget.

    The loss is RegretFormer's: minus the objective (the center's premiums, or
    welfare), plus a multiplier times the regret, with the multiplier updated by
    RegretFormer's rule against a budget that shrinks from `budget[0]` to
    `budget[1]`. RegretFormer sets that budget as a fraction of revenue; here it is a
    fraction of the jobs' available surplus (each job's best value over cost), since
    premiums can rightly fall to zero where nodes are free, and a budget on them then
    drives the multiplier without bound. A penalty weighted by `capacity` charges the
    nodes the relaxed allocation overbooks, and the free nodes it leaves idle while
    a job that fits them waits, since the market places every job that fits.
    Misreports come from the item-wise grid on the relaxed outcome, for up to
    `regret_jobs` jobs per window, instead of gradient ascent: a report that crosses
    a posted price changes the job's candidates, which gradients cannot see. Returns
    the trained mechanism and each step's history, whose regret is the relaxed
    network's: measure the deployed mechanism with `refined_regret`.
    """
    import numpy as np
    import torch

    from . import learned
    from .regretformer import RegretFormer

    if objective not in ("revenue", "welfare"):
        raise ValueError("objective must be 'revenue' or 'welfare'")

    def outcome(window, prices, truth):
        channels, candidate = window.features(prices)
        probabilities, fractions = net(channels, candidate, window.jobs)
        allocation = probabilities[..., : prices.shape[-1]]
        premium = fractions[..., None] * (window.value(prices) - window.cost())
        gain = window.value(truth) - window.cost() - premium
        waiting = probabilities[..., -1:] * candidate
        return allocation, waiting, premium, (allocation * gain).sum(-1)

    def misreported(window, truth, rows, owners, prices):
        reports = truth[rows].clone()
        reports[torch.arange(len(rows)), owners] = prices
        utility = outcome(window.take(rows), reports, truth[rows])[-1]
        return utility[torch.arange(len(rows)), owners]

    def regret(window, truth, utility, scale):
        picked, rows, owners, variants = [], [], [], []
        for row in range(len(truth)):
            present = np.flatnonzero(window.jobs[row].numpy())
            count = min(regret_jobs, len(present))
            posted = window.posted[row].tolist()
            for job in rng.choice(present, size=count, replace=False):
                picked.append((row, int(job), len(present) / count))
                prices = truth[row, job].tolist()
                for _, variant in _misreports(prices, posted, points, span):
                    rows.append(row)
                    owners.append(int(job))
                    variants.append(variant)
        rows, owners = torch.tensor(rows), torch.tensor(owners)
        variants = torch.tensor(variants, dtype=torch.float64)
        with torch.no_grad():
            scores = torch.cat(
                [
                    misreported(
                        window,
                        truth,
                        rows[first : first + _CHUNK],
                        owners[first : first + _CHUNK],
                        variants[first : first + _CHUNK],
                    )
                    for first in range(0, len(rows), _CHUNK)
                ]
            )
        best = []
        for row, job, _ in picked:
            mine = torch.nonzero((rows == row) & (owners == job)).squeeze(1)
            best.append(variants[mine[scores[mine].argmax()]])
        at = torch.tensor([row for row, _, _ in picked])
        who = torch.tensor([job for _, job, _ in picked])
        spread = torch.tensor([share for _, _, share in picked], dtype=torch.float64)
        gain = misreported(window, truth, at, who, torch.stack(best)) - utility[at, who]
        return (torch.relu(gain) * spread / scale[at]).sum() / len(truth)

    rng = np.random.default_rng(seed)
    mechanism = RegretFormer(seed=seed, **shape)
    net = mechanism.net.train()
    optimizer = torch.optim.Adam(net.parameters(), lr=rate)
    multiplier, (target, end) = 1.0, budget
    shrink = (end / target) ** (1.5 / steps)
    keys = ("objective", "regret", "multiplier", "overbooked", "idle")
    history = {key: [] for key in keys}
    for _ in range(steps):
        chosen = rng.choice(len(windows), size=min(batch, len(windows)), replace=False)
        window = learned.window([windows[index] for index in chosen], platforms)
        truth, scale = window.prices, window.scale()
        allocation, waiting, premium, utility = outcome(window, truth, truth)
        if objective == "revenue":
            earned = allocation * premium
        else:
            earned = allocation * (window.value(truth) - window.cost())
        gained = (earned.sum((1, 2)) / scale).mean()
        nodes = window.nodes[..., None]
        demand = (allocation * nodes).sum(1)
        overbooked = (torch.relu(demand - window.free) / window.exposed).sum(-1).mean()
        # The market places every job that fits, so waiting may hold back a job only
        # when its platforms are full: charge the free nodes such waiting leaves idle.
        held = torch.minimum(torch.relu(window.free - demand), (waiting * nodes).sum(1))
        idle = (held / window.exposed).sum(-1).mean()
        lost = regret(window, truth, utility, scale)
        loss = -gained + multiplier * lost + capacity * (overbooked + idle)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()
        surplus = (window.value(truth) - window.cost()) * window.candidates(truth)
        available = (surplus.amax(-1).sum(-1) / scale).mean().item()
        ratio = lost.item() / (available + 1.0e-8)
        change = math.log(ratio) - math.log(target) if ratio > 0 else -math.inf
        multiplier = max(0.0, multiplier + budget_rate * change)
        target = max(target * shrink, end)
        values = (
            gained.item(),
            lost.item(),
            multiplier,
            overbooked.item(),
            idle.item(),
        )
        for key, value in zip(keys, values):
            history[key].append(value)
    net.eval()
    return mechanism, history
