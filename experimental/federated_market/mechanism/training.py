################################################################################
#         Copyright 2023 Lawrence Livermore National Security, LLC             #
#         See the top-level LICENSE file for details.                          #
#                                                                              #
#         SPDX-License-Identifier: MIT                                         #
################################################################################

"""Training RegretFormer under a regret budget, with grid misreports."""

import math

from .regret import _misreports

_CHUNK = 2048


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
    under_price=False,
    device="cpu",
    **shape,
):
    """Train RegretFormer's network on market windows under a regret budget.

    The loss is minus the objective (the premiums over cost, or welfare), plus a
    multiplier times the regret, plus a `capacity`-weighted penalty on overbooked and
    idle nodes. The multiplier follows RegretFormer's rule against a budget that
    shrinks from `budget[0]` to `budget[1]` of the jobs' available surplus. Regret
    comes from grid misreports for up to `regret_jobs` jobs per window, and counts
    reports under a posted price only with `under_price`. Returns the mechanism, back
    on the CPU, and each step's history, whose regret is the relaxed network's: judge
    the deployed mechanism with `refined_regret`.
    """
    import numpy as np
    import torch

    from . import learned
    from .regretformer import RegretFormer

    if objective not in ("revenue", "welfare"):
        raise ValueError("objective must be 'revenue' or 'welfare'")

    def outcome(window, prices, truth):
        channels, offer = window.features(prices)
        probabilities, fractions = net(channels, offer, window.jobs)
        allocation = probabilities[..., : prices.shape[-1]]
        cost = window.cost()
        charge = learned.charge(cost, window.value(prices), fractions[..., None])
        gain = window.value(truth) - charge
        waiting = probabilities[..., -1:] * offer
        return allocation, waiting, charge - cost, (allocation * gain).sum(-1)

    def misreported(window, truth, rows, owners, prices):
        reports = truth[rows].clone()
        index = torch.arange(len(rows), device=reports.device)
        reports[index, owners] = prices
        utility = outcome(window.take(rows), reports, truth[rows])[-1]
        return utility[index, owners]

    def regret(window, truth, utility, scale):
        # Misreports come from a grid, not gradients: a report of zero withdraws an
        # offer, and the charge rule changes at the posted price.
        picked, rows, owners, variants = [], [], [], []
        for row in range(len(truth)):
            present = np.flatnonzero(window.jobs[row].cpu().numpy())
            count = min(regret_jobs, len(present))
            posted = window.posted[row].tolist()
            for job in rng.choice(present, size=count, replace=False):
                picked.append((row, int(job), len(present) / count))
                prices = truth[row, job].tolist()
                for _, variant in _misreports(
                    prices, posted, points, span, under_price
                ):
                    rows.append(row)
                    owners.append(int(job))
                    variants.append(variant)
        rows = torch.tensor(rows, device=device)
        owners = torch.tensor(owners, device=device)
        variants = torch.tensor(variants, dtype=torch.float64, device=device)
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
        at = torch.tensor([row for row, _, _ in picked], device=device)
        who = torch.tensor([job for _, job, _ in picked], device=device)
        spread = torch.tensor(
            [share for _, _, share in picked], dtype=torch.float64, device=device
        )
        gain = misreported(window, truth, at, who, torch.stack(best)) - utility[at, who]
        return (torch.relu(gain) * spread / scale[at]).sum() / len(truth)

    rng = np.random.default_rng(seed)
    mechanism = RegretFormer(seed=seed, **shape)
    net = mechanism.net.to(device).train()
    optimizer = torch.optim.Adam(net.parameters(), lr=rate)
    multiplier, (target, end) = 1.0, budget
    shrink = (end / target) ** (1.5 / steps)
    keys = ("objective", "regret", "multiplier", "overbooked", "idle")
    history = {key: [] for key in keys}
    for _ in range(steps):
        chosen = rng.choice(len(windows), size=min(batch, len(windows)), replace=False)
        window = learned.window([windows[index] for index in chosen], platforms)
        window = window.to(device)
        truth, scale = window.prices, window.scale()
        allocation, waiting, premium, utility = outcome(window, truth, truth)
        if objective == "revenue":
            earned = allocation * premium
        else:
            earned = allocation * (window.value(truth) - window.cost())
        gained = (earned.sum((1, 2)) / scale).mean()
        nodes = window.nodes[..., None]
        demand = (allocation * nodes).sum(1)
        exposed = window.exposed.clamp(min=1.0)
        overbooked = (torch.relu(demand - window.free) / exposed).sum(-1).mean()
        # The market places every job that fits, so waiting may hold back a job only
        # when its platforms are full: charge the free nodes such waiting leaves idle.
        held = torch.minimum(torch.relu(window.free - demand), (waiting * nodes).sum(1))
        idle = (held / exposed).sum(-1).mean()
        lost = regret(window, truth, utility, scale)
        loss = -gained + multiplier * lost + capacity * (overbooked + idle)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()
        # The budget is a share of the jobs' surplus rather than of revenue: premiums
        # rightly fall to zero where nodes are free, and a budget on them would drive
        # the multiplier without bound.
        surplus = (window.value(truth) - window.cost()).clamp(min=0)
        surplus = surplus * window.offers(truth)
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
    net.to("cpu").eval()
    return mechanism, history
