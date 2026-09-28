################################################################################
#         Copyright 2023 Lawrence Livermore National Security, LLC             #
#         See the top-level LICENSE file for details.                          #
#                                                                              #
#         SPDX-License-Identifier: MIT                                         #
################################################################################

"""Regret by item-wise grid search and guided refinement, after You et al. (2026).

A job's regret is the most it gains by misreporting its prices while every other job
reports truthfully, measured on the decisions the market applies. A report is one
price per platform; a single bid is the same price on each. Every estimate here is a
lower bound on the regret.
"""

from dataclasses import dataclass, replace

_CHUNK = 512


@dataclass(frozen=True)
class Estimates:
    """One job's regret: from the grid, after refinement, and by gradient alone."""

    grid: float
    refined: float
    gradient: float


def _prices(job, names):
    return [job.price(name) or 0.0 for name in names]


def _misreports(prices, posted, points, span):
    # Pairs of (item, prices): each price moved alone, then the whole report scaled
    # (item -1). The levels include the posted prices, where the candidates change.
    top = max(prices)
    levels = [top * span * step / (points - 1) for step in range(points)]
    variants = []
    for item, cutoff in enumerate(posted):
        for level in levels + [cutoff]:
            variants.append((item, prices[:item] + [level] + prices[item + 1 :]))
    factors = [span * step / (points - 1) for step in range(points)]
    factors += [cutoff / price for cutoff, price in zip(posted, prices) if price > 0]
    variants += [(-1, [factor * price for price in prices]) for factor in factors]
    return variants


def _utility(job, decisions):
    for decision in decisions:
        if decision.job_id == job.job_id:
            price = job.price(decision.platform) or 0.0
            return price * job.num_nodes * job.limit_s / 3600 - decision.charge
    return 0.0


def grid_regret(mechanism, jobs, platforms, free_nodes, *, points=101, span=4.0):
    """Return each job's item-wise grid regret in one window, for any mechanism.

    One price at a time moves over `points` levels from zero to `span` times the
    job's highest price, and to that platform's posted price; then the whole report
    is scaled over the same range and to each posted price. The best gain over the
    truthful utility is a lower bound on the job's regret.
    """
    jobs = list(jobs)
    names = list(platforms)
    posted = [platforms[name].price_per_node_hour for name in names]
    truthful = mechanism.decide(jobs, platforms, free_nodes)
    regret = {}
    for index, job in enumerate(jobs):
        truth = _utility(job, truthful)
        best = truth
        for _, prices in _misreports(_prices(job, names), posted, points, span):
            report = replace(job, bid=dict(zip(names, prices)))
            reports = jobs[:index] + [report] + jobs[index + 1 :]
            decisions = mechanism.decide(reports, platforms, free_nodes)
            best = max(best, _utility(job, decisions))
        regret[job.job_id] = best - truth
    return regret


def refined_regret(
    mechanism,
    jobs,
    platforms,
    free_nodes,
    *,
    points=101,
    span=4.0,
    starts=80,
    sigma=0.6,
    steps=100,
    rate=0.02,
    gradient_steps=1000,
    seed=0,
) -> dict[str, Estimates]:
    """Return RegretFormer's regret per job: grid, refined, and gradient alone.

    `grid` is `grid_regret`, computed in batches through the network. `refined`
    follows You et al.: gradient ascent on the network's relaxed utility from the
    combined grid argmax, each single-item argmax, `starts` perturbations of that
    point and of the truth (`sigma` times the job's highest price) and `starts`
    uniform draws, with every end point scored on the deployed mechanism. `gradient`
    is RegretFormer's own protocol: ascent from one uniform draw, scored on the
    relaxed network.
    """
    import numpy as np
    import torch

    from . import learned

    jobs = list(jobs)
    net = mechanism.net
    base = learned.window([(jobs, free_nodes)], platforms)
    truth = base.prices[0]
    work = (base.nodes * base.hours)[0]
    cost = base.posted[0] * work[:, None]
    size, width = truth.shape
    top = truth.max(dim=1).values
    generator = torch.Generator().manual_seed(seed)

    def reports(rows, prices):
        grid = truth.expand(len(rows), size, width).clone()
        grid[torch.arange(len(rows)), torch.as_tensor(rows)] = prices
        return grid

    def deployed(rows, prices):
        utilities = []
        for first in range(0, len(rows), _CHUNK):
            part, offered = rows[first : first + _CHUNK], prices[first : first + _CHUNK]
            window = base.repeat(len(part))
            assignment, fractions = learned.deploy(net, window, reports(part, offered))
            index = np.arange(len(part))
            placed = torch.as_tensor(assignment[index, part])
            column = placed.clamp(min=0)
            job = torch.as_tensor(part)
            fraction = torch.as_tensor(fractions[index, part])
            charged = cost[job, column] + fraction * (
                offered[torch.arange(len(part)), column] * work[job] - cost[job, column]
            )
            gain = truth[job, column] * work[job] - charged
            utilities.append(torch.where(placed >= 0, gain, torch.zeros_like(gain)))
        return torch.cat(utilities)

    def relaxed(rows, prices):
        window = base.repeat(len(rows))
        channels, candidate = window.features(reports(rows, prices))
        probabilities, fractions = net(channels, candidate, window.jobs)
        index, job = torch.arange(len(rows)), torch.as_tensor(rows)
        charged = cost[job] + fractions[index, job, None] * (
            prices * work[job, None] - cost[job]
        )
        gain = truth[job] * work[job, None] - charged
        return (probabilities[index, job, :width] * gain).sum(-1)

    def ascend(rows, start, count):
        ends = []
        for first in range(0, len(rows), _CHUNK):
            part = rows[first : first + _CHUNK]
            scale = top[torch.as_tensor(part), None]
            level = (start[first : first + _CHUNK] / scale).requires_grad_(True)
            optimizer = torch.optim.Adam([level], lr=rate)
            for _ in range(count):
                loss = -relaxed(part, level * scale).sum()
                (level.grad,) = torch.autograd.grad(loss, level)
                optimizer.step()
                with torch.no_grad():
                    level.clamp_(0.0, span)
            ends.append((level * scale).detach())
        return torch.cat(ends)

    everyone = np.arange(size)
    truthful = deployed(everyone, truth)
    posted = base.posted[0].tolist()
    rows, items, variants = [], [], []
    for job in range(size):
        for item, prices in _misreports(truth[job].tolist(), posted, points, span):
            rows.append(job)
            items.append(item)
            variants.append(prices)
    rows, items = np.asarray(rows), np.asarray(items)
    variants = torch.tensor(variants, dtype=torch.float64)
    utilities = deployed(rows, variants)

    grid, combined = [], truth.clone()
    for job in range(size):
        mine = torch.as_tensor(rows == job)
        grid.append(float(max(utilities[mine].max(), truthful[job])))
        for item in range(width):
            chosen = torch.as_tensor(np.flatnonzero((rows == job) & (items == item)))
            best = chosen[int(utilities[chosen].argmax())]
            combined[job, item] = variants[best, item]

    portfolio, owners = [], []
    diagonal = torch.arange(width)
    for job in range(size):
        singles = truth[job].repeat(width, 1)
        singles[diagonal, diagonal] = combined[job]
        shape = (2, starts, width)
        noise = torch.randn(shape, generator=generator, dtype=torch.float64)
        noise = noise * sigma * top[job]
        uniform = torch.rand(shape[1:], generator=generator, dtype=torch.float64)
        block = torch.cat(
            [
                combined[job, None],
                singles,
                combined[job] + noise[0],
                truth[job] + noise[1],
                uniform * span * top[job],
            ]
        ).clamp(0.0, float(span * top[job]))
        portfolio.append(block)
        owners += [job] * len(block)
    owners = np.asarray(owners)
    scored = deployed(owners, ascend(owners, torch.cat(portfolio), steps))

    draws = torch.rand(size, width, generator=generator, dtype=torch.float64)
    found = ascend(everyone, draws * span * top[:, None], gradient_steps)
    with torch.no_grad():
        gradient = relaxed(everyone, found) - relaxed(everyone, truth)

    estimates = {}
    for job, item in enumerate(jobs):
        refined = max(grid[job], float(scored[torch.as_tensor(owners == job)].max()))
        estimates[item.job_id] = Estimates(
            grid[job] - float(truthful[job]),
            refined - float(truthful[job]),
            max(float(gradient[job]), 0.0),
        )
    return estimates
