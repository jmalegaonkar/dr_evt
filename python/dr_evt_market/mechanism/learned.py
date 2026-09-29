################################################################################
#         Copyright 2023 Lawrence Livermore National Security, LLC             #
#         See the top-level LICENSE file for details.                          #
#                                                                              #
#         SPDX-License-Identifier: MIT                                         #
################################################################################

"""RegretFormer's network over a window of jobs and platforms, and its outcomes."""

import math
from dataclasses import dataclass, fields

import numpy as np
import torch
from torch import nn

_TOLERANCE = 1.0e-9
_CHANNELS = 5


@dataclass(frozen=True)
class Window:
    """Padded tensors for market windows over one set of platforms."""

    nodes: torch.Tensor  # [B, N] nodes per job, zero for padding
    hours: torch.Tensor  # [B, N] limit per job in hours
    prices: torch.Tensor  # [B, N, M] reported price per node-hour, zero if none
    posted: torch.Tensor  # [B, M] posted price per node-hour
    speeds: torch.Tensor  # [B, N, M] platform speed for each job
    fits: torch.Tensor  # [B, N, M] hardware and exposed nodes fit
    free: torch.Tensor  # [B, M] free nodes
    exposed: torch.Tensor  # [B, M] exposed nodes
    jobs: torch.Tensor  # [B, N] true for jobs, false for padding

    def repeat(self, count: int) -> "Window":
        """Return a single window repeated `count` times along the batch."""
        tensors = (getattr(self, field.name) for field in fields(self))
        return Window(*(tensor.expand(count, *tensor.shape[1:]) for tensor in tensors))

    def take(self, index) -> "Window":
        """Return the windows at the given batch positions."""
        index = torch.as_tensor(index)
        return Window(*(getattr(self, field.name)[index] for field in fields(self)))

    def scale(self) -> torch.Tensor:
        """Return each window's mean posted cost over the cells that fit, [B]."""
        public = self.fits & self.jobs.unsqueeze(-1)
        total = (self.cost() * public).sum((1, 2))
        return (total / public.sum((1, 2)).clamp(min=1)).clamp(min=_TOLERANCE)

    def cost(self) -> torch.Tensor:
        """Return the posted cost of every job on every platform."""
        work = (self.nodes * self.hours).unsqueeze(-1) / self.speeds
        return self.posted.unsqueeze(1) * work

    def value(self, prices: torch.Tensor) -> torch.Tensor:
        """Return the value of every job on every platform at the given prices."""
        work = (self.nodes * self.hours).unsqueeze(-1) / self.speeds
        return prices * work

    def candidates(self, prices: torch.Tensor) -> torch.Tensor:
        """Return the cells that fit now and whose price covers the posted price."""
        return (
            self.fits
            & (self.nodes.unsqueeze(-1) <= self.free.unsqueeze(1))
            & (self.value(prices) + _TOLERANCE >= self.cost())
        )

    def features(self, prices: torch.Tensor):
        """Return the network's input channels and the candidate cells."""
        cost = self.cost()
        scale = self.scale().view(-1, 1, 1)
        candidate = self.candidates(prices)
        channels = torch.stack(
            [
                self.value(prices) / scale,
                cost / scale,
                (self.nodes.unsqueeze(-1) / self.exposed.unsqueeze(1)).expand_as(cost),
                (self.free / self.exposed).unsqueeze(1).expand_as(cost),
                candidate.double(),
            ],
            dim=-1,
        )
        return channels.float(), candidate


def window(batches, platforms) -> Window:
    """Stack market windows, each a batch of jobs and its free nodes, as tensors."""
    names = list(platforms)
    shape = (len(batches), max(len(jobs) for jobs, _ in batches))
    nodes = torch.zeros(shape, dtype=torch.float64)
    hours = torch.zeros(shape, dtype=torch.float64)
    prices = torch.zeros(shape + (len(names),), dtype=torch.float64)
    speeds = torch.ones(shape + (len(names),), dtype=torch.float64)
    fits = torch.zeros(shape + (len(names),), dtype=torch.bool)
    present = torch.zeros(shape, dtype=torch.bool)
    for row, (jobs, _) in enumerate(batches):
        present[row, : len(jobs)] = True
        for column, job in enumerate(jobs):
            nodes[row, column] = job.num_nodes
            hours[row, column] = (job.requested_s or job.limit_s) / 3600
            hardware = "gpu" if "gpu" in job.requires else "cpu"
            for index, name in enumerate(names):
                platform = platforms[name]
                speed = platform.speed.get(hardware, 1.0)
                price = job.price(name) or 0.0
                speeds[row, column, index] = speed
                prices[row, column, index] = (
                    price if isinstance(job.bid, dict) else price * speed
                )
                fits[row, column, index] = platform.fits(job)
    posted = [platforms[name].price_per_node_hour for name in names]
    exposed = [platforms[name].exposed_nodes for name in names]
    return Window(
        nodes,
        hours,
        prices,
        torch.tensor([posted] * len(batches), dtype=torch.float64),
        speeds,
        fits,
        torch.tensor(
            [[free[name] for name in names] for _, free in batches],
            dtype=torch.float64,
        ),
        torch.tensor([exposed] * len(batches), dtype=torch.float64),
        present,
    )


def _mean_over_jobs(x, jobs):
    weight = jobs[:, :, None, None].to(x.dtype)
    return (x * weight).sum(1, keepdim=True) / weight.sum(1, keepdim=True).clamp(min=1)


class _Exchangeable(nn.Module):
    def __init__(self, inputs, outputs):
        super().__init__()
        self.cell = nn.Linear(inputs, outputs)
        self.over_jobs = nn.Linear(inputs, outputs, bias=False)
        self.over_platforms = nn.Linear(inputs, outputs, bias=False)
        self.over_all = nn.Linear(inputs, outputs, bias=False)

    def forward(self, x, jobs):
        over_jobs = _mean_over_jobs(x, jobs)
        return (
            self.cell(x)
            + self.over_jobs(over_jobs)
            + self.over_platforms(x.mean(2, keepdim=True))
            + self.over_all(over_jobs.mean(2, keepdim=True))
        )


class _Block(nn.Module):
    def __init__(self, hid, heads):
        super().__init__()
        self.norm_platforms = nn.LayerNorm(hid, eps=1.0e-6)
        self.norm_jobs = nn.LayerNorm(hid, eps=1.0e-6)
        self.across_platforms = nn.MultiheadAttention(
            hid, heads, bias=False, batch_first=True
        )
        self.across_jobs = nn.MultiheadAttention(
            hid, heads, bias=False, batch_first=True
        )
        self.fuse = nn.Linear(2 * hid, hid)

    def forward(self, x, jobs):
        batch, size, platforms, hid = x.shape
        rows = self.norm_platforms(x).reshape(batch * size, platforms, hid)
        rows = self.across_platforms(rows, rows, rows, need_weights=False)[0]
        rows = rows.reshape(x.shape) + x
        columns = (
            self.norm_jobs(x).transpose(1, 2).reshape(batch * platforms, size, hid)
        )
        padding = (~jobs).repeat_interleave(platforms, dim=0)
        columns = self.across_jobs(
            columns, columns, columns, key_padding_mask=padding, need_weights=False
        )[0]
        columns = columns.reshape(batch, platforms, size, hid).transpose(1, 2) + x
        return self.fuse(torch.tanh(torch.cat([rows, columns], dim=-1))) + x


class Network(nn.Module):
    """RegretFormer's network: where each job goes, and its payment fraction."""

    def __init__(self, hid=32, heads=2, layers=1) -> None:
        """Build the layers; the defaults are the RegretFormer paper's."""
        super().__init__()
        self.shape = {"hid": hid, "heads": heads, "layers": layers}
        self.embed = _Exchangeable(_CHANNELS, hid)
        self.blocks = nn.ModuleList(_Block(hid, heads) for _ in range(layers))
        self.norm = nn.LayerNorm(hid, eps=1.0e-6)
        self.encode_jobs = nn.Linear(hid, hid, bias=False)
        self.encode_platforms = nn.Linear(hid, hid, bias=False)
        self.payment = nn.Linear(hid, 1)

    def forward(self, channels, candidate, jobs):
        """Return option probabilities, waiting last, and payment fractions."""
        x = torch.tanh(self.embed(channels, jobs))
        for block in self.blocks:
            x = torch.tanh(block(x, jobs))
        x = self.norm(x)
        per_job = self.encode_jobs(x).mean(2)
        per_platform = _mean_over_jobs(self.encode_platforms(x), jobs).squeeze(1)
        logits = per_job @ per_platform.transpose(1, 2) / math.sqrt(x.shape[-1])
        logits = logits.masked_fill(~candidate, float("-inf"))
        waiting = -logits.masked_fill(~candidate, 0.0).sum(-1, keepdim=True)
        probabilities = torch.softmax(torch.cat([logits, waiting], dim=-1), dim=-1)
        return probabilities, torch.sigmoid(self.payment(per_job)).squeeze(-1)


def network(seed=0, **shape) -> Network:
    """Return an untrained network with weights drawn from the seed."""
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        return Network(**shape).eval()


def save(net: Network, path, **notes) -> None:
    """Save a network's shape, weights and free-form notes."""
    torch.save({"shape": net.shape, "state": net.state_dict(), "notes": notes}, path)


def load(path) -> Network:
    """Load a network saved by save."""
    checkpoint = torch.load(path, weights_only=True)
    net = Network(**checkpoint["shape"])
    net.load_state_dict(checkpoint["state"])
    return net.eval()


def _round(probabilities, candidate, nodes, free):
    size, platforms = candidate.shape
    left = free.copy()
    assignment = np.full(size, -1)
    order = np.argsort(-probabilities[:, :platforms], axis=None, kind="stable")
    for flat in order:
        job, platform = divmod(int(flat), platforms)
        if (
            assignment[job] < 0
            and candidate[job, platform]
            and nodes[job] <= left[platform]
        ):
            assignment[job] = platform
            left[platform] -= nodes[job]
    return assignment


@torch.no_grad()
def deploy(net: Network, window: Window, prices: torch.Tensor):
    """Return each job's platform (-1 to stay queued) and payment fraction.

    Cells are taken greedily from the most probable, and a job goes to the first of
    its platforms that still has the nodes. Its waiting probability only lowers its
    place in that order: a job stays in the queue only when none of its platforms
    has room left.
    """
    channels, candidate = window.features(prices)
    probabilities, fractions = net(channels, candidate, window.jobs)
    probabilities = probabilities.numpy()
    candidate = candidate.numpy()
    nodes = window.nodes.numpy()
    free = window.free.numpy()
    assignment = np.stack(
        [
            _round(probabilities[row], candidate[row], nodes[row], free[row])
            for row in range(len(probabilities))
        ]
    )
    return assignment, fractions.double().numpy()
