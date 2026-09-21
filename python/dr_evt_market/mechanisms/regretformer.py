################################################################################
#         Copyright 2023 Lawrence Livermore National Security, LLC             #
#         See the top-level LICENSE file for details.                          #
#                                                                              #
#         SPDX-License-Identifier: MIT                                         #
################################################################################

"""RegretFormer as a mechanism: a network over the (job, candidate) grid.

Checkpoints are dictionaries written by ``torch.save`` with ``state_dict``,
``in_channels``, ``hid``, ``hid_att``, ``n_layers``, ``n_heads``,
``trained_on`` (free text) and ``created`` (an ISO date).
"""

from __future__ import annotations

from pathlib import Path

from .base import Decision, Mechanism, Window

_CHECKPOINT_FIELDS = {
    "state_dict",
    "in_channels",
    "hid",
    "hid_att",
    "n_layers",
    "n_heads",
    "trained_on",
    "created",
}


class RegretFormer(Mechanism):
    """Deploy a seeded or checkpointed RegretFormer network."""

    name = "regretformer"

    def __init__(
        self,
        checkpoint: str | Path | None,
        *,
        hid: int = 64,
        hid_att: int = 32,
        n_layers: int = 2,
        n_heads: int = 4,
        seed: int = 0,
        device: str = "cpu",
    ) -> None:
        """Build a seeded random network or load a checkpoint."""
        import torch

        from ..learned import deploy, net, windows

        self._torch, self._deploy, self._windows = torch, deploy, windows
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError("seed must be an integer")
        self.seed = seed
        self.device = torch.device(device)
        torch.manual_seed(seed)
        state_dict = None
        self.trained_on = "random initialization"
        self.created = None
        in_channels = windows.IN_CHANNELS
        if checkpoint is not None:
            payload = torch.load(
                Path(checkpoint), map_location=self.device, weights_only=True
            )
            missing = sorted(
                _CHECKPOINT_FIELDS - set(payload if isinstance(payload, dict) else ())
            )
            if missing:
                raise ValueError(f"checkpoint is missing fields: {', '.join(missing)}")
            in_channels, hid, hid_att = (
                int(payload["in_channels"]),
                int(payload["hid"]),
                int(payload["hid_att"]),
            )
            n_layers, n_heads = int(payload["n_layers"]), int(payload["n_heads"])
            self.trained_on, self.created = str(payload["trained_on"]), str(
                payload["created"]
            )
            state_dict = payload["state_dict"]
        self.in_channels, self.hid, self.hid_att = in_channels, hid, hid_att
        self.n_layers, self.n_heads = n_layers, n_heads
        self.net = net.RegretFormerNet(
            in_channels=in_channels,
            hid=hid,
            hid_att=hid_att,
            n_layers=n_layers,
            n_heads=n_heads,
        ).to(self.device)
        if state_dict is not None:
            self.net.load_state_dict(state_dict)
        self.net.eval()

    def decide(self, window: Window) -> list[Decision]:
        """Round one network pass into feasible, individually rational decisions."""
        if not window.jobs:
            return []
        structure = self._windows.structure_from_window(window)
        batch = self._windows.WindowBatch((structure,), self.device)
        assignments, payments = self._deploy.deployed_outcome(
            self.net, batch, batch.true_values
        )
        decisions = []
        for job_index, candidate_index in enumerate(assignments[0]):
            if candidate_index < 0:
                continue
            job = window.jobs[job_index]
            placement = window.candidates[job.job_id][candidate_index]
            if placement.value_credits < placement.cost_credits:
                continue
            charge = min(
                placement.value_credits,
                max(placement.cost_credits, float(payments[0, job_index])),
            )
            decisions.append(Decision(job.job_id, placement, charge))
        return decisions


__all__ = ["RegretFormer"]
