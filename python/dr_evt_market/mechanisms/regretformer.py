################################################################################
#         Copyright 2023 Lawrence Livermore National Security, LLC             #
#         See the top-level LICENSE file for details.                          #
#                                                                              #
#         SPDX-License-Identifier: MIT                                         #
################################################################################

"""RegretFormer deployment as a market mechanism.

Checkpoints are dictionaries written by ``torch.save`` with ``state_dict``,
``in_channels``, ``hid``, ``hid_att``, ``n_layers``, ``n_heads``, ``trained_on``,
and ``created`` entries. ``trained_on`` is free text and ``created`` is an ISO
date.
"""

from __future__ import annotations

from pathlib import Path

from .base import Decision, MarketObservation, Mechanism


class RegretFormer(Mechanism):
    """Deploy a seeded or checkpointed RegretFormer network."""

    import torch as _torch

    from ..learned import deploy as _deploy
    from ..learned import net as _net
    from ..learned import windows as _windows

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
        """Build a seeded network or load its architecture and parameters."""
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError("seed must be an integer")
        self.seed = seed
        self.device = self._torch.device(device)
        self._torch.manual_seed(seed)

        state_dict = None
        self.trained_on = "random initialization"
        self.created = None
        if checkpoint is not None:
            checkpoint_path = Path(checkpoint)
            payload = self._torch.load(
                checkpoint_path,
                map_location=self.device,
                weights_only=True,
            )
            required = {
                "state_dict",
                "in_channels",
                "hid",
                "hid_att",
                "n_layers",
                "n_heads",
                "trained_on",
                "created",
            }
            available = set(payload) if isinstance(payload, dict) else set()
            if not required <= available:
                missing = sorted(required - available)
                raise ValueError(
                    f"checkpoint is missing fields: {', '.join(missing)}"
                )
            in_channels = int(payload["in_channels"])
            hid = int(payload["hid"])
            hid_att = int(payload["hid_att"])
            n_layers = int(payload["n_layers"])
            n_heads = int(payload["n_heads"])
            self.trained_on = str(payload["trained_on"])
            self.created = str(payload["created"])
            state_dict = payload["state_dict"]
        else:
            in_channels = self._windows.IN_CHANNELS

        self.in_channels = in_channels
        self.hid = hid
        self.hid_att = hid_att
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.net = self._net.RegretFormerNet(
            in_channels=in_channels,
            hid=hid,
            hid_att=hid_att,
            n_layers=n_layers,
            n_heads=n_heads,
        ).to(self.device)
        if state_dict is not None:
            self.net.load_state_dict(state_dict)
        self.net.eval()

    def decide(self, obs: MarketObservation) -> list[Decision]:
        """Round one network pass into feasible, individually rational decisions."""
        if not obs.jobs:
            return []
        structure = self._windows.structure_from_observation(obs)
        batch = self._windows.WindowBatch((structure,), self.device)
        assignments, payments = self._deploy.deployed_outcome(
            self.net,
            batch,
            batch.true_values,
        )
        decisions = []
        for job_index, candidate_index in enumerate(assignments[0]):
            if candidate_index < 0:
                continue
            job_id = structure.job_ids[job_index]
            placement_id = structure.candidate_ids[job_index][candidate_index]
            candidate = obs.candidate(job_id, placement_id)
            value = obs.value(job_id, placement_id)
            if value is None or value < candidate.resource_cost_credits:
                continue
            charge = min(
                value,
                max(
                    candidate.resource_cost_credits,
                    float(payments[0, job_index]),
                ),
            )
            decisions.append(Decision(
                job_id,
                placement_id,
                charge,
            ))
        return decisions
