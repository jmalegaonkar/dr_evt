################################################################################
#         Copyright 2023 Lawrence Livermore National Security, LLC             #
#         See the top-level LICENSE file for details.                          #
#                                                                              #
#         SPDX-License-Identifier: MIT                                         #
################################################################################

"""Train RegretFormer on synthetic or controller-logged market windows."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from math import exp, log
from pathlib import Path

import numpy as np
import torch

from .deploy import deployed_outcome
from .net import RegretFormerNet
from .regret import DEFAULT_GRID, guided_refinement_regret, summarize_regret
from .windows import (
    ValueSamplingSpec,
    WindowBatch,
    WindowStructure,
    sample_values,
)


@dataclass(frozen=True)
class TrainConfig:
    """Configure optimization, regret control, sampling, and evaluation."""

    objective: str = "welfare"
    epochs: int = 20
    batch_size: int = 8
    learning_rate: float = 1.0e-3
    misreport_steps: int = 10
    misreport_learning_rate: float = 0.05
    regret_jobs_per_batch: int = 4
    regret_target_start: float = 0.05
    regret_target_end: float = 0.005
    dual_learning_rate: float = 0.5
    capacity_penalty_weight: float = 5.0
    value_sampling: ValueSamplingSpec = field(
        default_factory=ValueSamplingSpec
    )
    seed: int = 0
    device: str = "cpu"
    hid: int = 64
    hid_att: int = 32
    n_layers: int = 2
    n_heads: int = 4
    evaluation_grid: tuple[float, ...] = DEFAULT_GRID
    evaluation_steps: int = 20
    evaluation_perturbations: int = 2

    def __post_init__(self) -> None:
        if self.objective not in {"welfare", "revenue"}:
            raise ValueError("objective must be 'welfare' or 'revenue'")
        integer_fields = (
            self.epochs,
            self.batch_size,
            self.regret_jobs_per_batch,
            self.hid,
            self.hid_att,
            self.n_layers,
            self.n_heads,
        )
        if any(value < 1 for value in integer_fields):
            raise ValueError("training and network sizes must be positive")
        if self.misreport_steps < 0 or self.evaluation_steps < 0:
            raise ValueError("ascent step counts must be non-negative")
        positive = (
            self.learning_rate,
            self.misreport_learning_rate,
            self.regret_target_start,
            self.regret_target_end,
            self.dual_learning_rate,
        )
        if any(value <= 0.0 for value in positive):
            raise ValueError("learning rates and regret targets must be positive")
        if self.capacity_penalty_weight < 0.0:
            raise ValueError("capacity_penalty_weight must be non-negative")


class Trainer:
    """Optimize one RegretFormer network over a collection of window structures."""

    def __init__(
        self,
        config: TrainConfig,
        structures: list[WindowStructure],
    ) -> None:
        """Create deterministic batches, network weights, and optimizer state."""
        if not structures:
            raise ValueError("Trainer requires at least one window structure")
        self.config = config
        self.structures = list(structures)
        torch.manual_seed(config.seed)
        self.rng = np.random.default_rng(config.seed)
        self.batches = [
            WindowBatch(
                self.structures[index:index + config.batch_size],
                config.device,
            )
            for index in range(0, len(self.structures), config.batch_size)
        ]
        self.net = RegretFormerNet(
            hid=config.hid,
            hid_att=config.hid_att,
            n_layers=config.n_layers,
            n_heads=config.n_heads,
        ).to(config.device)
        self.optimizer = torch.optim.Adam(
            self.net.parameters(),
            lr=config.learning_rate,
        )
        self.dual = 0.0
        self.history: list[dict[str, float | int]] = []
        self.evaluation: dict[str, float] = {}

    def _sample_dimensions(self, batch: WindowBatch) -> torch.Tensor:
        sampled = sample_values(
            batch.structures,
            self.rng,
            self.config.value_sampling,
        )
        values = torch.zeros_like(batch.dimension_values)
        for batch_index, jobs in enumerate(sampled):
            for job_index, dimensions in enumerate(jobs):
                values[
                    batch_index,
                    job_index,
                    :len(dimensions),
                ] = torch.tensor(
                    dimensions,
                    dtype=values.dtype,
                    device=values.device,
                )
        return values

    @staticmethod
    def _relaxed(
        net: torch.nn.Module,
        batch: WindowBatch,
        dimension_values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        candidate_values = batch.candidate_values(dimension_values)
        probabilities, fractions = net(
            batch.input_tensor(candidate_values),
            batch.slot_mask,
            batch.job_mask,
        )
        costs = batch.public_channels[..., 0] * batch.scales.unsqueeze(-1)
        charges = torch.minimum(
            candidate_values,
            torch.maximum(costs, fractions.unsqueeze(-1) * candidate_values),
        )
        utilities = (
            probabilities * (candidate_values - charges)
        ).sum(dim=-1)
        return probabilities, charges, candidate_values, utilities

    @staticmethod
    def _unilateral(
        true_values: torch.Tensor,
        batch_index: int,
        job_index: int,
        factors: torch.Tensor,
    ) -> torch.Tensor:
        selector = torch.zeros_like(true_values)
        selector[batch_index, job_index] = 1.0
        return true_values * (1.0 + selector * (factors - 1.0))

    def _regret_targets(self, batch: WindowBatch) -> list[tuple[int, int]]:
        targets = [
            (batch_index, job_index)
            for batch_index in range(batch.job_mask.shape[0])
            for job_index in range(batch.job_mask.shape[1])
            if bool(batch.job_mask[batch_index, job_index])
        ]
        count = min(self.config.regret_jobs_per_batch, len(targets))
        selected = self.rng.choice(len(targets), size=count, replace=False)
        return [targets[int(index)] for index in sorted(selected)]

    def _find_misreports(
        self,
        batch: WindowBatch,
        true_values: torch.Tensor,
        targets: list[tuple[int, int]],
    ) -> list[torch.Tensor]:
        results = []
        for batch_index, job_index in targets:
            valid = batch.dimension_mask[batch_index, job_index]
            factors = torch.ones(
                true_values.shape[2],
                dtype=true_values.dtype,
                device=true_values.device,
                requires_grad=True,
            )
            optimizer = torch.optim.Adam(
                (factors,),
                lr=self.config.misreport_learning_rate,
            )
            for _ in range(self.config.misreport_steps):
                optimizer.zero_grad()
                bounded = torch.where(
                    valid,
                    factors.clamp(0.0, 4.0),
                    torch.ones_like(factors),
                )
                reported = self._unilateral(
                    true_values,
                    batch_index,
                    job_index,
                    bounded,
                )
                utility = self._relaxed(
                    self.net,
                    batch,
                    reported,
                )[3][batch_index, job_index]
                gradient = torch.autograd.grad(-utility, factors)[0]
                factors.grad = gradient
                optimizer.step()
            results.append(torch.where(
                valid,
                factors.detach().clamp(0.0, 4.0),
                torch.ones_like(factors),
            ))
        return results

    def _relaxed_regret(
        self,
        batch: WindowBatch,
        true_values: torch.Tensor,
        targets: list[tuple[int, int]],
        factors: list[torch.Tensor],
        truthful_utilities: torch.Tensor,
    ) -> torch.Tensor:
        gains = []
        for (batch_index, job_index), report_factors in zip(targets, factors):
            reported = self._unilateral(
                true_values,
                batch_index,
                job_index,
                report_factors,
            )
            utility = self._relaxed(
                self.net,
                batch,
                reported,
            )[3][batch_index, job_index]
            gains.append(torch.relu(
                utility - truthful_utilities[batch_index, job_index]
            ))
        if not gains:
            return torch.zeros((), device=true_values.device)
        return torch.stack(gains).mean()

    def _target(self, epoch: int) -> float:
        if self.config.epochs == 1:
            return self.config.regret_target_end
        fraction = epoch / (self.config.epochs - 1)
        return exp(
            log(self.config.regret_target_start) * (1.0 - fraction)
            + log(self.config.regret_target_end) * fraction
        )

    def train(self) -> dict[str, object]:
        """Train all epochs, evaluate deployment, and return run history."""
        for epoch in range(self.config.epochs):
            totals = {
                "objective": 0.0,
                "regret": 0.0,
                "capacity_penalty": 0.0,
                "loss": 0.0,
            }
            order = self.rng.permutation(len(self.batches))
            for batch_index in order:
                batch = self.batches[int(batch_index)]
                true_values = self._sample_dimensions(batch)
                targets = self._regret_targets(batch)
                factors = self._find_misreports(batch, true_values, targets)
                self.optimizer.zero_grad()
                probabilities, charges, candidate_values, utilities = (
                    self._relaxed(self.net, batch, true_values)
                )
                welfare = (
                    probabilities
                    * (
                        candidate_values
                        - batch.public_channels[..., 0]
                        * batch.scales.unsqueeze(-1)
                    )
                ).sum(dim=(1, 2)).mean()
                revenue = (probabilities * charges).sum(dim=(1, 2)).mean()
                objective = welfare if self.config.objective == "welfare" else revenue
                used = torch.einsum(
                    "bns,bnsp->bp",
                    probabilities,
                    batch.demands,
                )
                capacity_penalty = torch.relu(
                    used - batch.capacities
                ).square().sum(dim=1).mean()
                regret = self._relaxed_regret(
                    batch,
                    true_values,
                    targets,
                    factors,
                    utilities,
                )
                target = self._target(epoch)
                self.dual = max(
                    0.0,
                    self.dual
                    + self.config.dual_learning_rate
                    * (float(regret.detach()) - target),
                )
                loss = (
                    -objective
                    + self.dual * regret
                    + self.config.capacity_penalty_weight * capacity_penalty
                )
                loss.backward()
                self.optimizer.step()
                totals["objective"] += float(objective.detach())
                totals["regret"] += float(regret.detach())
                totals["capacity_penalty"] += float(capacity_penalty.detach())
                totals["loss"] += float(loss.detach())
            divisor = len(self.batches)
            self.history.append({
                "epoch": epoch + 1,
                "objective": totals["objective"] / divisor,
                "relaxed_regret": totals["regret"] / divisor,
                "capacity_penalty": totals["capacity_penalty"] / divisor,
                "dual": self.dual,
                "regret_target": self._target(epoch),
                "loss": totals["loss"] / divisor,
            })
        self.evaluation = self.evaluate()
        return {
            "epochs": list(self.history),
            "evaluation": dict(self.evaluation),
        }

    @staticmethod
    def _optimal_welfare(
        batch: WindowBatch,
        batch_index: int,
        candidate_values: np.ndarray,
    ) -> float:
        demand = batch.demands[batch_index].cpu().numpy()
        capacities = batch.capacities[batch_index].cpu().numpy()
        costs = (
            batch.public_channels[batch_index, ..., 0]
            * batch.scales[batch_index].unsqueeze(-1)
        ).cpu().numpy()
        mask = batch.candidate_mask[batch_index].cpu().numpy()
        best = 0.0

        def search(job_index: int, remaining: np.ndarray, welfare: float) -> None:
            nonlocal best
            if job_index == mask.shape[0]:
                best = max(best, welfare)
                return
            search(job_index + 1, remaining, welfare)
            for candidate_index in np.flatnonzero(mask[job_index]):
                need = demand[job_index, candidate_index]
                net_value = (
                    candidate_values[job_index, candidate_index]
                    - costs[job_index, candidate_index]
                )
                if net_value < 0.0 or np.any(need > remaining):
                    continue
                search(
                    job_index + 1,
                    remaining - need,
                    welfare + float(net_value),
                )

        search(0, capacities, 0.0)
        return best

    def evaluate(self) -> dict[str, float]:
        """Measure deployed VCG coverage and grid-plus-ascent regret."""
        coverages = []
        regret_values = []
        truthful_utilities = []
        for batch_number, batch in enumerate(self.batches):
            dimensions = batch.dimension_values
            candidate_values = batch.candidate_values(dimensions)
            assignments, _ = deployed_outcome(
                self.net,
                batch,
                candidate_values,
            )
            candidate_array = candidate_values.cpu().numpy()
            costs = (
                batch.public_channels[..., 0]
                * batch.scales.unsqueeze(-1)
            ).cpu().numpy()
            for index, assignment in enumerate(assignments):
                deployed = sum(
                    max(
                        candidate_array[index, job, candidate]
                        - costs[index, job, candidate],
                        0.0,
                    )
                    for job, candidate in enumerate(assignment)
                    if candidate >= 0
                )
                optimum = self._optimal_welfare(
                    batch,
                    index,
                    candidate_array[index],
                )
                coverages.append(deployed / optimum if optimum > 0.0 else 1.0)
            regret = guided_refinement_regret(
                self.net,
                batch,
                dimensions,
                grid=self.config.evaluation_grid,
                steps=self.config.evaluation_steps,
                n_perturb=self.config.evaluation_perturbations,
                seed=self.config.seed + batch_number,
            )
            summary = summarize_regret(regret, batch, dimensions, self.net)
            regret_values.extend(
                regret[batch.job_mask].detach().cpu().numpy().tolist()
            )
            truthful_utilities.append(
                summary["mean_regret_over_mean_truthful_utility"]
            )
        mean_regret = float(np.mean(regret_values)) if regret_values else 0.0
        max_regret = float(np.max(regret_values)) if regret_values else 0.0
        return {
            "vcg_welfare_coverage": float(np.mean(coverages)),
            "mean_regret": mean_regret,
            "max_regret": max_regret,
            "mean_regret_over_mean_truthful_utility": float(
                np.mean(truthful_utilities)
            ),
        }

    def save(self, path: str | Path) -> Path:
        """Write a loadable RegretFormer checkpoint and return its path."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.net.state_dict(),
                "in_channels": self.batches[0].in_channels,
                "hid": self.config.hid,
                "hid_att": self.config.hid_att,
                "n_layers": self.config.n_layers,
                "n_heads": self.config.n_heads,
                "trained_on": f"{len(self.structures)} market windows",
                "created": date.today().isoformat(),
            },
            destination,
        )
        return destination
