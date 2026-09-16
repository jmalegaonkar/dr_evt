################################################################################
#         Copyright 2023 Lawrence Livermore National Security, LLC             #
#         See the top-level LICENSE file for details.                          #
#                                                                              #
#         SPDX-License-Identifier: MIT                                         #
################################################################################

"""Pure start-time prediction from one platform snapshot."""

from collections.abc import Sequence
from dataclasses import dataclass

from .base import PlatformSnapshot


@dataclass(frozen=True)
class Prediction:
    """Describe a predicted start and the rule that produced it."""

    start_s: int | None
    source: str


def _first_release_start(
    now: int,
    free_nodes: int,
    num_nodes: int,
    releases: Sequence[tuple[float, int]],
    *,
    limit_s: int | None = None,
    shadow_time_s: float | None = None,
) -> int | None:
    for release_time, nodes_released in sorted(releases):
        if release_time <= now:
            continue
        if shadow_time_s is not None and release_time >= shadow_time_s:
            break
        free_nodes += nodes_released
        if free_nodes < num_nodes:
            continue
        if shadow_time_s is None:
            return int(release_time)
        if limit_s is not None and release_time + limit_s < shadow_time_s:
            return int(release_time)
    return None


def predict_start(
    snapshot: PlatformSnapshot,
    num_nodes: int,
    limit_s: int,
    ledger_ends: Sequence[tuple[float, int]],
) -> Prediction:
    """Predict one job's start from capacity and projected ends.

    ``free_now`` is exact when the queue is empty. ``backfill`` and
    ``after_release`` are exact when no earlier-queued job competes for the
    same nodes. ``behind_head`` is the head's reservation and can be beaten by
    a backfill this rule does not model.
    """
    now = snapshot.time_s
    if snapshot.shadow_time_s < 0:
        if num_nodes <= snapshot.free_nodes:
            return Prediction(now, "free_now")

        releases = [
            (release.time_s, release.nodes_released)
            for release in snapshot.releases
        ]
        if not releases:
            releases = list(ledger_ends)

        release_start = _first_release_start(
            now,
            snapshot.free_nodes,
            num_nodes,
            releases,
        )
        if release_start is not None:
            return Prediction(release_start, "after_release")
        return Prediction(None, "unknown")

    if (
        num_nodes <= snapshot.free_nodes
        and now + limit_s < snapshot.shadow_time_s
    ):
        return Prediction(now, "backfill")

    releases = [
        (release.time_s, release.nodes_released)
        for release in snapshot.releases
    ]
    release_start = _first_release_start(
        now,
        snapshot.free_nodes,
        num_nodes,
        releases,
        limit_s=limit_s,
        shadow_time_s=snapshot.shadow_time_s,
    )
    if release_start is not None:
        return Prediction(release_start, "after_release")
    return Prediction(int(snapshot.shadow_time_s), "behind_head")
