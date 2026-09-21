################################################################################
#         Copyright 2023 Lawrence Livermore National Security, LLC             #
#         See the top-level LICENSE file for details.                          #
#                                                                              #
#         SPDX-License-Identifier: MIT                                         #
################################################################################

"""The persona rule that turns a user into a bid."""

import hashlib
import math


def persona_bid(
    seed: int, source: str, user: str, job_id: str, per_platform
) -> tuple[str, float | dict[str, float]]:
    """Return the deterministic persona and bid for one job."""
    import numpy

    def generator(key: str):
        digest = hashlib.sha256(key.encode()).digest()[:8]
        return numpy.random.default_rng(int.from_bytes(digest, "big"))

    user_rng = generator(f"{seed}:{source}:{user}")
    job_rng = generator(f"{seed}:{source}:{user}:{job_id}")
    draw = user_rng.random()
    heavy = user_rng.random() < 0.2
    if draw < 0.45:
        persona, multiplier = "sticker", 1.0
    elif draw < 0.80:
        urgent = job_rng.random() < 0.2
        persona = "tier"
        multiplier = 4.0 if urgent and heavy else 2.0 if urgent else 1.0
    elif draw < 0.95:
        value = job_rng.lognormal(math.log(3.0), 0.5)
        persona, multiplier = "value", round(value, 4)
    else:
        persona, multiplier = "whale", 10.0
    if per_platform is None:
        return persona, multiplier
    bid = {
        name: round(multiplier * math.exp(user_rng.normal(0.0, 0.3)), 4)
        for name in per_platform
    }
    return persona, bid
