################################################################################
#         Copyright 2023 Lawrence Livermore National Security, LLC             #
#         See the top-level LICENSE file for details.                          #
#                                                                              #
#         SPDX-License-Identifier: MIT                                         #
################################################################################

"""The persona rule that turns a user into a bid."""

import hashlib
import math


def job_generator(seed: int, source: str, row_identity: str):
    """Return the deterministic random generator for one job."""
    import numpy

    key = f"{seed}:{source}:{row_identity}"
    digest = hashlib.sha256(key.encode()).digest()[:8]
    return numpy.random.default_rng(int.from_bytes(digest, "big"))


def persona_bid(
    seed: int,
    source: str,
    user: str,
    row_identity: str,
    reference_price: float,
    per_platform,
    *,
    home_platform=None,
    job_rng=None,
) -> tuple[str, float | dict[str, float]]:
    """Return the deterministic persona and bid for one job."""
    import numpy

    def generator(key: str):
        digest = hashlib.sha256(key.encode()).digest()[:8]
        return numpy.random.default_rng(int.from_bytes(digest, "big"))

    user_rng = generator(f"{seed}:{source}:{user}")
    if job_rng is None:
        job_rng = job_generator(seed, source, row_identity)
    draw = user_rng.random()
    heavy = user_rng.random() < 0.2
    if draw < 0.45:
        persona, multiplier = "sticker", 1.0
    elif draw < 0.80:
        urgent = job_rng.random() < 0.2
        persona = "tier"
        multiplier = 4.0 if urgent and heavy else 2.0 if urgent else 1.0
    elif draw < 0.95:
        persona = "value"
        multiplier = job_rng.lognormal(math.log(3.0), 0.5)
    else:
        persona, multiplier = "whale", 10.0
    persona_price = reference_price * multiplier
    if per_platform is None:
        return persona, round(persona_price, 4)
    bid = {}
    for name in per_platform:
        preference = 1.0
        if name != home_platform:
            preference = math.exp(user_rng.normal(0.0, 0.3))
        bid[name] = round(persona_price * preference, 4)
    return persona, bid
