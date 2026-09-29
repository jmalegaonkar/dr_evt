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


def persona_terms(
    seed: int,
    source: str,
    user: str,
    row_identity: str,
    reference_price: float,
    *,
    job_rng=None,
) -> tuple[str, float, object]:
    """Draw a persona price and return its preference generator."""
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
    return persona, reference_price * multiplier, user_rng


def price_bid(
    persona_price: float,
    per_platform,
    *,
    speeds=None,
    home_platform=None,
    preference_rng=None,
) -> float | dict[str, float]:
    """Price one persona as a scalar or platform-specific bid."""
    if per_platform is None:
        return round(persona_price, 4)
    if speeds is None:
        raise ValueError("per-platform bids need platform speeds")
    bid = {}
    for name in per_platform:
        preference = 1.0
        if name != home_platform:
            preference = math.exp(preference_rng.normal(0.0, 0.3))
        if name not in speeds:
            continue
        bid[name] = round(persona_price * speeds[name] * preference, 4)
    return bid


def persona_bid(
    seed: int,
    source: str,
    user: str,
    row_identity: str,
    reference_price: float,
    per_platform,
    *,
    speeds=None,
    home_platform=None,
    job_rng=None,
) -> tuple[str, float | dict[str, float]]:
    """Return the deterministic persona and bid for one job."""
    persona, price, preference_rng = persona_terms(
        seed,
        source,
        user,
        row_identity,
        reference_price,
        job_rng=job_rng,
    )
    bid = price_bid(
        price,
        per_platform,
        speeds=speeds,
        home_platform=home_platform,
        preference_rng=preference_rng,
    )
    return persona, bid
