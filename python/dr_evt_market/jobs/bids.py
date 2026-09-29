################################################################################
#         Copyright 2023 Lawrence Livermore National Security, LLC             #
#         See the top-level LICENSE file for details.                          #
#                                                                              #
#         SPDX-License-Identifier: MIT                                         #
################################################################################

"""The persona rule that turns a user into a bid."""

import hashlib
import math


def generator(seed: int, source: str, key: str):
    """Return the deterministic random generator for one key of one source."""
    import numpy

    digest = hashlib.sha256(f"{seed}:{source}:{key}".encode()).digest()[:8]
    return numpy.random.default_rng(int.from_bytes(digest, "big"))


def persona_terms(
    seed: int, source: str, user: str | None, reference_price: float, job_rng
) -> tuple[str, float, object]:
    """Draw a persona price and return the generator of the user's preferences."""
    # A job without a user is its own user: every draw comes from its own stream.
    user_rng = job_rng if user is None else generator(seed, source, user)
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
    persona_price: float, per_platform, speeds, preference_rng
) -> float | dict[str, float]:
    """Price one persona as a scalar bid or a bid for each named platform."""
    if per_platform is None:
        return round(persona_price, 4)
    bid = {}
    for name in per_platform:
        # Drawn for every name, so a user's preference for a machine does not
        # depend on the machines the job's hardware allows.
        preference = math.exp(preference_rng.normal(0.0, 0.3))
        if name in speeds:
            bid[name] = round(persona_price * speeds[name] * preference, 4)
    return bid
