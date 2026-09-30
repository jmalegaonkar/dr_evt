################################################################################
#         Copyright 2023 Lawrence Livermore National Security, LLC             #
#         See the top-level LICENSE file for details.                          #
#                                                                              #
#         SPDX-License-Identifier: MIT                                         #
################################################################################

"""The persona rule that turns a user into a single bid or a bid per platform."""

import hashlib
import math
from dataclasses import dataclass

PERSONAS = ("sticker", "tier", "value", "whale")


def generator(seed: int, source: str, key: str):
    """Return the deterministic random generator for one key of one source."""
    import numpy

    digest = hashlib.sha256(f"{seed}:{source}:{key}".encode()).digest()[:8]
    return numpy.random.default_rng(int.from_bytes(digest, "big"))


def _persona(draw: float) -> str:
    """Return the persona that a uniform draw picks."""
    if draw < 0.45:
        return "sticker"
    if draw < 0.80:
        return "tier"
    if draw < 0.95:
        return "value"
    return "whale"


@dataclass(frozen=True)
class Terms:
    """A job's bid terms: its user's persona and heavy flag, its urgency and value."""

    persona: str
    heavy: bool
    urgent: bool
    value: float

    def multiple(self, persona: str | None = None) -> float:
        """Return the multiple a persona bids for this job, the user's by default."""
        persona = persona or self.persona
        if persona == "sticker":
            return 1.0
        if persona == "tier":
            return 4.0 if self.urgent and self.heavy else 2.0 if self.urgent else 1.0
        if persona == "value":
            return self.value
        return 10.0


def terms(seed: int, source: str, user: str | None, job_rng) -> Terms:
    """Draw a job's bid terms after its hardware draw."""
    # The job's own draws keep fixed places in its stream, whatever the persona.
    urgent = job_rng.random() < 0.2
    value = float(job_rng.lognormal(math.log(3.0), 0.5))
    # A job without a user is its own user: its user draws follow in its stream.
    user_rng = job_rng if user is None else generator(seed, source, user)
    persona = _persona(user_rng.random())
    heavy = user_rng.random() < 0.2
    return Terms(persona, heavy, urgent, value)


def single_bid(job_terms: Terms, level: float) -> float:
    """Return one bid per reference node-hour: a price level times the multiple."""
    return round(level * job_terms.multiple(), 4)


def multi_bid(seed, source, owner, job_terms: Terms, prices) -> dict[str, float]:
    """Return a bid on each platform, a multiple of that platform's own price."""
    bids = {}
    for name, price in prices.items():
        rng = generator(seed, source, f"{owner}@{name}")
        # Half the time a platform takes the user's own persona, else one of its own.
        own, draw = rng.random() < 0.5, rng.random()
        persona = job_terms.persona if own else _persona(draw)
        preference = math.exp(rng.normal(0.0, 0.3))
        bids[name] = round(price * job_terms.multiple(persona) * preference, 4)
    return bids
