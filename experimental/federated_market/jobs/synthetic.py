################################################################################
#         Copyright 2023 Lawrence Livermore National Security, LLC             #
#         See the top-level LICENSE file for details.                          #
#                                                                              #
#         SPDX-License-Identifier: MIT                                         #
################################################################################

"""Synthetic days: real groups of jobs, picked at random, at their times of day."""

from collections import defaultdict

from .bids import generator
from .traces import Row

_GROUP_GAP = 60
_DAY = 86400


def groups(records) -> list[list[Row]]:
    """Split one source's jobs into groups a user submitted within a minute."""
    by_user = defaultdict(list)
    for record in records:
        by_user[record.identity if record.user is None else record.user].append(record)
    found = []
    for jobs in by_user.values():
        current = [jobs[0]]
        for record in jobs[1:]:
            if record.submit - current[-1].submit <= _GROUP_GAP:
                current.append(record)
            else:
                found.append(current)
                current = [record]
        found.append(current)
    return sorted(found, key=lambda group: (group[0].submit, group[0].order))


def synthesize(kept, lower, upper, seed) -> tuple[list[Row], dict[str, int]]:
    """Return one synthetic day from the interval, and each source's group count.

    For each source, the number of groups is Poisson with the interval's mean per day;
    the groups are drawn at random, each at most once, and each keeps its time of day.
    """
    days = (upper - lower) / _DAY
    by_source = defaultdict(list)
    for record in kept:
        by_source[record.source].append(record)
    rows, counts = [], {}
    for source, records in by_source.items():
        found = groups(records)
        counts[source] = len(found)
        rng = generator(seed, source, "#day")
        count = min(len(found), int(rng.poisson(len(found) / days)))
        for index in sorted(rng.choice(len(found), size=count, replace=False)):
            group = found[index]
            shift = lower + (group[0].submit - lower) % _DAY - group[0].submit
            rows += [row._replace(submit=row.submit + shift) for row in group]
    rows.sort(key=lambda row: (row.submit, row.source, row.order))
    return rows, counts
