################################################################################
#         Copyright 2023 Lawrence Livermore National Security, LLC             #
#         See the top-level LICENSE file for details.                          #
#                                                                              #
#         SPDX-License-Identifier: MIT                                         #
################################################################################

"""Reading raw traces: the LC format and DR_EVT's simple format."""

import csv
import math
from pathlib import Path


def _read_trace(source: str, path: str | Path, trace_format: str) -> list[dict]:
    records = []
    with Path(path).open(newline="", encoding="utf-8") as stream:
        for order, row in enumerate(csv.DictReader(stream)):
            if trace_format == "lc":
                raw_nodes = row["job.node.count"]
                nodes = None if raw_nodes in {"", "-"} else int(float(raw_nodes))
                no_start = row["t_run"].strip() == "-"
                if no_start:
                    runtime = None
                else:
                    begin = math.floor(float(row["t_run"]))
                    end = math.floor(float(row["t_inactive"]))
                    runtime = end - begin
                try:
                    limit = float(row["user_time_limit"])
                except (TypeError, ValueError):
                    limit = float(row["time_limit"])
                submit = math.floor(float(row["t_submit"]))
                user = row["user.name"]
            else:
                raw_nodes = row["num_nodes"]
                nodes = None if raw_nodes in {"", "-"} else int(float(raw_nodes))
                limit = float(row["time_limit"])
                raw_runtime = row.get("actual_run_time")
                runtime = (
                    None
                    if raw_runtime in (None, "")
                    else math.floor(float(raw_runtime))
                )
                no_start = False
                submit = math.floor(float(row["job_submit_time"]))
                user = row.get("user")
            records.append(
                dict(
                    source=source,
                    order=order,
                    submit=submit,
                    nodes=nodes,
                    limit=math.floor(limit),
                    runtime=runtime,
                    user=user,
                    no_start=no_start,
                )
            )
    return records
