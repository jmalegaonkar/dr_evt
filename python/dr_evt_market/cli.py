################################################################################
#         Copyright 2023 Lawrence Livermore National Security, LLC             #
#         See the top-level LICENSE file for details.                          #
#                                                                              #
#         SPDX-License-Identifier: MIT                                         #
################################################################################

"""The command line: run a market, prepare a trace."""

import argparse
import sys
from pathlib import Path

from .jobs import prepare, read_jobs, write_jobs
from .market import run, write_outputs
from .mechanism import MECHANISMS
from .platform import DEFAULT_FEDERATION, PLATFORMS, federation

_PROFILES = {profile.name: profile for profile in PLATFORMS}


def _parser():
    parser = argparse.ArgumentParser(prog="dr_evt_market")
    commands = parser.add_subparsers(dest="command", required=True)

    run_parser = commands.add_parser("run", help="run a market")
    run_parser.add_argument("--jobs", required=True, type=Path)
    run_parser.add_argument("--out", required=True, type=Path)
    run_parser.add_argument("--share", type=float, default=1.0)
    run_parser.add_argument("--prefix", type=int, default=32)
    run_parser.add_argument("--window", type=int, default=60)
    run_parser.add_argument("--mechanism", choices=MECHANISMS, default="vcg")
    run_parser.add_argument("--platforms", default=",".join(DEFAULT_FEDERATION))

    prepare_parser = commands.add_parser("prepare", help="prepare trace jobs")
    prepare_parser.add_argument("--trace", action="append", required=True)
    prepare_parser.add_argument("--out", required=True, type=Path)
    prepare_parser.add_argument("--format", choices=("lc", "simple"), default="lc")
    prepare_parser.add_argument("--start", type=float)
    prepare_parser.add_argument("--hours", type=float)
    prepare_parser.add_argument("--seed", type=int, default=0)
    prepare_parser.add_argument("--requires", default="gpu")
    prepare_parser.add_argument("--per-platform")
    prepare_parser.add_argument(
        "--limit-from", choices=("runtime", "request"), default="runtime"
    )
    return parser


def _names(value):
    names = tuple(value.split(","))
    unknown = next((name for name in names if name not in _PROFILES), None)
    if unknown is not None:
        raise ValueError(f"unknown platform {unknown!r}")
    return names


def _run(args):
    platforms = federation(
        args.out / "platforms", args.share, names=_names(args.platforms)
    )
    result = run(
        read_jobs(args.jobs),
        platforms,
        MECHANISMS[args.mechanism](),
        window_s=args.window,
        prefix=args.prefix,
    )
    paths = write_outputs(result, args.out)
    windows = max((row.window for row in result.routed), default=-1) + 1
    print(f"windows={windows}")
    print(f"routed={len(result.routed)}")
    print(f"rejected={len(result.rejected)}")
    print(f"routed_sha256={paths['sha256']}")


def _prepare(args):
    traces = {}
    home_prices = {}
    for item in args.trace:
        if "=" not in item:
            raise ValueError(f"trace must be name=path: {item!r}")
        name, path = item.split("=", 1)
        if name not in _PROFILES:
            raise ValueError(f"unknown platform {name!r}")
        traces[name] = Path(path)
        home_prices[name] = _PROFILES[name].price_per_node_hour
    per_platform = None
    if args.per_platform is not None:
        per_platform = _names(args.per_platform)
    jobs, summary = prepare(
        traces,
        home_prices=home_prices,
        trace_format=args.format,
        start=args.start,
        hours=args.hours,
        seed=args.seed,
        requires=args.requires,
        per_platform=per_platform,
        limit_from=args.limit_from,
    )
    write_jobs(jobs, args.out)
    for key, value in summary.items():
        print(f"{key}={value}")


def main(argv=None) -> int:
    """Parse arguments, run the selected command, and return its status."""
    args = _parser().parse_args(argv)
    try:
        _run(args) if args.command == "run" else _prepare(args)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0
