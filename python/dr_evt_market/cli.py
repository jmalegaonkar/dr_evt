################################################################################
#         Copyright 2023 Lawrence Livermore National Security, LLC             #
#         See the top-level LICENSE file for details.                          #
#                                                                              #
#         SPDX-License-Identifier: MIT                                         #
################################################################################

"""The command line: run a market, train RegretFormer, prepare a trace."""

import argparse
import sys
import tempfile
from pathlib import Path

from .jobs import prepare, read_jobs, write_jobs
from .market import run, write_outputs
from .mechanism import MECHANISMS, record_windows, train_regretformer
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
    run_parser.add_argument("--checkpoint", type=Path)
    run_parser.add_argument("--platforms", default=",".join(DEFAULT_FEDERATION))

    train_parser = commands.add_parser("train", help="train RegretFormer")
    train_parser.add_argument("--jobs", required=True, type=Path)
    train_parser.add_argument("--out", required=True, type=Path)
    train_parser.add_argument("--share", type=float, default=1.0)
    train_parser.add_argument("--prefix", type=int, default=32)
    train_parser.add_argument("--window", type=int, default=60)
    train_parser.add_argument("--platforms", default=",".join(DEFAULT_FEDERATION))
    train_parser.add_argument(
        "--objective", choices=("revenue", "welfare"), default="revenue"
    )
    train_parser.add_argument("--steps", type=int, default=2000)
    train_parser.add_argument("--seed", type=int, default=0)

    prepare_parser = commands.add_parser("prepare", help="prepare trace jobs")
    prepare_parser.add_argument("--trace", action="append", required=True)
    prepare_parser.add_argument("--out", required=True, type=Path)
    prepare_parser.add_argument("--format", choices=("lc", "simple"), default="lc")
    prepare_parser.add_argument("--start", type=float)
    prepare_parser.add_argument("--hours", type=float)
    prepare_parser.add_argument("--seed", type=int, default=0)
    prepare_parser.add_argument("--gpu-fraction", type=float, default=0.5)
    prepare_parser.add_argument("--requires")
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


def _mechanism(args):
    if args.mechanism != "regretformer":
        return MECHANISMS[args.mechanism]()
    if args.checkpoint is None:
        raise ValueError("--mechanism regretformer needs --checkpoint")
    return MECHANISMS[args.mechanism](args.checkpoint)


def _run(args):
    mechanism = _mechanism(args)
    platforms = federation(
        args.out / "platforms", args.share, names=_names(args.platforms)
    )
    result = run(
        read_jobs(args.jobs),
        platforms,
        mechanism,
        window_s=args.window,
        prefix=args.prefix,
    )
    paths = write_outputs(result, args.out)
    windows = max((row.window for row in result.routed), default=-1) + 1
    print(f"windows={windows}")
    print(f"routed={len(result.routed)}")
    print(f"waiting={len(result.waiting)}")
    print(f"routed_sha256={paths['sha256']}")


def _train(args):
    with tempfile.TemporaryDirectory() as directory:
        platforms = federation(directory, args.share, names=_names(args.platforms))
        windows = record_windows(
            read_jobs(args.jobs), platforms, window_s=args.window, prefix=args.prefix
        )
        mechanism, history = train_regretformer(
            windows,
            platforms,
            objective=args.objective,
            steps=args.steps,
            seed=args.seed,
        )
    mechanism.save(
        args.out,
        objective=args.objective,
        steps=args.steps,
        windows=len(windows),
        share=args.share,
    )
    print(f"windows={len(windows)}")
    for key in ("objective", "regret", "multiplier"):
        last = history[key][-100:]
        print(f"{key}={sum(last) / len(last):.6g}")


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
        gpu_fraction=args.gpu_fraction,
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
        {"run": _run, "train": _train, "prepare": _prepare}[args.command](args)
    except (ImportError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0
