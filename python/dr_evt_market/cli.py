################################################################################
#         Copyright 2023 Lawrence Livermore National Security, LLC             #
#         See the top-level LICENSE file for details.                          #
#                                                                              #
#         SPDX-License-Identifier: MIT                                         #
################################################################################

"""The command line: prepare traces, run a market, harvest windows, train."""

import argparse
import sys
import tempfile
from pathlib import Path

from .harvest import harvest, read_windows, write_windows
from .jobs import prepare, read_jobs, write_jobs
from .market import run, write_outputs
from .mechanism import MECHANISMS, train_regretformer
from .platform import DEFAULT_FEDERATION, PROFILES, federation


def _share(value):
    """Parse one federation share or a share for each named platform."""
    try:
        return float(value)
    except ValueError:
        pass
    shares = {}
    for item in value.split(","):
        name, separator, raw_share = item.partition("=")
        if not separator or name not in PROFILES:
            raise argparse.ArgumentTypeError(f"invalid platform share {item!r}")
        try:
            shares[name] = float(raw_share)
        except ValueError as error:
            raise argparse.ArgumentTypeError(
                f"invalid platform share {item!r}"
            ) from error
    return shares


def _parser():
    parser = argparse.ArgumentParser(prog="dr_evt_market")
    commands = parser.add_subparsers(dest="command", required=True)

    run_parser = commands.add_parser("run", help="run a market")
    run_parser.add_argument("--jobs", required=True, type=Path)
    run_parser.add_argument("--out", required=True, type=Path)
    run_parser.add_argument("--share", type=_share, default=1.0)
    run_parser.add_argument("--prefix", type=int, default=32)
    run_parser.add_argument("--window", type=int, default=60)
    run_parser.add_argument("--mechanism", choices=MECHANISMS, default="vcg")
    run_parser.add_argument("--checkpoint", type=Path)
    run_parser.add_argument("--platforms", default=",".join(DEFAULT_FEDERATION))

    harvest_parser = commands.add_parser("harvest", help="record windows to train on")
    harvest_parser.add_argument("--jobs", action="append", required=True, type=Path)
    harvest_parser.add_argument("--out", required=True, type=Path)
    harvest_parser.add_argument("--mechanism", action="append", choices=MECHANISMS)
    harvest_parser.add_argument("--checkpoint", type=Path)
    harvest_parser.add_argument("--share", type=_share, default=1.0)
    harvest_parser.add_argument("--prefix", type=int, default=32)
    harvest_parser.add_argument("--window", type=int, default=60)
    harvest_parser.add_argument("--platforms", default=",".join(DEFAULT_FEDERATION))

    train_parser = commands.add_parser("train", help="train RegretFormer")
    train_parser.add_argument("--windows", action="append", required=True, type=Path)
    train_parser.add_argument("--out", required=True, type=Path)
    train_parser.add_argument(
        "--objective", choices=("revenue", "welfare"), default="revenue"
    )
    train_parser.add_argument("--steps", type=int, default=2000)
    train_parser.add_argument("--seed", type=int, default=0)
    train_parser.add_argument("--under-price", action="store_true")
    train_parser.add_argument("--device", default="cpu")

    prepare_parser = commands.add_parser("prepare", help="prepare trace jobs")
    prepare_parser.add_argument("--trace", action="append", required=True)
    prepare_parser.add_argument("--out", required=True, type=Path)
    prepare_parser.add_argument("--format", choices=("lc", "simple"), default="lc")
    prepare_parser.add_argument("--start", type=float)
    prepare_parser.add_argument("--hours", type=float)
    prepare_parser.add_argument("--seed", type=int, default=0)
    prepare_parser.add_argument("--gpu-fraction", type=float, default=0.5)
    prepare_parser.add_argument("--requires")
    prepare_parser.add_argument("--platforms", default=",".join(DEFAULT_FEDERATION))
    prepare_parser.add_argument("--bids", choices=("multi", "single"), default="multi")
    prepare_parser.add_argument("--synthetic", action="store_true")
    return parser


def _names(value):
    names = tuple(value.split(","))
    unknown = next((name for name in names if name not in PROFILES), None)
    if unknown is not None:
        raise ValueError(f"unknown platform {unknown!r}")
    return names


def _maker(name, checkpoint):
    if name != "regretformer":
        return MECHANISMS[name]
    if checkpoint is None:
        raise ValueError("--mechanism regretformer needs --checkpoint")
    return lambda: MECHANISMS[name](checkpoint)


def _run(args):
    mechanism = _maker(args.mechanism, args.checkpoint)()
    with tempfile.TemporaryDirectory() as directory:
        platforms = federation(directory, args.share, names=_names(args.platforms))
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


def _harvest(args):
    streams = {}
    for path in args.jobs:
        if path.stem in streams:
            raise ValueError(f"two jobs files named {path.stem!r}")
        streams[path.stem] = read_jobs(path)
    names = args.mechanism or ["vcg"]
    windows = harvest(
        streams,
        {name: _maker(name, args.checkpoint) for name in names},
        share=args.share,
        names=_names(args.platforms),
        window_s=args.window,
        prefix=args.prefix,
    )
    with tempfile.TemporaryDirectory() as directory:
        platforms = federation(directory, args.share, names=_names(args.platforms))
        write_windows(args.out, windows, platforms)
    print(f"windows={len(windows)}")
    for name in names:
        print(
            f"windows:{name}={sum(window['mechanism'] == name for window in windows)}"
        )
    sizes = [len(window["jobs"]) for window in windows]
    print(f"one_job={sizes.count(1)}")
    print(f"full_batch={sizes.count(args.prefix)}")


def _train(args):
    with tempfile.TemporaryDirectory() as directory:
        platforms, windows = read_windows(args.windows, directory)
        mechanism, history = train_regretformer(
            [(window["jobs"], window["free"]) for window in windows],
            platforms,
            objective=args.objective,
            steps=args.steps,
            seed=args.seed,
            under_price=args.under_price,
            device=args.device,
        )
    mechanism.save(
        args.out,
        objective=args.objective,
        steps=args.steps,
        windows=len(windows),
        files=[str(path) for path in args.windows],
        under_price=args.under_price,
    )
    print(f"windows={len(windows)}")
    for key in ("objective", "regret", "multiplier"):
        last = history[key][-100:]
        print(f"{key}={sum(last) / len(last):.6g}")


def _prepare(args):
    traces = {}
    for item in args.trace:
        if "=" not in item:
            raise ValueError(f"trace must be name=path: {item!r}")
        name, path = item.split("=", 1)
        traces[name] = Path(path)
    jobs, summary = prepare(
        traces,
        platforms=[PROFILES[name] for name in _names(args.platforms)],
        bids=args.bids,
        trace_format=args.format,
        start=args.start,
        hours=args.hours,
        seed=args.seed,
        gpu_fraction=args.gpu_fraction,
        requires=args.requires,
        synthetic=args.synthetic,
    )
    write_jobs(jobs, args.out)
    for key, value in summary.items():
        print(f"{key}={value}")


def main(argv=None) -> int:
    """Parse arguments, run the selected command, and return its status."""
    args = _parser().parse_args(argv)
    commands = {"prepare": _prepare, "run": _run, "harvest": _harvest, "train": _train}
    try:
        commands[args.command](args)
    except (ImportError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0
