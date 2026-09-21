################################################################################
#         Copyright 2023 Lawrence Livermore National Security, LLC             #
#         See the top-level LICENSE file for details.                          #
#                                                                              #
#         SPDX-License-Identifier: MIT                                         #
################################################################################

"""Command line: run a market, or train the learned mechanism."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from contextlib import ExitStack
from pathlib import Path
import sys

from .controller import Controller, write_outputs
from .inputs import read_jobs, read_platforms
from .mechanisms import Mechanism, Platform, Vcg
from .platforms.base import PlatformSession


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dr_evt_market",
        description="Run a fixed-window market over DR_EVT platforms.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="run one market to completion")
    run.add_argument("--jobs", required=True, type=Path)
    run.add_argument("--platforms", required=True, type=Path)
    run.add_argument("--out", required=True, type=Path)
    run.add_argument("--window", type=int, default=300)
    run.add_argument("--mechanism", choices=("vcg",), default="vcg")
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--server-binary", type=Path)
    run.add_argument("--start-servers", action="store_true")
    run.add_argument("--log-windows", action="store_true")
    train = commands.add_parser("train", help="train a RegretFormer checkpoint")
    train.add_argument("--windows", required=True)
    train.add_argument("--out", required=True, type=Path)
    train.add_argument("--epochs", type=int, default=20)
    train.add_argument("--seed", type=int, default=0)
    return parser


def _mechanism(name: str) -> Mechanism:
    if name == "vcg":
        return Vcg()
    raise ValueError(f"unknown mechanism {name!r}")


def _session(platform: Platform, out_dir: Path) -> PlatformSession:
    if platform.address:
        from .platforms.grpc import GrpcPlatform

        return GrpcPlatform(
            platform.name,
            platform.total_nodes,
            platform.address,
            out_dir / "servers" / platform.name,
            session_name=platform.name,
        )
    from .platforms.inprocess import InProcessPlatform

    return InProcessPlatform(
        platform.name, platform.total_nodes, out_dir / "platforms" / platform.name
    )


def _run(args: argparse.Namespace) -> int:
    out_dir = args.out.resolve()
    platforms = {platform.name: platform for platform in read_platforms(args.platforms)}
    jobs = read_jobs(args.jobs)
    with ExitStack() as stack:
        if args.start_servers:
            from .platforms.server import ServerProcess

            for platform in platforms.values():
                if platform.address:
                    stack.enter_context(
                        ServerProcess(
                            args.server_binary,
                            out_dir / "servers" / platform.name,
                            address=platform.address,
                        )
                    )
        sessions = {
            name: _session(platform, out_dir) for name, platform in platforms.items()
        }
        report = Controller(
            sessions,
            platforms,
            _mechanism(args.mechanism),
            jobs,
            window_s=args.window,
            seed=args.seed,
            log_dir=out_dir if args.log_windows else None,
        ).run()
        paths = write_outputs(report, out_dir)
    print(f"windows={len(report.windows)}")
    print(f"routed={len(report.routed)}")
    print(f"rejected={len(report.rejected)}")
    print(f"routed_sha256={paths['routed_sha256']}")
    print(f"windows_sha256={paths['windows_sha256']}")
    return 0


def _train(args: argparse.Namespace) -> int:
    from .learned.harvest import harvest_structures
    from .learned.synthetic import synthetic_structures
    from .learned.train import TrainConfig, Trainer

    if args.windows == "synthetic":
        structures = synthetic_structures(20, seed=args.seed)
    else:
        structures = harvest_structures(Path(args.windows))
    trainer = Trainer(TrainConfig(epochs=args.epochs, seed=args.seed), structures)
    history = trainer.train()
    checkpoint = trainer.save(args.out)
    print(f"epochs={len(history['epochs'])}")
    print(f"checkpoint={checkpoint.resolve()}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Parse the command line and run the selected command."""
    args = _parser().parse_args(argv)
    try:
        if args.command == "run":
            return _run(args)
        if args.command == "train":
            return _train(args)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 2
