################################################################################
#         Copyright 2023 Lawrence Livermore National Security, LLC             #
#         See the top-level LICENSE file for details.                          #
#                                                                              #
#         SPDX-License-Identifier: MIT                                         #
################################################################################

"""Launch and stop a local DR_EVT gRPC server process."""

import os
from pathlib import Path
import socket
import subprocess
from types import TracebackType
from typing import TextIO

from .base import ConfigurationError, InfrastructureFailure


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _find_server_binary(binary: str | Path | None) -> Path:
    if binary is not None:
        candidates = [Path(binary)]
    else:
        configured_prefix = os.environ.get("CMAKE_INSTALL_PREFIX")
        if configured_prefix:
            candidates = [Path(configured_prefix) / "bin" / "dr_evt_server"]
        else:
            root = _repository_root()
            candidates = [
                root / "install" / "bin" / "dr_evt_server",
                root / "build" / "dr_evt_server",
            ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise ConfigurationError(
        "dr_evt_server binary not found: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def _free_local_address() -> str:
    try:
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            return f"127.0.0.1:{listener.getsockname()[1]}"
    except OSError as error:
        raise InfrastructureFailure(
            f"cannot select a free local server port: {error}"
        ) from error


class ServerProcess:
    """Manage one local ``dr_evt_server`` subprocess for a context block."""

    def __init__(
        self,
        binary: str | Path | None,
        work_dir: str | Path,
        address: str | None = None,
    ) -> None:
        """Configure a server, selecting a free local port when needed."""
        self.binary = _find_server_binary(binary)
        self.work_dir = Path(work_dir).resolve()
        self.address = address or _free_local_address()
        if not isinstance(self.address, str) or not self.address:
            raise ConfigurationError("server address must be a non-empty string")
        try:
            self.work_dir.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise InfrastructureFailure(
                f"cannot create server work directory {self.work_dir}"
            ) from error
        self._log_path = self.work_dir / "server.log"
        self._log_file: TextIO | None = None
        self._process: subprocess.Popen[str] | None = None

    def _log_tail(self, limit: int = 4096) -> str:
        try:
            with self._log_path.open("rb") as log_file:
                log_file.seek(0, 2)
                log_file.seek(max(log_file.tell() - limit, 0))
                return log_file.read().decode("utf-8", errors="replace").strip()
        except OSError:
            return ""

    def start(self) -> "ServerProcess":
        """Start the process and wait until its gRPC channel is ready."""
        if self._process is not None:
            raise InfrastructureFailure("DR_EVT server process is already started")
        try:
            import grpc
        except ImportError as error:
            raise InfrastructureFailure(
                "Python gRPC dependencies are missing; install dr_evt_market[grpc]"
            ) from error

        try:
            self._log_file = self._log_path.open("w", encoding="utf-8")
            self._process = subprocess.Popen(
                [str(self.binary), self.address],
                cwd=self.work_dir,
                stdout=self._log_file,
                stderr=self._log_file,
                text=True,
            )
        except OSError as error:
            if self._log_file is not None:
                self._log_file.close()
                self._log_file = None
            raise InfrastructureFailure(
                f"cannot start DR_EVT server {self.binary}: {error}"
            ) from error

        channel = grpc.insecure_channel(self.address)
        try:
            grpc.channel_ready_future(channel).result(timeout=15)
        except grpc.FutureTimeoutError as error:
            self.stop()
            detail = self._log_tail()
            raise InfrastructureFailure(
                f"DR_EVT server did not become ready at {self.address}; "
                f"server.log tail: {detail!r}"
            ) from error
        finally:
            channel.close()
        return self

    def stop(self) -> None:
        """Terminate the owned process and wait for it to exit."""
        process = self._process
        self._process = None
        try:
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
        finally:
            if self._log_file is not None:
                self._log_file.close()
                self._log_file = None

    def __enter__(self) -> "ServerProcess":
        """Start the server and return this process manager."""
        return self.start()

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Stop the server regardless of how the context exits."""
        del exception_type, exception, traceback
        self.stop()
