"""Run a command with a Linux memory cap and a system RAM reserve."""

from __future__ import annotations

import argparse
import math
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

_GIB = 1024**3


def _available_memory(path: Path = Path("/proc/meminfo")) -> int:
    for line in path.read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("Linux MemAvailable is unavailable; cannot calculate a safe memory budget.")


def _memory_budget(available: int, reserve: int, maximum: int | None) -> int:
    budget = available - reserve
    if maximum is not None:
        budget = min(budget, maximum)
    if budget < 32 * 1024**2:
        raise ValueError("Not enough available RAM to preserve the requested reserve. Close other applications first.")
    return budget


def _scope_command(command: list[str], budget: int, unit: str) -> list[str]:
    return [
        "systemd-run",
        "--user",
        "--scope",
        "--quiet",
        f"--unit={unit}",
        f"--property=MemoryHigh={int(budget * 0.9)}",
        f"--property=MemoryMax={budget}",
        "--property=MemorySwapMax=0",
        "--property=OOMPolicy=kill",
        "--",
        *command,
    ]


def _stop_scope(unit: str, process: subprocess.Popen) -> None:
    subprocess.run(
        ["systemctl", "--user", "stop", "--no-block", unit],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=10,
    )
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        subprocess.run(
            ["systemctl", "--user", "kill", "--signal=SIGKILL", unit],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
        process.wait(timeout=5)


def run_limited(command: list[str], *, reserve: int, maximum: int | None = None) -> int:
    """Cap a command and its children, stopping them if available system RAM drops below the reserve."""
    if sys.platform != "linux" or not all(shutil.which(name) for name in ("systemd-run", "systemctl")):
        raise RuntimeError("Memory-limited execution requires Linux with a running systemd user session.")
    budget = _memory_budget(_available_memory(), reserve, maximum)
    unit = f"sec-rag-{uuid.uuid4().hex[:12]}.scope"
    print(
        f"RAM cap: {budget / _GIB:.2f} GiB; system reserve: {reserve / _GIB:.2f} GiB; swap disabled.",
        flush=True,
    )
    env = os.environ.copy()
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    process = subprocess.Popen(_scope_command(command, budget, unit), env=env)
    try:
        while True:
            try:
                returncode = process.wait(timeout=1)
                if returncode:
                    print(
                        "Command stopped. If the RAM cap was reached, reduce model memory use or raise the cap.",
                        file=sys.stderr,
                    )
                return returncode if returncode >= 0 else 128 - returncode
            except subprocess.TimeoutExpired:
                if _available_memory() < reserve:
                    print("Stopping command: available system RAM fell below the reserve.", file=sys.stderr)
                    _stop_scope(unit, process)
                    return 75
    except KeyboardInterrupt:
        _stop_scope(unit, process)
        return 130
    finally:
        if process.poll() is None:
            _stop_scope(unit, process)


def _positive_gib(value: str) -> int:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("RAM values must be positive finite numbers in GiB.")
    return int(number * _GIB)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-ram-gib", type=_positive_gib, help="Maximum RAM for the command and all its children.")
    parser.add_argument(
        "--reserve-ram-gib", type=_positive_gib, default=2 * _GIB, help="System RAM to keep available (default: 2 GiB)."
    )
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Command to run, preceded by --.")
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("Provide a command, for example: -- api")
    try:
        code = run_limited(command, reserve=args.reserve_ram_gib, maximum=args.max_ram_gib)
    except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as exc:
        parser.exit(2, f"Cannot run with memory limits: {exc}\n")
    raise SystemExit(code)


if __name__ == "__main__":
    main()
