"""Memory budget and process supervision checks without loading local models."""

import argparse
import subprocess
from unittest import mock

import pytest

from src.core import memory_limit

GIB = 1024**3


def test_available_memory_uses_linux_reclaimable_memory(tmp_path):
    path = tmp_path / "meminfo"
    path.write_text("MemFree: 100 kB\nMemAvailable: 123456 kB\n")
    assert memory_limit._available_memory(path) == 123456 * 1024


def test_budget_preserves_reserve_and_honors_explicit_maximum():
    assert memory_limit._memory_budget(8 * GIB, 2 * GIB, None) == 6 * GIB
    assert memory_limit._memory_budget(8 * GIB, 2 * GIB, 3 * GIB) == 3 * GIB
    assert memory_limit._memory_budget(3 * GIB, 2 * GIB, 4 * GIB) == GIB


@pytest.mark.parametrize("available", [GIB, 2 * GIB])
def test_budget_refuses_to_start_without_headroom(available):
    with pytest.raises(ValueError, match="reserve"):
        memory_limit._memory_budget(available, 2 * GIB, None)


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf"])
def test_cli_rejects_invalid_limits(value):
    with pytest.raises(argparse.ArgumentTypeError):
        memory_limit._positive_gib(value)


def test_scope_caps_all_children_and_disables_swap():
    command = memory_limit._scope_command(["python", "-m", "pytest"], GIB, "test.scope")
    assert f"--property=MemoryMax={GIB}" in command
    assert "--property=MemorySwapMax=0" in command
    assert "--property=OOMPolicy=kill" in command
    assert command[-4:] == ["--", "python", "-m", "pytest"]


def _mock_runtime(monkeypatch, available):
    monkeypatch.setattr(memory_limit.sys, "platform", "linux")
    monkeypatch.setattr(memory_limit.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(memory_limit, "_available_memory", mock.Mock(side_effect=available))
    process = mock.Mock()
    process.poll.return_value = 0
    popen = mock.Mock(return_value=process)
    monkeypatch.setattr(memory_limit.subprocess, "Popen", popen)
    return process, popen


def test_supervisor_preserves_child_exit_code(monkeypatch):
    process, _ = _mock_runtime(monkeypatch, [4 * GIB])
    process.wait.return_value = 7
    assert memory_limit.run_limited(["command"], reserve=2 * GIB, maximum=GIB) == 7


def test_supervisor_stops_scope_when_other_apps_use_reserve(monkeypatch):
    process, _ = _mock_runtime(monkeypatch, [4 * GIB, GIB])
    process.wait.side_effect = subprocess.TimeoutExpired("command", 1)
    stop = mock.Mock()
    monkeypatch.setattr(memory_limit, "_stop_scope", stop)

    assert memory_limit.run_limited(["command"], reserve=2 * GIB, maximum=GIB) == 75
    stop.assert_called_once()


def test_supervisor_refuses_unsupported_platform(monkeypatch):
    monkeypatch.setattr(memory_limit.sys, "platform", "darwin")
    with pytest.raises(RuntimeError, match="Linux"):
        memory_limit.run_limited(["command"], reserve=2 * GIB)
