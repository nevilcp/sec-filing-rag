"""Unit tests for the Phoenix evaluation CLI entry point."""

from unittest import mock

from src.eval import run_phoenix_evals


def test_main_configures_logging_before_running_evals():
    calls = []

    def fake_run(coroutine):
        calls.append("run")
        coroutine.close()

    with (
        mock.patch.object(run_phoenix_evals, "configure_logging", lambda: calls.append("configure")),
        mock.patch.object(run_phoenix_evals.asyncio, "run", fake_run),
    ):
        run_phoenix_evals.main()

    assert calls == ["configure", "run"]
