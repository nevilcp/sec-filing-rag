"""Unit tests for the Phoenix evaluation CLI entry point."""

import asyncio
from unittest import mock

import pandas as pd

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


def test_evaluator_only_grades_successful_query_answers():
    frame = pd.DataFrame(
        [
            {"name": "query_endpoint", "span_kind": "CHAIN", "status_code": "OK"},
            {"name": "retrieve_and_reason", "span_kind": "CHAIN", "status_code": "OK"},
            {"name": "query_endpoint", "span_kind": "CHAIN", "status_code": "ERROR"},
            {"name": "ChatGroq", "span_kind": "LLM", "status_code": "OK"},
        ],
        index=["answer", "internal-state", "blocked-input", "model-call"],
    )
    with (
        mock.patch.object(run_phoenix_evals, "Client") as client,
        mock.patch.object(run_phoenix_evals, "LLM"),
        mock.patch.object(run_phoenix_evals, "ClassificationEvaluator"),
        mock.patch.object(run_phoenix_evals, "async_evaluate_dataframe", new_callable=mock.AsyncMock) as evaluate,
        mock.patch.object(run_phoenix_evals, "to_annotation_dataframe") as annotations,
    ):
        client.return_value.spans.get_spans_dataframe.return_value = frame
        asyncio.run(run_phoenix_evals.main_async())

        assert list(evaluate.call_args.kwargs["dataframe"].index) == ["answer"]
        client.return_value.spans.log_span_annotations_dataframe.assert_called_once_with(
            dataframe=annotations.return_value
        )


def test_evaluator_skips_traces_without_completed_answers():
    frame = pd.DataFrame([{"name": "query_endpoint", "span_kind": "CHAIN", "status_code": "ERROR"}])
    with mock.patch.object(run_phoenix_evals, "Client") as client, mock.patch.object(run_phoenix_evals, "LLM") as llm:
        client.return_value.spans.get_spans_dataframe.return_value = frame

        asyncio.run(run_phoenix_evals.main_async())

        llm.assert_not_called()
