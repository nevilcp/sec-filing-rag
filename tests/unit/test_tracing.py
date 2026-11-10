"""
Unit tests for the tracing helpers.

Responsibilities:
  - Verify chain_span sets the OpenInference attributes on a standard
    OpenTelemetry tracer.
  - Verify the guardrails run with no tracer provider installed, which
    is the default when Phoenix is disabled.
"""

from unittest import mock

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from src.core import tracing
from src.guardrails.input_guard import validate_input
from src.guardrails.output_guard import validate_output


@pytest.fixture
def exporter(monkeypatch):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(tracing, "_tracer", provider.get_tracer("test"))
    return exporter


def test_chain_span_sets_openinference_input_attributes(exporter):
    with tracing.chain_span("step", "hello"):
        pass

    (span,) = exporter.get_finished_spans()
    assert span.name == "step"
    assert span.attributes["openinference.span.kind"] == "CHAIN"
    assert span.attributes["input.value"] == "hello"
    assert span.attributes["input.mime_type"] == "text/plain"


def test_set_span_output_sets_output_value(exporter):
    with tracing.chain_span("step", "hello") as span:
        tracing.set_span_output(span, "done")

    (finished,) = exporter.get_finished_spans()
    assert finished.attributes["output.value"] == "done"


@mock.patch("src.guardrails.input_guard._is_prompt_injection", return_value=False)
def test_validate_input_no_tracer_provider_returns_valid(mock_injection):
    assert validate_input("What are Apple's risk factors?") == (True, "")


@mock.patch("src.guardrails.output_guard._is_definitive_legal_advice", return_value=False)
@mock.patch("src.guardrails.output_guard._anonymize_pii", return_value=("text", []))
def test_validate_output_no_tracer_provider_returns_sanitized(mock_anonymize, mock_legal):
    assert validate_output("text") == ("text", [])
