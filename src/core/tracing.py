"""Tracing helpers that work with or without Phoenix."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes
from opentelemetry import trace
from opentelemetry.trace import Span

_tracer = trace.get_tracer(__name__)


@contextmanager
def chain_span(name: str, input_value: str) -> Iterator[Span]:
    """Open a CHAIN span with its input recorded, leaving output and status to the caller."""
    with _tracer.start_as_current_span(name) as span:
        span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.CHAIN.value)
        span.set_attribute(SpanAttributes.INPUT_VALUE, input_value)
        span.set_attribute(SpanAttributes.INPUT_MIME_TYPE, "text/plain")
        yield span


def set_span_output(span: Span, value: str) -> None:
    span.set_attribute(SpanAttributes.OUTPUT_VALUE, value)
