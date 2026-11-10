"""
Tracing helpers that work with or without Phoenix.

Responsibilities:
  - Open CHAIN spans through the standard OpenTelemetry API, so code
    runs unchanged when no tracer provider is installed.
  - Set OpenInference input and output attributes on those spans.

Not responsible for:
  - Registering the tracer provider (see api/run.py).
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes
from opentelemetry import trace
from opentelemetry.trace import Span

_tracer = trace.get_tracer(__name__)


@contextmanager
def chain_span(name: str, input_value: str) -> Iterator[Span]:
    """Open a CHAIN span that records its input.

    Args:
        name: Span name.
        input_value: Text recorded as the span input.

    Yields:
        The active span. The caller sets its output and status.
    """
    with _tracer.start_as_current_span(name) as span:
        span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.CHAIN.value)
        span.set_attribute(SpanAttributes.INPUT_VALUE, input_value)
        span.set_attribute(SpanAttributes.INPUT_MIME_TYPE, "text/plain")
        yield span


def set_span_output(span: Span, value: str) -> None:
    """Record the output of a span.

    Args:
        span: The span to update.
        value: Text recorded as the span output.
    """
    span.set_attribute(SpanAttributes.OUTPUT_VALUE, value)
