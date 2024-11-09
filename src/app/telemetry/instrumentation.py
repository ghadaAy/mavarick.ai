"""
OpenTelemetry tracer configuration module.

This module configures and OpenTelemetry for tracing across the application, enabling observability
and performance monitoring.

Typical usage:
    from app.telemetry import get_tracer
    tracer = get_tracer(__name__)
"""

from fastapi import FastAPI
from opentelemetry import trace
from opentelemetry.instrumentation.asyncio import AsyncioInstrumentor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.instrumentation.system_metrics import SystemMetricsInstrumentor
from opentelemetry.instrumentation.threading import ThreadingInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
)

from app.core.config import settings

from .custom_logger import get_logger
from .exporter import StructlogConsoleExporter

__all__ = ("get_tracer",)
logger = get_logger()
resource = Resource.create(attributes={"service.name": settings.PROJECT_NAME})

processor = BatchSpanProcessor(StructlogConsoleExporter())
provider = TracerProvider(resource=resource)
provider.add_span_processor(processor)


def configure_instrumentation(
    app: FastAPI,
) -> None:
    FastAPIInstrumentor.instrument_app(app, tracer_provider=provider)
    HTTPXClientInstrumentor().instrument(tracer_provider=provider)
    ThreadingInstrumentor().instrument(tracer_provider=provider)
    AsyncioInstrumentor().instrument(tracer_provider=provider)
    SystemMetricsInstrumentor().instrument(tracer_provider=provider)
    trace.set_tracer_provider(provider)


def get_tracer(name: str) -> trace.Tracer:
    """
    utility factory method which create a opentelemetry tracer instance.

    Args:
        name (str):  uniquely identifiable name for instrumentation scope,
        such as instrumentation library, package, module or class name

    Returns:
        trace.Tracer: tracer for telemetry purposes.

    """
    return trace.get_tracer(instrumenting_module_name=name)
