"""
Logging configuration module using structlog.

This module configures structured logging with structlog, setting up
processors, formatters, and handlers to standardize log output
across the application. Supports JSON and console output formats,
and integrates with OpenTelemetry for tracing.

Typical usage:
    from app.telemetry.api import get_logger
    logger = get_logger()
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from logging import _nameToLevel
from typing import Any

import msgspec
import structlog
from opentelemetry import trace
from structlog.types import EventDict, Processor

from app.core.config import LogFormat, LogLevel, settings

__all__ = ("get_logger",)


def add_open_telemetry_spans(_: Any, __: Any, event_dict: EventDict) -> EventDict:  # noqa: ANN401
    """
    Structlog processor to enrich logs with opentelemetry span info.

    Notes:
        see https://structlog.org/en/stable/frameworks.html.

    """
    span = trace.get_current_span()
    if not span.is_recording():
        return event_dict

    ctx = span.get_span_context()
    parent = getattr(span, "parent", None)

    event_dict["span_id"] = trace.format_span_id(ctx.span_id)
    event_dict["trace_id"] = trace.format_trace_id(ctx.trace_id)
    event_dict["parent_span_id"] = (
        None if not parent else trace.format_span_id(parent.span_id)
    )

    event_dict["service"] = None
    if (resource := getattr(trace.get_tracer_provider(), "resource", None)) and (
        service_name := resource.attributes.get("service.name", None)
    ):
        event_dict["service"] = service_name

    return event_dict


def _drop_color_message_key(_: Any, __: Any, event_dict: EventDict) -> EventDict:  # noqa: ANN401
    """
    Uvicorn logs the message a second time in the extra `color_message`, but we don't
    need it. This processor drops the key from the event dict if it exists.
    """
    event_dict.pop("color_message", None)
    return event_dict


def setup_logging(
    log_format: LogFormat = settings.LOG_FORMAT,
    log_level: LogLevel = settings.LOG_LEVEL,
) -> None:
    """
    Configures structured logging with specified settings.

    Args:
        log_format(LogFormat): how the logs should be formatted.
        log_level(LogLevel): the application wide log level.

    """
    if structlog.is_configured():
        return
    processors: list[Processor] = [
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.CallsiteParameterAdder(
            [
                structlog.processors.CallsiteParameter.FUNC_NAME,
                structlog.processors.CallsiteParameter.LINENO,
                structlog.processors.CallsiteParameter.THREAD_NAME,
                structlog.processors.CallsiteParameter.PROCESS,
            ]
        ),
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.stdlib.ExtraAdder(),
        _drop_color_message_key,
        add_open_telemetry_spans,
    ]
    assert log_format in (
        "json",
        "console",
    ), f"unexpected log destination found {log_format}"

    log_renderer: Processor
    if log_format == "json":
        # Format the exception only for JSON logs, as we want to pretty-print them when
        # using the ConsoleRenderer
        processors.append(structlog.processors.format_exc_info)
        log_renderer = structlog.processors.JSONRenderer(msgspec.json.decode)
    else:
        log_renderer = structlog.dev.ConsoleRenderer(
            exception_formatter=structlog.dev.plain_traceback
        )

    formatter = structlog.stdlib.ProcessorFormatter(
        # These run ONLY on `logging` entries that do NOT originate within
        # structlog.
        foreign_pre_chain=processors,
        # These run on ALL entries after the pre_chain is done.
        processors=[
            # Remove _record & _from_structlog.
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            log_renderer,
        ],
    )
    log_level_int = _nameToLevel[log_level]
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(log_level_int),
        processors=[
            *processors,
            # Prepare event dict for `ProcessorFormatter`.
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    handler = logging.StreamHandler()
    # Use OUR `ProcessorFormatter` to format all `logging` entries.
    handler.setFormatter(formatter)
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    root_logger.setLevel(log_level_int)

    httpx_logger = logging.getLogger("httpx")
    httpx_logger.handlers.clear()
    httpx_logger.propagate = False

    uvicorn_access = logging.getLogger("uvicorn.access")
    uvicorn_access.disabled = True
    for name in [
        "uvicorn",
        "uvicorn.access",
        "uvicorn.error",
    ]:
        # Clear the log handlers for uvicorn loggers, and enable propagation
        # so the messages are caught by our root logger and formatted correctly
        # by structlog
        logger = logging.getLogger(name)
        logger.setLevel(log_level_int)
        logger.handlers.clear()
        logger.propagate = True


class TypedLogger(structlog.stdlib.BoundLogger, ABC):
    """
    This is only meant to improve type hinting, as the original BoundLogger
    methods return Any and cause bugs when using 'await' with non async logging
    methods which can't be caught by type_checker.
    """

    @abstractmethod
    def debug(self, event: str | None = None, *args: str, **kw: Any) -> None: ...  # noqa: ANN401

    @abstractmethod
    def info(self, event: str | None = None, *args: str, **kw: Any) -> None: ...  # noqa: ANN401

    @abstractmethod
    def warning(self, event: str | None = None, *args: str, **kw: Any) -> None: ...  # noqa: ANN401

    @abstractmethod
    def error(self, event: str | None = None, *args: str, **kw: Any) -> None: ...  # noqa: ANN401

    @abstractmethod
    def critical(self, event: str | None = None, *args: str, **kw: Any) -> None: ...  # noqa: ANN401

    @abstractmethod
    def exception(
        self,
        event: str | None = None,
        exc_info: BaseException | None = None,
        *args: str,
        **kw: Any,  # noqa: ANN401
    ) -> None: ...

    @abstractmethod
    def log(
        self,
        level: int,
        event: str | None = None,
        *args: str,
        **kw: Any,  # noqa: ANN401
    ) -> None: ...

    fatal = critical

    @abstractmethod
    def bind(self, **new_values: Any) -> TypedLogger: ...  # noqa: ANN401

    @abstractmethod
    def unbind(self, *keys: str) -> TypedLogger: ...

    @abstractmethod
    def try_unbind(self, *keys: str) -> TypedLogger: ...

    @abstractmethod
    def new(self, **new_values: Any) -> TypedLogger: ...  # noqa: ANN401


def get_logger(*args: str, **kwargs: Any) -> TypedLogger:  # noqa: ANN401
    """
    Initializes and returns a structured logger instance.

    Args:
        *args (str): arguments to include in the logger output.
        **kwargs (Any): attributes to include in the logger output.

    Returns:
        TypedLogger: Configured logger instance.

    """
    setup_logging()
    return structlog.stdlib.get_logger(*args, *kwargs)  # type: ignore[return-value]
