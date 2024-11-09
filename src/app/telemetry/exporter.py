import logging  # noqa: D100
from collections.abc import Sequence

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

from .custom_logger import get_logger

logger = get_logger()


class StructlogConsoleExporter(SpanExporter):
    """Exports spans using structlog for human-readable console output."""

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        """
        Override the export method to forward spans to structlog.

        Args:
            spans (Sequence[ReadableSpan]): batched spans.

        Returns:
            SpanExportResult: flag indicating the export status.

        """
        for span in spans:
            log_level = self.determine_log_level(span)
            status = span.status.status_code.name if span.status else None
            logger.log(
                log_level,
                span.name,
                trace_id=f"{span.context.trace_id:016x}" if span.context else None,
                span_id=f"{span.context.span_id:016x}" if span.context else None,
                parent_id=f"{span.parent.span_id:016x}" if span.parent else None,
                start_time=span.start_time,
                end_time=span.end_time,
                status_code=status,
                events=span.events,
                **(span.attributes or {}),
            )
        return SpanExportResult.SUCCESS

    @staticmethod
    def determine_log_level(span: ReadableSpan) -> int:
        """Determine log level based on span status and severity attribute."""
        severity = span.attributes.get("severity") if span.attributes else "not_set"
        status_code = span.status.status_code if span.status else None

        if (
            severity == "CRITICAL"
            or (status_code and status_code.name == "ERROR")
            or severity == "ERROR"
        ):
            return logging.ERROR
        if severity == "WARNING":
            return logging.WARNING
        if severity == "INFO" or (status_code and status_code.name == "OK"):
            return logging.INFO
        if severity == "DEBUG":
            return logging.DEBUG
        return logging.INFO
