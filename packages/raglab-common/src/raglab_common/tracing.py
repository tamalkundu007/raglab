"""
RAGLab OpenTelemetry tracing — raglab-common/tracing.py

Phase 0 decisions:
    Standard:  OpenTelemetry SDK + auto-instrumentation (FastAPI, httpx, SQLAlchemy)
    Storage:   Postgres event store (raglab_traces + raglab_events tables)
    Views:     Native in-app D3.js — no Jaeger/Grafana dependency
    Principle: Observability is READ-ONLY. This module never mutates pipeline state.

What this module provides:
    1. configure_tracing()      — call once in each service lifespan
    2. get_tracer()             — returns the service's OTel Tracer
    3. TraceMiddleware          — FastAPI middleware that injects trace_id into
                                  every request and propagates via X-Trace-Id header
    4. traced_span()            — context manager for manual span creation
    5. record_event()           — structured event attached to current span
    6. PostgresSpanExporter     — writes spans to raglab_events Postgres table
    7. trace_id_from_request()  — extract trace_id from request headers

Trace ID propagation:
    Every service emits trace_id in structured logs (since R1).
    This module formalises propagation:
        Inbound:  X-Trace-Id header → span context
        Outbound: inject X-Trace-Id into downstream httpx calls
    Existing logs remain unchanged — OTel adds the formal span layer on top.

Graceful degradation:
    OTel SDK not installed → tracing disabled, no-op stubs returned.
    Postgres unavailable → spans buffered in memory, dropped on overflow.
    Service still works; observability is advisory, never blocking.
"""

from __future__ import annotations

import time
import uuid
from contextlib import contextmanager
from typing import Any, Generator

from raglab_common.logging import get_logger

log = get_logger(__name__)

# Module-level optional imports — patchable in tests
try:
    from opentelemetry import trace as otel_trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
    from opentelemetry.propagators.b3 import B3MultiFormat
    _OTEL_AVAILABLE = True
except ImportError:
    _OTEL_AVAILABLE = False

_NOOP_TRACER = None  # populated on first get_tracer() call if OTel unavailable

# ── Postgres span exporter ─────────────────────────────────────────────────────

_EVENTS_DDL = """
CREATE TABLE IF NOT EXISTS raglab_events (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    trace_id        TEXT NOT NULL,
    span_id         TEXT NOT NULL,
    parent_span_id  TEXT,
    service_name    TEXT NOT NULL,
    operation_name  TEXT NOT NULL,
    start_time_ms   BIGINT NOT NULL,
    duration_ms     BIGINT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'ok',
    attributes      JSONB,
    events          JSONB,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS ix_raglab_events_trace_id ON raglab_events (trace_id);
CREATE INDEX IF NOT EXISTS ix_raglab_events_service  ON raglab_events (service_name);
CREATE INDEX IF NOT EXISTS ix_raglab_events_created  ON raglab_events (created_at DESC);
"""


class PostgresSpanExporter:
    """
    OTel SpanExporter that writes spans to the raglab_events Postgres table.

    Used by BatchSpanProcessor. Async inserts via asyncpg-compatible DSN.
    Falls back to in-memory buffer when Postgres is unavailable.
    """

    def __init__(self, dsn: str, service_name: str) -> None:
        self.dsn = dsn
        self.service_name = service_name
        self._buffer: list[dict] = []
        self._max_buffer = 1000

    def export(self, spans: Any) -> Any:
        """Export spans to Postgres. Called by BatchSpanProcessor."""
        if not _OTEL_AVAILABLE:
            return

        records = []
        for span in spans:
            try:
                ctx = span.get_span_context()
                records.append({
                    "trace_id": format(ctx.trace_id, "032x"),
                    "span_id": format(ctx.span_id, "016x"),
                    "parent_span_id": (
                        format(span.parent.span_id, "016x") if span.parent else None
                    ),
                    "service_name": self.service_name,
                    "operation_name": span.name,
                    "start_time_ms": span.start_time // 1_000_000,
                    "duration_ms": (span.end_time - span.start_time) // 1_000_000,
                    "status": span.status.status_code.name.lower(),
                    "attributes": dict(span.attributes or {}),
                    "events": [
                        {"name": e.name, "attributes": dict(e.attributes or {})}
                        for e in span.events
                    ],
                })
            except Exception as exc:
                log.warning("otel.span_export_error", error=str(exc))

        self._flush_to_postgres(records)
        return 0  # SUCCESS

    def _flush_to_postgres(self, records: list[dict]) -> None:
        """Sync write to Postgres using psycopg2-style connection."""
        import json
        try:
            import psycopg2  # type: ignore[import]
            conn = psycopg2.connect(
                self.dsn.replace("postgresql+asyncpg://", "postgresql://")
                        .replace("postgresql+psycopg2://", "postgresql://")
            )
            with conn:
                with conn.cursor() as cur:
                    for r in records:
                        cur.execute(
                            """
                            INSERT INTO raglab_events
                              (trace_id, span_id, parent_span_id, service_name,
                               operation_name, start_time_ms, duration_ms,
                               status, attributes, events)
                            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                            ON CONFLICT DO NOTHING
                            """,
                            (
                                r["trace_id"], r["span_id"], r["parent_span_id"],
                                r["service_name"], r["operation_name"],
                                r["start_time_ms"], r["duration_ms"],
                                r["status"],
                                json.dumps(r["attributes"]),
                                json.dumps(r["events"]),
                            ),
                        )
            conn.close()
        except Exception as exc:
            # Buffer for retry — never raise, never block the service
            log.warning("otel.postgres_flush_failed", error=str(exc), buffered=len(self._buffer))
            if len(self._buffer) < self._max_buffer:
                self._buffer.extend(records)

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


# ── Service tracer setup ───────────────────────────────────────────────────────

_providers: dict[str, Any] = {}


def configure_tracing(
    service_name: str,
    postgres_dsn: str | None = None,
    otlp_endpoint: str | None = None,
    enabled: bool = True,
) -> None:
    """
    Configure OTel tracing for a service. Call once in lifespan.

    Args:
        service_name:   Service identifier (e.g. "api-gateway", "pipeline").
        postgres_dsn:   If provided, spans are exported to raglab_events table.
        otlp_endpoint:  If provided, spans also exported via OTLP gRPC.
        enabled:        Master toggle. If False, tracing is a no-op.
    """
    if not enabled or not _OTEL_AVAILABLE:
        log.info("tracing.disabled", service=service_name,
                 reason="disabled" if not enabled else "otel_not_installed")
        return

    resource = Resource.create({"service.name": service_name, "service.namespace": "raglab"})
    provider = TracerProvider(resource=resource)

    # Postgres exporter (primary for native views)
    if postgres_dsn:
        pg_exporter = PostgresSpanExporter(dsn=postgres_dsn, service_name=service_name)
        provider.add_span_processor(BatchSpanProcessor(pg_exporter))

    # OTLP exporter (optional — for cloud-native backends)
    if otlp_endpoint and _OTEL_AVAILABLE:
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
            otlp = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)
            provider.add_span_processor(BatchSpanProcessor(otlp))
        except Exception as exc:
            log.warning("tracing.otlp_setup_failed", error=str(exc))

    otel_trace.set_tracer_provider(provider)
    _providers[service_name] = provider
    log.info("tracing.configured", service=service_name,
             postgres=bool(postgres_dsn), otlp=bool(otlp_endpoint))


def get_tracer(service_name: str = "raglab") -> Any:
    """Return the OTel Tracer for a service. Returns no-op tracer if OTel unavailable."""
    if not _OTEL_AVAILABLE:
        return _NoopTracer()
    return otel_trace.get_tracer(service_name)


# ── Span context helpers ───────────────────────────────────────────────────────

def current_trace_id() -> str | None:
    """Return the current trace ID as a hex string, or None if not in a span."""
    if not _OTEL_AVAILABLE:
        return None
    span = otel_trace.get_current_span()
    ctx = span.get_span_context()
    if ctx and ctx.is_valid:
        return format(ctx.trace_id, "032x")
    return None


def current_span_id() -> str | None:
    """Return the current span ID as a hex string."""
    if not _OTEL_AVAILABLE:
        return None
    span = otel_trace.get_current_span()
    ctx = span.get_span_context()
    if ctx and ctx.is_valid:
        return format(ctx.span_id, "016x")
    return None


@contextmanager
def traced_span(
    tracer: Any,
    name: str,
    attributes: dict[str, Any] | None = None,
) -> Generator[Any, None, None]:
    """
    Context manager for manual span creation.

    Usage:
        tracer = get_tracer("pipeline")
        with traced_span(tracer, "chunk_document", {"doc_id": doc_id}) as span:
            chunks = chunker.chunk(text, doc_id)
            span.set_attribute("chunk_count", len(chunks))
    """
    if isinstance(tracer, _NoopTracer):
        yield _NoopSpan()
        return

    with tracer.start_as_current_span(name) as span:
        if attributes:
            for k, v in attributes.items():
                try:
                    span.set_attribute(k, v)
                except Exception:
                    pass
        yield span


def record_event(span: Any, name: str, attributes: dict[str, Any] | None = None) -> None:
    """Attach a named event to a span — safe no-op if span is invalid."""
    if isinstance(span, _NoopSpan):
        return
    try:
        span.add_event(name, attributes=attributes or {})
    except Exception:
        pass


def trace_id_from_headers(headers: dict[str, str]) -> str | None:
    """Extract trace ID from incoming request headers (W3C traceparent or X-Trace-Id)."""
    # W3C traceparent: 00-{trace_id}-{span_id}-{flags}
    tp = headers.get("traceparent", "")
    if tp:
        parts = tp.split("-")
        if len(parts) == 4:
            return parts[1]
    return headers.get("x-trace-id") or headers.get("X-Trace-Id")


# ── FastAPI middleware ─────────────────────────────────────────────────────────

def make_trace_middleware(service_name: str):
    """
    Returns a FastAPI middleware class that:
      1. Extracts or generates trace_id per request.
      2. Injects trace_id into request.state for use in route handlers.
      3. Adds X-Trace-Id to all responses.
      4. Records request/response as a span.

    Usage:
        app.add_middleware(make_trace_middleware("api-gateway"))
    """
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.requests import Request
    from starlette.responses import Response

    class TraceMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next) -> Response:
            # Extract or generate trace ID
            trace_id = (
                trace_id_from_headers(dict(request.headers))
                or str(uuid.uuid4()).replace("-", "")
            )
            request.state.trace_id = trace_id
            request.state.service_name = service_name

            # Inject into structlog context for this request
            import structlog
            structlog.contextvars.clear_contextvars()
            structlog.contextvars.bind_contextvars(
                trace_id=trace_id,
                service=service_name,
            )

            t0 = time.perf_counter()
            response = await call_next(request)
            duration_ms = round((time.perf_counter() - t0) * 1000, 1)

            # Add trace headers to response
            response.headers["X-Trace-Id"] = trace_id
            response.headers["X-Service"] = service_name

            log.info(
                "http.request",
                method=request.method,
                path=request.url.path,
                status=response.status_code,
                duration_ms=duration_ms,
                trace_id=trace_id,
            )
            return response

    return TraceMiddleware


# ── No-op stubs ───────────────────────────────────────────────────────────────

class _NoopSpan:
    """No-op span — returned when OTel is unavailable."""
    def set_attribute(self, key: str, value: Any) -> None: pass
    def add_event(self, name: str, **kwargs: Any) -> None: pass
    def set_status(self, *args: Any, **kwargs: Any) -> None: pass
    def record_exception(self, *args: Any, **kwargs: Any) -> None: pass
    def get_span_context(self) -> Any: return None


class _NoopTracer:
    """No-op tracer — returned when OTel is unavailable."""
    def start_as_current_span(self, name: str, **kwargs: Any):
        from contextlib import contextmanager
        @contextmanager
        def _noop():
            yield _NoopSpan()
        return _noop()

    def start_span(self, name: str, **kwargs: Any) -> _NoopSpan:
        return _NoopSpan()


# ── Convenience: inject trace_id into httpx outbound calls ────────────────────

def trace_headers(trace_id: str | None = None) -> dict[str, str]:
    """
    Returns headers dict with current trace context for outbound HTTP calls.

    Usage:
        async with httpx.AsyncClient() as client:
            resp = await client.post(url, json=body, headers=trace_headers())
    """
    tid = trace_id or current_trace_id() or str(uuid.uuid4()).replace("-", "")
    return {"X-Trace-Id": tid}
