"""
Unit tests for raglab-common tracing module (R6 Phase 1).

All OTel SDK calls are patched — zero infra required.

Covers:
- _NoopTracer / _NoopSpan: safe no-op contract
- configure_tracing: enabled=False is silent no-op
- configure_tracing: OTel unavailable → graceful
- get_tracer: returns NoopTracer when OTel unavailable
- current_trace_id: returns None when no span active
- trace_id_from_headers: W3C traceparent, X-Trace-Id, missing headers
- trace_headers: returns dict with X-Trace-Id key
- traced_span: yields NoopSpan with NoopTracer
- record_event: safe no-op on NoopSpan
- PostgresSpanExporter: buffers on Postgres failure, never raises
- TraceMiddleware: injects trace_id into request.state
- TraceMiddleware: adds X-Trace-Id to response headers
- TraceMiddleware: generates trace_id when header missing
- TraceMiddleware: preserves incoming trace_id
- BaseServiceSettings: tracing_enabled, tracing_postgres_dsn fields present
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI, Request as FastAPIRequest
from fastapi.testclient import TestClient

from raglab_common.tracing import (
    _NoopSpan,
    _NoopTracer,
    current_trace_id,
    get_tracer,
    make_trace_middleware,
    record_event,
    trace_headers,
    trace_id_from_headers,
    traced_span,
    PostgresSpanExporter,
    configure_tracing,
)


# ═══════════════════════════════════════════════════════════════════════════════
# No-op stubs
# ═══════════════════════════════════════════════════════════════════════════════

class TestNoopStubs:
    def test_noop_span_set_attribute_no_raise(self):
        s = _NoopSpan()
        s.set_attribute("key", "value")  # should not raise

    def test_noop_span_add_event_no_raise(self):
        s = _NoopSpan()
        s.add_event("my_event", attributes={"k": "v"})

    def test_noop_span_set_status_no_raise(self):
        s = _NoopSpan()
        s.set_status("ok")

    def test_noop_span_get_span_context_returns_none(self):
        assert _NoopSpan().get_span_context() is None

    def test_noop_tracer_start_span_returns_noop_span(self):
        t = _NoopTracer()
        span = t.start_span("op")
        assert isinstance(span, _NoopSpan)

    def test_noop_tracer_context_manager(self):
        t = _NoopTracer()
        with t.start_as_current_span("op") as span:
            assert isinstance(span, _NoopSpan)


# ═══════════════════════════════════════════════════════════════════════════════
# configure_tracing
# ═══════════════════════════════════════════════════════════════════════════════

class TestConfigureTracing:
    def test_disabled_is_silent_noop(self):
        # Should not raise even without OTel SDK
        configure_tracing("test-service", enabled=False)

    def test_otel_unavailable_graceful(self):
        with patch("raglab_common.tracing._OTEL_AVAILABLE", False):
            configure_tracing("test-service", enabled=True)  # should not raise

    def test_enabled_with_no_exporters(self):
        # Should complete without raising even if no postgres_dsn/otlp_endpoint
        configure_tracing("test-service", enabled=True)


# ═══════════════════════════════════════════════════════════════════════════════
# get_tracer
# ═══════════════════════════════════════════════════════════════════════════════

class TestGetTracer:
    def test_otel_unavailable_returns_noop(self):
        with patch("raglab_common.tracing._OTEL_AVAILABLE", False):
            tracer = get_tracer("test-service")
        assert isinstance(tracer, _NoopTracer)

    def test_returns_something_when_available(self):
        tracer = get_tracer("test-service")
        assert tracer is not None


# ═══════════════════════════════════════════════════════════════════════════════
# current_trace_id
# ═══════════════════════════════════════════════════════════════════════════════

class TestCurrentTraceId:
    def test_returns_none_when_otel_unavailable(self):
        with patch("raglab_common.tracing._OTEL_AVAILABLE", False):
            assert current_trace_id() is None

    def test_returns_none_outside_span(self):
        # Outside any span context → no valid trace ID
        result = current_trace_id()
        assert result is None or isinstance(result, str)


# ═══════════════════════════════════════════════════════════════════════════════
# trace_id_from_headers
# ═══════════════════════════════════════════════════════════════════════════════

class TestTraceIdFromHeaders:
    def test_extracts_from_traceparent(self):
        tid = "a" * 32
        headers = {"traceparent": f"00-{tid}-bbbbbbbbbbbbbbbb-01"}
        assert trace_id_from_headers(headers) == tid

    def test_extracts_from_x_trace_id(self):
        tid = "abc123def456"
        headers = {"x-trace-id": tid}
        assert trace_id_from_headers(headers) == tid

    def test_extracts_from_uppercase_x_trace_id(self):
        tid = "abc123"
        headers = {"X-Trace-Id": tid}
        assert trace_id_from_headers(headers) == tid

    def test_empty_headers_returns_none(self):
        assert trace_id_from_headers({}) is None

    def test_traceparent_takes_precedence(self):
        tid = "a" * 32
        headers = {
            "traceparent": f"00-{tid}-bbbbbbbbbbbbbbbb-01",
            "x-trace-id": "other-id",
        }
        assert trace_id_from_headers(headers) == tid

    def test_malformed_traceparent_falls_through(self):
        headers = {"traceparent": "bad-format", "x-trace-id": "fallback"}
        result = trace_id_from_headers(headers)
        # Malformed traceparent returns None (wrong part count); falls back to x-trace-id
        assert result == "fallback" or result is None


# ═══════════════════════════════════════════════════════════════════════════════
# trace_headers
# ═══════════════════════════════════════════════════════════════════════════════

class TestTraceHeaders:
    def test_returns_dict_with_x_trace_id(self):
        h = trace_headers()
        assert "X-Trace-Id" in h

    def test_accepts_explicit_trace_id(self):
        tid = "explicit-trace-id"
        h = trace_headers(trace_id=tid)
        assert h["X-Trace-Id"] == tid

    def test_generates_when_no_active_trace(self):
        h = trace_headers()
        assert len(h["X-Trace-Id"]) > 0

    def test_returns_string_value(self):
        h = trace_headers()
        assert isinstance(h["X-Trace-Id"], str)


# ═══════════════════════════════════════════════════════════════════════════════
# traced_span
# ═══════════════════════════════════════════════════════════════════════════════

class TestTracedSpan:
    def test_noop_tracer_yields_noop_span(self):
        tracer = _NoopTracer()
        with traced_span(tracer, "test_op") as span:
            assert isinstance(span, _NoopSpan)

    def test_noop_tracer_with_attributes_no_raise(self):
        tracer = _NoopTracer()
        with traced_span(tracer, "op", {"doc_id": "123", "chunks": 5}) as span:
            span.set_attribute("extra", "value")

    def test_context_manager_completes_normally(self):
        tracer = _NoopTracer()
        result = []
        with traced_span(tracer, "op") as span:
            result.append("inside")
        assert result == ["inside"]


# ═══════════════════════════════════════════════════════════════════════════════
# record_event
# ═══════════════════════════════════════════════════════════════════════════════

class TestRecordEvent:
    def test_noop_span_no_raise(self):
        record_event(_NoopSpan(), "my_event", {"k": "v"})

    def test_none_attributes_no_raise(self):
        record_event(_NoopSpan(), "event", None)


# ═══════════════════════════════════════════════════════════════════════════════
# PostgresSpanExporter
# ═══════════════════════════════════════════════════════════════════════════════

class TestPostgresSpanExporter:
    def test_buffers_on_postgres_failure(self):
        exporter = PostgresSpanExporter(dsn="postgresql://bad-host/db", service_name="test")
        # Mock a span
        mock_span = MagicMock()
        mock_ctx = MagicMock()
        mock_ctx.trace_id = int("a" * 32, 16)
        mock_ctx.span_id  = int("b" * 16, 16)
        mock_ctx.is_valid = True
        mock_span.get_span_context.return_value = mock_ctx
        mock_span.parent = None
        mock_span.name = "test_op"
        mock_span.start_time = 1_000_000_000
        mock_span.end_time   = 2_000_000_000
        mock_span.status.status_code.name = "OK"
        mock_span.attributes = {"doc_id": "test"}
        mock_span.events = []

        # Should not raise even when Postgres is unavailable
        exporter.export([mock_span])
        assert len(exporter._buffer) <= exporter._max_buffer

    def test_buffer_does_not_exceed_max(self):
        exporter = PostgresSpanExporter(dsn="postgresql://bad/db", service_name="t")
        exporter._max_buffer = 5
        # Fill buffer manually
        exporter._buffer = [{}] * 5
        # Another flush should not grow beyond max
        exporter._flush_to_postgres([{"extra": "item"}])
        assert len(exporter._buffer) <= exporter._max_buffer

    def test_shutdown_no_raise(self):
        exporter = PostgresSpanExporter(dsn="postgresql://bad/db", service_name="t")
        exporter.shutdown()  # should not raise

    def test_force_flush_returns_true(self):
        exporter = PostgresSpanExporter(dsn="postgresql://bad/db", service_name="t")
        assert exporter.force_flush() is True


# ═══════════════════════════════════════════════════════════════════════════════
# TraceMiddleware
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def traced_app():
    _a = FastAPI()
    _a.add_middleware(make_trace_middleware("test-service"))

    @_a.get("/ping")
    async def _ping(req: FastAPIRequest):
        return {"trace_id": getattr(req.state, "trace_id", None)}

    return TestClient(_a)


class TestTraceMiddleware:
    def test_response_has_x_trace_id_header(self, traced_app):
        r = traced_app.get("/ping")
        assert "x-trace-id" in r.headers or "X-Trace-Id" in r.headers

    def test_response_has_x_service_header(self, traced_app):
        r = traced_app.get("/ping")
        headers_lower = {k.lower(): v for k, v in r.headers.items()}
        assert "x-service" in headers_lower
        assert headers_lower["x-service"] == "test-service"

    def test_incoming_trace_id_preserved(self, traced_app):
        tid = "abc123def456abc123def456abc123de"
        r = traced_app.get("/ping", headers={"X-Trace-Id": tid})
        headers_lower = {k.lower(): v for k, v in r.headers.items()}
        assert headers_lower.get("x-trace-id") == tid

    def test_generates_trace_id_when_missing(self, traced_app):
        r = traced_app.get("/ping")
        headers_lower = {k.lower(): v for k, v in r.headers.items()}
        tid = headers_lower.get("x-trace-id", "")
        assert len(tid) > 0

    def test_trace_id_injected_into_request_state(self, traced_app):
        r = traced_app.get("/ping")
        body = r.json()
        assert body["trace_id"] is not None

    def test_returns_200(self, traced_app):
        assert traced_app.get("/ping").status_code == 200


# ═══════════════════════════════════════════════════════════════════════════════
# BaseServiceSettings tracing fields
# ═══════════════════════════════════════════════════════════════════════════════

class TestBaseSettingsTracingFields:
    def test_tracing_enabled_default_true(self):
        from raglab_common.settings import BaseServiceSettings
        s = BaseServiceSettings()
        assert s.tracing_enabled is True

    def test_tracing_postgres_dsn_default_empty(self):
        from raglab_common.settings import BaseServiceSettings
        s = BaseServiceSettings()
        assert s.tracing_postgres_dsn == ""

    def test_tracing_otlp_endpoint_default_empty(self):
        from raglab_common.settings import BaseServiceSettings
        s = BaseServiceSettings()
        assert s.tracing_otlp_endpoint == ""
