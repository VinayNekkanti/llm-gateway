"""Logs and metrics.

Logs: structlog, one JSON line per request (or readable colored lines with LOG_FORMAT=console).
Metrics: Prometheus counters and histograms, served at GET /metrics.

Metric labels are kept to small, fixed sets (route templates, model names from config) -
a label with unbounded values, like the raw URL path, would create a new time series for
every random URL someone requests.
"""

import logging
import os
import sys

import structlog
from prometheus_client import Counter, Histogram

HTTP_REQUESTS = Counter(
    "llmgw_http_requests_total", "HTTP requests handled", ["route", "method", "status"]
)
HTTP_LATENCY = Histogram(
    "llmgw_http_request_duration_seconds",
    "Time to first response byte (for streams: until streaming starts)",
    ["route"],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60),
)
CACHE_LOOKUPS = Counter("llmgw_cache_lookups_total", "Cache results for chat requests", ["result"])
UPSTREAM_ATTEMPTS = Counter(
    "llmgw_upstream_attempts_total",
    "Calls to providers, including retries",
    ["provider", "model", "outcome"],
)
UPSTREAM_LATENCY = Histogram(
    "llmgw_upstream_duration_seconds",
    "Provider call duration (non-streaming: full answer; streaming: until headers)",
    ["provider"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120),
)
FALLBACKS = Counter(
    "llmgw_fallbacks_total", "Requests moved to a fallback model", ["from_model", "to_model"]
)
TOKENS = Counter("llmgw_tokens_total", "Tokens used, by model", ["model", "kind"])
COST_USD = Counter("llmgw_cost_usd_total", "Estimated spend in US dollars", ["model"])
RATE_LIMITED = Counter("llmgw_rate_limited_total", "Requests rejected by the rate limiter")


def setup_logging() -> None:
    """JSON logs by default (for log systems); LOG_FORMAT=console for local reading."""
    level = getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper(), logging.INFO)
    console = os.environ.get("LOG_FORMAT", "json").lower() == "console"
    renderer = structlog.dev.ConsoleRenderer() if console else structlog.processors.JSONRenderer()
    structlog.configure(
        processors=[
            # Adds request_id (and anything else bound for this request) to every line
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        # Not cached, so tests can capture log lines; the cost is negligible
        cache_logger_on_first_use=False,
    )
