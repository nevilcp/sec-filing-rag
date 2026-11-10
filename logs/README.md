# Logs Directory — SEC Filing RAG

This directory contains structured (JSON-per-line) logs emitted by the application.
Each subdirectory maps to a specific subsystem channel; log files are rotated daily
at UTC midnight and retained for **14 days** (~2 weeks of rolling history).

---

## Directory Layout

```
logs/
├── app/           # Application lifecycle (startup, shutdown, config summary)
│   └── app.log
├── api/           # HTTP access log (method, path, status, latency)
│   └── access.log
├── rag/           # One query_complete record per answered query
│   └── pipeline.log
├── phoenix/       # Reserved, nothing writes here yet
│   └── phoenix.log
├── system/        # Reserved, nothing writes here yet
│   └── system.log
└── audit/         # Security-sensitive: input/output guard decisions (NOT on stdout)
    └── audit.log
```

---

## Log Format

Every line is a JSON object. Common fields present on all records:

| Field       | Type   | Description                                    |
|-------------|--------|------------------------------------------------|
| `timestamp` | string | ISO-8601 UTC timestamp                         |
| `level`     | string | `DEBUG` / `INFO` / `WARNING` / `ERROR`         |
| `logger`    | string | Python logger name (`sec_rag.rag`, etc.)       |
| `channel`   | string | Subsystem channel (`rag`, `api`, `audit`, …)   |
| `message`   | string | Human-readable event description               |
| `module`    | string | Python module name                             |
| `function`  | string | Calling function name                          |
| `line`      | int    | Source line number                             |

Additional structured fields are merged into the top-level object by each
channel (see examples below).

---

## Channel Details

### `app` — Application lifecycle

Emitted by `src/api/main.py` at startup and shutdown.

```json
{
  "timestamp": "2026-05-21T06:00:01.123456+00:00",
  "level": "INFO",
  "channel": "app",
  "message": "SEC RAG API starting up",
  "host": "0.0.0.0",
  "port": 8000,
  "phoenix_enabled": true
}
```

### `api` — HTTP access log

One record per HTTP request. Emitted by `_AccessLogMiddleware`.

```json
{
  "timestamp": "2026-05-21T06:01:15.000000+00:00",
  "level": "INFO",
  "channel": "api",
  "message": "http_request",
  "method": "POST",
  "path": "/query",
  "status_code": 200,
  "latency_ms": 1342.7,
  "client": "127.0.0.1"
}
```

### `rag` — Pipeline traces

One `query_complete` record per answered query, emitted by `src/api/routes.py`. Per-node events from `src/agent/nodes.py` go to stdout only, not to this file.

```json
{
  "timestamp": "2026-05-21T06:01:16.500000+00:00",
  "level": "INFO",
  "channel": "rag",
  "message": "query_complete",
  "session_id": "abc-123",
  "query_category": "compliance_check",
  "cache_hit": false,
  "retrieval_grade": "relevant",
  "retrieved_doc_count": 8,
  "rewrite_count": 0,
  "warnings": []
}
```

### `phoenix` — Phoenix / OTEL events

Reserved for Phoenix and OpenTelemetry events. Nothing writes to this channel yet, so `phoenix.log` stays empty. The record below shows the intended shape.

```json
{
  "timestamp": "2026-05-21T06:00:02.000000+00:00",
  "level": "INFO",
  "channel": "phoenix",
  "message": "Phoenix tracing enabled",
  "grpc_endpoint": "http://localhost:4317",
  "http_endpoint": "http://localhost:6006/v1/traces",
  "ui": "http://localhost:6006"
}
```

### `system` — Infrastructure / rate limiter

Reserved for infrastructure events. Nothing writes to this channel yet, so `system.log` stays empty. Rate limiter and embedding messages go to stdout only.

### `audit` — Security decisions ⚠️

Input and output guard verdicts. **This channel does NOT propagate to stdout**
to avoid leaking sensitive prompt content into terminal logs.

```json
{
  "timestamp": "2026-05-21T06:01:15.200000+00:00",
  "level": "INFO",
  "channel": "audit",
  "message": "input_guard_decision",
  "valid": true,
  "reason": null,
  "session_id": "abc-123"
}
```

---

## Retention Policy

| Setting                | Value                          |
|------------------------|--------------------------------|
| Rotation schedule      | Daily at UTC midnight          |
| Retained history       | 14 days (configurable via `LOG_ROTATION_BACKUP_COUNT` in `config.py`) |
| Archive naming         | `<base>.log.YYYY-MM-DD`        |

To adjust retention, modify `LOG_ROTATION_BACKUP_COUNT` in `src/core/config.py`
and restart the API server.

---

## Git Policy

Log **files** (`.log`, `.log.YYYY-MM-DD`) are excluded from version control
via `.gitignore`.  The subdirectory `.gitkeep` placeholders are committed so
the directory structure is preserved in the repository.

---

## Querying Logs

Since every line is valid JSON, you can use standard tools:

```bash
# All ERROR-level events today
jq 'select(.level == "ERROR")' logs/app/app.log

# All audit blocks (invalid queries)
jq 'select(.message == "input_guard_decision" and .valid == false)' logs/audit/audit.log

# API requests slower than 2 seconds
jq 'select(.latency_ms > 2000)' logs/api/access.log

# RAG queries that triggered a rewrite
jq 'select(.rewrite_count > 0)' logs/rag/pipeline.log

# Tail all channels simultaneously
tail -f logs/**/*.log | jq .
```
