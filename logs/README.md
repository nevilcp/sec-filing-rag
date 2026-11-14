# Runtime logs

The API creates these directories at startup and writes one JSON record per line.

| File | Records |
| --- | --- |
| `app/app.log` | API startup and shutdown |
| `api/access.log` | HTTP method, path, status, latency, and client address |
| `rag/pipeline.log` | One `query_complete` event per completed query |
| `audit/audit.log` | Input and output guard decisions; excluded from stdout |
| `phoenix/phoenix.log` | Reserved; currently no events are emitted to this channel |
| `system/system.log` | Reserved; currently no events are emitted to this channel |

Module-level events, including graph node, embedding, and rate-limit messages, go to
stdout. They are not routed to the reserved Phoenix or system files.

## Format and retention

Records contain `timestamp`, `level`, `logger`, `channel`, `message`, `module`, `function`,
and `line`. Timestamps are UTC. Structured fields from `extra` are merged into the record;
exception details appear under `exception` when present.

Files rotate at UTC midnight, with up to 14 archived files per channel by default.
Archives use `<name>.log.YYYY-MM-DD`. Change `LOG_ROTATION_WHEN` or
`LOG_ROTATION_BACKUP_COUNT` in [configuration](../src/core/config.py) and restart the API
to adjust rotation. Generated logs and archives are ignored by Git.

## Queries

```bash
# Errors.
jq 'select(.level == "ERROR")' logs/app/app.log

# Rejected queries.
jq 'select(.message == "input_guard_decision" and .valid == false)' logs/audit/audit.log

# Requests over two seconds.
jq 'select(.latency_ms > 2000)' logs/api/access.log

# Queries with rewrites.
jq 'select(.rewrite_count > 0)' logs/rag/pipeline.log
```
