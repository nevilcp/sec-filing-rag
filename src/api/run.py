"""
api/run.py

Uvicorn server entry point and Phoenix observability launcher.

Responsibilities:
  - Register the Arize Phoenix OTel TracerProvider before uvicorn imports
    any instrumented library, preventing a provider-registration race.
  - Launch the Uvicorn ASGI server for the FastAPI application.
  - Provide a companion entry point for starting the Phoenix server.

Not responsible for:
  - FastAPI app configuration (see api/main.py).
  - Logging setup (see core/logging.py).

Dependencies:
  - uvicorn
  - openinference-instrumentation-langchain (optional, when PHOENIX_ENABLED)
  - arize-phoenix (optional, when PHOENIX_ENABLED)
"""

import os

import uvicorn

from src.core import config


def main() -> None:
    """Start the SEC RAG API server, optionally enabling Phoenix tracing.

    Phoenix OTel registration must happen here — before uvicorn.run() —
    so the TracerProvider is in place before the transformers library is
    imported transitively by the LLM stack. Transformers locks the global
    OTel provider on import, so late registration would be silently ignored.
    """
    if os.getenv("PHOENIX_ENABLED", "").lower() == "true":
        from openinference.instrumentation.langchain import LangChainInstrumentor
        from phoenix.otel import register

        os.environ.setdefault(
            "PHOENIX_COLLECTOR_ENDPOINT",
            f"http://{config.PHOENIX_HOST}:{config.PHOENIX_PORT}",
        )
        register(project_name="sec-filing-rag", protocol="http/protobuf", verbose=False)
        LangChainInstrumentor().instrument()

    uvicorn.run(
        "src.api.main:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=os.getenv("UVICORN_RELOAD", "false").lower() == "true",
    )


def run_phoenix() -> None:
    """CLI entry point: start the Arize Phoenix observability server.

    Passes the configured port via sys.argv so the Phoenix CLI picks it
    up without requiring manual flag injection by the caller.
    """
    import sys

    from phoenix.server.main import main as phoenix_main

    if len(sys.argv) == 1:
        sys.argv.extend(["serve", "--port", str(config.PHOENIX_PORT)])
    sys.exit(phoenix_main())


if __name__ == "__main__":
    main()
