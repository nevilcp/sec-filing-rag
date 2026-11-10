"""Uvicorn server entry point and Phoenix observability launcher.

Responsibilities:
  - Register the Arize Phoenix OTel TracerProvider before uvicorn
    imports any instrumented library, which avoids a provider
    registration race.
  - Launch the Uvicorn ASGI server for the FastAPI application.
  - Provide a companion entry point that starts the Phoenix server.

Not responsible for:
  - FastAPI app configuration (see api/main.py).
  - Logging setup (see core/logging.py).

Dependencies:
  - openinference-instrumentation-langchain and arize-phoenix, needed
    only when PHOENIX_ENABLED is true.
"""

import os

import uvicorn

from src.core import config


def main() -> None:
    """Start the SEC RAG API server, with Phoenix tracing if enabled.

    Phoenix OTel registration must happen here, before uvicorn.run(),
    so the TracerProvider is in place before the transformers library is
    imported by the LLM stack. Transformers locks the global OTel
    provider on import, so late registration would be silently ignored.
    """
    if config.PHOENIX_ENABLED:
        from openinference.instrumentation.langchain import LangChainInstrumentor
        from phoenix.otel import register

        os.environ.setdefault(
            "PHOENIX_COLLECTOR_ENDPOINT",
            f"http://{config.PHOENIX_HOST}:{config.PHOENIX_PORT}",
        )
        register(project_name=config.PHOENIX_PROJECT_NAME, protocol="http/protobuf", verbose=False)
        LangChainInstrumentor().instrument()

    uvicorn.run(
        "src.api.main:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=config.UVICORN_RELOAD,
    )


def run_phoenix() -> None:
    """Start the Arize Phoenix observability server as a CLI command.

    Adds the configured port to sys.argv when no arguments were given,
    so the Phoenix CLI picks it up.
    """
    import sys

    from phoenix.server.main import main as phoenix_main

    if len(sys.argv) == 1:
        sys.argv.extend(["serve", "--port", str(config.PHOENIX_PORT)])
    sys.exit(phoenix_main())


if __name__ == "__main__":
    main()
