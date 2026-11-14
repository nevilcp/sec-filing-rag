"""Uvicorn server entry point and Phoenix observability launcher."""

import os

import uvicorn

from src.core import config


def main() -> None:
    """Register Phoenix tracing before importing the model stack, then start the API server."""
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
    """Start Phoenix on the configured port unless CLI arguments override it."""
    import sys

    from phoenix.server.main import main as phoenix_main

    if len(sys.argv) == 1:
        sys.argv.extend(["serve", "--port", str(config.PHOENIX_PORT)])
    sys.exit(phoenix_main())


if __name__ == "__main__":
    main()
