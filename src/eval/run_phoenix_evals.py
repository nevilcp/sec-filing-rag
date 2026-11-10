"""Arize Phoenix evaluation runner using a Groq-hosted judge model.

Responsibilities:
  - Connect to the local Phoenix server and fetch recorded traces.
  - Run answer-relevance classification over CHAIN-level spans.
  - Log evaluation results back to Phoenix as span annotations.

Not responsible for:
  - Defining evaluation prompt templates (see eval/prompts.py).
  - Starting the Phoenix server (see api/run.py or the phoenix CLI).

Dependencies:
  - A running Phoenix server with recorded traces.
  - GROQ_API_KEY must be set in the environment.
"""

import asyncio
import os

from phoenix.client import Client
from phoenix.evals import LLM, ClassificationEvaluator, async_evaluate_dataframe
from phoenix.evals.utils import to_annotation_dataframe
from phoenix.trace import suppress_tracing

from src.core import config
from src.core.logging import configure_logging, get_logger
from src.eval.prompts import (
    ANSWER_RELEVANCE_TEMPLATE,
)

logger = get_logger(__name__)


async def main_async() -> None:
    """Fetch traces from Phoenix and run LLM-as-a-judge evaluations.

    Connects to the local Phoenix instance, retrieves all CHAIN spans
    from the Phoenix project, runs an answer-relevance classification
    pass with config.EVAL_MODEL and logs the scored annotations back to
    Phoenix. It returns early, after logging, if there are no traces or
    no CHAIN spans.

    Concurrency is limited by config.EVAL_CONCURRENCY to respect Groq
    free-tier rate limits. Context relevance and faithfulness
    evaluations are omitted because they need
    attributes.retrieval.documents on the CHAIN span, which depends on
    the instrumentation configuration.

    Raises:
        KeyError: If GROQ_API_KEY is not set.
    """
    base_url = f"http://{config.PHOENIX_HOST}:{config.PHOENIX_PORT}"
    logger.info("Connecting to Phoenix Client at %s...", base_url)
    px_client = Client(base_url=base_url)

    logger.info("Fetching traces...")
    df = px_client.spans.get_spans_dataframe(project_name=config.PHOENIX_PROJECT_NAME)

    if df.empty:
        logger.warning(
            "No traces found in the '%s' project. Run the app and generate traces first.",
            config.PHOENIX_PROJECT_NAME,
        )
        return

    parent_spans = df[df["span_kind"] == "CHAIN"]
    if parent_spans.empty:
        logger.warning("No CHAIN spans found. Unable to evaluate.")
        return

    logger.info("Initializing Groq LLM Evaluator...")
    llm = LLM(
        provider="openai",
        model=config.EVAL_MODEL,
        base_url=config.EVAL_BASE_URL,
        api_key=os.environ["GROQ_API_KEY"],
    )

    logger.info("Setting up Evaluator for Answer Relevance...")
    relevance_evaluator = ClassificationEvaluator(
        name="answer_relevance",
        prompt_template=ANSWER_RELEVANCE_TEMPLATE,
        llm=llm,
        choices={"relevant": 1.0, "irrelevant": 0.0},
    )

    evaluators = [relevance_evaluator]

    logger.info("Running evaluations...")
    with suppress_tracing():
        results_df = await async_evaluate_dataframe(
            dataframe=parent_spans,
            evaluators=evaluators,
            concurrency=config.EVAL_CONCURRENCY,
        )

    logger.info("Logging results to Phoenix...")
    evaluations = to_annotation_dataframe(dataframe=results_df)
    px_client.spans.log_span_annotations_dataframe(dataframe=evaluations)
    logger.info("Evaluation completed successfully.")


def main() -> None:
    """Run the async evaluation loop as a CLI command.

    Configures logging first, because nothing else does for this entry
    point and the progress messages would otherwise be dropped.
    """
    configure_logging()
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
