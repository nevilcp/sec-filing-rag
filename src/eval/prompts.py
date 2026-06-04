"""
eval/prompts.py

LLM-as-a-judge prompt templates for Arize Phoenix evaluations.

Responsibilities:
  - Define prompt templates for context relevance, faithfulness,
    and answer relevance evaluations.
  - Templates use Phoenix's attribute interpolation syntax so they can
    be passed directly to ClassificationEvaluator.

Not responsible for:
  - Evaluation orchestration or Phoenix client setup
    (see eval/run_phoenix_evals.py).

Dependencies:
  - None (plain string constants).
"""

CONTEXT_RELEVANCE_TEMPLATE = """
You are evaluating the relevance of a retrieved document to a user's query.
User Query: {attributes.input.value}
Retrieved Context: {attributes.retrieval.documents}

To be marked as "relevant", the retrieved context MUST contain information that directly helps answer the user's query.
If the context is completely unrelated to the query, it is "irrelevant".

Respond with ONLY one word: "relevant" or "irrelevant".
Then provide a brief explanation
"""

FAITHFULNESS_TEMPLATE = """
You are evaluating whether a generated answer is fully grounded in the provided context.
Retrieved Context: {attributes.retrieval.documents}
Generated Answer: {attributes.output.value}

To be marked as "faithful", ALL claims made in the generated answer MUST be supported by the retrieved context.
If the answer hallucinates information or includes external facts not present in the context, it is "unfaithful".

Respond with ONLY one word: "faithful" or "unfaithful".
Then provide a brief explanation
"""

ANSWER_RELEVANCE_TEMPLATE = """
You are evaluating whether a generated answer directly addresses the user's query.
User Query: {attributes.input.value}
Generated Answer: {attributes.output.value}

To be marked as "relevant", the answer MUST directly and adequately address the user's question based on the premise of the question.
If the answer is evasive, off-topic, or fails to address the core of the query, it is "irrelevant".

Respond with ONLY one word: "relevant" or "irrelevant".
Then provide a brief explanation
"""
