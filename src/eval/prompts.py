"""LLM-as-a-judge prompt templates for Arize Phoenix evaluations."""

ANSWER_RELEVANCE_TEMPLATE = """
You are evaluating whether a generated answer directly addresses the user's query.
User Query: {attributes.input.value}
Generated Answer: {attributes.output.value}

To be marked as "relevant", the answer MUST directly and adequately address the user's question based on the premise of the question.
If the answer is evasive, off-topic, or fails to address the core of the query, it is "irrelevant".

Respond with ONLY one word: "relevant" or "irrelevant".
Then provide a brief explanation
"""
