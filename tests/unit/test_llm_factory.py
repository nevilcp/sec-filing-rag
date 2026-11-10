"""Unit tests for the LLM client builders.

Responsibilities:
  - Verify the SDK retry count is bounded separately from the tenacity
    retry loop, so the two do not multiply.
"""

from unittest import mock

from src.core import config, llm_factory


@mock.patch("src.core.llm_factory.ChatGroq")
def test_build_groq_llm_uses_sdk_retry_constant(mock_chat):
    llm_factory._build_groq_llm("key", "model", 0.0)

    assert mock_chat.call_args.kwargs["max_retries"] == config.LLM_SDK_MAX_RETRIES


@mock.patch("src.core.llm_factory.ChatGoogleGenerativeAI")
def test_build_gemini_llm_uses_sdk_retry_constant(mock_chat):
    llm_factory._build_gemini_llm("key", "model", 0.0)

    assert mock_chat.call_args.kwargs["max_retries"] == config.LLM_SDK_MAX_RETRIES
