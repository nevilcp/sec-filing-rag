"""Unit tests for output sanitization and the output guardrail."""

from unittest import mock

import pytest

from src.core import config
from src.core.rate_limiter import RateLimitExhausted, estimate_llm_tokens
from src.guardrails.output_guard import (
    _LEGAL_ADVICE_PROMPT,
    _anonymize_pii,
    _classifier_chunk_size,
    _is_definitive_legal_advice,
    validate_output,
)


def test_anonymize_pii_no_pii():
    text, entities = _anonymize_pii("The company reported strong Q3 earnings.")
    assert text == "The company reported strong Q3 earnings."
    assert not entities


def test_anonymize_pii_with_email():
    text, entities = _anonymize_pii("Contact me at john.doe@example.com.")
    assert "<EMAIL_ADDRESS>" in text
    assert "john.doe@example.com" not in text
    assert "EMAIL_ADDRESS" in entities


@mock.patch("src.guardrails.output_guard.invoke_with_rate_limit")
def test_is_definitive_legal_advice_violation(mock_invoke):
    mock_invoke.return_value = "VIOLATION"
    assert _is_definitive_legal_advice("You must file Form 8-K.")


@mock.patch("src.guardrails.output_guard.invoke_with_rate_limit")
def test_is_definitive_legal_advice_compliant(mock_invoke):
    mock_invoke.return_value = "COMPLIANT"
    assert not _is_definitive_legal_advice("Companies usually file Form 8-K.")


@mock.patch("src.guardrails.output_guard._is_definitive_legal_advice")
@mock.patch("src.guardrails.output_guard._anonymize_pii")
def test_validate_output_clean(mock_anonymize, mock_is_legal_advice):
    mock_anonymize.return_value = ("Clean text", [])
    mock_is_legal_advice.return_value = False

    sanitized, warnings = validate_output("Clean text")
    assert sanitized == "Clean text"
    assert not warnings


@mock.patch("src.guardrails.output_guard._is_definitive_legal_advice")
@mock.patch("src.guardrails.output_guard._anonymize_pii")
def test_validate_output_with_warnings(mock_anonymize, mock_is_legal_advice):
    mock_anonymize.return_value = ("Sanitized <EMAIL_ADDRESS>", ["EMAIL_ADDRESS"])
    mock_is_legal_advice.return_value = True

    sanitized, warnings = validate_output("Sanitized john@example.com")

    assert sanitized == "Sanitized <EMAIL_ADDRESS>"
    assert len(warnings) == 2
    assert any("PII" in w for w in warnings)
    assert any("legal advice" in w for w in warnings)


@mock.patch("src.guardrails.output_guard.invoke_with_rate_limit")
def test_validate_output_pii_response_keeps_pii_out_of_classifier_prompt(mock_invoke):
    mock_invoke.return_value = "COMPLIANT"

    sanitized, _ = validate_output("Reach the filer at john.doe@example.com for details.")

    prompt = mock_invoke.call_args.args[1]
    assert "john.doe@example.com" not in prompt
    assert "john.doe@example.com" not in sanitized


@mock.patch("src.guardrails.output_guard.invoke_with_rate_limit")
def test_long_output_is_classified_in_bounded_overlapping_chunks(mock_invoke):
    from src.core.rate_limiter import get_model_limits

    mock_invoke.return_value = "COMPLIANT"
    text = "Informational disclosure. " * 1500 + "END OF ANSWER"

    assert not _is_definitive_legal_advice(text)

    prompts = [call.args[1] for call in mock_invoke.call_args_list]
    assert len(prompts) > 1
    budget = get_model_limits(config.OUTPUT_GUARD_MODEL)[2]
    assert all(estimate_llm_tokens(prompt) + config.OUTPUT_GUARD_MAX_OUTPUT_TOKENS <= budget for prompt in prompts)
    assert _LEGAL_ADVICE_PROMPT.format(text=text[: _classifier_chunk_size(budget)]) == prompts[0]
    assert "END OF ANSWER" in prompts[-1]


@mock.patch("src.guardrails.output_guard.invoke_with_rate_limit")
def test_long_output_detects_violation_in_later_chunk(mock_invoke):
    mock_invoke.side_effect = ["COMPLIANT", "VIOLATION"]

    assert _is_definitive_legal_advice("Informational disclosure. " * 700)
    assert mock_invoke.call_count == 2


@pytest.mark.parametrize("budget", [1000, 2000, 8000])
def test_classifier_chunks_reserve_room_for_completion(budget):
    size = _classifier_chunk_size(budget)
    prompt = _LEGAL_ADVICE_PROMPT.format(text="x" * size)
    assert estimate_llm_tokens(prompt) + config.OUTPUT_GUARD_MAX_OUTPUT_TOKENS <= budget


def test_classifier_rejects_budget_smaller_than_policy():
    with pytest.raises(RateLimitExhausted):
        _classifier_chunk_size(300)


@mock.patch("src.guardrails.output_guard.invoke_with_rate_limit", return_value="")
def test_truncated_classifier_reply_is_not_accepted_as_compliant(mock_invoke):
    with pytest.raises(RuntimeError, match="classification"):
        _is_definitive_legal_advice("Informational answer")
