"""Unit tests for output sanitization and the output guardrail."""

from unittest import mock

from src.guardrails.output_guard import (
    _anonymize_pii,
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
