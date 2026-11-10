"""Gradio frontend for the SEC compliance RAG API.

Responsibilities:
  - Provide a chat interface for compliance analysts.
  - Render HTML citation cards and a legal-advice disclaimer.
  - Relay user input to the FastAPI backend over HTTP.

Dependencies:
  - A running API server at config.API_BASE_URL.
"""

import html
import uuid

import gradio as gr
import requests

from src.core import config

QUERY_URL = f"{config.API_BASE_URL}/query"
INGEST_URL = f"{config.API_BASE_URL}/ingest/trigger"
STATUS_URL = f"{config.API_BASE_URL}/corpus/status"
REQUEST_TIMEOUT_SECONDS = 60

ALL_COMPANIES = "All companies"
NO_DEBUG_INFO = "No debug info yet."

# Each entry is (text colour, background colour) for a ticker badge.
_TICKER_PALETTE = [
    ("#818cf8", "#312e81"),
    ("#22d3ee", "#164e63"),
    ("#34d399", "#064e3b"),
    ("#fbbf24", "#78350f"),
    ("#f87171", "#7f1d1d"),
    ("#a78bfa", "#4c1d95"),
    ("#f472b6", "#831843"),
    ("#a3e635", "#3f6212"),
]


def _ticker_colours(ticker: str, seen: dict) -> tuple[str, str]:
    """Return (text_colour, bg_colour) for a ticker within a panel."""
    if ticker not in seen:
        idx = len(seen) % len(_TICKER_PALETTE)
        seen[ticker] = _TICKER_PALETTE[idx]
    return seen[ticker]


def _edgar_url(accession_number: str) -> str:
    """Build the EDGAR filing index URL from an accession number.

    The CIK in the URL is taken from the accession number prefix.
    """
    clean = accession_number.replace("-", "")
    cik = clean[:10].lstrip("0")
    return f"https://www.sec.gov/Archives/edgar/data/{cik}/{clean}/{accession_number}-index.htm"


_PANEL_STYLE = """
<style>
  .citation-panel {
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
    margin-top: 8px;
    color: #e5e7eb;
  }
  .citation-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 12px;
  }
  .citation-header-title {
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    color: #9ca3af;
  }
  .citation-count-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    background: #374151;
    color: #f3f4f6;
    font-size: 11px;
    font-weight: 600;
    border-radius: 9999px;
    padding: 1px 8px;
  }
  .citation-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 10px;
  }
  .citation-card {
    background: #1f2937;
    border: 1px solid #374151;
    border-radius: 10px;
    padding: 14px 16px;
    transition: box-shadow 0.15s ease, border-color 0.15s ease;
  }
  .citation-card:hover {
    box-shadow: 0 4px 12px rgba(0,0,0,0.5);
    border-color: #4b5563;
  }
  .citation-card-header {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    margin-bottom: 10px;
  }
  .ticker-badge {
    display: inline-block;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.06em;
    padding: 2px 8px;
    border-radius: 6px;
    flex-shrink: 0;
    margin-top: 1px;
  }
  .company-name {
    font-size: 13px;
    font-weight: 600;
    color: #f9fafb;
    line-height: 1.35;
  }
  .citation-meta {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }
  .meta-row {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 12px;
    color: #9ca3af;
  }
  .meta-label {
    font-weight: 600;
    color: #6b7280;
    min-width: 80px;
    flex-shrink: 0;
  }
  .meta-value {
    color: #d1d5db;
  }
  .form-chip {
    display: inline-block;
    background: #374151;
    color: #e5e7eb;
    font-size: 11px;
    font-weight: 600;
    padding: 1px 7px;
    border-radius: 4px;
  }
  .section-value {
    font-style: italic;
    color: #d1d5db;
  }
  .edgar-link {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    margin-top: 10px;
    font-size: 11px;
    font-weight: 600;
    color: #818cf8;
    text-decoration: none;
    border: 1px solid #4f46e5;
    border-radius: 6px;
    padding: 3px 9px;
    background: #312e81;
    transition: background 0.15s;
    word-break: break-all;
  }
  .edgar-link:hover {
    background: #4338ca;
  }
  .citation-empty {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 28px 16px;
    border: 1px dashed #4b5563;
    border-radius: 10px;
    color: #6b7280;
    font-size: 13px;
    gap: 6px;
    background: #111827;
  }
  .citation-empty-icon {
    font-size: 24px;
    line-height: 1;
  }
</style>
"""


def build_citations_html(sources: list[dict]) -> str:
    """Render a styled HTML citation panel from SourceCitation dicts.

    Repeated (accession, section) pairs are shown once. All values from
    the API are HTML-escaped.

    Args:
        sources: Citation metadata dicts from the API response.

    Returns:
        An HTML string with one card per distinct source, or an empty
        state panel when there are none.
    """
    if not sources:
        return """
        <div class="citation-panel">
          <div class="citation-header">
            <span class="citation-header-title">Sources</span>
            <span class="citation-count-badge">0</span>
          </div>
          <div class="citation-empty">
            <span class="citation-empty-icon">📄</span>
            <span>No source documents were retrieved for this response.</span>
          </div>
        </div>
        """

    seen_tickers: dict = {}
    seen_cards: set = set()
    cards_html = ""
    valid_source_count = 0

    for s in sources:
        accession = s.get("accession_number", "")
        section = html.escape(s.get("section_name", "—"))

        card_key = (accession, section)
        if card_key in seen_cards:
            continue
        seen_cards.add(card_key)
        valid_source_count += 1

        ticker = s.get("ticker", "—")
        company = html.escape(s.get("company_name", "—"))
        form = html.escape(s.get("form_type", "—"))
        filing_date = html.escape(s.get("filing_date", "—"))
        period = html.escape(s.get("period_of_report", "—"))

        text_col, bg_col = _ticker_colours(ticker, seen_tickers)
        edgar_href = html.escape(_edgar_url(accession), quote=True) if accession else "#"

        cards_html += f"""
        <div class="citation-card">
          <div class="citation-card-header">
            <span class="ticker-badge" style="color:{text_col}; background:{bg_col};">{html.escape(ticker)}</span>
            <span class="company-name">{company}</span>
          </div>
          <div class="citation-meta">
            <div class="meta-row">
              <span class="meta-label">Form</span>
              <span class="form-chip">{form}</span>
            </div>
            <div class="meta-row">
              <span class="meta-label">Filed</span>
              <span class="meta-value">{filing_date}</span>
            </div>
            <div class="meta-row">
              <span class="meta-label">Period</span>
              <span class="meta-value">{period}</span>
            </div>
            <div class="meta-row">
              <span class="meta-label">Section</span>
              <span class="meta-value section-value">{section}</span>
            </div>
          </div>
          <a class="edgar-link" href="{edgar_href}" target="_blank" rel="noopener noreferrer">
            🔗 {html.escape(accession)}
          </a>
        </div>
        """

    company_count = len(seen_tickers)
    plural = "companies" if company_count != 1 else "company"
    subtitle = f"{valid_source_count} passage{'s' if valid_source_count != 1 else ''} · {company_count} {plural}"

    return f"""
    <div class="citation-panel">
      <div class="citation-header">
        <span class="citation-header-title">Sources</span>
        <span class="citation-count-badge">{subtitle}</span>
      </div>
      <div class="citation-grid">
        {cards_html}
      </div>
    </div>
    """


_EMPTY_CITATIONS = build_citations_html([])


_LEGAL_KEYWORDS = (
    "legal advice",
    "legal opinion",
    "not legal advice",
    "regulatory analysis",
    "consult",
)


def _has_legal_warning(warnings: list[str]) -> bool:
    """Return True if any warning contains a legal-advice keyword.

    Args:
        warnings: Warning strings from the backend.

    Returns:
        True if a keyword in _LEGAL_KEYWORDS is present.
    """
    lowered = " ".join(warnings).lower()
    return any(kw in lowered for kw in _LEGAL_KEYWORDS)


def build_legal_banner_html(warnings: list[str]) -> str:
    """Build the legal-advice disclaimer banner as HTML.

    Args:
        warnings: Warning strings from the backend.

    Returns:
        The banner HTML, or an empty string when no warning calls for
        it.
    """
    if not _has_legal_warning(warnings):
        return ""

    return """
    <style>
      .legal-banner {
        display: flex;
        align-items: flex-start;
        gap: 12px;
        background: #fffbeb;
        border: 1px solid #fcd34d;
        border-left: 4px solid #f59e0b;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 10px;
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
      }
      .legal-banner-icon {
        font-size: 18px;
        line-height: 1.4;
        flex-shrink: 0;
      }
      .legal-banner-body {
        font-size: 13px;
        line-height: 1.55;
        color: #78350f;
      }
      .legal-banner-title {
        font-weight: 700;
        margin-bottom: 2px;
        color: #92400e;
      }
    </style>
    <div class="legal-banner" role="note" aria-label="Regulatory analysis disclaimer">
      <span class="legal-banner-icon">⚖️</span>
      <div class="legal-banner-body">
        <div class="legal-banner-title">Regulatory Analysis — Not Legal Advice</div>
        This response contains regulatory or compliance analysis derived from SEC filings.
        It does not constitute legal advice and should not be relied upon as such.
        All claims should be verified against the original source documents linked
        in the citation panel below.
      </div>
    </div>
    """


def get_companies() -> list[str]:
    """Fetch the indexed company list from the corpus status endpoint.

    Falls back to ['All companies'] if the backend is unreachable or
    returns an error, so the UI stays usable while the API starts.

    Returns:
        The tickers, preceded by 'All companies'.
    """
    try:
        response = requests.get(STATUS_URL, timeout=REQUEST_TIMEOUT_SECONDS)
        if response.status_code == 200:
            return [ALL_COMPANIES] + response.json().get("indexed_companies", [])
        gr.Warning(f"Backend returned HTTP {response.status_code} for corpus status.")
    except (requests.RequestException, ValueError):
        gr.Warning("Could not reach backend to load company list; defaulting to 'All companies'.")
    return [ALL_COMPANIES]


def trigger_ingestion() -> None:
    """Start a background ingestion job through the backend API."""
    try:
        response = requests.post(INGEST_URL, timeout=REQUEST_TIMEOUT_SECONDS)
        if response.status_code == 200:
            gr.Info("Ingestion started.")
        else:
            gr.Warning(f"Failed to trigger ingestion (HTTP {response.status_code}).")
    except (requests.RequestException, ValueError):
        gr.Warning("Could not reach backend ingestion endpoint.")


def clear_chat() -> tuple:
    """Reset the chat UI to its initial empty state.

    Returns:
        A tuple of (empty_history, new_session_id, debug_placeholder,
        empty_citations_html, empty_banner_html).
    """
    return [], str(uuid.uuid4()), NO_DEBUG_INFO, _EMPTY_CITATIONS, ""


def submit_query(
    user_text: str,
    history: list,
    sess_id: str,
    comp_filter: str,
) -> tuple:
    """Send a user query to the backend API and update the chat UI.

    Connection errors, 400 guardrail rejections, other non-200 responses
    and invalid JSON are reported as an error message in the chat
    history instead of raising into the Gradio runtime.

    Args:
        user_text: The raw text entered by the user.
        history: The current Gradio chatbot message history.
        sess_id: The session UUID, sent as the LangGraph thread id.
        comp_filter: The selected company filter dropdown value.

    Returns:
        A tuple of (cleared_input, updated_history, debug_markdown,
        citations_html, legal_banner_html).
    """
    if not user_text:
        return "", history, NO_DEBUG_INFO, _EMPTY_CITATIONS, ""

    payload = {"query": user_text, "session_id": sess_id}
    if comp_filter and comp_filter != ALL_COMPANIES:
        payload["company_filter"] = comp_filter

    try:
        response = requests.post(QUERY_URL, json=payload, timeout=REQUEST_TIMEOUT_SECONDS)
    except requests.exceptions.RequestException:
        error_msg = f"Could not connect to the backend API at {config.API_BASE_URL}."
        history.append({"role": "user", "content": user_text})
        history.append({"role": "assistant", "content": f"**Error**: {error_msg}"})
        return "", history, "Error: Connection Failed", _EMPTY_CITATIONS, ""

    if response.status_code == 400:
        reason = "Input rejected by guardrails."
        try:
            reason = response.json().get("detail", reason)
        except ValueError:
            pass
        history.append({"role": "user", "content": user_text})
        history.append({"role": "assistant", "content": f"**Validation Error**: {reason}"})
        return "", history, f"Error: Validation Failed - {reason}", _EMPTY_CITATIONS, ""

    if response.status_code != 200:
        history.append({"role": "user", "content": user_text})
        history.append(
            {
                "role": "assistant",
                "content": f"**Error**: Backend returned HTTP {response.status_code}.",
            }
        )
        return "", history, f"Error: HTTP {response.status_code}", _EMPTY_CITATIONS, ""

    try:
        data = response.json()
    except ValueError:
        history.append({"role": "user", "content": user_text})
        history.append(
            {
                "role": "assistant",
                "content": "**Error**: Backend returned invalid JSON.",
            }
        )
        return "", history, "Error: Invalid JSON", _EMPTY_CITATIONS, ""

    assistant_text = data.get("response", "")
    history.append({"role": "user", "content": user_text})
    history.append({"role": "assistant", "content": assistant_text})

    cache_hit = data.get("cache_hit", False)
    grade = data.get("retrieval_grade", "")
    warnings = data.get("warnings", [])

    debug_md = f"""
**Cache Hit**: {cache_hit}
**Retrieval Grade**: {grade}
**Query Category**: {data.get("query_category", "general")}
**Retrieved Doc Count**: {data.get("retrieved_doc_count", 0)}
**Rewrite Count**: {data.get("rewrite_count", 0)}
**Warnings**: {", ".join(warnings) if warnings else "None"}
    """.strip()

    sources = data.get("sources", [])
    citations_html = build_citations_html(sources)
    banner_html = build_legal_banner_html(warnings)

    return "", history, debug_md, citations_html, banner_html


def build_demo() -> gr.Blocks:
    """Build the Gradio UI.

    Fetches the company list from the backend, so call it at launch time
    rather than at import time.

    Returns:
        The assembled Gradio Blocks app.
    """
    with gr.Blocks(title="SEC Filing RAG") as demo:
        gr.Markdown("# SEC Filing RAG")
        gr.Markdown("Frontend client for FastAPI backend")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Session")
                session_id = gr.Textbox(value=lambda: str(uuid.uuid4()), label="Session ID", interactive=False)
                company_filter = gr.Dropdown(choices=get_companies(), value=ALL_COMPANIES, label="Company Filter")
                clear_btn = gr.Button("Clear Conversation")
                ingest_btn = gr.Button("Trigger Ingestion")

            with gr.Column(scale=3):
                legal_banner = gr.HTML(value="")

                chatbot = gr.Chatbot(label="Conversation", height=500)
                with gr.Row():
                    user_input = gr.Textbox(
                        show_label=False,
                        placeholder="Ask a compliance question...",
                        scale=4,
                    )
                    submit_btn = gr.Button("Submit", scale=1)

                citations_panel = gr.HTML(value=_EMPTY_CITATIONS, label="Sources")

                with gr.Accordion(label="Show Debug Details", open=False):
                    debug_info = gr.Markdown(NO_DEBUG_INFO)

        gr.on(
            triggers=[user_input.submit, submit_btn.click],
            fn=submit_query,
            inputs=[user_input, chatbot, session_id, company_filter],
            outputs=[user_input, chatbot, debug_info, citations_panel, legal_banner],
        )

        clear_btn.click(
            clear_chat,
            inputs=[],
            outputs=[chatbot, session_id, debug_info, citations_panel, legal_banner],
        )

        ingest_btn.click(trigger_ingestion, inputs=[], outputs=[])

    return demo


def main() -> None:
    """Launch the Gradio application on config.GRADIO_PORT."""
    build_demo().launch(
        server_name=config.GRADIO_HOST,
        server_port=config.GRADIO_PORT,
        share=False,
        theme=gr.themes.Default(),
        head=_PANEL_STYLE,
    )


if __name__ == "__main__":
    main()
