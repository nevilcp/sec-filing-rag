"""Streamlit frontend for SEC Compliance RAG API (HTTP client only)."""

from __future__ import annotations

import uuid

import requests
import streamlit as st

API_BASE_URL = "http://localhost:8000"
QUERY_URL = f"{API_BASE_URL}/query"
INGEST_URL = f"{API_BASE_URL}/ingest/trigger"
REQUEST_TIMEOUT_SECONDS = 60


def _ensure_session_state() -> None:
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "messages" not in st.session_state:
        st.session_state.messages = []


def _status_chip(label: str, color: str) -> str:
    return f":{color}[**{label}**]"


def _render_debug_panel(debug: dict) -> None:
    cache_hit = bool(debug.get("cache_hit", False))
    retrieval_grade = str(debug.get("retrieval_grade", ""))
    warnings = debug.get("warnings", []) or []

    cache_chip = _status_chip(
        f"cache_hit={cache_hit}",
        "green" if cache_hit else "gray",
    )
    grade_color = "yellow" if retrieval_grade == "irrelevant" else "green"
    grade_chip = _status_chip(f"retrieval_grade={retrieval_grade}", grade_color)
    warning_chip = _status_chip(
        f"warnings={len(warnings)}",
        "red" if warnings else "green",
    )

    st.markdown(f"{cache_chip} &nbsp;&nbsp; {grade_chip} &nbsp;&nbsp; {warning_chip}")
    st.write(f"query_category: {debug.get('query_category', 'general')}")
    st.write(f"retrieved_doc_count: {debug.get('retrieved_doc_count', 0)}")
    st.write(f"rewrite_count: {debug.get('rewrite_count', 0)}")
    if warnings:
        for warning in warnings:
            st.error(warning)


def _render_messages() -> None:
    for item in st.session_state.messages:
        role = item.get("role", "assistant")
        with st.chat_message(role):
            st.write(item.get("content", ""))
            if role == "assistant" and "debug" in item:
                with st.expander("Show Your Work", expanded=False):
                    _render_debug_panel(item["debug"])


def _post_query(query_text: str) -> tuple[bool, dict]:
    payload = {"query": query_text, "session_id": st.session_state.session_id}
    try:
        response = requests.post(QUERY_URL, json=payload, timeout=REQUEST_TIMEOUT_SECONDS)
    except requests.exceptions.RequestException:
        return (
            False,
            {
                "error_type": "connection",
                "message": (
                    "Could not connect to the backend API at http://localhost:8000. "
                    "Start the backend first (e.g., `python -m src.api.run`)."
                ),
            },
        )

    if response.status_code == 422:
        reason = "Input rejected by guardrails."
        try:
            reason = response.json().get("detail", reason)
        except ValueError:
            pass
        return False, {"error_type": "validation", "message": str(reason)}

    if response.status_code != 200:
        return (
            False,
            {
                "error_type": "server",
                "message": f"Backend returned HTTP {response.status_code}.",
            },
        )

    try:
        return True, response.json()
    except ValueError:
        return False, {"error_type": "server", "message": "Backend returned invalid JSON."}


def _trigger_ingestion() -> None:
    with st.spinner("Triggering ingestion..."):
        try:
            response = requests.post(INGEST_URL, timeout=REQUEST_TIMEOUT_SECONDS)
        except requests.exceptions.RequestException:
            st.error(
                "Could not reach backend ingestion endpoint. "
                "Ensure API is running at http://localhost:8000."
            )
            return

    if response.status_code == 200:
        st.success("Ingestion started.")
    else:
        st.error(f"Failed to trigger ingestion (HTTP {response.status_code}).")


def main() -> None:
    st.set_page_config(page_title="SEC Compliance RAG", page_icon=":scroll:")
    _ensure_session_state()

    st.title("SEC Compliance RAG")
    st.caption("Frontend client for FastAPI backend")

    with st.sidebar:
        st.subheader("Session")
        st.code(st.session_state.session_id, language=None)
        if st.button("Clear Conversation", use_container_width=True):
            st.session_state.messages = []
            st.session_state.session_id = str(uuid.uuid4())
            st.rerun()
        if st.button("Trigger Ingestion", use_container_width=True):
            _trigger_ingestion()

    _render_messages()

    user_input = st.chat_input("Ask a compliance question...")
    if not user_input:
        return

    success, payload = _post_query(user_input)
    if not success:
        if payload.get("error_type") == "validation":
            st.error(str(payload.get("message", "Input rejected.")))
        else:
            st.error(str(payload.get("message", "Request failed.")))
        return

    assistant_text = str(payload.get("response", ""))
    debug_payload = {
        "cache_hit": bool(payload.get("cache_hit", False)),
        "query_category": str(payload.get("query_category", "general")),
        "retrieval_grade": str(payload.get("retrieval_grade", "irrelevant")),
        "retrieved_doc_count": int(payload.get("retrieved_doc_count", 0)),
        "rewrite_count": int(payload.get("rewrite_count", 0)),
        "warnings": list(payload.get("warnings", [])),
    }

    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.messages.append(
        {"role": "assistant", "content": assistant_text, "debug": debug_payload}
    )
    st.rerun()


if __name__ == "__main__":
    main()
