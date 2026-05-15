"""Gradio frontend for SEC Compliance RAG API."""

import uuid
import requests
import gradio as gr

API_BASE_URL = "http://localhost:8000"
QUERY_URL = f"{API_BASE_URL}/query"
INGEST_URL = f"{API_BASE_URL}/ingest/trigger"
STATUS_URL = f"{API_BASE_URL}/corpus/status"
REQUEST_TIMEOUT_SECONDS = 60

def get_companies():
    try:
        response = requests.get(STATUS_URL, timeout=REQUEST_TIMEOUT_SECONDS)
        if response.status_code == 200:
            return ["All companies"] + response.json().get("indexed_companies", [])
    except Exception:
        pass
    return ["All companies"]

def trigger_ingestion():
    try:
        response = requests.post(INGEST_URL, timeout=REQUEST_TIMEOUT_SECONDS)
        if response.status_code == 200:
            gr.Info("Ingestion started.")
        else:
            gr.Warning(f"Failed to trigger ingestion (HTTP {response.status_code}).")
    except Exception:
        gr.Warning("Could not reach backend ingestion endpoint.")

def clear_chat():
    return [], str(uuid.uuid4()), "No debug info yet.", [["No sources available", "-", "-", "-", "-", "-"]]

def submit_query(user_text, history, sess_id, comp_filter):
    if not user_text:
        return "", history, "No debug info yet.", [["No sources available", "-", "-", "-", "-", "-"]]

    payload = {"query": user_text, "session_id": sess_id}
    if comp_filter and comp_filter != "All companies":
        payload["company_filter"] = comp_filter

    try:
        response = requests.post(QUERY_URL, json=payload, timeout=REQUEST_TIMEOUT_SECONDS)
    except requests.exceptions.RequestException:
        error_msg = "Could not connect to the backend API at http://localhost:8000."
        history.append((user_text, f"**Error**: {error_msg}"))
        return "", history, "Error: Connection Failed", [["-", "-", "-", "-", "-", "-"]]

    if response.status_code == 422:
        reason = "Input rejected by guardrails."
        try:
            reason = response.json().get("detail", reason)
        except ValueError:
            pass
        history.append((user_text, f"**Validation Error**: {reason}"))
        return "", history, f"Error: Validation Failed - {reason}", [["-", "-", "-", "-", "-", "-"]]

    if response.status_code != 200:
        history.append((user_text, f"**Error**: Backend returned HTTP {response.status_code}."))
        return "", history, f"Error: HTTP {response.status_code}", [["-", "-", "-", "-", "-", "-"]]

    try:
        data = response.json()
    except ValueError:
        history.append((user_text, "**Error**: Backend returned invalid JSON."))
        return "", history, "Error: Invalid JSON", [["-", "-", "-", "-", "-", "-"]]

    assistant_text = data.get("response", "")
    history.append((user_text, assistant_text))
    
    cache_hit = data.get("cache_hit", False)
    grade = data.get("retrieval_grade", "")
    warnings = data.get("warnings", [])
    
    debug_md = f"""
**Cache Hit**: {cache_hit}  
**Retrieval Grade**: {grade}  
**Query Category**: {data.get("query_category", "general")}  
**Retrieved Doc Count**: {data.get("retrieved_doc_count", 0)}  
**Rewrite Count**: {data.get("rewrite_count", 0)}  
**Warnings**: {', '.join(warnings) if warnings else 'None'}
    """.strip()
    
    sources = data.get("sources", [])
    if not sources:
        source_rows = [["No sources available", "-", "-", "-", "-", "-"]]
    else:
        source_rows = []
        for s in sources:
            source_rows.append([
                s.get("ticker", ""),
                s.get("company_name", ""),
                s.get("form_type", ""),
                s.get("filing_date", ""),
                s.get("section_name", ""),
                s.get("accession_number", "")
            ])

    return "", history, debug_md, source_rows

with gr.Blocks(title="SEC Compliance RAG") as demo:
    gr.Markdown("# SEC Compliance RAG")
    gr.Markdown("Frontend client for FastAPI backend")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Session")
            session_id = gr.Textbox(value=lambda: str(uuid.uuid4()), label="Session ID", interactive=False)
            company_filter = gr.Dropdown(choices=get_companies(), value="All companies", label="Company Filter")
            clear_btn = gr.Button("Clear Conversation")
            ingest_btn = gr.Button("Trigger Ingestion")
            
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(label="Conversation", height=500)
            with gr.Row():
                user_input = gr.Textbox(show_label=False, placeholder="Ask a compliance question...", scale=4)
                submit_btn = gr.Button("Submit", scale=1)
            
            with gr.Accordion(label="Show Details", open=False):
                debug_info = gr.Markdown("No debug info yet.")
                sources_table = gr.Dataframe(
                    headers=["Ticker", "Company", "Form", "Filing Date", "Section", "Accession Number"],
                    value=[["No sources available", "-", "-", "-", "-", "-"]],
                    interactive=False
                )

    user_input.submit(
        submit_query, 
        inputs=[user_input, chatbot, session_id, company_filter],
        outputs=[user_input, chatbot, debug_info, sources_table]
    )
    submit_btn.click(
        submit_query, 
        inputs=[user_input, chatbot, session_id, company_filter],
        outputs=[user_input, chatbot, debug_info, sources_table]
    )
    
    clear_btn.click(
        clear_chat,
        inputs=[],
        outputs=[chatbot, session_id, debug_info, sources_table]
    )
    
    ingest_btn.click(
        trigger_ingestion,
        inputs=[],
        outputs=[]
    )

def main():
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)

if __name__ == "__main__":
    main()
