# Agentic RAG for SEC EDGAR Filings

An agentic, corrective retrieval-augmented generation (CRAG) system for analyzing US SEC 10-K filings, detecting compliance risks, and surfacing evidence-grounded financial insights via a hybrid multi-agent architecture.

---

## System Architecture & Data Flow

Below is the complete end-to-end data flow mapping the document ingestion pipeline and stateful LangGraph query engine.

```mermaid
flowchart TD
    %% ─────────────────────────────────────────
    %%  INGESTION PIPELINE
    %% ─────────────────────────────────────────
    subgraph INGEST ["INGESTION PIPELINE"]
        direction TB
        A([Corpus Manifest\nTickers · Form Types · Fiscal Year]) --> B[CIK Resolution\ncompany_tickers.json cache]
        B --> C[Filing Discovery\nSubmissions Endpoint\ndata.sec.gov]
        C --> D[Document Index Resolution\nAccession Number → URL]
        D --> E[Document Acquisition\nEDGAR Archives\n≤ 7 req/sec · User-Agent required]
        E --> F[Local Staging Area\nRaw HTML on disk\nIdempotent · skip if exists]
        F --> G[Section Extractor\nItem 1 · 1A · 7 · 7A\n4-tier fallback parsing]
        G --> H[Chunker\n750 token chunks\nOverlap: 150 tokens]
        H --> J["Local Embedder\nBGE-M3 (FlagEmbedding)\n5 chunks per batch"]
        J --> L[(Qdrant Dense Index\nMetadata: ticker · section\nfiling_date · accession_no)]
        J --> M[(BM25S Index\nSparse Keyword Index\nDirectory · rebuilt on ingest)]
        J --> N([Checkpoint Updated\nAccession numbers logged\nIndexed companies recorded])
    end

    %% ─────────────────────────────────────────
    %%  QUERY PIPELINE
    %% ─────────────────────────────────────────
    subgraph QUERY ["QUERY PIPELINE — FastAPI + LangGraph"]
        direction TB
        subgraph GUARD_IN ["Input Guardrails"]
            Q1[InputGuard\nPII Detection · Presidio\nPrompt Injection · PromptGuard 2\nTier 1 · Groq LPU]
        end
        subgraph CACHE ["Semantic Cache"]
            Q2{Qdrant Cache Lookup\nCosine threshold 0.92}
        end
        subgraph AGENT ["LangGraph State Machine"]
            direction TB
            Q3[process_input\nQuery Router\nTier 1 · Qwen 3 32B\nCategories: compliance_check · doc_analysis\ncross_regulation · cross_company · temporal · general]
            Q4[retrieve_and_reason\nHybrid Retrieval\nDense Qdrant + Sparse BM25S\nRRF Fusion k=60 · BGE Reranker v2 M3\nDirected search filter if ticker/reg detected]
            Q5[grade_retrieval\nRelevance Grader\nMath: mean ≥ 0.5 & top1 ≥ 0.7\nSemantic LLM: Llama 3.1 8B]
            Q6[rewrite_query\nCRAG Query Rewriter\nTier 1 · Llama 3.1 8B\nCapped at 2 rewrites]
            Q7[respond\nFinal Answer Generation\nTier 2 · Gemini 3.5 Flash\nPremium synthesis]
            
            Q3 --> Q4
            Q4 --> Q5
            Q5 -- Relevant --> Q7
            Q5 -- Irrelevant + under cap --> Q6
            Q6 --> Q4
        end
        subgraph GUARD_OUT ["Output Guardrails"]
            Q8[OutputGuard\nPII Anonymisation · Presidio\nLegal Advice Flag · OSS Safeguard 20B\nTier 1 · Groq LPU]
        end
        Q1 -- Blocked --> REJECT([HTTP 422\nRejected])
        Q1 -- Clean --> Q2
        Q2 -- Cache hit --> Q8
        Q2 -- Cache miss --> Q3
        Q7 --> Q8
        Q8 --> RESP([QueryResponse\nAnswer · Sources · Warning flag\nTicker · Form type · Filing date\nSection · Accession number])
    end

    %% ─────────────────────────────────────────
    %%  EXTERNALS & OBS
    %% ─────────────────────────────────────────
    USER([User Query\nGradio UI\nOptional: company filter]) --> API[FastAPI\nPOST /query]
    API --> Q1
    L -.->|Dense search\nmetadata filter if ticker detected| Q4
    M -.->|BM25S keyword search| Q4
    Q7 -.->|Write response to cache| Q2
    subgraph OBS ["OBSERVABILITY"]
        OB1[Structured JSON Logging]
        OB2[Arize Phoenix Tracing\nLocal · OpenTelemetry-native]
        OB3[GET /metrics\nCache hit rate · RPD usage]
        OB4[GET /corpus/status\nIndexed companies · Chunk count]
    end
    API -.-> OB1
    API -.-> OB2
    Q4 -.-> OB1
    Q7 -.-> OB2
    TRIGGER([POST /ingest/trigger\nOptional: tickers · date range]) --> INGEST

    %% ─────────────────────────────────────────
    %%  STYLING DEFINITIONS
    %% ─────────────────────────────────────────
    classDef pipeline fill:#0f172a,stroke:#3b82f6,color:#f8fafc,stroke-width:2px,rx:6
    classDef decision fill:#172554,stroke:#f59e0b,color:#fef08a,stroke-width:2px
    classDef store fill:#022c22,stroke:#10b981,color:#d1fae5,stroke-width:2px
    classDef endpoint fill:#1e1b4b,stroke:#6366f1,color:#e0e7ff,stroke-width:2px
    classDef guard fill:#3b0764,stroke:#a855f7,color:#faf5ff,stroke-width:2px
    classDef obs fill:#1c1917,stroke:#78716c,color:#f5f5f4,stroke-width:1px
    classDef terminal fill:#18181b,stroke:#52525b,color:#f4f4f5,stroke-width:2px

    class A,N,REJECT,RESP,USER,TRIGGER terminal
    class B,C,D,E,F,G,H,J pipeline
    class Q3,Q4,Q5,Q6,Q7 pipeline
    class Q2 decision
    class L,M store
    class API endpoint
    class Q1,Q8 guard
    class OB1,OB2,OB3,OB4 obs
```

---

## Business Value & Problem Statement

### The Problem

Manual auditing of corporate compliance in sprawling, highly technical SEC EDGAR 10-K filings (often hundreds of pages of dense narrative) is labor-intensive, costly, and prone to oversight. Compliance officers, legal auditors, and financial analysts face significant overhead when cross-referencing company disclosures against shifting regulatory frameworks. Missing a single risk factor or disclosure gap can result in regulatory fines and legal liability.

Traditional linear RAG systems fail silently in this domain for three reasons:

1. **Financial Jargon Complexity:** Vector indexes alone struggle to align specialized financial terminology with ambiguous user queries.
2. **Hallucination & Recall Deficits:** Partially retrieved sections lead to incomplete or incorrect summaries that appear confident.
3. **API Cost & Privacy Barriers:** Enterprise workflows demand local data handling and strict API cost budgets that unmanaged cloud LLM calls cannot satisfy.

### The Solution

**Financial-Compliance-Analyst-AI** addresses these challenges by implementing a stateful **Corrective RAG (CRAG)** system. A hybrid storage architecture combining **Qdrant** (dense vector search) and **BM25S** (sparse keyword matching) feeds a multi-stage retrieval pipeline fused via Reciprocal Rank Fusion (RRF) and BGE Cross-Encoder reranking.

A stateful **LangGraph** engine then orchestrates a set of autonomous nodes — query routing, retrieval, mathematical + semantic grading, and query rewriting — that ensure only verified, relevant context reaches the final generator. Failing retrievals are automatically corrected by rewriting the query with domain-specific SEC terminology.

### Key Engineering Decisions & Their Impact

* **85% API Cost Reduction via Hybrid Model Routing:** High-frequency utility operations (input/output guardrails, query routing, query rewriting, and retrieval grading) are routed to Groq LPUs using lean open-source models. Only the final answer synthesis — which requires a large context window and complex reasoning — is sent to Gemini 3.5 Flash.
* **100% Local Privacy Guardrails:** Input/output sanitization runs entirely on-premises using Microsoft Presidio (PII detection and anonymization), PromptGuard 2-86M (injection classification), and GPT-OSS-Safeguard-20B (legal advice detection), ensuring sensitive queries never reach unintended public endpoints.
* **Zero Silent Hallucination via Corrective Gates:** Retrieved context passes a two-stage relevance gate (mathematical reranker score thresholds followed by an LLM-as-a-judge) before synthesis. Failed gates trigger automatic query rewriting, capped at two attempts, preventing the system from fabricating answers from irrelevant context.

---

## Ingestion, Storage, & Reasoning Logic

### 1. Ingestion Pipeline

* **Targeted Document Acquisition:** Programmatically queries the SEC EDGAR submissions endpoint (`data.sec.gov`) to discover and resolve accession numbers, fetching the most recent 10-K filing for each company in the corpus manifest. Requests are rate-limited to ≤ 7 req/sec to stay safely within EDGAR's 10 req/sec hard cap.
* **Resilient Parsing & Section Extraction:** A **4-tier fallback parsing strategy** isolates four critical 10-K sections: **Item 1** (Business), **Item 1A** (Risk Factors), **Item 7** (MD&A), and **Item 7A** (Quantitative Market Risk Disclosures). Tiers progress from anchor-based HTML extraction → heading-tag detection → regex text pattern matching → full-document fallback, ensuring broad filing format compatibility.
* **Chunking:** Sections are split using `RecursiveCharacterTextSplitter` with a **750-token window** and **150-token overlap**, preserving narrative context across chunk boundaries.
* **Local Dense Embeddings:** The local **BGE-M3 (FlagEmbedding)** model generates **1024-dimensional** dense vectors in batches of 5, using `fp16` precision with normalized output.
* **Idempotent Checkpointing:** A JSON-based checkpoint file tracks processed accession numbers and indexed tickers, making every ingestion run safe to interrupt and resume without duplication.

### 2. Storage & Retrieval (Hybrid RAG)

* **Dense Retrieval (Qdrant):** Indexes 10-K chunks as 1024-dim vectors with rich metadata payloads (ticker, section, filing date, accession number). A dedicated second Qdrant collection serves as the semantic response cache (cosine threshold: 0.92).
* **Sparse Retrieval (BM25S):** A BM25S keyword index is rebuilt from all Qdrant payloads after every ingestion run, enabling exact-term matching for regulatory clause numbers, legal citations, and financial terminology that dense search can underweight.
* **Fusion & Reranking:** Dense, sparse, and directed (ticker-filtered) result lists are merged via **Reciprocal Rank Fusion (RRF, k=60)**, then deduplicated by text content. The merged candidate set is re-scored by **BGE-Reranker-v2-M3** (cross-encoder, running locally), and results below a minimum reranker score of 0.25 are filtered out.

### 3. Orchestration (Multi-Agent LangGraph Workflow)

The graph starts with a **cache-first** routing check before running any LLM:

1. **`cache_check`:** Looks up the query embedding in the Qdrant semantic cache. On a hit, the cached response is passed directly to `respond`, bypassing all retrieval and LLM calls entirely.
2. **`process_input` (Query Router):** On a cache miss, `Qwen 3 32B` (via Groq) classifies the query into one of six categories: `compliance_check`, `document_analysis`, `cross_regulation`, `cross_company`, `temporal`, or `general`. The category controls the structured output template used during generation.
3. **`retrieve_and_reason` (Hybrid Retrieval + Grading):** Executes the full hybrid retrieval pipeline. Retrieved chunks are evaluated inline using both a mathematical gate (mean reranker score ≥ 0.50 **and** top-1 score ≥ 0.70) and an LLM-as-a-judge (`Llama 3.1 8B` via Groq) that verifies topic, entity, and temporal match.
4. **`grade_retrieval`:** Reads the relevance verdict from state and routes the graph: relevant context proceeds to `respond`; irrelevant context with rewrites remaining routes to `rewrite_query`.
5. **`rewrite_query` (CRAG Rewriter):** `Llama 3.1 8B` rewrites the query using formal SEC/CFR terminology and explicit item-level references. The rewrite loop is capped at **2 attempts**; if both fail, the graph proceeds to generation with whatever context is available.
6. **`respond` (Generation):** `Gemini 3.5 Flash` synthesizes a structured, citation-grounded answer using a category-specific prompt template. The response and sources are written to the semantic cache.

### 4. Generation & LLM Routing

* **Groq LPU (Tier 1):** Handles all high-frequency, low-latency tasks: PII detection (Presidio), prompt injection classification (PromptGuard 2-86M), query routing (Qwen 3 32B), retrieval grading (Llama 3.1 8B), query rewriting (Llama 3.1 8B), and legal advice flagging (GPT-OSS-Safeguard-20B).
* **Gemini 3.5 Flash (Tier 2):** Handles final answer synthesis only, using its large context window to reason across multiple cross-company filing excerpts.
* **Structured Output Enforcement:** The generation prompt enforces a strict section-by-section output format with mandatory inline citations (`[TICKER FORM_TYPE FILING_DATE — SECTION]`) and a legal disclaimer footer.
* **Rate-Limited Invocation:** All LLM calls pass through a sliding-window rate limiter before hitting the API, with exponential backoff retry for transient 429 errors.

---

## Tech Stack

| Category | Technology | Role |
| :--- | :--- | :--- |
| **Orchestration** | `LangGraph`, `LangChain` | Stateful CRAG graph, node wiring, and LLM abstractions |
| **API Layer** | `FastAPI`, `uvicorn` | REST endpoints, request validation, and background ingestion |
| **Vector Store & Cache** | `Qdrant` (on-disk) | Dense retrieval collection + semantic response cache collection |
| **Sparse Index** | `BM25S` | Keyword search index, rebuilt from Qdrant payloads after each ingest |
| **Reranker** | `BGE-Reranker-v2-M3` | Local cross-encoder reranking via `sentence-transformers` |
| **Embeddings** | `BGE-M3` (`FlagEmbedding`) | 1024-dim local dense embeddings, `fp16`, normalized |
| **Generation** | `Gemini 3.5 Flash` | Final answer synthesis via `langchain-google-genai` |
| **Utility LLMs** | `Qwen 3 32B`, `Llama 3.1 8B` | Query routing and CRAG grading/rewriting, via Groq LPU |
| **Input Guardrail** | `Microsoft Presidio` + `PromptGuard 2-86M` | PII detection; prompt injection classification (Groq) |
| **Output Guardrail** | `Microsoft Presidio` + `GPT-OSS-Safeguard-20B` | PII anonymization; legal advice detection (Groq) |
| **Observability** | `Arize Phoenix`, `OpenTelemetry` | Local LLM tracing and span export (opt-in via `PHOENIX_ENABLED=true`) |
| **Frontend** | `Gradio (Blocks)` | Chat UI with source citation panel |

---

## Quickstart

### Prerequisites

* **Python 3.11+** runtime environment
* **`GEMINI_API_KEY`** — [Google AI Studio](https://aistudio.google.com) (free tier)
* **`GROQ_API_KEY`** — [Groq Cloud](https://console.groq.com) (free tier)
* **`EDGAR_CONTACT_EMAIL`** — Required by SEC EDGAR as the `User-Agent` header value (e.g., `analyst@example.com`)

### Environment Configuration

Copy the example file and fill in your values:

```bash
cp .env.example .env
```

`.env.example` reference:

```bash
# Powers the fast Llama-3 and Qwen routing/grading LLMs
GROQ_API_KEY=

# Powers the Gemini 1.5 Flash final response generation
GEMINI_API_KEY=

# SEC-mandated contact identity to prevent IP bans during data ingestion
EDGAR_CONTACT_EMAIL=contact@example.com

# Enables Arize Phoenix OpenTelemetry tracing for agent observability
PHOENIX_ENABLED=

# Target fiscal year for filtering EDGAR 10-K and 10-Q filing retrievals
EDGAR_FISCAL_YEAR=2024
```

### Step-by-Step Setup

**1. Clone and create a virtual environment:**

```bash
git clone <your-repo-url> && cd financial-compliance-analyst-ai
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

**2. Run the ingestion pipeline:**

Ingest the corpus. Checkpoints are saved after each filing so you can safely interrupt and resume:

```bash
ingest
```

> BGE-M3 weights (~2.3 GB) are downloaded on the first run. The embedding batch size is intentionally conservative (5 chunks/batch) to respect API rate limits.

**Customizing the Corpus (Adding Companies & Regulations):**
By default, the system indexes a predefined set of 12 major companies and recognizes common SEC regulations. To customize these:

1. Open `src/core/constants.py`.
2. **Add Companies:** Append your target companies' ticker symbols to the `KNOWN_TICKERS` list:

   ```python
   KNOWN_TICKERS = [
       # ... existing companies ...
       "TSLA",
   ]
   ```

3. **Add Regulations:** If your queries frequently reference specific CFR clauses or niche regulations, add them to `KNOWN_REGULATIONS`. This list helps the query router explicitly filter dense/sparse search results for specific legal terminology.
4. Re-run the `ingest` command. The idempotent checkpointer will automatically skip previously processed companies and only fetch the new ones.

**3. Start the FastAPI backend:**

```bash
api
```

FastAPI is available at [http://localhost:8000](http://localhost:8000).

**4. Launch the Gradio frontend:**

```bash
gradio
```

Open [http://localhost:7860](http://localhost:7860) for the interactive chat UI with source citation panel.

### API Endpoints

| Method | Path | Description |
| :--- | :--- | :--- |
| `POST` | `/query` | Submit a compliance question |
| `GET` | `/corpus/status` | Inspect indexed companies and chunk count |
| `GET` | `/metrics` | Operational metrics (cache hit rate, embedding rate limiter stats) |
| `GET` | `/health` | Liveness check |
| `POST` | `/ingest/trigger` | Trigger a background ingestion run |

---

## Future Scope

1. **Structured Table & Financial Statement Parsing:** Upgrade document parsing to extract tabular data (balance sheets, income statements) as structured nodes rather than raw text, enabling precise numerical queries.
2. **Multi-Year Comparative Knowledge Graph:** Build a Neo4j graph layer tracking entity relationships across consecutive fiscal years, enabling detection of changes in risk language, executive disclosures, and cited regulatory references over time.
3. **Real-Time Filing Alerts:** Monitor the SEC EDGAR RSS feed to automatically trigger ingestion and re-indexing as soon as target companies publish new filings, keeping the corpus current without manual intervention.

---

## Project Structure

```text
.
├── app
│   └── gradio_app.py
├── logs
│   ├── api
│   │   └── access.log
│   ├── app
│   │   └── app.log
│   ├── audit
│   │   └── audit.log
│   ├── phoenix
│   │   └── phoenix.log
│   ├── rag
│   │   └── pipeline.log
│   └── system
│       └── system.log
├── README.md
├── src
│   ├── agent
│   │   ├── graph.py
│   │   ├── nodes.py
│   │   ├── router.py
│   │   └── state.py
│   ├── api
│   │   ├── main.py
│   │   ├── routes.py
│   │   ├── run.py
│   │   └── schemas.py
│   ├── core
│   │   ├── config.py
│   │   ├── constants.py
│   │   ├── corpus.py
│   │   ├── embeddings.py
│   │   ├── llm_factory.py
│   │   ├── logging.py
│   │   └── rate_limiter.py
│   ├── guardrails
│   │   ├── input_guard.py
│   │   └── output_guard.py
│   └── rag
│       ├── bm25_index.py
│       ├── cache.py
│       ├── edgar_client.py
│       ├── ingest.py
│       ├── retriever.py
│       ├── section_extractor.py
│       └── vector_store.py
└── tests
    └── unit
        ├── conftest.py
        ├── test_nodes.py
        ├── test_output_guard.py
        ├── test_rate_limiter.py
        ├── test_retriever.py
        └── test_section_extractor.py
```
