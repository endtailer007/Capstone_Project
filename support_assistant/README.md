# Zepto Support Assistant

A RAG-powered customer-support pipeline built with **LangGraph + ChromaDB + FastAPI**.

---

## RAG Pipeline Architecture

### Pipeline overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        INGESTION  (once at startup)                 │
│  doc1…doc8.txt  ──►  _load_documents()  ──►  _split_into_chunks()  │
│                              zepto_assistant.py / chunk()           │
└───────────────────────────────────┬─────────────────────────────────┘
                                    │ 19 text chunks
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                           EMBEDDING  (once at startup)              │
│  SentenceTransformer("all-MiniLM-L6-v2").encode(chunks)             │
│  → 384-dim float vectors                                            │
│                              zepto_assistant.py / chunk()           │
└───────────────────────────────────┬─────────────────────────────────┘
                                    │ embeddings + chunk text + IDs
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     VECTOR STORE  (persisted on disk)               │
│  ChromaDB PersistentClient  →  collection "zepto_support_docs"      │
│  hnsw:space = cosine  |  19 documents stored                        │
└─────────────────────────────────────────────────────────────────────┘

          ──────────────── per-request path ─────────────────

POST /ask {"query": "..."}   (app.py → FastAPI)
                │
                ▼
┌───────────────────────────────────┐
│  Node 1 — classify_intent         │  zepto_assistant.py
│  MOCK : keyword heuristic         │  ★ branches on MOCK_LLM
│  REAL : LLM classification call   │
└──────────┬──────────────┬─────────┘
           │              │
    policy_question  general_question
           │              │
           ▼              ▼
┌──────────────────┐  ┌─────────────────────────────────┐
│  Node 2          │  │  Node 3 — direct_answer          │
│  retrieve_and_   │  │  MOCK : fixed canned string       │  ★ branches
│  answer          │  │  REAL : free-form LLM answer      │  on MOCK_LLM
│                  │  └────────────────┬────────────────┘
│  RETRIEVAL       │                   │
│  (always real):  │                   │
│  embed query     │                   │
│  → ChromaDB      │                   │
│    top-3 chunks  │                   │
│                  │                   │
│  GENERATION      │                   │
│  MOCK : canned   │                   │
│   template from  │                   │  ★ branches
│   top chunk      │                   │  on MOCK_LLM
│  REAL : LLM call │                   │
│   via build_     │                   │
│   prompt()       │                   │
└────────┬─────────┘                   │
         └────────────┬────────────────┘
                      ▼
         SupportAnswer (Pydantic)
         { answer, sources, confidence }
                      │
                      ▼
              HTTP 200 JSON response
```

---

### Stage-by-stage walkthrough

#### Stage 1 — Ingestion

**Files/functions:** `zepto_assistant.py` → `chunk()` → `_load_documents()` + `_split_into_chunks()`

At server startup (FastAPI lifespan hook in `app.py`), `chunk()` reads all eight
`doc*.txt` policy files from disk using `_load_documents()`. Each document is then
passed to `_split_into_chunks()`, which breaks it into segments of at most 300
characters, preferring sentence boundaries. The result is 19 text chunks spread
across the 8 documents. Each chunk is assigned a deterministic ID of the form
`<filename>_chunk_<n>` (e.g. `doc1_delivery_policy.txt_chunk_0`).

---

#### Stage 2 — Embedding

**Files/functions:** `zepto_assistant.py` → `chunk()` → `SentenceTransformer("all-MiniLM-L6-v2").encode()`

The 19 chunks are passed in a single batch to the `all-MiniLM-L6-v2`
sentence-transformer model (loaded via the `sentence-transformers` library).
The model converts each chunk into a 384-dimensional float vector that captures
its semantic meaning. These vectors are what make similarity search possible —
two chunks about the same topic will produce vectors that are close together in
cosine space, regardless of exact wording.

This stage runs **entirely locally**, with no API key and no network call once
the model weights are downloaded.

---

#### Stage 3 — Retrieval

**Files/functions:** `zepto_assistant.py` → `retrieve_and_answer()` → `_get_collection()` → `collection.query()`

**Vector store:** ChromaDB `PersistentClient`, collection `zepto_support_docs`
(cosine similarity, stored in `./chroma_db/`).

When a `policy_question` query arrives, the `retrieve_and_answer` LangGraph node
embeds the incoming query with the same `all-MiniLM-L6-v2` model, then calls
`collection.query(query_embeddings=..., n_results=3)` to find the three most
semantically similar chunks. ChromaDB returns the chunk texts, their IDs, and
their cosine distances. The chunk IDs become the `sources` field in the final
`SupportAnswer`.

> This retrieval step **always runs for real** in both mock and real-LLM modes,
> since ChromaDB needs no API key and no external network access.

---

#### Stage 4 — Generation

**Files/functions:** `zepto_assistant.py` → `retrieve_and_answer()` or `direct_answer()`
**Prompt template:** `build_prompt()` (role–context–task–format–length skeleton with negative constraints and few-shot examples)

The generation step is where the **MOCK_LLM toggle** has its primary effect:

| `MOCK_LLM` value | `classify_intent` | `retrieve_and_answer` (generation only) | `direct_answer` |
|---|---|---|---|
| `1` / unset **(default, graded)** | Keyword heuristic — checks if query contains any of `"delivery"`, `"return"`, `"refund"`, `"membership"`, `"tracking"`, `"cancel"`, `"gift card"`, `"support hours"` | Returns `"Based on the retrieved context: <first 200 chars of top chunk>"` — **no LLM call** | Returns fixed string `"I can only answer questions about Zepto policies right now."` — **no LLM call** |
| `0` *(optional extension)* | LLM classifies intent | Calls LLM with the structured `build_prompt()` template, retrying up to 2 times if `SupportAnswer` validation fails | Calls LLM with the raw query, retrying up to 2 times on schema validation failure |

In both modes, the final answer is validated against the `SupportAnswer` Pydantic
model (`answer: str`, `sources: list[str]`, `confidence: float 0–1`) before being
returned. In mock mode the schema is populated deterministically: `sources` = the
three retrieved chunk IDs (or `[]` for `general_question`), `confidence` = `1.0`.

---

### Data flow summary

```
doc*.txt  →  chunks (str)  →  384-dim vectors  →  ChromaDB
                                                       ↑ stored once at startup
query (str)  →  384-dim vector  →  ChromaDB top-3  →  retrieved_chunks (str[])
                                                       ↓
                                              build_prompt() (REAL mode)
                                                       ↓
                                         LLM / canned template (MOCK mode)
                                                       ↓
                                          SupportAnswer (Pydantic validated)
                                                       ↓
                                              HTTP 200 JSON
```

---

## Project structure

```
support_assistant/
├── app.py                          # FastAPI application
├── zepto_assistant.py              # Core pipeline (chunk, embed, graph, prompt)
├── README.md                       # This file
├── doc1_delivery_policy.txt
├── doc2_return_and_refund.txt
├── doc3_membership_tiers.txt
├── doc4_order_tracking.txt
├── doc5_order_cancellation_policy.txt
├── doc6_damaged_or_missing_items.txt
├── doc7_gift_cards.txt
├── doc8_customer_support_hours.txt
└── .gitignore                      # excludes chroma_db/ binaries
```

---

## Setup

```bash
pip install chromadb sentence-transformers langgraph pydantic fastapi uvicorn
```

---

## Running the server

```bash
cd support_assistant
python -m uvicorn app:app --port 8000
```

On startup the server automatically:
1. Loads and chunks all 8 policy documents.
2. Embeds every chunk with `all-MiniLM-L6-v2` (384-dim).
3. Stores embeddings in a local ChromaDB collection (`zepto_support_docs`).
4. Compiles the LangGraph StateGraph.

---

## MOCK_LLM toggle

| Value | Behaviour |
|-------|-----------|
| `MOCK_LLM=1` (default) | Keyword classifier + canned answers — **no API key needed** |
| `MOCK_LLM=0` | Real-LLM classification and generation (wire up SDK in `zepto_assistant.py`) |

---

## API Reference

### `POST /ask`

**Request**
```json
{ "query": "<customer question>" }
```

**Response** — `SupportAnswer`
```json
{
  "answer":     "<string>",
  "sources":    ["<chunk_id>", ...],
  "confidence": 1.0
}
```

| Field        | Type         | Description |
|--------------|--------------|-------------|
| `answer`     | `string`     | Human-readable answer |
| `sources`    | `[string]`   | ChromaDB chunk IDs used; empty for general questions |
| `confidence` | `float 0–1`  | Mock mode always returns `1.0`; real-LLM mode is LLM-reported |

---

## Example calls & recorded responses

### Call 1 — Policy question (triggers retrieval)

**Request**
```bash
curl -X POST http://127.0.0.1:8000/ask \
     -H "Content-Type: application/json" \
     -d '{"query": "What is the refund timeline for approved returns?"}'
```

**Response**
```json
{
  "answer": "Based on the retrieved context: refund is processed.",
  "sources": [
    "doc6_damaged_or_missing_items.txt_chunk_2",
    "doc2_return_and_refund.txt_chunk_1",
    "doc2_return_and_refund.txt_chunk_0"
  ],
  "confidence": 1.0
}
```

→ `intent = policy_question` — ChromaDB retrieved 3 chunks from the return/refund and damaged-items documents. Sources are populated with their chunk IDs.

---

### Call 2 — General question (no retrieval)

**Request**
```bash
curl -X POST http://127.0.0.1:8000/ask \
     -H "Content-Type: application/json" \
     -d '{"query": "Who won the cricket match yesterday?"}'
```

**Response**
```json
{
  "answer": "I can only answer questions about Zepto policies right now.",
  "sources": [],
  "confidence": 1.0
}
```

→ `intent = general_question` — no retrieval step; `sources` is empty; fixed canned response returned.

---

## Interactive docs

With the server running, open:

- **Swagger UI**: http://127.0.0.1:8000/docs
- **ReDoc**:      http://127.0.0.1:8000/redoc
