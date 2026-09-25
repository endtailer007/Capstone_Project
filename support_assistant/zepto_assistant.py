"""
zepto_assistant.py
──────────────────
Loads the 8 Zepto support documents, chunks them, embeds each chunk with
the all-MiniLM-L6-v2 sentence-transformer model, and stores the embeddings
in a persistent ChromaDB collection.

Also provides:
  - build_prompt()  : structured role-context-task-format-length prompt template.
  - build_graph()   : LangGraph StateGraph with 3 nodes and conditional routing.
                      Respects the MOCK_LLM env-var toggle.
  - SupportAnswer   : Pydantic output schema (answer, sources, confidence).
"""

import os
import json
from typing import Literal, Optional
from typing_extensions import TypedDict
from pydantic import BaseModel, Field, field_validator
import chromadb
from sentence_transformers import SentenceTransformer
from langgraph.graph import StateGraph, END

#Configuration

DOCS_DIR = os.path.dirname(os.path.abspath(__file__))

DOCUMENT_FILES = [
    "doc1_delivery_policy.txt",
    "doc2_return_and_refund.txt",
    "doc3_membership_tiers.txt",
    "doc4_order_tracking.txt",
    "doc5_order_cancellation_policy.txt",
    "doc6_damaged_or_missing_items.txt",
    "doc7_gift_cards.txt",
    "doc8_customer_support_hours.txt",
]

CHROMA_DB_PATH  = os.path.join(DOCS_DIR, "chroma_db")
COLLECTION_NAME = "zepto_support_docs"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Maximum characters per chunk
CHUNK_SIZE = 300

# LangGraph MOCK_LLM toggle.
# Set env var MOCK_LLM=0 to enable real-LLM branches (requires API key).
# Any other value (or unset) uses the mock/keyword baseline.
MOCK_LLM = os.environ.get("MOCK_LLM", "1") != "0"

# Keywords that classify a query as a policy question
POLICY_KEYWORDS = {
    "delivery", "return", "refund", "membership",
    "tracking", "cancel", "gift card", "support hours",
}


# Pydantic output schema
class SupportAnswer(BaseModel):
    """
    Enforced JSON output schema for every graph response.

    Fields
    ------
    answer     : The human-readable answer text.
    sources    : IDs of the ChromaDB chunks used to generate the answer.
                 Empty list for general_question answers (no retrieval).
    confidence : A float in [0, 1] representing answer confidence.
                 Mock mode always returns 1.0 deterministically.
    """
    answer:     str            = Field(..., description="Human-readable answer to the customer's question.")
    sources:    list[str]      = Field(default_factory=list, description="ChromaDB chunk IDs used; empty for general questions.")
    confidence: float          = Field(..., ge=0.0, le=1.0, description="Confidence score between 0 and 1.")

    @field_validator("confidence")
    @classmethod
    def _clamp_confidence(cls, v: float) -> float:
        """Silently clamp confidence to [0, 1] before validation fails."""
        return max(0.0, min(1.0, v))


#Helpers

def _load_documents():
    """Read every document file and return a list of {source, text} dicts."""
    documents = []
    for filename in DOCUMENT_FILES:
        filepath = os.path.join(DOCS_DIR, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read().strip()
        documents.append({"source": filename, "text": text})
        print(f"  [load] {filename}  ({len(text)} chars)")
    return documents


def _split_into_chunks(text, chunk_size=CHUNK_SIZE):
    """
    Split text into non-overlapping chunks of at most chunk_size characters,
    breaking on sentence boundaries where possible.

    Strategy:
      1. Split text into sentences (naively on ". ").
      2. Accumulate sentences into a chunk until adding the next sentence would
         exceed chunk_size; then start a new chunk.
      3. If a single sentence exceeds chunk_size it is split at word boundaries.
    """
    raw_sentences = text.split(". ")
    sentences = [s + ". " for s in raw_sentences[:-1]] + [raw_sentences[-1]]

    chunks = []
    current = ""

    for sentence in sentences:
        if len(current) + len(sentence) <= chunk_size:
            current += sentence
        else:
            if current:
                chunks.append(current.strip())
            if len(sentence) > chunk_size:
                words = sentence.split()
                sub = ""
                for word in words:
                    if len(sub) + len(word) + 1 <= chunk_size:
                        sub += (" " if sub else "") + word
                    else:
                        if sub:
                            chunks.append(sub.strip())
                        sub = word
                if sub:
                    current = sub + " "
                else:
                    current = ""
            else:
                current = sentence

    if current.strip():
        chunks.append(current.strip())

    return chunks


# Main function 

def chunk():
    """
    Load the 8 Zepto support documents, chunk each one, embed every chunk with
    all-MiniLM-L6-v2, and persist the embeddings in a ChromaDB collection.

    Returns
    -------
    chromadb.Collection
        The populated ChromaDB collection so callers can query it directly.
    """
    print("=" * 60)
    print("Step 1 - Loading documents ...")
    documents = _load_documents()

    #Build chunk list
    print("\nStep 2 - Chunking documents ...")
    all_chunks    = []
    all_ids       = []
    all_metadatas = []

    for doc in documents:
        chunks = _split_into_chunks(doc["text"], CHUNK_SIZE)
        print(f"  {doc['source']}  ->  {len(chunks)} chunk(s)")
        for i, chunk_text in enumerate(chunks):
            chunk_id = f"{doc['source']}_chunk_{i}"
            all_chunks.append(chunk_text)
            all_ids.append(chunk_id)
            all_metadatas.append({"source": doc["source"], "chunk_index": i})

    print(f"\n  Total chunks: {len(all_chunks)}")

    # Embed
    print(f"\nStep 3 - Embedding with '{EMBEDDING_MODEL}' ...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = model.encode(all_chunks, show_progress_bar=True).tolist()
    print(f"  Embedding dimension: {len(embeddings[0])}")

    #Store in ChromaDB
    print(f"\nStep 4 - Storing in ChromaDB collection '{COLLECTION_NAME}' ...")
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    # Drop and recreate so re-runs stay idempotent.
    try:
        client.delete_collection(COLLECTION_NAME)
        print("  Existing collection deleted.")
    except Exception:
        pass  # Collection did not exist yet - that is fine.

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},  # cosine similarity for MiniLM
    )

    collection.add(
        ids=all_ids,
        documents=all_chunks,
        embeddings=embeddings,
        metadatas=all_metadatas,
    )

    print(f"  Stored {collection.count()} chunks successfully.")
    print("=" * 60)
    print("Done!  ChromaDB persisted at:", CHROMA_DB_PATH)

    return collection


# ── Prompt template ───────────────────────────────────────────────────────────

FEW_SHOT_EXAMPLES = """\
---- FEW-SHOT EXAMPLES (for illustration only — do NOT treat as real context) ----

Example 1
User question : Can I return an opened shampoo bottle?
Retrieved context: Personal care items that have been opened are non-returnable
                   except in the case of a manufacturing defect.
Answer        : No, opened personal care items cannot be returned unless there
                is a manufacturing defect.

Example 2
User question : How long does Zepto take to deliver?
Retrieved context: Zepto delivers orders within 10 minutes in serviceable areas
                   using localized dark stores.
Answer        : Zepto typically delivers within 10 minutes in areas it services.

---- END OF EXAMPLES ----
"""


def build_prompt(user_question: str, retrieved_chunks: list) -> str:
    """
    Build a fully structured LLM prompt for the Zepto customer-support assistant.

    The prompt follows the role-context-task-format-length skeleton and embeds:
      * An explicit negative constraint (no hallucination beyond given context).
      * Two few-shot examples that teach the desired answer style.

    Parameters
    ----------
    user_question : str
        The raw question typed by the customer.
    retrieved_chunks : list of str
        The top-k text chunks retrieved from the ChromaDB collection.

    Returns
    -------
    str
        A ready-to-send prompt string for any chat or completion LLM endpoint.
    """

    # ── ROLE ──────────────────────────────────────────────────────────────────
    role = (
        "You are a knowledgeable and empathetic Zepto customer-support assistant. "
        "Your sole purpose is to help customers resolve queries about Zepto's "
        "policies — including delivery, returns, refunds, membership tiers, "
        "order tracking, cancellations, damaged items, gift cards, and support hours."
    )

    # ── CONTEXT ───────────────────────────────────────────────────────────────
    numbered_chunks = "\n\n".join(
        f"[Context {i + 1}]\n{chunk.strip()}"
        for i, chunk in enumerate(retrieved_chunks)
    )
    context_block = (
        "The following excerpts have been retrieved from Zepto's official "
        "policy documents. They are the ONLY source of truth you may use "
        "when composing your answer:\n\n"
        + numbered_chunks
    )

    # ── TASK ──────────────────────────────────────────────────────────────────
    task = (
        f"Using exclusively the context provided above, answer the customer's "
        f"question below as helpfully and accurately as possible.\n\n"
        f"Customer question: {user_question}"
    )

    # ── NEGATIVE CONSTRAINT ───────────────────────────────────────────────────
    constraint = (
        "IMPORTANT CONSTRAINTS\n"
        "1. Do NOT answer using any information that is not explicitly present "
        "in the retrieved context above. If the context does not contain enough "
        "information to answer the question, respond with: "
        "'I\'m sorry, I don\'t have information about that in my knowledge base. "
        "Please contact Zepto support directly for further help.'\n"
        "2. Do NOT speculate, infer, or make up policies, fees, or timelines.\n"
        "3. Do NOT reference competitor services or external websites."
    )

    # ── FORMAT ────────────────────────────────────────────────────────────────
    fmt = (
        "FORMAT RULES\n"
        "* Write in plain, friendly prose — no bullet lists unless the answer "
        "genuinely contains multiple distinct steps.\n"
        "* Do not expose internal chunk labels like [Context 1] in your answer.\n"
        "* If a specific rupee amount, time window, or condition is mentioned in "
        "the context, quote it exactly."
    )

    # ── LENGTH ────────────────────────────────────────────────────────────────
    length = (
        "LENGTH\n"
        "Keep the answer between 1 and 4 sentences. "
        "Do not pad the response with unnecessary filler or repetition."
    )

    # ── Assemble full prompt ───────────────────────────────────────────────────
    prompt = "\n\n".join([
        f"[ROLE]\n{role}",
        f"[CONTEXT]\n{context_block}",
        f"[FEW-SHOT EXAMPLES]\n{FEW_SHOT_EXAMPLES.strip()}",
        f"[TASK]\n{task}",
        f"[CONSTRAINTS]\n{constraint}",
        f"[FORMAT]\n{fmt}",
        f"[LENGTH]\n{length}",
    ])

    return prompt


#LangGraph — StateGraph implementation

class ZeptoState(TypedDict):
    """
    Shared state that flows through every node in the graph.

    Fields
    ------
    query            : the original customer question
    intent           : classification result — 'policy_question' or 'general_question'
    chunks           : top-k text chunks retrieved from ChromaDB (populated by retrieve_and_answer)
    chunk_ids        : ChromaDB IDs for the retrieved chunks (used to populate SupportAnswer.sources)
    answer           : the final answer string produced by a terminal node
    structured_answer: the validated SupportAnswer dict (JSON-serialisable)
    """
    query:             str
    intent:            str
    chunks:            list
    chunk_ids:         list
    answer:            str
    structured_answer: dict


def _get_collection() -> chromadb.Collection:
    """Open the existing persistent ChromaDB collection (must have been built first)."""
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    return client.get_collection(COLLECTION_NAME)


# Node 1 — classify_intent
def classify_intent(state: ZeptoState) -> ZeptoState:
    """
    Classify the incoming query as 'policy_question' or 'general_question'.

    Mock mode (MOCK_LLM=1 or unset — graded baseline)
    ---------------------------------------------------
    Uses a keyword heuristic: if any policy keyword appears in the
    lowercased query, classify as policy_question; otherwise general_question.
    No LLM call is made.

    Optional real-LLM mode (MOCK_LLM=0)
    -------------------------------------
    Calls the configured LLM to classify the intent.
    """
    query = state["query"]

    if MOCK_LLM:
        # Keyword heuristic — no API call
        lower = query.lower()
        intent = (
            "policy_question"
            if any(kw in lower for kw in POLICY_KEYWORDS)
            else "general_question"
        )
        print(f"  [classify_intent | MOCK] intent='{intent}'")
    else:
        # Optional real-LLM extension
        # Replace the block below with your preferred LLM SDK call.
        # Example (Gemini):
        #   import google.generativeai as genai
        #   model = genai.GenerativeModel("gemini-1.5-flash")
        #   response = model.generate_content(
        #       f"Classify the following customer query as exactly one of "
        #       f"'policy_question' or 'general_question'.\nQuery: {query}\nAnswer:"
        #   )
        #   intent = response.text.strip().lower()
        raise NotImplementedError(
            "Real-LLM classify_intent not wired up yet. "
            "Set MOCK_LLM=1 or implement your LLM call above."
        )

    return {**state, "intent": intent}


# Routing function — wires the conditional edge out of classify_intent
def _route_intent(state: ZeptoState) -> Literal["retrieve_and_answer", "direct_answer"]:
    """
    Pure routing function: reads the intent set by classify_intent and
    returns the name of the next node to execute.
    This function does NOT depend on MOCK_LLM — only generation steps do.
    """
    return "retrieve_and_answer" if state["intent"] == "policy_question" else "direct_answer"


# Node 2 — retrieve_and_answer
def retrieve_and_answer(state: ZeptoState) -> ZeptoState:
    """
    For policy_question queries:
      1. Embeds the query and retrieves the top-3 chunks from ChromaDB.
         (This retrieval step always runs for real in both modes.)
      2. Generates an answer.

    Mock mode (MOCK_LLM=1 or unset — graded baseline)
    ---------------------------------------------------
    Returns a canned templated answer:
        'Based on the retrieved context: <first ~200 chars of top chunk>'
    No LLM call is made.

    Optional real-LLM mode (MOCK_LLM=0)
    -------------------------------------
    Calls the LLM using build_prompt() with the retrieved chunks.
    """
    query = state["query"]

    # Always-real: embed + retrieve from ChromaDB
    embed_model = SentenceTransformer(EMBEDDING_MODEL)
    query_vec = embed_model.encode([query]).tolist()

    collection = _get_collection()
    results = collection.query(query_embeddings=query_vec, n_results=3)
    retrieved_chunks  = results["documents"][0]  # list of 3 chunk strings
    retrieved_ids     = results["ids"][0]         # list of 3 chunk ID strings

    print(f"  [retrieve_and_answer] retrieved {len(retrieved_chunks)} chunk(s): {retrieved_ids}")

    if MOCK_LLM:
        # Mock answer — canned template using the top chunk
        top_chunk_snippet = retrieved_chunks[0][:200] if retrieved_chunks else ""
        answer_text = f"Based on the retrieved context: {top_chunk_snippet}"

        # Build SupportAnswer deterministically — no LLM output to validate
        structured = SupportAnswer(
            answer=answer_text,
            sources=retrieved_ids,   # IDs of the 3 retrieved chunks
            confidence=1.0,          # fixed value in mock mode
        )
        print(f"  [retrieve_and_answer | MOCK] SupportAnswer built")
    else:
        # Optional real-LLM extension with retry-on-validation-failure
        #
        # Pseudocode (wire up your LLM SDK here):
        #
        #   import google.generativeai as genai
        #   llm = genai.GenerativeModel("gemini-1.5-flash")
        #   prompt = build_prompt(query, retrieved_chunks)
        #   schema_instruction = (
        #       "Respond ONLY with a JSON object matching this schema — no extra text:\n"
        #       '{"answer": "<str>", "sources": ["<chunk_id>", ...], "confidence": <float 0-1>}'
        #   )
        #   full_prompt = prompt + "\n\n" + schema_instruction
        #
        #   structured = None
        #   for attempt in range(3):  # initial + 2 retries
        #       raw = llm.generate_content(full_prompt).text.strip()
        #       try:
        #           structured = SupportAnswer.model_validate_json(raw)
        #           break
        #       except Exception as exc:
        #           if attempt < 2:
        #               full_prompt = (
        #                   f"Your previous response failed schema validation: {exc}\n"
        #                   "Please correct and respond ONLY with valid JSON matching the schema.\n\n"
        #                   + schema_instruction
        #               )
        #           else:
        #               structured = SupportAnswer(
        #                   answer="[ERROR] Could not generate a valid structured response after 3 attempts.",
        #                   sources=retrieved_ids,
        #                   confidence=0.0,
        #               )
        raise NotImplementedError(
            "Real-LLM retrieve_and_answer not wired up yet. "
            "Set MOCK_LLM=1 or implement your LLM call above."
        )

    return {
        **state,
        "chunks":            retrieved_chunks,
        "chunk_ids":         retrieved_ids,
        "answer":            structured.answer,
        "structured_answer": structured.model_dump(),
    }


# Node 3 — direct_answer
def direct_answer(state: ZeptoState) -> ZeptoState:
    """
    For general_question queries:

    Mock mode (MOCK_LLM=1 or unset — graded baseline)
    ---------------------------------------------------
    Returns a fixed canned string. No LLM call is made.

    Optional real-LLM mode (MOCK_LLM=0)
    -------------------------------------
    Prompts the LLM directly (no retrieval).
    """
    if MOCK_LLM:
        answer_text = "I can only answer questions about Zepto policies right now."

        # Build SupportAnswer deterministically — no LLM output to validate
        structured = SupportAnswer(
            answer=answer_text,
            sources=[],     # no retrieval for general questions
            confidence=1.0, # fixed value in mock mode
        )
        print(f"  [direct_answer | MOCK] SupportAnswer built")
    else:
        # Optional real-LLM extension with retry-on-validation-failure
        #
        # Pseudocode (wire up your LLM SDK here):
        #
        #   import google.generativeai as genai
        #   llm = genai.GenerativeModel("gemini-1.5-flash")
        #   schema_instruction = (
        #       "Respond ONLY with a JSON object matching this schema — no extra text:\n"
        #       '{"answer": "<str>", "sources": [], "confidence": <float 0-1>}'
        #   )
        #   full_prompt = state["query"] + "\n\n" + schema_instruction
        #
        #   structured = None
        #   for attempt in range(3):  # initial + 2 retries
        #       raw = llm.generate_content(full_prompt).text.strip()
        #       try:
        #           structured = SupportAnswer.model_validate_json(raw)
        #           break
        #       except Exception as exc:
        #           if attempt < 2:
        #               full_prompt = (
        #                   f"Your previous response failed schema validation: {exc}\n"
        #                   "Please correct and respond ONLY with valid JSON matching the schema.\n\n"
        #                   + schema_instruction
        #               )
        #           else:
        #               structured = SupportAnswer(
        #                   answer="[ERROR] Could not generate a valid structured response after 3 attempts.",
        #                   sources=[],
        #                   confidence=0.0,
        #               )
        raise NotImplementedError(
            "Real-LLM direct_answer not wired up yet. "
            "Set MOCK_LLM=1 or implement your LLM call above."
        )

    return {
        **state,
        "answer":            structured.answer,
        "structured_answer": structured.model_dump(),
    }


def build_graph() -> StateGraph:
    """
    Assemble and compile the Zepto support LangGraph StateGraph.

    Graph topology
    --------------
    START
      │
      ▼
    classify_intent
      │
      ├── (policy_question)  ──► retrieve_and_answer ──► END
      └── (general_question) ──► direct_answer       ──► END

    Returns
    -------
    CompiledGraph
        A compiled LangGraph graph ready to invoke with a ZeptoState dict.
    """
    graph = StateGraph(ZeptoState)

    # Register nodes
    graph.add_node("classify_intent",    classify_intent)
    graph.add_node("retrieve_and_answer", retrieve_and_answer)
    graph.add_node("direct_answer",       direct_answer)

    # Entry edge
    graph.set_entry_point("classify_intent")

    # Conditional edge out of classify_intent — routing does NOT depend on MOCK_LLM
    graph.add_conditional_edges(
        "classify_intent",
        _route_intent,
        {
            "retrieve_and_answer": "retrieve_and_answer",
            "direct_answer":       "direct_answer",
        },
    )

    # Terminal edges
    graph.add_edge("retrieve_and_answer", END)
    graph.add_edge("direct_answer",        END)

    return graph.compile()


#Entry point
if __name__ == "__main__":
    # Build / refresh the ChromaDB collection
    chunk()

    # Demo the LangGraph pipeline
    print(f"\nMOCK_LLM = {MOCK_LLM}")
    app = build_graph()

    test_queries = [
        "What is the delivery fee for orders below INR 149?",  # policy_question
        "What is the weather like today?",                     # general_question
        "How do I cancel my order?",                           # policy_question
    ]

    for q in test_queries:
        print(f"\n{'='*60}")
        print(f"Query : {q}")
        initial_state: ZeptoState = {
            "query":             q,
            "intent":            "",
            "chunks":            [],
            "chunk_ids":         [],
            "answer":            "",
            "structured_answer": {},
        }
        result = app.invoke(initial_state)
        print(f"Intent: {result['intent']}")
        print(f"Structured JSON output:")
        print(json.dumps(result["structured_answer"], indent=2))
    print(f"{'='*60}")

