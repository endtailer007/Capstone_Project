"""
app.py
------
FastAPI application wrapping the Zepto support LangGraph pipeline.

Endpoints
---------
POST /ask    -- Accepts {"query": str}, returns SupportAnswer JSON.
GET  /health -- Liveness check.

Run locally:
    python -m uvicorn app:app --port 8000
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from zepto_assistant import SupportAnswer, ZeptoState, build_graph, chunk


# ---------------------------------------------------------------------------
# Request model
# ---------------------------------------------------------------------------
class AskRequest(BaseModel):
    """Request body for POST /ask."""
    query: str


# ---------------------------------------------------------------------------
# App lifecycle — build ChromaDB + compile graph once at startup
# ---------------------------------------------------------------------------
_app_state: dict = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[startup] Building ChromaDB collection ...")
    chunk()
    print("[startup] Compiling LangGraph graph ...")
    _app_state["graph"] = build_graph()
    print("[startup] Ready.")
    yield
    _app_state.clear()


app = FastAPI(
    title="Zepto Support Assistant",
    description="RAG-powered customer support API backed by LangGraph + ChromaDB.",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.post(
    "/ask",
    response_model=SupportAnswer,
    summary="Answer a customer support question",
)
def ask(request: AskRequest) -> SupportAnswer:
    """
    Accepts a customer question, classifies it, optionally retrieves relevant
    policy chunks from ChromaDB, and returns a validated SupportAnswer JSON.
    """
    if not request.query.strip():
        raise HTTPException(status_code=422, detail="query must not be blank.")
    graph = _app_state.get("graph")
    if graph is None:
        raise HTTPException(status_code=503, detail="Graph not initialised.")

    initial_state: ZeptoState = {
        "query":             request.query,
        "intent":            "",
        "chunks":            [],
        "chunk_ids":         [],
        "answer":            "",
        "structured_answer": {},
    }
    result = graph.invoke(initial_state)
    return SupportAnswer(**result["structured_answer"])


@app.get("/health", summary="Health check")
def health():
    return {"status": "ok"}
