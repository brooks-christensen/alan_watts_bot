from __future__ import annotations

import json
import os
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple, cast

import faiss
import httpx
import numpy as np
import yaml
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from openai import OpenAI
from pydantic import BaseModel, Field, field_validator
from starlette.responses import JSONResponse


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
TURNSTILE_VERIFY_URL = "https://challenges.cloudflare.com/turnstile/v0/siteverify"


@dataclass
class RetrieveConfig:
    index_dir: Path
    embedding_model: str
    top_k: int


@dataclass
class GenerateConfig:
    model: str
    temperature: float
    max_output_tokens: int
    system_prompt: str
    top_k: int
    max_context_chars_per_chunk: int


@dataclass
class AppConfig:
    retrieve: RetrieveConfig
    generate: GenerateConfig
    max_query_chars: int
    rate_limit_requests: int
    rate_limit_window_seconds: int
    allowed_origins: List[str]
    require_turnstile: bool


class AskRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=600)
    turnstile_token: Optional[str] = Field(default=None, max_length=4096)
    top_k: Optional[int] = Field(default=None, ge=1, le=5)

    @field_validator("query")
    @classmethod
    def normalize_query(cls, value: str) -> str:
        value = " ".join(value.split())
        if len(value) < 3:
            raise ValueError("Query is too short.")
        return value


class AskResponse(BaseModel):
    answer: str
    retrieved_context: List[Dict[str, Any]]
    generation_model: str
    embedding_model: str
    top_k: int


class HealthResponse(BaseModel):
    status: str
    index_vectors: int
    metadata_rows: int
    embedding_model: str
    generation_model: str


class AppState:
    def __init__(self) -> None:
        self.config: Optional[AppConfig] = None
        self.client: Optional[OpenAI] = None
        self.index: Optional[faiss.Index] = None
        self.metadata: Optional[List[Dict[str, Any]]] = None
        self.manifest: Optional[Dict[str, Any]] = None
        self.rate_buckets: Dict[str, Deque[float]] = defaultdict(deque)


APP_STATE = AppState()


def load_config(config_path: Path) -> AppConfig:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}

    retrieve_section = raw.get("retrieve", {})
    generate_section = raw.get("generate", {})
    api_section = raw.get("api", {})

    retrieve_config = RetrieveConfig(
        index_dir=PROJECT_ROOT / retrieve_section.get("index_dir", "data/indexes"),
        embedding_model=retrieve_section.get("embedding_model", "text-embedding-3-small"),
        top_k=int(retrieve_section.get("top_k", 3)),
    )

    generate_config = GenerateConfig(
        model=generate_section.get("model", "gpt-5.4-mini"),
        temperature=float(generate_section.get("temperature", 0.2)),
        max_output_tokens=int(generate_section.get("max_output_tokens", 320)),
        system_prompt=generate_section.get(
            "system_prompt",
            (
                "You are a reflective conversational assistant grounded in Alan Watts source material. "
                "Answer the user's question using the provided excerpts as your primary basis. "
                "Be calm, lucid, and conceptually clear. "
                "Do not imitate theatrically. "
                "Do not mention transcripts, chunks, retrieval, or source documents. "
                "Do not invent outside facts or biography. "
                "If the source material supports multiple angles, synthesize them clearly and naturally. "
                "Keep the answer concise but meaningful."
            ),
        ),
        top_k=int(generate_section.get("top_k", retrieve_config.top_k)),
        max_context_chars_per_chunk=int(generate_section.get("max_context_chars_per_chunk", 1800)),
    )

    allowed_origins = api_section.get(
        "allowed_origins",
        [
            "https://brookschristensen.com",
            "https://www.brookschristensen.com",
            "http://localhost:3000",
            "http://127.0.0.1:3000",
        ],
    )

    return AppConfig(
        retrieve=retrieve_config,
        generate=generate_config,
        max_query_chars=int(api_section.get("max_query_chars", 600)),
        rate_limit_requests=int(api_section.get("rate_limit_requests", 12)),
        rate_limit_window_seconds=int(api_section.get("rate_limit_window_seconds", 300)),
        allowed_origins=[str(origin).rstrip("/") for origin in allowed_origins],
        require_turnstile=bool(api_section.get("require_turnstile", True)),
    )


def load_metadata(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Metadata JSONL not found: {path}")

    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_num} of {path}: {e}") from e

    return rows


def load_manifest(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Index manifest not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def validate_index_alignment(
    index: faiss.Index,
    metadata: List[Dict[str, Any]],
    manifest: Dict[str, Any],
) -> None:
    if index.ntotal != len(metadata):
        raise ValueError(
            f"Index/metadata mismatch: index has {index.ntotal} vectors, metadata has {len(metadata)} rows"
        )

    manifest_vectors = manifest.get("num_vectors")
    if manifest_vectors is not None and manifest_vectors != len(metadata):
        raise ValueError(
            f"Manifest/metadata mismatch: manifest says {manifest_vectors}, metadata has {len(metadata)} rows"
        )

    manifest_dim = manifest.get("vector_dim")
    if manifest_dim is not None and manifest_dim != index.d:
        raise ValueError(
            f"Manifest/index dimension mismatch: manifest says {manifest_dim}, index has dimension {index.d}"
        )


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm < 1e-12:
        raise ValueError("Query embedding norm is too small to normalize.")
    return vector / norm


def embed_query(client: OpenAI, query: str, model: str) -> np.ndarray:
    response = client.embeddings.create(model=model, input=query)
    vector = np.asarray(response.data[0].embedding, dtype=np.float32)

    if vector.ndim != 1:
        raise ValueError(f"Expected 1D embedding vector, got shape {vector.shape}")

    vector = np.ascontiguousarray(l2_normalize(vector).reshape(1, -1), dtype=np.float32)
    return vector


def retrieve_top_k(
    query_vector: np.ndarray,
    index: faiss.Index,
    metadata: List[Dict[str, Any]],
    top_k: int,
) -> List[Dict[str, Any]]:
    scores, indices = cast(Any, index).search(query_vector, top_k)

    top_scores = scores[0]
    top_indices = indices[0]

    results: List[Dict[str, Any]] = []
    for rank, (score, idx) in enumerate(zip(top_scores, top_indices), start=1):
        if idx < 0:
            continue

        row = metadata[int(idx)]
        results.append(
            {
                "rank": rank,
                "score": float(score),
                "chunk_id": row["chunk_id"],
                "source": row["source"],
                "text": row["text"],
            }
        )

    return results


def truncate_context_text(text: str, max_chars: int) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + " ..."


def build_context_block(retrieved_chunks: List[Dict[str, Any]], max_chars_per_chunk: int) -> str:
    parts: List[str] = []
    for chunk in retrieved_chunks:
        chunk_text = truncate_context_text(chunk["text"], max_chars_per_chunk)
        parts.append(
            "\n".join(
                [
                    f"[Source Rank {chunk['rank']}]",
                    f"chunk_id: {chunk['chunk_id']}",
                    f"source: {chunk['source']}",
                    f"score: {chunk['score']:.4f}",
                    "excerpt:",
                    chunk_text,
                ]
            )
        )
    return "\n\n---\n\n".join(parts)


def build_user_prompt(query: str, retrieved_chunks: List[Dict[str, Any]], max_chars_per_chunk: int) -> str:
    context_block = build_context_block(retrieved_chunks, max_chars_per_chunk=max_chars_per_chunk)
    return f"""Answer the user's question using the retrieved Alan Watts source excerpts below.

User question:
{query}

Retrieved source excerpts:
{context_block}

Instructions:
- Use the excerpts as the primary grounding for your answer.
- Synthesize naturally rather than quoting excessively.
- Do not mention retrieval, chunks, transcripts, or source excerpts explicitly.
- Avoid unsupported outside claims.
- Keep the answer conversational, reflective, and clear.
"""


def generate_answer(
    client: OpenAI,
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_output_tokens: int,
) -> str:
    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "input_text", "text": user_prompt}]},
        ],
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )

    if not getattr(response, "output_text", None):
        raise ValueError("Generation model returned no output_text")

    return response.output_text.strip()


def get_client_ip(request: Request) -> str:
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    if request.client:
        return request.client.host
    return "unknown"


def enforce_rate_limit(request: Request) -> None:
    config = APP_STATE.config
    if config is None:
        raise RuntimeError("App config is not loaded.")

    client_ip = get_client_ip(request)
    now = time.time()
    bucket = APP_STATE.rate_buckets[client_ip]
    window_start = now - config.rate_limit_window_seconds

    while bucket and bucket[0] < window_start:
        bucket.popleft()

    if len(bucket) >= config.rate_limit_requests:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=(
                "Too many requests from this client. Please wait a few minutes and try again."
            ),
        )

    bucket.append(now)


async def verify_turnstile_token(request: Request, token: Optional[str]) -> None:
    config = APP_STATE.config
    if config is None:
        raise RuntimeError("App config is not loaded.")

    if not config.require_turnstile:
        return

    secret = os.getenv("TURNSTILE_SECRET_KEY")
    if not secret:
        raise RuntimeError("TURNSTILE_SECRET_KEY is not configured.")

    if not token:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing Turnstile token.",
        )

    payload = {
        "secret": secret,
        "response": token,
        "remoteip": get_client_ip(request),
    }

    async with httpx.AsyncClient(timeout=5.0) as client:
        response = await client.post(TURNSTILE_VERIFY_URL, data=payload)
        response.raise_for_status()
        body = response.json()

    if not body.get("success", False):
        logger.warning("Turnstile validation failed: {}", body.get("error-codes", []))
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Human verification failed. Please refresh the page and try again.",
        )


def redact_context(chunks: List[Dict[str, Any]], max_return_chars: int = 500) -> List[Dict[str, Any]]:
    redacted: List[Dict[str, Any]] = []
    for chunk in chunks:
        redacted.append(
            {
                "rank": chunk["rank"],
                "score": chunk["score"],
                "chunk_id": chunk["chunk_id"],
                "source": chunk["source"],
                "text": truncate_context_text(chunk["text"], max_return_chars),
            }
        )
    return redacted


@asynccontextmanager
async def lifespan(_: FastAPI):
    config = load_config(Path(os.getenv("APP_CONFIG_PATH", DEFAULT_CONFIG_PATH)))

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    index_path = config.retrieve.index_dir / "rag_faiss.index"
    metadata_path = config.retrieve.index_dir / "rag_metadata.jsonl"
    manifest_path = config.retrieve.index_dir / "rag_index_manifest.json"

    logger.info("Loading FAISS index from {}", index_path)
    index = faiss.read_index(str(index_path))

    logger.info("Loading metadata from {}", metadata_path)
    metadata = load_metadata(metadata_path)

    logger.info("Loading manifest from {}", manifest_path)
    manifest = load_manifest(manifest_path)

    validate_index_alignment(index=index, metadata=metadata, manifest=manifest)

    APP_STATE.config = config
    APP_STATE.client = OpenAI(api_key=api_key)
    APP_STATE.index = index
    APP_STATE.metadata = metadata
    APP_STATE.manifest = manifest

    logger.info(
        "Application startup complete. index_vectors={}, embedding_model={}, generation_model={}",
        index.ntotal,
        config.retrieve.embedding_model,
        config.generate.model,
    )
    yield


BOOT_CONFIG_PATH = Path(os.getenv("APP_CONFIG_PATH", DEFAULT_CONFIG_PATH))
BOOT_CONFIG = load_config(BOOT_CONFIG_PATH) if BOOT_CONFIG_PATH.exists() else None
BOOT_ALLOWED_ORIGINS = (
    BOOT_CONFIG.allowed_origins
    if BOOT_CONFIG is not None
    else [
        "https://brookschristensen.com",
        "https://www.brookschristensen.com",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]
)

app = FastAPI(title="Alan Watts Chatbot API", version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def security_headers_middleware(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Cache-Control"] = "no-store"
    return response


# CORS is intentionally narrow. Add origins explicitly rather than using '*'.
app.add_middleware(
    CORSMiddleware,
    allow_origins=BOOT_ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["Content-Type"],
)


@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.exception_handler(Exception)
async def unhandled_exception_handler(_: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled application error: {}", exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error."},
    )


@app.get("/health", response_model=HealthResponse)
async def healthz() -> HealthResponse:
    if APP_STATE.config is None or APP_STATE.index is None or APP_STATE.metadata is None:
        raise HTTPException(status_code=503, detail="Application not ready.")

    return HealthResponse(
        status="ok",
        index_vectors=APP_STATE.index.ntotal,
        metadata_rows=len(APP_STATE.metadata),
        embedding_model=APP_STATE.config.retrieve.embedding_model,
        generation_model=APP_STATE.config.generate.model,
    )


@app.post("/ask", response_model=AskResponse)
async def ask(payload: AskRequest, request: Request) -> AskResponse:
    if APP_STATE.config is None or APP_STATE.client is None or APP_STATE.index is None or APP_STATE.metadata is None:
        raise HTTPException(status_code=503, detail="Application not ready.")

    enforce_rate_limit(request)
    await verify_turnstile_token(request, payload.turnstile_token)

    config = APP_STATE.config
    query = payload.query.strip()
    if len(query) > config.max_query_chars:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Query exceeds max length of {config.max_query_chars} characters.",
        )

    top_k = payload.top_k if payload.top_k is not None else config.generate.top_k
    top_k = min(max(top_k, 1), 5)

    client_ip = get_client_ip(request)
    logger.info(
        "Processing ask request: ip={}, query_len={}, top_k={}",
        client_ip,
        len(query),
        top_k,
    )

    query_vector = embed_query(
        client=APP_STATE.client,
        query=query,
        model=config.retrieve.embedding_model,
    )

    retrieved_chunks = retrieve_top_k(
        query_vector=query_vector,
        index=APP_STATE.index,
        metadata=APP_STATE.metadata,
        top_k=top_k,
    )

    user_prompt = build_user_prompt(
        query=query,
        retrieved_chunks=retrieved_chunks,
        max_chars_per_chunk=config.generate.max_context_chars_per_chunk,
    )

    answer = generate_answer(
        client=APP_STATE.client,
        model=config.generate.model,
        system_prompt=config.generate.system_prompt,
        user_prompt=user_prompt,
        temperature=config.generate.temperature,
        max_output_tokens=config.generate.max_output_tokens,
    )

    return AskResponse(
        answer=answer,
        retrieved_context=redact_context(retrieved_chunks),
        generation_model=config.generate.model,
        embedding_model=config.retrieve.embedding_model,
        top_k=top_k,
    )
