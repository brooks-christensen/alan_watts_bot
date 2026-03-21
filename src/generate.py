from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import faiss
import numpy as np
import yaml
from loguru import logger
from openai import OpenAI


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


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


def load_config(config_path: Path) -> tuple[RetrieveConfig, GenerateConfig]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}

    retrieve_section = raw.get("retrieve", {})
    generate_section = raw.get("generate", {})

    retrieve_config = RetrieveConfig(
        index_dir=PROJECT_ROOT / retrieve_section.get("index_dir", "data/indexes"),
        embedding_model=retrieve_section.get("embedding_model", "text-embedding-3-small"),
        top_k=int(retrieve_section.get("top_k", 5)),
    )

    generate_config = GenerateConfig(
        model=generate_section.get("model", "gpt-5.4"),
        temperature=float(generate_section.get("temperature", 0.2)),
        max_output_tokens=int(generate_section.get("max_output_tokens", 400)),
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
        max_context_chars_per_chunk=int(generate_section.get("max_context_chars_per_chunk", 2500)),
    )

    return retrieve_config, generate_config


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
            f"Index/metadata mismatch: index has {index.ntotal} vectors, "
            f"metadata has {len(metadata)} rows"
        )

    manifest_vectors = manifest.get("num_vectors")
    if manifest_vectors is not None and manifest_vectors != len(metadata):
        raise ValueError(
            f"Manifest/metadata mismatch: manifest says {manifest_vectors}, "
            f"metadata has {len(metadata)} rows"
        )

    manifest_dim = manifest.get("vector_dim")
    if manifest_dim is not None and manifest_dim != index.d:
        raise ValueError(
            f"Manifest/index dimension mismatch: manifest says {manifest_dim}, "
            f"index has dimension {index.d}"
        )


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm < 1e-12:
        raise ValueError("Query embedding norm is too small to normalize.")
    return vector / norm


def embed_query(client: OpenAI, query: str, model: str) -> np.ndarray:
    response = client.embeddings.create(
        model=model,
        input=query,
    )
    vector = np.asarray(response.data[0].embedding, dtype=np.float32)

    if vector.ndim != 1:
        raise ValueError(f"Expected 1D embedding vector, got shape {vector.shape}")

    return l2_normalize(vector).reshape(1, -1)


def retrieve_top_k(
    query_vector: np.ndarray,
    index: faiss.Index,
    metadata: List[Dict[str, Any]],
    top_k: int,
) -> List[Dict[str, Any]]:
    scores, indices = index.search(query_vector, top_k)

    top_scores = scores[0]
    top_indices = indices[0]

    results: List[Dict[str, Any]] = []
    for rank, (score, idx) in enumerate(zip(top_scores, top_indices), start=1):
        if idx < 0:
            continue

        row = metadata[idx]
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


def build_context_block(
    retrieved_chunks: List[Dict[str, Any]],
    max_chars_per_chunk: int,
) -> str:
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


def build_user_prompt(
    query: str,
    retrieved_chunks: List[Dict[str, Any]],
    max_chars_per_chunk: int,
) -> str:
    context_block = build_context_block(
        retrieved_chunks=retrieved_chunks,
        max_chars_per_chunk=max_chars_per_chunk,
    )

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
            {
                "role": "system",
                "content": [{"type": "input_text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": [{"type": "input_text", "text": user_prompt}],
            },
        ],
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )

    if not getattr(response, "output_text", None):
        raise ValueError("Generation model returned no output_text")

    return response.output_text.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate grounded Alan Watts-style answers with RAG.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="The user query to answer.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Optional override for number of retrieved chunks.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    retrieve_config, generate_config = load_config(Path(args.config))

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set.")

    client = OpenAI(api_key=api_key)

    index_path = retrieve_config.index_dir / "rag_faiss.index"
    metadata_path = retrieve_config.index_dir / "rag_metadata.jsonl"
    manifest_path = retrieve_config.index_dir / "rag_index_manifest.json"

    logger.info(f"Loading FAISS index from {index_path}")
    index = faiss.read_index(str(index_path))

    logger.info(f"Loading metadata from {metadata_path}")
    metadata = load_metadata(metadata_path)

    logger.info(f"Loading manifest from {manifest_path}")
    manifest = load_manifest(manifest_path)

    validate_index_alignment(index=index, metadata=metadata, manifest=manifest)

    top_k = args.top_k if args.top_k is not None else generate_config.top_k
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    logger.info(f"Embedding query with model: {retrieve_config.embedding_model}")
    query_vector = embed_query(
        client=client,
        query=args.query,
        model=retrieve_config.embedding_model,
    )

    logger.info(f"Retrieving top {top_k} chunks")
    retrieved_chunks = retrieve_top_k(
        query_vector=query_vector,
        index=index,
        metadata=metadata,
        top_k=top_k,
    )

    user_prompt = build_user_prompt(
        query=args.query,
        retrieved_chunks=retrieved_chunks,
        max_chars_per_chunk=generate_config.max_context_chars_per_chunk,
    )

    logger.info(f"Generating answer with model: {generate_config.model}")
    answer = generate_answer(
        client=client,
        model=generate_config.model,
        system_prompt=generate_config.system_prompt,
        user_prompt=user_prompt,
        temperature=generate_config.temperature,
        max_output_tokens=generate_config.max_output_tokens,
    )

    output = {
        "query": args.query,
        "retrieval_embedding_model": retrieve_config.embedding_model,
        "generation_model": generate_config.model,
        "top_k": top_k,
        "answer": answer,
        "retrieved_context": retrieved_chunks,
    }

    if args.pretty:
        print(json.dumps(output, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(output, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("Interrupted by user.")
        sys.exit(1)