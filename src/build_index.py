from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import faiss
import numpy as np
import yaml
from loguru import logger
from openai import OpenAI
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


@dataclass
class BuildIndexConfig:
    input_path: Path
    output_dir: Path
    embedding_model: str
    batch_size: int
    overwrite: bool
    sleep_seconds: float
    max_retries: int


def load_config(config_path: Path) -> BuildIndexConfig:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    section = raw.get("build_index", {})

    return BuildIndexConfig(
        input_path=PROJECT_ROOT / section.get("rag_chunks_path", "data/processed/rag_chunks.jsonl"),
        output_dir=PROJECT_ROOT / section.get("index_output_dir", "data/indexes"),
        embedding_model=section.get("embedding_model", "text-embedding-3-small"),
        batch_size=int(section.get("embedding_batch_size", 32)),
        overwrite=bool(section.get("index_overwrite", False)),
        sleep_seconds=float(section.get("embedding_sleep_seconds", 0.1)),
        max_retries=int(section.get("embedding_max_retries", 3)),
    )


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Input JSONL not found: {path}")

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

def split_text_for_embedding(
    text: str,
    max_chars: int = 24000,
    overlap_chars: int = 1000,
) -> List[str]:
    """
    Approximate safeguard against embedding token limits.

    We do not have exact token counting here, so we use a conservative
    character-based window. This is usually sufficient to stay under the
    8192-token limit for English prose.

    If the text is short enough, return it unchanged.
    """
    text = text.strip()
    if len(text) <= max_chars:
        return [text]

    pieces: List[str] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + max_chars, text_len)
        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)

        if end >= text_len:
            break

        start = max(end - overlap_chars, 0)

    return pieces


def embed_single_text(
    client: OpenAI,
    text: str,
    model: str,
    max_retries: int,
) -> np.ndarray:
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            response = client.embeddings.create(
                model=model,
                input=text,
            )
            vector = np.asarray(response.data[0].embedding, dtype=np.float32)
            if vector.ndim != 1:
                raise ValueError(f"Expected 1D embedding vector, got shape {vector.shape}")
            return vector

        except Exception as e:
            last_error = e
            logger.warning(
                f"Single embedding failed on attempt {attempt}/{max_retries}: {e}"
            )
            time.sleep(min(2.0 * attempt, 10.0))

    raise RuntimeError(f"Single embedding failed after {max_retries} attempts: {last_error}")


def embed_record_text(
    client: OpenAI,
    text: str,
    model: str,
    max_retries: int,
) -> np.ndarray:
    """
    Embed one logical chunk.

    If it is too large, split it into embedding-safe pieces, embed each piece,
    and mean-pool the piece vectors into one final vector for the original chunk.
    """
    pieces = split_text_for_embedding(text)

    if len(pieces) == 1:
        return embed_single_text(
            client=client,
            text=pieces[0],
            model=model,
            max_retries=max_retries,
        )

    logger.info(f"Large chunk detected; embedding as {len(pieces)} sub-pieces and pooling.")

    piece_vectors: List[np.ndarray] = []
    for piece in pieces:
        vec = embed_single_text(
            client=client,
            text=piece,
            model=model,
            max_retries=max_retries,
        )
        piece_vectors.append(vec)

    stacked = np.vstack(piece_vectors).astype(np.float32)
    pooled = stacked.mean(axis=0)
    return pooled


def validate_chunk_record(record: Dict[str, Any]) -> None:
    required = {"chunk_id", "text", "source"}
    missing = required - record.keys()
    if missing:
        raise ValueError(f"Chunk record missing fields: {sorted(missing)}")

    if not isinstance(record["chunk_id"], int):
        raise ValueError(f"chunk_id must be int, got {type(record['chunk_id'])}")

    if not isinstance(record["text"], str) or not record["text"].strip():
        raise ValueError("text must be a non-empty string")

    if not isinstance(record["source"], str) or not record["source"].strip():
        raise ValueError("source must be a non-empty string")


def chunked(seq: Sequence[Dict[str, Any]], batch_size: int) -> List[List[Dict[str, Any]]]:
    return [list(seq[i:i + batch_size]) for i in range(0, len(seq), batch_size)]


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=1e-12, a_max=None)
    return vectors / norms


def write_metadata_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_index_manifest(
    path: Path,
    *,
    input_path: Path,
    index_path: Path,
    metadata_path: Path,
    embedding_model: str,
    vector_dim: int,
    num_vectors: int,
    normalized: bool,
    batch_size: int,
) -> None:
    manifest = {
        "input_path": str(input_path),
        "index_path": str(index_path),
        "metadata_path": str(metadata_path),
        "embedding_model": embedding_model,
        "vector_dim": vector_dim,
        "num_vectors": num_vectors,
        "normalized": normalized,
        "batch_size": batch_size,
        "faiss_index_type": "IndexFlatIP",
    }
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build FAISS index for Alan Watts RAG chunks.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to YAML config file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(Path(args.config))

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set.")

    client = OpenAI(api_key=api_key)

    logger.info(f"Loading chunk records from {config.input_path}")
    records = load_jsonl(config.input_path)

    for record in records:
        validate_chunk_record(record)

    logger.info(f"Loaded {len(records)} chunk records")
    logger.info(f"Using embedding model: {config.embedding_model}")
    logger.info(f"Batch size: {config.batch_size}")

    config.output_dir.mkdir(parents=True, exist_ok=True)

    index_path = config.output_dir / "rag_faiss.index"
    metadata_path = config.output_dir / "rag_metadata.jsonl"
    manifest_path = config.output_dir / "rag_index_manifest.json"

    existing_outputs = [index_path, metadata_path, manifest_path]
    if any(p.exists() for p in existing_outputs) and not config.overwrite:
        raise FileExistsError(
            "Index output files already exist. "
            "Set index_overwrite: true in config.yaml if you want to replace them."
        )

    batches = chunked(records, config.batch_size)

    all_vectors: List[np.ndarray] = []
    metadata_records: List[Dict[str, Any]] = []

    for batch in tqdm(batches, desc="Embedding chunks"):
        batch_vectors: List[np.ndarray] = []

        for r in batch:
            vec = embed_record_text(
                client=client,
                text=r["text"],
                model=config.embedding_model,
                max_retries=config.max_retries,
            )
            batch_vectors.append(vec)

            metadata_records.append(
                {
                    "chunk_id": r["chunk_id"],
                    "source": r["source"],
                    "text": r["text"],
                    "embedded_with_pooling": len(split_text_for_embedding(r["text"])) > 1,
                }
            )

            time.sleep(config.sleep_seconds)

        all_vectors.append(np.vstack(batch_vectors).astype(np.float32))

    vectors = np.vstack(all_vectors).astype(np.float32)

    if len(vectors) != len(records):
        raise RuntimeError(
            f"Vector count mismatch: got {len(vectors)} vectors for {len(records)} records"
        )

    logger.info(f"Embedding matrix shape before normalization: {vectors.shape}")

    # Normalize and use inner product index => cosine similarity retrieval.
    vectors = l2_normalize(vectors)
    vector_dim = vectors.shape[1]

    index = faiss.IndexFlatIP(vector_dim)

    vectors = np.ascontiguousarray(vectors, dtype=np.float32)

    if vectors.ndim != 2:
        raise ValueError(f"Expected 2D embedding matrix, got shape {vectors.shape}")

    index.add(vectors)

    logger.info(f"Built FAISS index with {index.ntotal} vectors and dimension {vector_dim}")

    faiss.write_index(index, str(index_path))
    write_metadata_jsonl(metadata_path, metadata_records)
    write_index_manifest(
        manifest_path,
        input_path=config.input_path,
        index_path=index_path,
        metadata_path=metadata_path,
        embedding_model=config.embedding_model,
        vector_dim=vector_dim,
        num_vectors=len(records),
        normalized=True,
        batch_size=config.batch_size,
    )

    logger.info(f"Wrote FAISS index to {index_path}")
    logger.info(f"Wrote metadata to {metadata_path}")
    logger.info(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("Interrupted by user.")
        sys.exit(1)