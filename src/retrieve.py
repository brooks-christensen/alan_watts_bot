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


def load_config(config_path: Path) -> RetrieveConfig:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    section = raw.get("retrieve", {})

    return RetrieveConfig(
        index_dir=PROJECT_ROOT / section.get("index_dir", "data/indexes"),
        embedding_model=section.get("embedding_model", "text-embedding-3-small"),
        top_k=int(section.get("top_k", 5)),
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrieve top-k Alan Watts chunks for a query.")
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
        help="The user query to retrieve against.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Optional override for number of results to return.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(Path(args.config))

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set.")

    client = OpenAI(api_key=api_key)

    index_path = config.index_dir / "rag_faiss.index"
    metadata_path = config.index_dir / "rag_metadata.jsonl"
    manifest_path = config.index_dir / "rag_index_manifest.json"

    logger.info(f"Loading FAISS index from {index_path}")
    index = faiss.read_index(str(index_path))

    logger.info(f"Loading metadata from {metadata_path}")
    metadata = load_metadata(metadata_path)

    logger.info(f"Loading manifest from {manifest_path}")
    manifest = load_manifest(manifest_path)

    validate_index_alignment(index=index, metadata=metadata, manifest=manifest)

    top_k = args.top_k if args.top_k is not None else config.top_k
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    logger.info(f"Embedding query with model: {config.embedding_model}")
    query_vector = embed_query(
        client=client,
        query=args.query,
        model=config.embedding_model,
    )

    logger.info(f"Searching top {top_k} results")
    results = retrieve_top_k(
        query_vector=query_vector,
        index=index,
        metadata=metadata,
        top_k=top_k,
    )

    output = {
        "query": args.query,
        "embedding_model": config.embedding_model,
        "top_k": top_k,
        "results": results,
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