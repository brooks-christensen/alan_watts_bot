from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "dialogic_training_pairs_full.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "data" / "splits"

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10
RANDOM_SEED = 42


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_num}: {e}") from e
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    if abs((TRAIN_RATIO + VAL_RATIO + TEST_RATIO) - 1.0) > 1e-9:
        raise ValueError("TRAIN_RATIO + VAL_RATIO + TEST_RATIO must equal 1.0")

    rows = load_jsonl(INPUT_PATH)

    # Group all records by chunk_id to prevent leakage across splits.
    by_chunk: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        metadata = row.get("metadata", {})
        if "chunk_id" not in metadata:
            raise KeyError("Each record must contain metadata.chunk_id")
        chunk_id = int(metadata["chunk_id"])
        by_chunk[chunk_id].append(row)

    chunk_ids = sorted(by_chunk.keys())
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(chunk_ids)

    n_chunks = len(chunk_ids)
    n_train = int(n_chunks * TRAIN_RATIO)
    n_val = int(n_chunks * VAL_RATIO)
    # Put the remainder in test so total is exact.
    n_test = n_chunks - n_train - n_val

    train_chunk_ids = set(chunk_ids[:n_train])
    val_chunk_ids = set(chunk_ids[n_train:n_train + n_val])
    test_chunk_ids = set(chunk_ids[n_train + n_val:])

    train_rows: List[Dict[str, Any]] = []
    val_rows: List[Dict[str, Any]] = []
    test_rows: List[Dict[str, Any]] = []

    for chunk_id, group in by_chunk.items():
        if chunk_id in train_chunk_ids:
            train_rows.extend(group)
        elif chunk_id in val_chunk_ids:
            val_rows.extend(group)
        elif chunk_id in test_chunk_ids:
            test_rows.extend(group)
        else:
            raise RuntimeError(f"Chunk {chunk_id} was not assigned to any split")

    train_path = OUTPUT_DIR / "dialogic_train.jsonl"
    val_path = OUTPUT_DIR / "dialogic_val.jsonl"
    test_path = OUTPUT_DIR / "dialogic_test.jsonl"
    manifest_path = OUTPUT_DIR / "split_manifest.json"

    write_jsonl(train_path, train_rows)
    write_jsonl(val_path, val_rows)
    write_jsonl(test_path, test_rows)

    manifest = {
        "input_path": str(INPUT_PATH),
        "random_seed": RANDOM_SEED,
        "ratios": {
            "train": TRAIN_RATIO,
            "val": VAL_RATIO,
            "test": TEST_RATIO,
        },
        "counts": {
            "chunks_total": n_chunks,
            "chunks_train": len(train_chunk_ids),
            "chunks_val": len(val_chunk_ids),
            "chunks_test": len(test_chunk_ids),
            "records_total": len(rows),
            "records_train": len(train_rows),
            "records_val": len(val_rows),
            "records_test": len(test_rows),
        },
        "chunk_ids": {
            "train": sorted(train_chunk_ids),
            "val": sorted(val_chunk_ids),
            "test": sorted(test_chunk_ids),
        },
    }

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Total chunks: {n_chunks}")
    print(f"Train chunks: {len(train_chunk_ids)} | records: {len(train_rows)}")
    print(f"Val chunks:   {len(val_chunk_ids)} | records: {len(val_rows)}")
    print(f"Test chunks:  {len(test_chunk_ids)} | records: {len(test_rows)}")
    print(f"\nWrote:")
    print(f"  {train_path}")
    print(f"  {val_path}")
    print(f"  {test_path}")
    print(f"  {manifest_path}")


if __name__ == "__main__":
    main()