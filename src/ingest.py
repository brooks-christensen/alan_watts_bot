from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"

INPUT_FILE = DATA_DIR / "watts_rag_corpus_clean.txt"
OUTPUT_FILE = PROCESSED_DIR / "rag_chunks.jsonl"


def load_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    return path.read_text(encoding="utf-8")


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_paragraphs(text: str) -> List[str]:
    paragraphs = [p.strip() for p in text.split("\n\n")]
    return [p for p in paragraphs if p]


def chunk_paragraphs(
    paragraphs: List[str],
    target_chars: int = 1200,
    overlap_chars: int = 200,
) -> List[str]:
    chunks: List[str] = []
    current = ""

    for para in paragraphs:
        if not current:
            current = para
            continue

        if len(current) + 2 + len(para) <= target_chars:
            current += "\n\n" + para
        else:
            chunks.append(current)

            if overlap_chars > 0:
                overlap_text = current[-overlap_chars:]
                split_idx = overlap_text.find(" ")
                if split_idx != -1:
                    overlap_text = overlap_text[split_idx + 1 :]
                current = overlap_text + "\n\n" + para
            else:
                current = para

    if current:
        chunks.append(current)

    return [c.strip() for c in chunks if c.strip()]


def write_jsonl(chunks: List[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for idx, chunk in enumerate(chunks):
            record = {
                "chunk_id": idx,
                "text": chunk,
                "source": INPUT_FILE.name,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    raw_text = load_text(INPUT_FILE)
    clean_text = normalize_text(raw_text)
    paragraphs = split_paragraphs(clean_text)
    chunks = chunk_paragraphs(paragraphs)

    write_jsonl(chunks, OUTPUT_FILE)

    print(f"Loaded: {INPUT_FILE}")
    print(f"Paragraphs: {len(paragraphs)}")
    print(f"Chunks: {len(chunks)}")
    print(f"Wrote: {OUTPUT_FILE}")

    if chunks:
        print("\nSample chunk:\n")
        print(chunks[0][:1000])


if __name__ == "__main__":
    main()