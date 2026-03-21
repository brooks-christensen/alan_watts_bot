from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import yaml
from loguru import logger
from openai import OpenAI
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


ANSWER_SCHEMA: Dict[str, Any] = {
    "name": "dialogic_generation",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "qa_pairs": {
                "type": "array",
                "minItems": 1,
                "maxItems": 3,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "question": {"type": "string"},
                        "answer": {"type": "string"},
                    },
                    "required": ["question", "answer"],
                },
            }
        },
        "required": ["qa_pairs"],
    },
}


SYSTEM_PROMPT = """You are generating dialogic Alan Watts-style training examples from structured semantic analysis.

Your goals:
1. Answer each user question using ONLY the provided source materials.
2. Write in a calm, reflective, conceptually clear tone compatible with the source material.
3. Do NOT imitate theatrically, do NOT add fake biography, and do NOT mention transcripts, chunks, lectures, or source documents.
4. Stay grounded in the source_text, semantic_summary, core_claim, themes, and tone_notes.
5. Each answer must feel like a direct response to the specific user question.
6. Prefer concise, meaningful answers over long essays.
7. Do not introduce outside facts, names, or references not present in the provided materials.
8. Do not use phrases like "in this chunk", "the speaker says", "the passage suggests", or "Alan Watts says here".
9. Return valid JSON only.
"""


USER_PROMPT_TEMPLATE = """Generate one grounded answer for each candidate question.

chunk_id: {chunk_id}
source: {source}

semantic_summary:
{semantic_summary}

core_claim:
{core_claim}

themes:
{themes}

tone_notes:
{tone_notes}

source_text:
\"\"\"
{source_text}
\"\"\"

candidate_questions:
{candidate_questions}

Requirements:
- Return exactly one answer for each candidate question.
- Preserve the exact question text.
- Each answer should usually be about 70-160 words.
- Answers should be distinct where the questions are distinct.
- Answers should be grounded, concise, and usable for training a conversational model.
"""


@dataclass
class ValidateConfig:
    input_path: Path
    enriched_output_path: Path
    training_output_path: Path
    model: str
    temperature: float
    max_records: Optional[int]
    overwrite: bool
    sleep_seconds: float
    max_retries: int


def load_config(config_path: Path) -> ValidateConfig:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    section = raw.get("validate_dialogic", {})

    return ValidateConfig(
        input_path=PROJECT_ROOT / section.get(
            "input_path",
            "data/processed/dialogic_analysis_full.jsonl",
        ),
        enriched_output_path=PROJECT_ROOT / section.get(
            "enriched_output_path",
            "data/processed/dialogic_dataset_enriched.jsonl",
        ),
        training_output_path=PROJECT_ROOT / section.get(
            "training_output_path",
            "data/processed/dialogic_training_pairs.jsonl",
        ),
        model=section.get("model", "gpt-5.4"),
        temperature=float(section.get("temperature", 0.2)),
        max_records=section.get("max_records"),
        overwrite=bool(section.get("overwrite", False)),
        sleep_seconds=float(section.get("sleep_seconds", 0.25)),
        max_retries=int(section.get("max_retries", 2)),
    )


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Input JSONL not found: {path}")

    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_num} of {path}: {e}") from e
    return records


def load_completed_chunk_ids(path: Path) -> Set[int]:
    if not path.exists():
        return set()

    completed: Set[int] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if obj.get("generation_status") == "ok":
                    completed.add(int(obj["chunk_id"]))
            except Exception:
                continue
    return completed


def normalize_spaces(text: str) -> str:
    return " ".join(text.strip().split())


def dedupe_preserve_order(items: List[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for item in items:
        norm = normalize_spaces(item)
        if not norm:
            continue
        key = norm.lower()
        if key not in seen:
            seen.add(key)
            out.append(norm)
    return out


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z][a-zA-Z\-']+", text.lower())


def build_stopwords() -> Set[str]:
    return {
        "a", "an", "the", "and", "or", "but", "if", "then", "than", "so", "of", "to",
        "in", "on", "for", "with", "by", "as", "at", "from", "that", "this", "these",
        "those", "is", "are", "was", "were", "be", "been", "being", "it", "its", "into",
        "about", "through", "because", "what", "why", "how", "when", "where", "who",
        "whom", "which", "do", "does", "did", "can", "could", "would", "should", "may",
        "might", "will", "shall", "i", "you", "we", "they", "he", "she", "them", "his",
        "her", "our", "their", "your", "my", "me", "us", "not", "no", "yes", "all",
        "more", "most", "much", "many", "very", "just", "than", "also", "there", "here",
        "over", "under", "again", "still", "only", "such",
    }


STOPWORDS = build_stopwords()

BANNED_PHRASES = [
    "in this chunk",
    "this chunk",
    "the speaker says",
    "the passage says",
    "the passage suggests",
    "the transcript",
    "this lecture",
    "alan watts says here",
]


def content_overlap_ratio(answer: str, source_text: str, semantic_summary: str, core_claim: str) -> float:
    answer_tokens = {t for t in tokenize(answer) if t not in STOPWORDS}
    source_tokens = {
        t for t in tokenize(" ".join([source_text, semantic_summary, core_claim]))
        if t not in STOPWORDS
    }
    if not answer_tokens:
        return 0.0
    overlap = answer_tokens & source_tokens
    return len(overlap) / max(len(answer_tokens), 1)


def contains_banned_phrase(text: str) -> bool:
    low = text.lower()
    return any(phrase in low for phrase in BANNED_PHRASES)


def answer_word_count(text: str) -> int:
    return len(text.split())


def themes_present(answer: str, themes: List[str]) -> bool:
    low = answer.lower()
    return any(theme.lower() in low for theme in themes)


def validate_qa_pairs(
    record: Dict[str, Any],
    qa_pairs: List[Dict[str, str]],
) -> Tuple[bool, Optional[str], List[Dict[str, str]]]:
    input_questions = [normalize_spaces(q) for q in record["candidate_questions"]]
    output_questions = [normalize_spaces(p["question"]) for p in qa_pairs]

    if len(qa_pairs) != len(input_questions):
        return False, "qa_pair_count_mismatch", qa_pairs

    if output_questions != input_questions:
        return False, "question_order_or_text_mismatch", qa_pairs

    validated_pairs: List[Dict[str, str]] = []

    for pair in qa_pairs:
        q = normalize_spaces(pair["question"])
        a = normalize_spaces(pair["answer"])

        if not a:
            return False, f"empty_answer_for_question:{q}", qa_pairs

        wc = answer_word_count(a)
        if wc < 45:
            return False, f"answer_too_short_for_question:{q}", qa_pairs
        if wc > 220:
            return False, f"answer_too_long_for_question:{q}", qa_pairs

        if contains_banned_phrase(a):
            return False, f"banned_phrase_in_answer:{q}", qa_pairs

        theme_reflected = themes_present(a, record["themes"])
        overlap = content_overlap_ratio(
            answer=a,
            source_text=record["source_text"],
            semantic_summary=record["semantic_summary"],
            core_claim=record["core_claim"],
        )

        if overlap < 0.08:
            return False, f"low_source_overlap:{q}", qa_pairs

        if not theme_reflected and overlap < 0.14:
            return False, f"weak_grounding_and_theme_reflection:{q}", qa_pairs

        validated_pairs.append({"question": q, "answer": a})

    return True, None, validated_pairs


def build_input_messages(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    candidate_questions_json = json.dumps(record["candidate_questions"], ensure_ascii=False)
    themes_json = json.dumps(record["themes"], ensure_ascii=False)
    tone_notes_json = json.dumps(record["tone_notes"], ensure_ascii=False)

    user_prompt = USER_PROMPT_TEMPLATE.format(
        chunk_id=record["chunk_id"],
        source=record["source"],
        semantic_summary=record["semantic_summary"],
        core_claim=record["core_claim"],
        themes=themes_json,
        tone_notes=tone_notes_json,
        source_text=record["source_text"],
        candidate_questions=candidate_questions_json,
    )

    return [
        {
            "role": "system",
            "content": [{"type": "input_text", "text": SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [{"type": "input_text", "text": user_prompt}],
        },
    ]


def call_model(
    client: OpenAI,
    record: Dict[str, Any],
    model: str,
    temperature: float,
) -> List[Dict[str, str]]:
    response = client.responses.create(
        model=model,
        input=build_input_messages(record),
        temperature=temperature,
        text={
            "format": {
                "type": "json_schema",
                "name": ANSWER_SCHEMA["name"],
                "schema": ANSWER_SCHEMA["schema"],
                "strict": True,
            }
        },
    )

    if not getattr(response, "output_text", None):
        raise ValueError("Model returned no output_text")

    parsed = json.loads(response.output_text)
    qa_pairs = parsed["qa_pairs"]

    clean_pairs: List[Dict[str, str]] = []
    for pair in qa_pairs:
        clean_pairs.append(
            {
                "question": normalize_spaces(pair["question"]),
                "answer": normalize_spaces(pair["answer"]),
            }
        )
    return clean_pairs


def build_enriched_records(
    input_record: Dict[str, Any],
    qa_pairs: List[Dict[str, str]],
    model: str,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    for idx, pair in enumerate(qa_pairs):
        out.append(
            {
                "chunk_id": input_record["chunk_id"],
                "pair_id": f"{input_record['chunk_id']}_{idx}",
                "source": input_record["source"],
                "source_text": input_record["source_text"],
                "semantic_summary": input_record["semantic_summary"],
                "core_claim": input_record["core_claim"],
                "themes": input_record["themes"],
                "tone_notes": input_record["tone_notes"],
                "question": pair["question"],
                "answer": pair["answer"],
                "messages": [
                    {"role": "user", "content": pair["question"]},
                    {"role": "assistant", "content": pair["answer"]},
                ],
                "generation_model": model,
                "generation_status": "ok",
            }
        )

    return out


def build_training_records(enriched_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    training_records: List[Dict[str, Any]] = []
    for r in enriched_records:
        training_records.append(
            {
                "messages": r["messages"],
                "metadata": {
                    "chunk_id": r["chunk_id"],
                    "pair_id": r["pair_id"],
                    "themes": r["themes"],
                    "source": r["source"],
                },
            }
        )
    return training_records


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def iter_records(
    records: List[Dict[str, Any]],
    completed_chunk_ids: Set[int],
    max_records: Optional[int],
) -> Iterable[Dict[str, Any]]:
    count = 0
    for record in records:
        if record.get("analysis_status") != "ok":
            continue
        chunk_id = int(record["chunk_id"])
        if chunk_id in completed_chunk_ids:
            continue
        yield record
        count += 1
        if max_records is not None and count >= max_records:
            break


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pass-2 dialogic generation and validation for Alan Watts chunks."
    )
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

    if config.overwrite:
        for path in [config.enriched_output_path, config.training_output_path]:
            if path.exists():
                logger.warning(f"Overwrite enabled; removing existing output: {path}")
                path.unlink()

    client = OpenAI(api_key=api_key)

    input_records = load_jsonl(config.input_path)
    completed_chunk_ids = load_completed_chunk_ids(config.enriched_output_path)

    logger.info(f"Loaded {len(input_records)} analysis records from {config.input_path}")
    logger.info(f"Found {len(completed_chunk_ids)} already-completed chunk generations")
    logger.info(f"Writing enriched dataset to {config.enriched_output_path}")
    logger.info(f"Writing training dataset to {config.training_output_path}")
    logger.info(f"Using model: {config.model}")

    worklist = list(iter_records(input_records, completed_chunk_ids, config.max_records))
    logger.info(f"Chunks to process in this run: {len(worklist)}")

    for record in tqdm(worklist, desc="Pass-2 dialogic generation"):
        chunk_id = record["chunk_id"]
        success = False
        last_error: Optional[str] = None

        for attempt in range(1, config.max_retries + 1):
            try:
                qa_pairs = call_model(
                    client=client,
                    record=record,
                    model=config.model,
                    temperature=config.temperature,
                )

                ok, error_reason, validated_pairs = validate_qa_pairs(record, qa_pairs)
                if not ok:
                    raise ValueError(error_reason)

                enriched_records = build_enriched_records(record, validated_pairs, config.model)
                training_records = build_training_records(enriched_records)

                for enriched in enriched_records:
                    append_jsonl(config.enriched_output_path, enriched)

                for training in training_records:
                    append_jsonl(config.training_output_path, training)

                success = True
                break

            except Exception as e:
                last_error = str(e)
                logger.warning(
                    f"Chunk {chunk_id} failed on attempt {attempt}/{config.max_retries}: {e}"
                )
                time.sleep(min(2.0 * attempt, 10.0))

        if not success:
            error_record = {
                "chunk_id": chunk_id,
                "source": record.get("source", ""),
                "source_text": record.get("source_text", ""),
                "semantic_summary": record.get("semantic_summary", ""),
                "core_claim": record.get("core_claim", ""),
                "themes": record.get("themes", []),
                "tone_notes": record.get("tone_notes", []),
                "generation_status": "error",
                "error_message": last_error or "unknown error",
                "generation_model": config.model,
            }
            append_jsonl(config.enriched_output_path, error_record)

        time.sleep(config.sleep_seconds)

    logger.info("Pass-2 dialogic generation complete.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("Interrupted by user.")
        sys.exit(1)