#!/usr/bin/env python3
"""
Prepare a compact, chunk-safe OpenAI SFT dataset for the Alan Watts chatbot.

This v2 script is designed for a first-pass hosted fine-tune where we want:
1) clean chunk-level train / validation / test splits (80 / 10 / 10 by default),
2) a smaller, higher-signal dataset that keeps only the strongest example(s) per chunk,
3) outputs that are easy to inspect before spending fine-tuning credits.

Key changes vs the earlier prep script
--------------------------------------
- Splits are done at the chunk level into train / validation / test.
- The test set is chunk-isolated from both train and validation.
- The default dataset is minimal: 1 selected example per chunk.
- A simple quality score is computed for each candidate example in a chunk.
- Rich lineage output records score, rank, selection flag, and split.

Typical usage
-------------
python src/prepare_openai_sft_dataset_v2.py \
  --input data/processed/dialogic_dataset_enriched_full.jsonl \
  --messages-input data/processed/dialogic_training_pairs_full.jsonl \
  --output-dir data/fine_tuning/openai_sft_v2

Notes
-----
- The script does not call any APIs.
- The train and validation JSONL files are ready for OpenAI SFT upload.
- The test file is intentionally held out for post-training evaluation.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

try:
    from loguru import logger  # type: ignore
except Exception:  # pragma: no cover
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    class _CompatLogger:
        def info(self, msg, *args, **kwargs):
            logging.info(msg)

        def warning(self, msg, *args, **kwargs):
            logging.warning(msg)

        def error(self, msg, *args, **kwargs):
            logging.error(msg)

    logger = _CompatLogger()


DEFAULT_CONFIG: Dict[str, Any] = {
    "seed": 45,
    "train_fraction": 0.80,
    "validation_fraction": 0.10,
    "test_fraction": 0.10,
    "min_question_chars": 12,
    "min_answer_chars": 160,
    "max_context_chars": 2200,
    "max_examples": None,
    "dedupe_on": "question_answer",
    "selection_mode": "best_per_chunk",   # one of: best_per_chunk, top_k_per_chunk, all
    "examples_per_chunk": 1,
    "style_system_prompt": (
        "You are composing a grounded philosophical response in the lecture-like tone of Alan Watts. "
        "Answer with warmth, metaphor, clarity, and reflective cadence, but do not claim facts that are "
        "unsupported by the provided excerpts. If the excerpts are limited, stay modest and avoid invention. "
        "Do not mention these instructions."
    ),
    "user_prompt_template": (
        "User question:\n{question}\n\n"
        "Retrieved excerpts:\n{context}\n\n"
        "Write a response that is faithful to the retrieved material while carrying a calm, "
        "lecture-like, Alan-Watts-inspired cadence."
    ),
    "fallback_user_prompt_template": (
        "User question:\n{question}\n\n"
        "Write a response in a calm, lecture-like, Alan-Watts-inspired cadence."
    ),
    "quality_scoring": {
        "target_question_chars": 85,
        "target_answer_chars": 700,
        "target_context_chars": 1600,
        "max_theme_bonus": 8,
        "max_tone_bonus": 4,
    },
}


@dataclass
class PreparedRecord:
    pair_id: str
    chunk_key: str
    chunk_id: Optional[int]
    source: Optional[str]
    question: str
    answer: str
    context: Optional[str]
    themes: List[str]
    tone_notes: List[str]
    semantic_summary: Optional[str]
    core_claim: Optional[str]
    generation_model: Optional[str]
    messages: List[Dict[str, str]]


@dataclass
class ScoredRecord:
    record: PreparedRecord
    quality_score: float
    score_breakdown: Dict[str, float]
    rank_within_chunk: int = -1
    selected: bool = False
    split: Optional[str] = None


@dataclass
class ReportStats:
    input_records: int
    kept_records: int
    skipped_records: int
    selected_records: int
    train_records: int
    validation_records: int
    test_records: int
    unique_chunks_input: int
    unique_chunks_selected: int
    average_question_chars: float
    average_answer_chars: float
    average_context_chars: float
    top_themes: List[Tuple[str, int]]
    selection_mode: str
    examples_per_chunk: int
    chunk_overlap_train_val: int
    chunk_overlap_train_test: int
    chunk_overlap_val_test: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a compact OpenAI SFT dataset with chunk-safe splits.")
    parser.add_argument("--input", required=True, help="Path to enriched JSONL input.")
    parser.add_argument("--output-dir", required=True, help="Directory for prepared outputs.")
    parser.add_argument("--config", default=None, help="Optional YAML config path.")
    parser.add_argument(
        "--messages-input",
        default=None,
        help="Optional messages-only JSONL input. Used as a fallback / comparison source.",
    )
    return parser.parse_args()


def load_config(path: Optional[str]) -> Dict[str, Any]:
    config = dict(DEFAULT_CONFIG)
    if path:
        cfg_path = Path(path)
        if not cfg_path.exists():
            raise FileNotFoundError(f"Config file not found: {cfg_path}")
        if yaml is None:
            raise RuntimeError("PyYAML is not available, but a YAML config path was provided.")
        loaded = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            raise ValueError("Top-level YAML config must be a mapping/dictionary.")
        for key, value in loaded.items():
            if key == "quality_scoring" and isinstance(value, dict):
                merged = dict(config.get("quality_scoring", {}))
                merged.update(value)
                config["quality_scoring"] = merged
            else:
                config[key] = value

    total = float(config["train_fraction"]) + float(config["validation_fraction"]) + float(config["test_fraction"])
    if not math.isclose(total, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError("train_fraction + validation_fraction + test_fraction must sum to 1.0")
    return config


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning(f"Skipping malformed JSON at {path}:{line_number}: {exc}")


def normalize_whitespace(text: str) -> str:
    text = text.replace("\u00a0", " ")
    return re.sub(r"\s+", " ", text).strip()


def clip_text(text: str, max_chars: int) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    clipped = text[: max_chars - 1].rstrip()
    last_break = max(clipped.rfind(". "), clipped.rfind("? "), clipped.rfind("! "), clipped.rfind("; "))
    if last_break > max_chars * 0.55:
        clipped = clipped[: last_break + 1].rstrip()
    return clipped + " …"


def extract_messages_record(obj: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    messages = obj.get("messages") or []
    if not isinstance(messages, list):
        return None, None

    user_content = None
    assistant_content = None
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = message.get("content")
        if not isinstance(content, str):
            continue
        if role == "user" and user_content is None:
            user_content = content.strip()
        elif role == "assistant" and assistant_content is None:
            assistant_content = content.strip()
    return user_content, assistant_content


def safe_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if isinstance(item, (str, int, float))]


def to_prepared_record(obj: Dict[str, Any], config: Dict[str, Any]) -> Optional[PreparedRecord]:
    question = obj.get("question")
    answer = obj.get("answer")
    context = obj.get("source_text")

    if not isinstance(question, str) or not isinstance(answer, str):
        question_msg, answer_msg = extract_messages_record(obj)
        question = question if isinstance(question, str) else question_msg
        answer = answer if isinstance(answer, str) else answer_msg
        if context is None:
            context = obj.get("context")

    if not isinstance(question, str) or not isinstance(answer, str):
        return None

    question = normalize_whitespace(question)
    answer = normalize_whitespace(answer)
    context = normalize_whitespace(context) if isinstance(context, str) else None

    if len(question) < int(config["min_question_chars"]):
        return None
    if len(answer) < int(config["min_answer_chars"]):
        return None

    if context:
        context = clip_text(context, int(config["max_context_chars"]))

    metadata = obj.get("metadata") if isinstance(obj.get("metadata"), dict) else {}

    pair_id = str(obj.get("pair_id") or metadata.get("pair_id") or "unknown_pair")
    chunk_id_raw = obj.get("chunk_id", metadata.get("chunk_id"))
    try:
        chunk_id = int(chunk_id_raw) if chunk_id_raw is not None else None
    except (TypeError, ValueError):
        chunk_id = None

    chunk_key = f"chunk::{chunk_id}" if chunk_id is not None else f"pair::{pair_id}"

    source = obj.get("source") or metadata.get("source")
    themes = safe_list(obj.get("themes") or metadata.get("themes"))
    tone_notes = safe_list(obj.get("tone_notes"))
    semantic_summary = obj.get("semantic_summary")
    core_claim = obj.get("core_claim")
    generation_model = obj.get("generation_model")

    if context:
        user_content = config["user_prompt_template"].format(question=question, context=context)
    else:
        user_content = config["fallback_user_prompt_template"].format(question=question)

    messages = [
        {"role": "system", "content": str(config["style_system_prompt"])},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": answer},
    ]

    return PreparedRecord(
        pair_id=pair_id,
        chunk_key=chunk_key,
        chunk_id=chunk_id,
        source=source if isinstance(source, str) else None,
        question=question,
        answer=answer,
        context=context,
        themes=themes,
        tone_notes=tone_notes,
        semantic_summary=semantic_summary if isinstance(semantic_summary, str) else None,
        core_claim=core_claim if isinstance(core_claim, str) else None,
        generation_model=generation_model if isinstance(generation_model, str) else None,
        messages=messages,
    )


def dedupe_records(records: Sequence[PreparedRecord], dedupe_on: str) -> List[PreparedRecord]:
    seen = set()
    deduped: List[PreparedRecord] = []

    for record in records:
        if dedupe_on == "pair_id":
            key = record.pair_id
        elif dedupe_on == "question":
            key = record.question.lower()
        else:
            key = (record.question.lower(), record.answer.lower())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(record)

    return deduped


_WORD_RE = re.compile(r"[a-zA-Z']+")


def tokenize(text: str) -> List[str]:
    return [token.lower() for token in _WORD_RE.findall(text)]


def stopwords() -> set:
    return {
        "the", "a", "an", "and", "or", "but", "if", "then", "that", "this", "those", "these",
        "is", "are", "was", "were", "be", "been", "being", "to", "of", "in", "on", "for", "with",
        "as", "at", "by", "from", "into", "it", "its", "it's", "you", "your", "we", "they", "their",
        "he", "she", "his", "her", "i", "me", "my", "our", "ours", "them", "there", "here",
        "what", "why", "how", "when", "where", "who", "whom", "which", "do", "does", "did",
        "can", "could", "would", "should", "not", "no", "yes", "so", "than", "too", "very",
        "just", "really", "more", "most", "much", "many", "also", "about", "because", "while",
    }


def closeness_score(value: int, target: int, max_points: float, tolerance_fraction: float = 1.0) -> float:
    if target <= 0:
        return 0.0
    tolerance = max(1.0, target * tolerance_fraction)
    distance = abs(value - target)
    scaled = max(0.0, 1.0 - (distance / tolerance))
    return round(max_points * scaled, 4)


def jaccard_similarity(tokens_a: Sequence[str], tokens_b: Sequence[str]) -> float:
    set_a = set(tokens_a)
    set_b = set(tokens_b)
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def compute_quality_score(record: PreparedRecord, config: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
    q_cfg = config["quality_scoring"]
    question_len = len(record.question)
    answer_len = len(record.answer)
    context_len = len(record.context or "")
    theme_count = len(record.themes)
    tone_count = len(record.tone_notes)
    sentence_count = max(1, len(re.findall(r"[.!?]+", record.answer)))

    question_tokens = [t for t in tokenize(record.question) if t not in stopwords()]
    answer_tokens = [t for t in tokenize(record.answer) if t not in stopwords()]
    context_tokens = [t for t in tokenize(record.context or "") if t not in stopwords()]

    overlap_qa = jaccard_similarity(question_tokens, answer_tokens)
    overlap_ca = jaccard_similarity(context_tokens, answer_tokens)
    diversity = (len(set(answer_tokens)) / max(1, len(answer_tokens))) if answer_tokens else 0.0

    breakdown = {
        "question_length": closeness_score(question_len, int(q_cfg["target_question_chars"]), 10.0, 1.2),
        "answer_length": closeness_score(answer_len, int(q_cfg["target_answer_chars"]), 18.0, 1.0),
        "context_length": 8.0 if record.context else 0.0,
        "context_balance": closeness_score(context_len, int(q_cfg["target_context_chars"]), 6.0, 1.0) if record.context else 0.0,
        "question_shape": 4.0 if record.question.strip().endswith("?") else 1.0,
        "theme_bonus": min(float(theme_count), float(q_cfg["max_theme_bonus"])),
        "tone_bonus": min(float(tone_count), float(q_cfg["max_tone_bonus"])),
        "sentence_balance": closeness_score(sentence_count, 6, 7.0, 1.0),
        "grounding_overlap": round(min(overlap_ca * 20.0, 10.0), 4),
        "question_answer_overlap": round(min(overlap_qa * 12.0, 6.0), 4),
        "lexical_diversity": round(min(diversity * 10.0, 6.0), 4),
    }

    if record.semantic_summary:
        breakdown["semantic_summary_present"] = 1.5
    else:
        breakdown["semantic_summary_present"] = 0.0

    if record.core_claim:
        breakdown["core_claim_present"] = 1.5
    else:
        breakdown["core_claim_present"] = 0.0

    total = round(sum(breakdown.values()), 4)
    return total, breakdown


def select_records_by_chunk(records: Sequence[PreparedRecord], config: Dict[str, Any]) -> Tuple[List[ScoredRecord], List[ScoredRecord]]:
    mode = str(config["selection_mode"])
    examples_per_chunk = int(config.get("examples_per_chunk", 1))

    by_chunk: Dict[str, List[PreparedRecord]] = defaultdict(list)
    for record in records:
        by_chunk[record.chunk_key].append(record)

    all_scored: List[ScoredRecord] = []
    selected_scored: List[ScoredRecord] = []

    for chunk_key, bucket in by_chunk.items():
        scored_bucket: List[ScoredRecord] = []
        for record in bucket:
            score, breakdown = compute_quality_score(record, config)
            scored_bucket.append(ScoredRecord(record=record, quality_score=score, score_breakdown=breakdown))

        scored_bucket.sort(
            key=lambda sr: (
                -sr.quality_score,
                -len(sr.record.answer),
                -len(sr.record.context or ""),
                sr.record.pair_id,
            )
        )

        for rank, scored in enumerate(scored_bucket, start=1):
            scored.rank_within_chunk = rank

        if mode == "all":
            selected_count = len(scored_bucket)
        elif mode == "top_k_per_chunk":
            selected_count = max(1, examples_per_chunk)
        else:
            selected_count = 1

        for idx, scored in enumerate(scored_bucket):
            if idx < selected_count:
                scored.selected = True
                selected_scored.append(scored)

        all_scored.extend(scored_bucket)

    return all_scored, selected_scored


def allocate_counts(total: int, fractions: Sequence[float]) -> List[int]:
    raw = [total * frac for frac in fractions]
    base = [math.floor(value) for value in raw]
    remainder = total - sum(base)
    order = sorted(range(len(raw)), key=lambda i: (raw[i] - base[i]), reverse=True)
    for i in range(remainder):
        base[order[i % len(order)]] += 1
    return base


def split_selected_by_chunk(selected_records: Sequence[ScoredRecord], config: Dict[str, Any]) -> Tuple[List[ScoredRecord], List[ScoredRecord], List[ScoredRecord]]:
    by_chunk: Dict[str, List[ScoredRecord]] = defaultdict(list)
    for scored in selected_records:
        by_chunk[scored.record.chunk_key].append(scored)

    chunk_keys = list(by_chunk.keys())
    rng = random.Random(int(config["seed"]))
    rng.shuffle(chunk_keys)

    train_count, val_count, test_count = allocate_counts(
        len(chunk_keys),
        [
            float(config["train_fraction"]),
            float(config["validation_fraction"]),
            float(config["test_fraction"]),
        ],
    )

    train_keys = set(chunk_keys[:train_count])
    val_keys = set(chunk_keys[train_count : train_count + val_count])
    test_keys = set(chunk_keys[train_count + val_count : train_count + val_count + test_count])

    train: List[ScoredRecord] = []
    validation: List[ScoredRecord] = []
    test: List[ScoredRecord] = []

    for chunk_key, bucket in by_chunk.items():
        if chunk_key in train_keys:
            split_name = "train"
            target = train
        elif chunk_key in val_keys:
            split_name = "validation"
            target = validation
        elif chunk_key in test_keys:
            split_name = "test"
            target = test
        else:
            raise RuntimeError(f"Chunk key {chunk_key} was not assigned to any split.")

        for scored in bucket:
            scored.split = split_name
            target.append(scored)

    return train, validation, test


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_report(
    kept_records: Sequence[PreparedRecord],
    selected_records: Sequence[ScoredRecord],
    train: Sequence[ScoredRecord],
    validation: Sequence[ScoredRecord],
    test: Sequence[ScoredRecord],
    input_records: int,
    config: Dict[str, Any],
) -> ReportStats:
    themes = Counter()
    q_lengths: List[int] = []
    a_lengths: List[int] = []
    c_lengths: List[int] = []
    input_chunks = set()
    selected_chunks = set()

    for record in kept_records:
        themes.update(record.themes)
        q_lengths.append(len(record.question))
        a_lengths.append(len(record.answer))
        c_lengths.append(len(record.context or ""))
        input_chunks.add(record.chunk_key)

    for scored in selected_records:
        selected_chunks.add(scored.record.chunk_key)

    def avg(values: Sequence[int]) -> float:
        return round(float(statistics.mean(values)), 2) if values else 0.0

    train_chunks = {x.record.chunk_key for x in train}
    val_chunks = {x.record.chunk_key for x in validation}
    test_chunks = {x.record.chunk_key for x in test}

    return ReportStats(
        input_records=input_records,
        kept_records=len(kept_records),
        skipped_records=input_records - len(kept_records),
        selected_records=len(selected_records),
        train_records=len(train),
        validation_records=len(validation),
        test_records=len(test),
        unique_chunks_input=len(input_chunks),
        unique_chunks_selected=len(selected_chunks),
        average_question_chars=avg(q_lengths),
        average_answer_chars=avg(a_lengths),
        average_context_chars=avg(c_lengths),
        top_themes=themes.most_common(15),
        selection_mode=str(config["selection_mode"]),
        examples_per_chunk=int(config.get("examples_per_chunk", 1)),
        chunk_overlap_train_val=len(train_chunks & val_chunks),
        chunk_overlap_train_test=len(train_chunks & test_chunks),
        chunk_overlap_val_test=len(val_chunks & test_chunks),
    )


def render_markdown_report(
    stats: ReportStats,
    sample_train: Sequence[ScoredRecord],
    sample_validation: Sequence[ScoredRecord],
    sample_test: Sequence[ScoredRecord],
    config: Dict[str, Any],
) -> str:
    lines = [
        "# OpenAI SFT Dataset Preparation Report (v2)",
        "",
        "## Summary",
        "",
        f"- Input records: **{stats.input_records}**",
        f"- Kept records: **{stats.kept_records}**",
        f"- Skipped records: **{stats.skipped_records}**",
        f"- Selected records for compact dataset: **{stats.selected_records}**",
        f"- Train records: **{stats.train_records}**",
        f"- Validation records: **{stats.validation_records}**",
        f"- Test records: **{stats.test_records}**",
        f"- Unique chunks in input: **{stats.unique_chunks_input}**",
        f"- Unique chunks in selected dataset: **{stats.unique_chunks_selected}**",
        f"- Avg question length: **{stats.average_question_chars} chars**",
        f"- Avg answer length: **{stats.average_answer_chars} chars**",
        f"- Avg context length: **{stats.average_context_chars} chars**",
        f"- Selection mode: **{stats.selection_mode}**",
        f"- Examples per chunk: **{stats.examples_per_chunk}**",
        "",
        "## Split integrity",
        "",
        f"- Train ∩ Validation chunk overlap: **{stats.chunk_overlap_train_val}**",
        f"- Train ∩ Test chunk overlap: **{stats.chunk_overlap_train_test}**",
        f"- Validation ∩ Test chunk overlap: **{stats.chunk_overlap_val_test}**",
        "",
        "## Why this output matters",
        "",
        "This compact dataset is shaped for a first-pass fine-tune where style transfer should be",
        "strong enough to matter, but the redundancy should be low enough that we are not paying to",
        "teach the model near-duplicate answers from the same chunk repeatedly.",
        "",
        "## Most common themes",
        "",
    ]

    for theme, count in stats.top_themes:
        lines.append(f"- {theme}: {count}")

    lines.extend([
        "",
        "## Active prompt template",
        "",
        "```text",
        str(config["user_prompt_template"]),
        "```",
        "",
        "## System prompt",
        "",
        "```text",
        str(config["style_system_prompt"]),
        "```",
        "",
        "## Sample selected train examples",
        "",
    ])

    for scored in list(sample_train)[:2]:
        record = scored.record
        lines.extend([
            f"### {record.pair_id} (score={scored.quality_score})",
            "",
            f"**Question:** {record.question}",
            "",
            f"**Context excerpt:** {(record.context or '')[:500]}{'…' if record.context and len(record.context) > 500 else ''}",
            "",
            f"**Answer excerpt:** {record.answer[:500]}{'…' if len(record.answer) > 500 else ''}",
            "",
        ])

    lines.extend([
        "## Sample validation examples",
        "",
    ])

    for scored in list(sample_validation)[:2]:
        record = scored.record
        lines.extend([
            f"### {record.pair_id}",
            "",
            f"**Question:** {record.question}",
            "",
            f"**Themes:** {', '.join(record.themes[:8]) if record.themes else 'n/a'}",
            "",
        ])

    lines.extend([
        "## Sample test examples",
        "",
    ])

    for scored in list(sample_test)[:2]:
        record = scored.record
        lines.extend([
            f"### {record.pair_id}",
            "",
            f"**Question:** {record.question}",
            "",
            f"**Themes:** {', '.join(record.themes[:8]) if record.themes else 'n/a'}",
            "",
        ])

    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    rng = random.Random(int(config["seed"]))

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input JSONL not found: {input_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Reading enriched input from {input_path}")
    raw_objects = list(read_jsonl(input_path))
    input_records = len(raw_objects)
    logger.info(f"Loaded {input_records} raw records")

    prepared_records: List[PreparedRecord] = []
    for obj in raw_objects:
        record = to_prepared_record(obj, config)
        if record is not None:
            prepared_records.append(record)

    if args.messages_input:
        messages_path = Path(args.messages_input)
        if messages_path.exists():
            logger.info(f"Reading optional messages-only input from {messages_path}")
            for obj in read_jsonl(messages_path):
                record = to_prepared_record(obj, config)
                if record is not None:
                    prepared_records.append(record)

    prepared_records = dedupe_records(prepared_records, str(config["dedupe_on"]))
    rng.shuffle(prepared_records)

    max_examples = config.get("max_examples")
    if isinstance(max_examples, int) and max_examples > 0:
        prepared_records = prepared_records[:max_examples]

    if not prepared_records:
        raise RuntimeError("No valid records remained after filtering.")

    logger.info("Scoring and selecting examples within each chunk")
    all_scored, selected_scored = select_records_by_chunk(prepared_records, config)

    if not selected_scored:
        raise RuntimeError("No records were selected after chunk-level ranking.")

    logger.info("Creating chunk-safe 80/10/10 train/validation/test splits")
    train_records, validation_records, test_records = split_selected_by_chunk(selected_scored, config)

    if not train_records or not validation_records or not test_records:
        raise RuntimeError("One of the splits is empty. Adjust fractions or dataset size.")

    train_jsonl = output_dir / "openai_sft_train.jsonl"
    validation_jsonl = output_dir / "openai_sft_validation.jsonl"
    test_eval_jsonl = output_dir / "openai_sft_test_eval.jsonl"
    test_chat_jsonl = output_dir / "openai_sft_test_chat.jsonl"
    lineage_jsonl = output_dir / "openai_sft_lineage.jsonl"
    report_json = output_dir / "openai_sft_report.json"
    report_md = output_dir / "openai_sft_report.md"

    logger.info("Writing SFT train / validation JSONL files")
    write_jsonl(train_jsonl, ({"messages": sr.record.messages} for sr in train_records))
    write_jsonl(validation_jsonl, ({"messages": sr.record.messages} for sr in validation_records))

    logger.info("Writing held-out test artifacts")
    write_jsonl(
        test_eval_jsonl,
        (
            {
                "pair_id": sr.record.pair_id,
                "chunk_id": sr.record.chunk_id,
                "question": sr.record.question,
                "context": sr.record.context,
                "reference_answer": sr.record.answer,
                "themes": sr.record.themes,
                "quality_score": sr.quality_score,
            }
            for sr in test_records
        ),
    )
    write_jsonl(
        test_chat_jsonl,
        (
            {
                "messages": sr.record.messages,
                "pair_id": sr.record.pair_id,
                "chunk_id": sr.record.chunk_id,
                "quality_score": sr.quality_score,
            }
            for sr in test_records
        ),
    )

    logger.info("Writing lineage and reports")
    write_jsonl(
        lineage_jsonl,
        (
            {
                **asdict(sr.record),
                "quality_score": sr.quality_score,
                "score_breakdown": sr.score_breakdown,
                "rank_within_chunk": sr.rank_within_chunk,
                "selected": sr.selected,
                "split": sr.split,
            }
            for sr in all_scored
        ),
    )

    stats = build_report(prepared_records, selected_scored, train_records, validation_records, test_records, input_records, config)

    report_payload = asdict(stats)
    report_payload["config"] = config
    report_payload["examples_per_chunk_effective"] = Counter(sr.record.chunk_key for sr in selected_scored).most_common(5)

    report_json.write_text(json.dumps(report_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    report_md.write_text(
        render_markdown_report(stats, train_records, validation_records, test_records, config),
        encoding="utf-8",
    )

    logger.info("Done")
    logger.info(f"Train JSONL: {train_jsonl}")
    logger.info(f"Validation JSONL: {validation_jsonl}")
    logger.info(f"Test eval JSONL: {test_eval_jsonl}")
    logger.info(f"Test chat JSONL: {test_chat_jsonl}")
    logger.info(f"Lineage JSONL: {lineage_jsonl}")
    logger.info(f"Report JSON: {report_json}")
    logger.info(f"Report MD: {report_md}")


if __name__ == "__main__":
    main()
