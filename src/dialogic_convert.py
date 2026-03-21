from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

import yaml
from loguru import logger
from openai import OpenAI
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


ANALYSIS_SCHEMA: Dict[str, Any] = {
    "name": "dialogic_analysis",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "semantic_summary": {
                "type": "string",
                "description": (
                    "A faithful 2-3 sentence semantic summary grounded only in the source chunk. "
                    "Prefer faithful compression over elegant synthesis."
                ),
            },
            "core_claim": {
                "type": "string",
                "description": "A single-sentence statement of the chunk's main idea.",
            },
            "candidate_questions": {
                "type": "array",
                "description": (
                    "1-3 realistic, concept-centered questions a modern user might ask "
                    "that this chunk can answer."
                ),
                "minItems": 1,
                "maxItems": 3,
                "items": {"type": "string"},
            },
            "themes": {
                "type": "array",
                "description": (
                    "3-8 short, reusable, lowercase themes or keyword phrases, ideally 1-3 words each."
                ),
                "minItems": 3,
                "maxItems": 8,
                "items": {"type": "string"},
            },
            "tone_notes": {
                "type": "array",
                "description": (
                    "2-5 short lowercase style descriptors, ideally one or two words each."
                ),
                "minItems": 2,
                "maxItems": 5,
                "items": {"type": "string"},
            },
        },
        "required": [
            "semantic_summary",
            "core_claim",
            "candidate_questions",
            "themes",
            "tone_notes",
        ],
    },
}


SYSTEM_PROMPT = """You are converting monologic Alan Watts source material into structured semantic analysis for a later dialogic generation pass.

Your job in this pass is NOT to imitate Alan Watts theatrically.
Your job is to identify what the chunk means, what questions it can answer, and what tone it carries.

Rules:
1. Use ONLY the provided source_text. Do not add outside facts, biography, doctrine, or transcript context.
2. Treat the chunk as the only source of truth.
3. semantic_summary must be faithful, concise, non-hallucinatory, and limited to 2-3 compact sentences.
4. Prefer semantic compression over interpretive elegance.
5. candidate_questions must sound like realistic questions a modern user might ask.
6. Do not mention 'chunk', 'transcript', 'lecture', 'paragraph', or 'speaker' in the questions.
7. Do not generate trivia questions. Generate concept-centered questions.
8. themes must be short, atomic, reusable, and ideally 1-3 words each.
9. Prefer simple tags over compound phrases. Split broad themes into smaller reusable themes when possible.
10. tone_notes must describe the style of explanation, not the topic.
11. tone_notes should be short lowercase descriptors, ideally one word and at most two words.
12. If the chunk is repetitive, still produce a clean semantic distillation.
13. Return valid JSON only.
"""


USER_PROMPT_TEMPLATE = """Analyze the following source chunk and return the structured analysis.

chunk_id: {chunk_id}
source: {source}

source_text:
\"\"\"
{source_text}
\"\"\"
"""


@dataclass
class ConvertConfig:
    input_path: Path
    output_path: Path
    model: str
    temperature: float
    max_records: Optional[int]
    overwrite: bool
    sleep_seconds: float
    max_retries: int


def load_config(config_path: Path) -> ConvertConfig:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    section = raw.get("dialogic_convert", {})

    input_path = PROJECT_ROOT / section.get("input_path", "data/processed/rag_chunks.jsonl")
    output_path = PROJECT_ROOT / section.get("output_path", "data/processed/dialogic_analysis.jsonl")

    return ConvertConfig(
        input_path=input_path,
        output_path=output_path,
        model=section.get("model", "gpt-5.4"),
        temperature=float(section.get("temperature", 0.2)),
        max_records=section.get("max_records"),
        overwrite=bool(section.get("overwrite", False)),
        sleep_seconds=float(section.get("sleep_seconds", 0.25)),
        max_retries=int(section.get("max_retries", 3)),
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
                if obj.get("analysis_status") == "ok":
                    completed.add(int(obj["chunk_id"]))
            except Exception:
                continue
    return completed


def dedupe_preserve_order(items: List[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for item in items:
        norm = item.strip()
        if not norm:
            continue
        key = norm.lower()
        if key not in seen:
            seen.add(key)
            out.append(norm)
    return out


def sentence_count(text: str) -> int:
    parts = [p.strip() for p in re.split(r"[.!?]+", text) if p.strip()]
    return len(parts)


def normalize_spaces(text: str) -> str:
    return " ".join(text.strip().split())


def normalize_theme(theme: str) -> str:
    theme = normalize_spaces(theme).lower()
    theme = theme.strip(",;:.")
    return theme


def normalize_tone_note(note: str) -> str:
    note = normalize_spaces(note).lower()
    note = note.strip(",;:.")
    return note


def maybe_split_compound_theme(theme: str) -> List[str]:
    """
    Conservative heuristic:
    - split explicit contrast pairs like 'faith vs belief'
    - split 'x and y' only if both sides are short
    """
    theme = normalize_theme(theme)

    if " vs " in theme:
        parts = [normalize_theme(p) for p in theme.split(" vs ")]
        parts = [p for p in parts if p]
        return parts if len(parts) > 1 else [theme]

    if " versus " in theme:
        parts = [normalize_theme(p) for p in theme.split(" versus ")]
        parts = [p for p in parts if p]
        return parts if len(parts) > 1 else [theme]

    if " and " in theme:
        parts = [normalize_theme(p) for p in theme.split(" and ")]
        parts = [p for p in parts if p]
        if len(parts) == 2 and all(1 <= len(p.split()) <= 2 for p in parts):
            return parts

    return [theme]


def postprocess_themes(themes: List[str]) -> List[str]:
    expanded: List[str] = []
    for theme in themes:
        expanded.extend(maybe_split_compound_theme(theme))

    cleaned = dedupe_preserve_order(
        [normalize_theme(t) for t in expanded if normalize_theme(t)]
    )

    # Keep only the first 8 to stay within schema expectations downstream.
    return cleaned[:8]


def postprocess_tone_notes(tone_notes: List[str]) -> List[str]:
    cleaned = dedupe_preserve_order(
        [normalize_tone_note(t) for t in tone_notes if normalize_tone_note(t)]
    )
    return cleaned[:5]


def validate_analysis_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    required = {"semantic_summary", "core_claim", "candidate_questions", "themes", "tone_notes"}
    missing = required - payload.keys()
    if missing:
        raise ValueError(f"Missing required fields in model output: {sorted(missing)}")

    if not isinstance(payload["semantic_summary"], str) or not payload["semantic_summary"].strip():
        raise ValueError("semantic_summary must be a non-empty string")

    if not isinstance(payload["core_claim"], str) or not payload["core_claim"].strip():
        raise ValueError("core_claim must be a non-empty string")

    questions = payload["candidate_questions"]
    themes = payload["themes"]
    tone_notes = payload["tone_notes"]

    if not isinstance(questions, list):
        raise ValueError("candidate_questions must be a list")
    if not isinstance(themes, list):
        raise ValueError("themes must be a list")
    if not isinstance(tone_notes, list):
        raise ValueError("tone_notes must be a list")

    payload["semantic_summary"] = normalize_spaces(payload["semantic_summary"])
    payload["core_claim"] = normalize_spaces(payload["core_claim"])

    payload["candidate_questions"] = dedupe_preserve_order(
        [normalize_spaces(str(q)) for q in questions if normalize_spaces(str(q))]
    )
    payload["themes"] = postprocess_themes([str(t) for t in themes])
    payload["tone_notes"] = postprocess_tone_notes([str(t) for t in tone_notes])

    if not (1 <= len(payload["candidate_questions"]) <= 3):
        raise ValueError("candidate_questions must contain 1-3 unique non-empty items")

    if not (3 <= len(payload["themes"]) <= 8):
        raise ValueError("themes must contain 3-8 unique non-empty items")

    if not (2 <= len(payload["tone_notes"]) <= 5):
        raise ValueError("tone_notes must contain 2-5 unique non-empty items")

    # Summary should stay compact for pass 2.
    if sentence_count(payload["semantic_summary"]) > 3:
        raise ValueError("semantic_summary exceeds 3 sentences")

    # Encourage atomic reusable tags, but not so aggressively that we break good outputs.
    for theme in payload["themes"]:
        if len(theme.split()) > 3:
            raise ValueError(f"theme too long/non-atomic: {theme}")

    # Tone notes should be compact style descriptors.
    for note in payload["tone_notes"]:
        if len(note.split()) > 2:
            raise ValueError(f"tone_note too long/non-atomic: {note}")

    return payload


def build_input_messages(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    user_prompt = USER_PROMPT_TEMPLATE.format(
        chunk_id=record["chunk_id"],
        source=record["source"],
        source_text=record["text"],
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
) -> Dict[str, Any]:
    response = client.responses.create(
        model=model,
        input=build_input_messages(record),
        temperature=temperature,
        text={
            "format": {
                "type": "json_schema",
                "name": ANALYSIS_SCHEMA["name"],
                "schema": ANALYSIS_SCHEMA["schema"],
                "strict": True,
            }
        },
    )

    if not getattr(response, "output_text", None):
        raise ValueError("Model returned no output_text")

    parsed = json.loads(response.output_text)
    return validate_analysis_payload(parsed)


def build_output_record(
    input_record: Dict[str, Any],
    analysis: Dict[str, Any],
    model: str,
) -> Dict[str, Any]:
    return {
        "chunk_id": input_record["chunk_id"],
        "source": input_record["source"],
        "source_text": input_record["text"],
        "semantic_summary": analysis["semantic_summary"],
        "core_claim": analysis["core_claim"],
        "candidate_questions": analysis["candidate_questions"],
        "themes": analysis["themes"],
        "tone_notes": analysis["tone_notes"],
        "analysis_model": model,
        "analysis_status": "ok",
    }


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
        chunk_id = int(record["chunk_id"])
        if chunk_id in completed_chunk_ids:
            continue
        yield record
        count += 1
        if max_records is not None and count >= max_records:
            break


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pass-1 dialogic conversion for Alan Watts chunks.")
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

    if config.overwrite and config.output_path.exists():
        logger.warning(f"Overwrite enabled; removing existing output: {config.output_path}")
        config.output_path.unlink()

    client = OpenAI(api_key=api_key)

    input_records = load_jsonl(config.input_path)
    completed_chunk_ids = load_completed_chunk_ids(config.output_path)

    logger.info(f"Loaded {len(input_records)} input chunk records from {config.input_path}")
    logger.info(f"Found {len(completed_chunk_ids)} already-completed analysis records")
    logger.info(f"Writing pass-1 analysis to {config.output_path}")
    logger.info(f"Using model: {config.model}")

    worklist = list(iter_records(input_records, completed_chunk_ids, config.max_records))
    logger.info(f"Chunks to analyze in this run: {len(worklist)}")

    for record in tqdm(worklist, desc="Pass-1 dialogic analysis"):
        chunk_id = record["chunk_id"]

        success = False
        last_error: Optional[Exception] = None

        for attempt in range(1, config.max_retries + 1):
            try:
                analysis = call_model(
                    client=client,
                    record=record,
                    model=config.model,
                    temperature=config.temperature,
                )
                output_record = build_output_record(record, analysis, config.model)
                append_jsonl(config.output_path, output_record)
                success = True
                break

            except Exception as e:
                last_error = e
                logger.warning(
                    f"Chunk {chunk_id} failed on attempt {attempt}/{config.max_retries}: {e}"
                )
                time.sleep(min(2.0 * attempt, 10.0))

        if not success:
            error_record = {
                "chunk_id": chunk_id,
                "source": record.get("source", ""),
                "source_text": record.get("text", ""),
                "analysis_status": "error",
                "error_message": str(last_error) if last_error else "unknown error",
                "analysis_model": config.model,
            }
            append_jsonl(config.output_path, error_record)

        time.sleep(config.sleep_seconds)

    logger.info("Pass-1 dialogic analysis complete.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("Interrupted by user.")
        sys.exit(1)