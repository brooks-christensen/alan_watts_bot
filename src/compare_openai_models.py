#!/usr/bin/env python3
"""
Compare a baseline OpenAI model and a candidate fine-tuned model on a held-out
Alan Watts test set, writing side-by-side outputs and a markdown review sheet.

This script is intentionally simple and inspection-friendly:
- it does NOT auto-grade with another model
- it reuses the exact chat-style messages from your held-out test JSONL
- it sends the same input to both models so retrieval/context is fixed
- it writes rich outputs for manual review before any production change

Example:
    export OPENAI_API_KEY=...
    python src/compare_openai_models.py \
        --test-file data/fine_tuning/openai_sft/openai_sft_test_chat.jsonl \
        --baseline-model gpt-4.1-mini-2025-04-14 \
        --candidate-model ft:gpt-4.1-mini-2025-04-14:personal:alan-watts-sft-v1:DOXSTfMY \
        --output-dir data/fine_tuning/evals/sft_v1_compare \
        --max-cases 20
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from loguru import logger
from openai import OpenAI


@dataclass
class CaseResult:
    case_index: int
    pair_id: Optional[str]
    chunk_id: Optional[str]
    quality_score: Optional[float]
    baseline_model: str
    candidate_model: str
    question: str
    context_excerpt: str
    gold_answer_excerpt: str
    baseline_output: str
    candidate_output: str
    baseline_response_id: Optional[str]
    candidate_response_id: Optional[str]


def configure_logging(verbose: bool) -> None:
    logger.remove()
    level = "DEBUG" if verbose else "INFO"
    logger.add(sys.stderr, level=level)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare baseline and fine-tuned OpenAI models on held-out cases.")
    parser.add_argument("--test-file", required=True, help="Path to held-out chat-format JSONL test set.")
    parser.add_argument("--baseline-model", required=True, help="Current production or comparison model ID.")
    parser.add_argument("--candidate-model", required=True, help="Fine-tuned model ID to evaluate.")
    parser.add_argument("--output-dir", required=True, help="Directory for JSONL/JSON/Markdown outputs.")
    parser.add_argument("--max-cases", type=int, default=20, help="Maximum number of cases to evaluate.")
    parser.add_argument("--seed", type=int, default=45, help="Random seed for case sampling.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature for both models.")
    parser.add_argument("--max-output-tokens", type=int, default=500, help="Max output tokens for both models.")
    parser.add_argument("--sleep-seconds", type=float, default=0.0, help="Optional delay between API calls.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser.parse_args()


def extract_text(response: Any) -> str:
    if hasattr(response, "output_text") and response.output_text:
        return response.output_text

    texts: List[str] = []
    output = getattr(response, "output", None) or []
    for item in output:
        content = getattr(item, "content", None) or []
        for block in content:
            text = getattr(block, "text", None)
            if text:
                texts.append(text)
    return "\n".join(texts).strip()


def parse_question_and_context(user_content: str) -> tuple[str, str]:
    question = user_content
    context = ""

    q_marker = "User question:\n"
    c_marker = "\n\nRetrieved excerpts:\n"
    if q_marker in user_content and c_marker in user_content:
        after_q = user_content.split(q_marker, 1)[1]
        question, rest = after_q.split(c_marker, 1)
        context = rest
        tail = "\n\nWrite a response that is faithful"
        if tail in context:
            context = context.split(tail, 1)[0]
    return question.strip(), context.strip()


def load_cases(path: Path) -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no}: {exc}") from exc
            messages = obj.get("messages")
            if not isinstance(messages, list) or len(messages) < 2:
                raise ValueError(f"Line {line_no} missing valid messages array")
            cases.append(obj)
    return cases


def sample_cases(cases: List[Dict[str, Any]], max_cases: int, seed: int) -> List[Dict[str, Any]]:
    if max_cases >= len(cases):
        return list(cases)
    rng = random.Random(seed)
    idxs = list(range(len(cases)))
    rng.shuffle(idxs)
    chosen = sorted(idxs[:max_cases])
    return [cases[i] for i in chosen]


def build_responses_input(messages: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    inputs: List[Dict[str, Any]] = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        inputs.append({
            "role": role,
            "content": [{"type": "input_text", "text": content}],
        })
    return inputs


def call_model(
    client: OpenAI,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_output_tokens: int,
) -> tuple[str, Optional[str]]:
    response = client.responses.create(
        model=model,
        input=build_responses_input(messages),
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )
    text = extract_text(response).strip()
    response_id = getattr(response, "id", None)
    return text, response_id


def truncate(text: str, max_chars: int = 500) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def render_markdown(results: List[CaseResult], baseline_model: str, candidate_model: str) -> str:
    lines: List[str] = []
    lines.append("# Alan Watts Model Comparison\n")
    lines.append(f"- Baseline model: `{baseline_model}`")
    lines.append(f"- Candidate model: `{candidate_model}`")
    lines.append(f"- Cases compared: **{len(results)}**\n")
    lines.append("## Suggested review rubric\n")
    lines.append("For each case, score both outputs on:")
    lines.append("- grounding to retrieved excerpts")
    lines.append("- Alan-like cadence / lecture feel")
    lines.append("- warmth / reassurance")
    lines.append("- coherence / flow")
    lines.append("- restraint when excerpts are limited\n")
    for r in results:
        lines.append(f"## Case {r.case_index}: `{r.pair_id}` / `{r.chunk_id}`\n")
        lines.append(f"**Question:** {r.question}\n")
        lines.append(f"**Context excerpt:** {truncate(r.context_excerpt, 900)}\n")
        lines.append(f"**Gold answer excerpt:** {truncate(r.gold_answer_excerpt, 700)}\n")
        lines.append(f"### Baseline: `{baseline_model}`\n")
        lines.append(r.baseline_output or "<empty>")
        lines.append("")
        lines.append(f"### Candidate: `{candidate_model}`\n")
        lines.append(r.candidate_output or "<empty>")
        lines.append("")
        lines.append("### Notes\n")
        lines.append("- Better grounded: ")
        lines.append("- More Alan-like: ")
        lines.append("- More coherent: ")
        lines.append("- Winner: ")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set.")

    client = OpenAI(api_key=api_key)

    test_path = Path(args.test_file)
    if not test_path.exists():
        raise FileNotFoundError(f"Test file not found: {test_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading held-out cases from {}", test_path)
    cases = load_cases(test_path)
    selected = sample_cases(cases, args.max_cases, args.seed)
    logger.info("Selected {} case(s) for comparison", len(selected))

    results: List[CaseResult] = []

    for idx, case in enumerate(selected, start=1):
        messages = case["messages"]
        user_content = next((m["content"] for m in messages if m.get("role") == "user"), "")
        gold_answer = next((m["content"] for m in messages if m.get("role") == "assistant"), "")
        question, context = parse_question_and_context(user_content)

        logger.info("Case {} / {} -> {}", idx, len(selected), case.get("pair_id", f"row_{idx}"))

        baseline_output, baseline_response_id = call_model(
            client=client,
            model=args.baseline_model,
            messages=messages[:2],  # system + user only
            temperature=args.temperature,
            max_output_tokens=args.max_output_tokens,
        )
        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

        candidate_output, candidate_response_id = call_model(
            client=client,
            model=args.candidate_model,
            messages=messages[:2],
            temperature=args.temperature,
            max_output_tokens=args.max_output_tokens,
        )
        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

        results.append(
            CaseResult(
                case_index=idx,
                pair_id=case.get("pair_id"),
                chunk_id=case.get("chunk_id"),
                quality_score=case.get("quality_score"),
                baseline_model=args.baseline_model,
                candidate_model=args.candidate_model,
                question=question,
                context_excerpt=context,
                gold_answer_excerpt=gold_answer,
                baseline_output=baseline_output,
                candidate_output=candidate_output,
                baseline_response_id=baseline_response_id,
                candidate_response_id=candidate_response_id,
            )
        )

    results_jsonl = output_dir / "comparison_results.jsonl"
    results_json = output_dir / "comparison_results.json"
    review_md = output_dir / "comparison_review.md"
    config_json = output_dir / "comparison_config.json"

    write_jsonl(results_jsonl, (asdict(r) for r in results))
    results_json.write_text(json.dumps([asdict(r) for r in results], indent=2, ensure_ascii=False), encoding="utf-8")
    review_md.write_text(render_markdown(results, args.baseline_model, args.candidate_model), encoding="utf-8")
    config_json.write_text(
        json.dumps(
            {
                "test_file": str(test_path),
                "baseline_model": args.baseline_model,
                "candidate_model": args.candidate_model,
                "max_cases": args.max_cases,
                "seed": args.seed,
                "temperature": args.temperature,
                "max_output_tokens": args.max_output_tokens,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    logger.info("Wrote {}", results_jsonl)
    logger.info("Wrote {}", results_json)
    logger.info("Wrote {}", review_md)
    logger.info("Wrote {}", config_json)


if __name__ == "__main__":
    main()
