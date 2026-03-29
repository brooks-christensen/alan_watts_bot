
#!/usr/bin/env python3
"""
Launch and manage a supervised fine-tuning (SFT) job for the Alan Watts chatbot.

This script is intentionally conservative:
- Validates local JSONL chat-format files before upload.
- Uploads training and validation files to OpenAI with purpose="fine-tune".
- Creates an SFT job on a supported base model (default: gpt-4.1-mini-2025-04-14).
- Writes a local run manifest so the job can be resumed, inspected, or rolled back.
- Optionally waits for completion and records the resulting fine-tuned model ID.

Typical usage:

    python src/launch_openai_sft_job.py create \
      --train-file data/fine_tuning/openai_sft/openai_sft_train.jsonl \
      --validation-file data/fine_tuning/openai_sft/openai_sft_validation.jsonl \
      --run-dir data/fine_tuning/openai_sft/runs \
      --suffix alan-watts-sft-v1 \
      --wait

    python src/launch_openai_sft_job.py status \
      --job-id ftjob-abc123

Environment:
- OPENAI_API_KEY must be set.
- OPENAI_PROJECT is optional but useful if your account uses multiple projects.

Dependencies:
- openai
- loguru

This script assumes the dataset is already prepared and chunk-safe.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from loguru import logger
from openai import OpenAI


SUPPORTED_TERMINAL_STATUSES = {"succeeded", "failed", "cancelled"}
DEFAULT_MODEL = "gpt-4.1-mini-2025-04-14"


@dataclass
class LocalExampleCheck:
    line_number: int
    ok: bool
    message: str


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def configure_logging(verbose: bool) -> None:
    logger.remove()
    level = "DEBUG" if verbose else "INFO"
    logger.add(sys.stderr, level=level)


def load_jsonl(path: Path) -> Iterable[Tuple[int, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                yield line_number, json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}: invalid JSON on line {line_number}: {exc}") from exc


def validate_chat_example(obj: Dict[str, Any], line_number: int) -> LocalExampleCheck:
    if not isinstance(obj, dict):
        return LocalExampleCheck(line_number, False, "Top-level item is not a JSON object.")
    messages = obj.get("messages")
    if not isinstance(messages, list) or not messages:
        return LocalExampleCheck(line_number, False, "Missing non-empty 'messages' list.")
    allowed_roles = {"system", "user", "assistant", "developer"}
    saw_assistant = False
    saw_user = False
    for idx, msg in enumerate(messages):
        if not isinstance(msg, dict):
            return LocalExampleCheck(line_number, False, f"messages[{idx}] is not an object.")
        role = msg.get("role")
        content = msg.get("content")
        if role not in allowed_roles:
            return LocalExampleCheck(line_number, False, f"messages[{idx}].role={role!r} is invalid.")
        if not isinstance(content, str) or not content.strip():
            return LocalExampleCheck(line_number, False, f"messages[{idx}].content must be a non-empty string.")
        if role == "user":
            saw_user = True
        if role == "assistant":
            saw_assistant = True
    if not saw_user:
        return LocalExampleCheck(line_number, False, "Example has no user message.")
    if not saw_assistant:
        return LocalExampleCheck(line_number, False, "Example has no assistant message.")
    return LocalExampleCheck(line_number, True, "ok")


def validate_chat_jsonl(path: Path, max_errors: int = 10) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if path.suffix.lower() != ".jsonl":
        raise ValueError(f"Expected a .jsonl file: {path}")

    record_count = 0
    errors: List[Dict[str, Any]] = []

    for line_number, obj in load_jsonl(path):
        record_count += 1
        check = validate_chat_example(obj, line_number)
        if not check.ok:
            errors.append(asdict(check))
            if len(errors) >= max_errors:
                break

    return {
        "path": str(path),
        "record_count": record_count,
        "ok": len(errors) == 0 and record_count > 0,
        "errors": errors,
    }


def create_openai_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set in the environment.")
    project = os.environ.get("OPENAI_PROJECT")
    if project:
        logger.info("Using OpenAI project from OPENAI_PROJECT={}", project)
        return OpenAI(api_key=api_key, project=project)
    return OpenAI(api_key=api_key)


def upload_file(client: OpenAI, path: Path) -> Any:
    logger.info("Uploading {} for fine-tuning", path)
    with path.open("rb") as handle:
        file_obj = client.files.create(file=handle, purpose="fine-tune")
    logger.info("Uploaded {} -> {}", path.name, file_obj.id)
    return file_obj


def serialize_openai_object(obj: Any) -> Dict[str, Any]:
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if isinstance(obj, dict):
        return obj
    return json.loads(json.dumps(obj, default=str))


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    ensure_parent_dir(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_run_name(prefix: str = "alan_watts_sft") -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{stamp}"


def create_manifest(
    *,
    run_name: str,
    train_file: Path,
    validation_file: Optional[Path],
    model: str,
    suffix: Optional[str],
    seed: Optional[int],
    method_payload: Dict[str, Any],
    metadata: Dict[str, str],
    local_validation: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "run_name": run_name,
        "created_at": utc_now_iso(),
        "model": model,
        "suffix": suffix,
        "seed": seed,
        "method": method_payload,
        "metadata": metadata,
        "local_files": {
            "train_file": str(train_file),
            "validation_file": str(validation_file) if validation_file else None,
        },
        "local_validation": local_validation,
        "openai_files": {
            "training_file_id": None,
            "validation_file_id": None,
        },
        "fine_tuning_job": {
            "job_id": None,
            "status": None,
            "fine_tuned_model": None,
            "result_files": [],
        },
    }


def build_method_payload(
    *,
    n_epochs: Optional[int],
    batch_size: Optional[int],
    learning_rate_multiplier: Optional[float],
) -> Dict[str, Any]:
    hyperparameters: Dict[str, Any] = {}
    if n_epochs is not None:
        hyperparameters["n_epochs"] = n_epochs
    if batch_size is not None:
        hyperparameters["batch_size"] = batch_size
    if learning_rate_multiplier is not None:
        hyperparameters["learning_rate_multiplier"] = learning_rate_multiplier

    payload: Dict[str, Any] = {"type": "supervised"}
    if hyperparameters:
        payload["supervised"] = {"hyperparameters": hyperparameters}
    return payload


def create_fine_tuning_job(
    client: OpenAI,
    *,
    model: str,
    training_file_id: str,
    validation_file_id: Optional[str],
    suffix: Optional[str],
    seed: Optional[int],
    method_payload: Dict[str, Any],
    metadata: Dict[str, str],
) -> Any:
    kwargs: Dict[str, Any] = {
        "model": model,
        "training_file": training_file_id,
        "method": method_payload,
    }
    if validation_file_id:
        kwargs["validation_file"] = validation_file_id
    if suffix:
        kwargs["suffix"] = suffix
    if seed is not None:
        kwargs["seed"] = seed
    if metadata:
        kwargs["metadata"] = metadata

    logger.info("Creating fine-tuning job on model={}", model)
    job = client.fine_tuning.jobs.create(**kwargs)
    logger.info("Created job {} with status={}", job.id, job.status)
    return job


def retrieve_job(client: OpenAI, job_id: str) -> Any:
    return client.fine_tuning.jobs.retrieve(job_id)


def poll_job(
    client: OpenAI,
    job_id: str,
    *,
    poll_seconds: int = 30,
    manifest_path: Optional[Path] = None,
) -> Dict[str, Any]:
    logger.info("Polling fine-tuning job {}", job_id)
    while True:
        job = retrieve_job(client, job_id)
        payload = serialize_openai_object(job)
        status = payload.get("status")
        ft_model = payload.get("fine_tuned_model")
        logger.info(
            "Job {} status={} fine_tuned_model={}",
            job_id,
            status,
            ft_model,
        )
        if manifest_path and manifest_path.exists():
            manifest = load_json(manifest_path)
            manifest["fine_tuning_job"] = {
                "job_id": payload.get("id"),
                "status": status,
                "fine_tuned_model": ft_model,
                "result_files": payload.get("result_files", []),
                "trained_tokens": payload.get("trained_tokens"),
                "estimated_finish": payload.get("estimated_finish"),
                "error": payload.get("error"),
                "raw": payload,
            }
            write_json(manifest_path, manifest)

        if status in SUPPORTED_TERMINAL_STATUSES:
            return payload

        time.sleep(poll_seconds)


def maybe_write_fine_tuned_model_file(
    output_path: Optional[Path],
    fine_tuned_model: Optional[str],
) -> None:
    if not output_path or not fine_tuned_model:
        return
    ensure_parent_dir(output_path)
    output_path.write_text(fine_tuned_model + "\n", encoding="utf-8")
    logger.info("Wrote fine-tuned model ID to {}", output_path)


def parse_metadata(items: List[str]) -> Dict[str, str]:
    metadata: Dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Metadata item must be key=value, got: {item}")
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise ValueError(f"Metadata key is empty in item: {item}")
        metadata[key] = value
    return metadata


def create_command(args: argparse.Namespace) -> int:
    client = create_openai_client()

    train_path = Path(args.train_file)
    validation_path = Path(args.validation_file) if args.validation_file else None
    run_dir = Path(args.run_dir)
    run_name = args.run_name or build_run_name()
    run_root = run_dir / run_name
    manifest_path = run_root / "ft_run_manifest.json"
    job_raw_path = run_root / "ft_job_create_response.json"

    run_root.mkdir(parents=True, exist_ok=True)

    logger.info("Validating local training file: {}", train_path)
    train_validation = validate_chat_jsonl(train_path)
    if not train_validation["ok"]:
        write_json(run_root / "train_validation_errors.json", train_validation)
        raise ValueError(f"Training JSONL validation failed. See {run_root / 'train_validation_errors.json'}")

    validation_validation = None
    if validation_path:
        logger.info("Validating local validation file: {}", validation_path)
        validation_validation = validate_chat_jsonl(validation_path)
        if not validation_validation["ok"]:
            write_json(run_root / "validation_validation_errors.json", validation_validation)
            raise ValueError(f"Validation JSONL validation failed. See {run_root / 'validation_validation_errors.json'}")

    method_payload = build_method_payload(
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        learning_rate_multiplier=args.learning_rate_multiplier,
    )
    metadata = parse_metadata(args.metadata or [])

    local_validation = {
        "train": train_validation,
        "validation": validation_validation,
    }
    manifest = create_manifest(
        run_name=run_name,
        train_file=train_path,
        validation_file=validation_path,
        model=args.model,
        suffix=args.suffix,
        seed=args.seed,
        method_payload=method_payload,
        metadata=metadata,
        local_validation=local_validation,
    )
    write_json(manifest_path, manifest)

    training_file_obj = upload_file(client, train_path)
    validation_file_obj = upload_file(client, validation_path) if validation_path else None

    manifest = load_json(manifest_path)
    manifest["openai_files"]["training_file_id"] = training_file_obj.id
    manifest["openai_files"]["validation_file_id"] = validation_file_obj.id if validation_file_obj else None
    write_json(manifest_path, manifest)

    job = create_fine_tuning_job(
        client,
        model=args.model,
        training_file_id=training_file_obj.id,
        validation_file_id=validation_file_obj.id if validation_file_obj else None,
        suffix=args.suffix,
        seed=args.seed,
        method_payload=method_payload,
        metadata=metadata,
    )
    job_payload = serialize_openai_object(job)
    write_json(job_raw_path, job_payload)

    manifest = load_json(manifest_path)
    manifest["fine_tuning_job"] = {
        "job_id": job_payload.get("id"),
        "status": job_payload.get("status"),
        "fine_tuned_model": job_payload.get("fine_tuned_model"),
        "result_files": job_payload.get("result_files", []),
        "trained_tokens": job_payload.get("trained_tokens"),
        "estimated_finish": job_payload.get("estimated_finish"),
        "error": job_payload.get("error"),
        "raw": job_payload,
    }
    write_json(manifest_path, manifest)

    logger.info("Run manifest written to {}", manifest_path)
    logger.info("Fine-tuning job id: {}", job_payload.get("id"))

    if args.wait:
        final_payload = poll_job(
            client,
            job_payload["id"],
            poll_seconds=args.poll_seconds,
            manifest_path=manifest_path,
        )
        if final_payload.get("status") == "succeeded":
            maybe_write_fine_tuned_model_file(
                Path(args.write_model_id_to) if args.write_model_id_to else None,
                final_payload.get("fine_tuned_model"),
            )
            logger.info("Fine-tuning succeeded: {}", final_payload.get("fine_tuned_model"))
            return 0
        logger.error("Fine-tuning finished with status={}", final_payload.get("status"))
        return 1

    return 0


def status_command(args: argparse.Namespace) -> int:
    client = create_openai_client()
    payload = serialize_openai_object(retrieve_job(client, args.job_id))
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def wait_command(args: argparse.Namespace) -> int:
    client = create_openai_client()
    manifest_path = Path(args.manifest_path) if args.manifest_path else None
    final_payload = poll_job(
        client,
        args.job_id,
        poll_seconds=args.poll_seconds,
        manifest_path=manifest_path,
    )
    if final_payload.get("status") == "succeeded":
        maybe_write_fine_tuned_model_file(
            Path(args.write_model_id_to) if args.write_model_id_to else None,
            final_payload.get("fine_tuned_model"),
        )
        return 0
    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch and track an OpenAI supervised fine-tuning job."
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    create_p = subparsers.add_parser("create", help="Upload files and create an SFT job.")
    create_p.add_argument("--train-file", required=True, help="Path to training JSONL.")
    create_p.add_argument("--validation-file", required=False, help="Path to validation JSONL.")
    create_p.add_argument("--run-dir", default="data/fine_tuning/openai_sft_runs", help="Directory for run artifacts.")
    create_p.add_argument("--run-name", default=None, help="Optional explicit run name.")
    create_p.add_argument("--model", default=DEFAULT_MODEL, help=f"Base model to fine-tune. Default: {DEFAULT_MODEL}")
    create_p.add_argument("--suffix", default="alan-watts-sft-v1", help="Optional suffix for the fine-tuned model.")
    create_p.add_argument("--seed", type=int, default=45, help="Optional job seed.")
    create_p.add_argument("--n-epochs", type=int, default=None, help="Optional supervised n_epochs override.")
    create_p.add_argument("--batch-size", type=int, default=None, help="Optional supervised batch size override.")
    create_p.add_argument(
        "--learning-rate-multiplier",
        type=float,
        default=None,
        help="Optional supervised learning rate multiplier override.",
    )
    create_p.add_argument(
        "--metadata",
        action="append",
        default=[],
        help="Metadata entries as key=value. Repeat as needed.",
    )
    create_p.add_argument("--wait", action="store_true", help="Poll until the job reaches a terminal state.")
    create_p.add_argument("--poll-seconds", type=int, default=30, help="Polling interval in seconds when --wait is used.")
    create_p.add_argument(
        "--write-model-id-to",
        default=None,
        help="Optional file path where the fine-tuned model ID will be written after success.",
    )
    create_p.set_defaults(func=create_command)

    status_p = subparsers.add_parser("status", help="Retrieve current job status.")
    status_p.add_argument("--job-id", required=True, help="OpenAI fine-tuning job ID.")
    status_p.set_defaults(func=status_command)

    wait_p = subparsers.add_parser("wait", help="Poll an existing job until it finishes.")
    wait_p.add_argument("--job-id", required=True, help="OpenAI fine-tuning job ID.")
    wait_p.add_argument("--poll-seconds", type=int, default=30, help="Polling interval in seconds.")
    wait_p.add_argument("--manifest-path", default=None, help="Optional manifest JSON path to update during polling.")
    wait_p.add_argument(
        "--write-model-id-to",
        default=None,
        help="Optional file path where the fine-tuned model ID will be written after success.",
    )
    wait_p.set_defaults(func=wait_command)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    configure_logging(args.verbose)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
