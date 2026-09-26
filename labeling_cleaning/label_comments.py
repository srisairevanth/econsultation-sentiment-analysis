"""One-time local LLM labeling of public comments via Ollama (qwen3:8b)."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    ALLOWED_LABELS,
    COMMENT_COLUMN,
    DEFAULT_FAILED_PATH,
    DEFAULT_MODEL,
    DEFAULT_OUTPUT_CSV,
    JSON_COMMENT_PATHS,
    MAX_RETRIES,
    OLLAMA_HOST,
    OLLAMA_MODEL,
    OLLAMA_SEED,
    OLLAMA_TEMPERATURE,
    PROJECT_ROOT,
    SAVE_EVERY,
    SYSTEM_PROMPT,
)

THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
LABEL_TOKEN = re.compile(r"\b(Positive|Negative|Neutral)\b", re.IGNORECASE)


class LabelingError(RuntimeError):
    """Raised when Ollama or the local environment is not usable."""


def is_blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    return str(value).strip() == ""


def nested_get(obj: Any, path: tuple[str, ...]) -> Any:
    current = obj
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def extract_comment_from_record(record: Any) -> str | None:
    """Pull comment text from a JSON object using known Regulations.gov fields."""
    if isinstance(record, str):
        return record if not is_blank(record) else None
    if not isinstance(record, dict):
        return None

    for path in JSON_COMMENT_PATHS:
        value = nested_get(record, path)
        if isinstance(value, str) and not is_blank(value):
            return value.strip()

    return None


def iter_json_records(payload: Any) -> Iterable[Any]:
    """Walk common JSON envelopes (list, JSON:API data[], nested lists)."""
    if payload is None:
        return
    if isinstance(payload, list):
        for item in payload:
            yield from iter_json_records(item)
        return
    if isinstance(payload, dict):
        if "data" in payload and isinstance(payload["data"], list):
            for item in payload["data"]:
                yield from iter_json_records(item)
            return
        if "comments" in payload and isinstance(payload["comments"], list):
            for item in payload["comments"]:
                yield from iter_json_records(item)
            return
        yield payload
        return
    yield payload


def load_comments_from_json(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    comments: list[str] = []
    for record in iter_json_records(payload):
        text = extract_comment_from_record(record)
        if text is not None:
            comments.append(text)
    return comments


def detect_csv_comment_column(fieldnames: list[str] | None, requested: str | None) -> str:
    if not fieldnames:
        raise LabelingError("CSV file has no header row; cannot find a comment column.")
    if requested:
        if requested not in fieldnames:
            raise LabelingError(
                f"CSV column '{requested}' was not found. Available columns: {', '.join(fieldnames)}"
            )
        return requested

    lowered = {name.lower(): name for name in fieldnames}
    for candidate in ("comment", "commenttext", "comment_text", "text", "body", "content"):
        if candidate in lowered:
            return lowered[candidate]
    raise LabelingError(
        "Could not auto-detect the comment column. Pass --comment-column. "
        f"Available columns: {', '.join(fieldnames)}"
    )


def load_comments_from_csv(path: Path, comment_column: str | None) -> tuple[list[str], str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        column = detect_csv_comment_column(reader.fieldnames, comment_column)
        comments: list[str] = []
        for row in reader:
            value = row.get(column)
            if is_blank(value):
                continue
            comments.append(str(value).strip())
    return comments, column


def load_valid_comments(input_path: Path, comment_column: str | None) -> list[str]:
    suffix = input_path.suffix.lower()
    if suffix == ".json":
        comments = load_comments_from_json(input_path)
        used_column = "attributes.comment (JSON auto-detect)"
    elif suffix == ".csv":
        comments, used_column = load_comments_from_csv(input_path, comment_column)
    else:
        raise LabelingError("Input must be a .json or .csv file.")

    print(f"Loaded {len(comments)} valid comment(s) from {input_path}", flush=True)
    print(f"Comment field: {used_column}", flush=True)
    return comments


def comment_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def checkpoint_path_for(output_path: Path) -> Path:
    return output_path.with_name(output_path.name + ".checkpoint.json")


def load_checkpoint(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    tmp.replace(path)


def load_existing_rows(output_path: Path) -> list[dict[str, str]]:
    if not output_path.exists():
        return []
    suffix = output_path.suffix.lower()
    if suffix == ".json":
        with output_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, list):
            raise LabelingError("Existing JSON output must be a list of {comment, label} objects.")
        rows: list[dict[str, str]] = []
        for item in data:
            if isinstance(item, dict) and "comment" in item and "label" in item:
                rows.append({"comment": str(item["comment"]), "label": str(item["label"])})
        return rows
    if suffix == ".csv":
        with output_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            return [
                {"comment": row.get("comment", ""), "label": row.get("label", "")}
                for row in reader
                if row.get("comment") is not None
            ]
    raise LabelingError("Output must be a .csv or .json file.")


def write_output(output_path: Path, rows: list[dict[str, str]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    if suffix == ".json":
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(rows, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
    elif suffix == ".csv":
        with tmp.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["comment", "label"], extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    else:
        raise LabelingError("Output must be a .csv or .json file.")
    tmp.replace(output_path)


def load_failed(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, list) else []


def write_failed(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    tmp.replace(path)


def parse_label(raw: str) -> str | None:
    if raw is None:
        return None
    cleaned = THINK_BLOCK.sub("", raw).strip()
    cleaned = cleaned.replace("```", "").strip().strip('"').strip("'")
    if cleaned in ALLOWED_LABELS:
        return cleaned
    lowered = cleaned.lower()
    for label in ALLOWED_LABELS:
        if lowered == label.lower():
            return label
    matches = LABEL_TOKEN.findall(cleaned)
    if len(matches) == 1:
        token = matches[0].capitalize()
        if token == "Neutral":
            return "Neutral"
        if token in ALLOWED_LABELS:
            return token
        mapping = {"positive": "Positive", "negative": "Negative", "neutral": "Neutral"}
        return mapping.get(matches[0].lower())
    return None


def ollama_base_url(host: str) -> str:
    parsed = urlparse(host if "://" in host else f"http://{host}")
    if parsed.scheme not in {"http", "https"}:
        raise LabelingError("OLLAMA_HOST must be a local http(s) URL, not a cloud AI endpoint.")
    return f"{parsed.scheme}://{parsed.netloc}"


def check_ollama(host: str, model: str) -> None:
    base = ollama_base_url(host)
    try:
        response = requests.get(f"{base}/api/tags", timeout=10)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise LabelingError(
            "Ollama does not appear to be running or the local API is not reachable at "
            f"{base}. Start Ollama, then retry.\n"
            f"Details: {exc}"
        ) from exc

    payload = response.json()
    models = payload.get("models") or []
    names = [
        item.get("name")
        for item in models
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    ]
    model_ok = any(name == model or name.startswith(f"{model}-") or name.startswith(f"{model}:") for name in names)
    if model in names:
        model_ok = True
    if not model_ok:
        raise LabelingError(
            f"Model '{model}' is not available locally."
            + (f" Installed models: {', '.join(names)}." if names else "")
            + "\nInstall it with:\n\n    ollama pull qwen3:8b\n"
        )
    print(f"Ollama is reachable at {base}")
    print(f"Using model: {model}")


def classify_comment(
    host: str,
    model: str,
    comment: str,
    temperature: float,
    seed: int,
) -> str:
    base = ollama_base_url(host)
    payload = {
        "model": model,
        "stream": False,
        "think": False,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "/no_think\n"
                    "Classify the overall sentiment of this public comment. "
                    "Reply with only one word: Positive, Negative, or Neutral.\n\n"
                    f"Comment:\n{comment}"
                ),
            },
        ],
        "options": {
            "temperature": temperature,
            "top_p": 0.1,
            "seed": seed,
            "num_predict": 24,
        },
    }
    response = requests.post(f"{base}/api/chat", json=payload, timeout=180)
    response.raise_for_status()
    body = response.json()
    message = body.get("message") or {}
    content = message.get("content") or body.get("response") or ""
    label = parse_label(str(content))
    if label is None:
        raise ValueError(f"Invalid model output: {content!r}")
    return label


def classify_with_retries(
    host: str,
    model: str,
    comment: str,
    temperature: float,
    seed: int,
    retries: int,
) -> tuple[str | None, str | None]:
    last_error = None
    attempts = max(1, retries)
    for attempt in range(1, attempts + 1):
        try:
            label = classify_comment(host, model, comment, temperature, seed)
            return label, None
        except (requests.RequestException, ValueError, json.JSONDecodeError) as exc:
            last_error = str(exc)
            print(f"  Attempt {attempt}/{attempts} failed: {last_error}")
            time.sleep(min(2 * attempt, 6))
    return None, last_error


def resume_start_index(
    comments: list[str],
    output_rows: list[dict[str, str]],
    checkpoint: dict[str, Any] | None,
    input_path: Path,
) -> int:
    if checkpoint:
        hashes = checkpoint.get("comment_hashes") or []
        start = int(checkpoint.get("processed_count") or 0)
        if start > len(comments):
            raise LabelingError(
                "Checkpoint processed_count is larger than the current input. "
                "Delete the checkpoint and output files to start over."
            )
        for index in range(min(start, len(hashes))):
            if comment_hash(comments[index]) != hashes[index]:
                raise LabelingError(
                    f"Input comments no longer match the checkpoint at index {index}. "
                    "Use the same input file, or remove the checkpoint/output to relabel."
                )
        if start:
            print(f"Resuming from comment {start + 1}/{len(comments)} using checkpoint data.")
        return start

    if output_rows:
        start = len(output_rows)
        if start > len(comments):
            raise LabelingError("Existing output has more rows than valid input comments.")
        for index, row in enumerate(output_rows):
            if row.get("comment") != comments[index]:
                raise LabelingError(
                    "Existing output comments do not match the input order. "
                    "Use a new --output path or remove the old labeled file."
                )
        print(f"Resuming from comment {start + 1}/{len(comments)} using existing output.")
        return start

    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Label public comments locally with Ollama qwen3:8b (offline, one-time labeling)."
    )
    parser.add_argument("--input", required=True, type=Path, help="Path to raw JSON or CSV comments")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help="Output labeled dataset (default: data/labeled/labeled_comments.csv)",
    )
    parser.add_argument(
        "--comment-column",
        default=None,
        help=f"CSV comment column name (default: {COMMENT_COLUMN} or auto-detect)",
    )
    parser.add_argument("--model", default=OLLAMA_MODEL, help=f"Ollama model name (default: {OLLAMA_MODEL})")
    parser.add_argument("--ollama-host", default=OLLAMA_HOST, help=f"Local Ollama URL (default: {OLLAMA_HOST})")
    parser.add_argument("--save-every", type=int, default=SAVE_EVERY, help="Write output every N labeled comments")
    parser.add_argument("--max-retries", type=int, default=MAX_RETRIES, help="Retries per comment on invalid/API failure")
    parser.add_argument(
        "--failed-output",
        type=Path,
        default=DEFAULT_FAILED_PATH,
        help="Where to record comments that could not be labeled",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N remaining comments (use this for a small test run)",
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip Ollama/model checks (not recommended)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    input_path = args.input if args.input.is_absolute() else PROJECT_ROOT / args.input
    output_path = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    failed_path = args.failed_output if args.failed_output.is_absolute() else PROJECT_ROOT / args.failed_output

    if not input_path.exists():
        print(f"Input file not found: {input_path}", file=sys.stderr)
        return 1
    if not input_path.is_file():
        print(f"Input path is not a file: {input_path}", file=sys.stderr)
        return 1

    if input_path.resolve() == output_path.resolve():
        print("Refusing to write output over the original input file.", file=sys.stderr)
        return 1

    comment_column = args.comment_column or COMMENT_COLUMN
    try:
        comments = load_valid_comments(input_path, comment_column)
    except LabelingError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if not comments:
        print("No valid comments found (null/empty/whitespace-only rows were skipped).")
        return 0

    checkpoint_file = checkpoint_path_for(output_path)
    existing_rows = load_existing_rows(output_path)
    checkpoint = load_checkpoint(checkpoint_file)
    try:
        start = resume_start_index(comments, existing_rows, checkpoint, input_path)
    except LabelingError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    labeled_rows = existing_rows[:start]
    hashes = [comment_hash(text) for text in comments[:start]]
    failed_rows = load_failed(failed_path)

    if not args.skip_preflight:
        try:
            check_ollama(args.ollama_host, args.model)
        except LabelingError as exc:
            print(str(exc), file=sys.stderr)
            return 1
    elif args.model != DEFAULT_MODEL:
        print(f"Using model: {args.model}")

    remaining = comments[start:]
    if args.limit is not None:
        remaining = remaining[: max(0, args.limit)]

    total = len(comments)
    processed_this_run = 0
    save_every = max(1, args.save_every)

    try:
        for offset, comment in enumerate(remaining):
            index = start + offset
            number = index + 1
            print(f"\nProcessing comment {number}/{total}")
            label, error = classify_with_retries(
                host=args.ollama_host,
                model=args.model,
                comment=comment,
                temperature=OLLAMA_TEMPERATURE,
                seed=OLLAMA_SEED,
                retries=args.max_retries,
            )
            if label is None:
                print("Label: FAILED (recorded; continuing)")
                failed_rows.append(
                    {
                        "index": index,
                        "comment": comment,
                        "error": error,
                        "retries": args.max_retries,
                    }
                )
                write_failed(failed_path, failed_rows)
            else:
                print(f"Label: {label}")
                labeled_rows.append({"comment": comment, "label": label})

            hashes.append(comment_hash(comment))
            processed_this_run += 1
            done_count = index + 1

            if processed_this_run % save_every == 0 or number == start + len(remaining):
                write_output(output_path, labeled_rows)
                save_checkpoint(
                    checkpoint_file,
                    {
                        "input_path": str(input_path.resolve()),
                        "output_path": str(output_path.resolve()),
                        "processed_count": done_count,
                        "comment_hashes": hashes,
                        "model": args.model,
                    },
                )
                print(f"Saved progress ({len(labeled_rows)} labeled row(s)) -> {output_path}")
    except KeyboardInterrupt:
        write_output(output_path, labeled_rows)
        save_checkpoint(
            checkpoint_file,
            {
                "input_path": str(input_path.resolve()),
                "output_path": str(output_path.resolve()),
                "processed_count": start + processed_this_run,
                "comment_hashes": hashes,
                "model": args.model,
            },
        )
        print("\nInterrupted. Progress saved; re-run the same command to resume.")
        return 130

    print(f"\nFinished. Labeled file: {output_path}")
    print(f"Labeled rows: {len(labeled_rows)}")
    if failed_rows:
        print(f"Failed comments: {len(failed_rows)} -> {failed_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
