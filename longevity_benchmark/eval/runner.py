"""Evaluation runner for OpenAI-compatible chat endpoints."""

from __future__ import annotations

import json
import os
import time

try:
    from dotenv import load_dotenv
    load_dotenv()
except ModuleNotFoundError:
    pass
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from sklearn.metrics import accuracy_score, balanced_accuracy_score

from ..io import load_jsonl
from .parsing import PARSERS, strip_think


DEFAULT_ENDPOINT = "https://swchnq0ekc3scmqw.us-east-2.aws.endpoints.huggingface.cloud/v1"
MODEL = "longevity-llm"
WORKERS = 6  # Stay <= 8 per hackathon guidance.

TASK_FILE_PREFIX = {
    "effect": "mgi_effect",
    "mcq": "mgi_mcq",
    "ternary": "mgi_ternary",
    "set": "mgi_set",
    "pairwise": "mgi_pairwise",
}


def make_client(endpoint: str):
    try:
        import httpx
        from openai import OpenAI
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Install evaluation dependencies first: pip install openai httpx scikit-learn"
        ) from exc

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("Set HF_TOKEN in your environment before running evaluation.")
    return OpenAI(
        base_url=endpoint,
        api_key=hf_token,
        http_client=httpx.Client(timeout=300),
    )


def call_model(row: dict, client, enable_thinking: bool) -> dict:
    """Send one row to the model and return a scored result dict."""
    messages = row["messages"][:-1]
    gold = row["messages"][-1]["content"].strip()
    fmt = row["format"]
    parser = PARSERS.get(fmt)
    if parser is None:
        raise ValueError(f"No parser registered for format: {fmt}")

    start = time.time()
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_tokens=2000 if enable_thinking else 200,
        temperature=0.0,
        extra_body={"chat_template_kwargs": {"enable_thinking": enable_thinking}},
    )
    elapsed = time.time() - start

    raw = response.choices[0].message.content or ""
    think, answer = strip_think(raw)
    pred = parser(answer)

    return {
        "lb_id": row["lb_id"],
        "format": fmt,
        "gold": gold,
        "raw_response": raw,
        "think": think,
        "answer": answer,
        "pred": pred,
        "correct": pred == gold,
        "elapsed_s": round(elapsed, 2),
        "prompt_tokens": response.usage.prompt_tokens,
        "total_tokens": response.usage.total_tokens,
        "metadata": row.get("metadata", {}),
    }


def run_eval(rows: list[dict], client, enable_thinking: bool, log_path: str | Path) -> list[dict]:
    """Run rows concurrently, stream results to log_path, and return results."""
    results = []
    n_errors = 0
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as log_file:
        with ThreadPoolExecutor(max_workers=WORKERS) as executor:
            futures = {
                executor.submit(call_model, row, client, enable_thinking): row for row in rows
            }
            done = 0
            for future in as_completed(futures):
                done += 1
                try:
                    result = future.result()
                    results.append(result)
                    log_file.write(json.dumps(result, ensure_ascii=False) + "\n")
                    log_file.flush()
                    mark = "OK" if result["correct"] else "NO"
                    print(
                        f"  [{done}/{len(rows)}] {mark}"
                        f"  pred={result['pred']!r:12s}"
                        f"  gold={result['gold']!r}"
                        f"  ({result['elapsed_s']}s)"
                    )
                except Exception as exc:  # noqa: BLE001
                    n_errors += 1
                    print(f"  [{done}/{len(rows)}] ERROR: {exc}")

    print(f"\n  Errors: {n_errors}/{len(rows)}")
    return results


def parse_set_value(value: str) -> set[str]:
    if not value:
        return set()
    return {item.strip() for item in value.split(",") if item.strip()}


def set_f1(gold: set[str], pred: set[str]) -> float:
    if not gold and not pred:
        return 1.0
    if not gold or not pred:
        return 0.0
    overlap = len(gold & pred)
    precision = overlap / len(pred)
    recall = overlap / len(gold)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def report_metrics(results: list[dict], task: str) -> tuple[float, str, float]:
    """Print per-class breakdown and return score, metric name, baseline."""
    preds = [result["pred"] for result in results]
    golds = [result["gold"] for result in results]

    print(f"\n{'=' * 50}")
    print(f"Task: {task}  |  n={len(results)}")

    if task in {"pairwise", "mcq"}:
        score = accuracy_score(golds, preds)
        metric = "accuracy"
        baseline = 0.5
        if task == "mcq":
            baseline = 0.25
        print(f"Accuracy        : {score:.3f}")
        print(f"Random baseline : {baseline:.3f}")
        print(f"Pred dist       : {dict(Counter(preds))}")
    elif task == "set":
        row_scores = [
            set_f1(parse_set_value(gold), parse_set_value(pred))
            for gold, pred in zip(golds, preds)
        ]
        score = sum(row_scores) / len(row_scores) if row_scores else 0.0
        metric = "mean_set_f1"
        baseline = 0.0
        exact = sum(gold == pred for gold, pred in zip(golds, preds))
        print(f"Mean set F1     : {score:.3f}")
        exact_rate = exact / len(results) if results else 0.0
        print(f"Exact match     : {exact}/{len(results)} = {exact_rate:.2f}")
        print(f"Pred dist       : {dict(Counter(preds))}")
    else:
        score = balanced_accuracy_score(golds, preds)
        metric = "balanced_accuracy"
        labels = sorted(set(golds))
        baseline = 1 / len(labels)
        print(f"Balanced accuracy : {score:.3f}")
        print(f"Random baseline   : {baseline:.3f}")
        for label in labels:
            sub = [(pred, gold) for pred, gold in zip(preds, golds) if gold == label]
            correct = sum(pred == gold for pred, gold in sub)
            print(f"  {label:12s}: {correct}/{len(sub)} = {correct / len(sub):.2f}")

    print(f"{'=' * 50}")
    return score, metric, baseline


def evaluate_task(
    task: str,
    split: str,
    input_dir: str,
    eval_dir: str,
    client,
    enable_thinking: bool,
    limit: int | None,
) -> None:
    data_path = Path(input_dir) / f"{TASK_FILE_PREFIX[task]}_{split}.jsonl"
    if not data_path.exists():
        print(f"File not found: {data_path} - skipping")
        return

    rows = load_jsonl(data_path)
    if limit:
        rows = rows[:limit]

    think_tag = "think" if enable_thinking else "nothink"
    log_path = Path(eval_dir) / f"results_{task}_{split}_{think_tag}.jsonl"
    print(f"\n{'=' * 50}")
    print(f"Running {task} | split={split} | n={len(rows)} | thinking={enable_thinking}")
    print(f"Logging -> {log_path}")
    print(f"{'=' * 50}")

    results = run_eval(rows, client, enable_thinking, log_path)
    score, metric, baseline = report_metrics(results, task)

    summary = {
        "task": task,
        "split": split,
        "thinking": enable_thinking,
        "n": len(results),
        "score": round(score, 4),
        "metric": metric,
        "baseline": round(baseline, 4),
    }
    summary_path = Path(eval_dir) / f"summary_{task}_{split}_{think_tag}.json"
    with open(summary_path, "w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2)
    print(f"Summary -> {summary_path}")
