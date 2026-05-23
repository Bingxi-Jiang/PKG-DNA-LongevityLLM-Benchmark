"""
Evaluate Longevity-LLM on the MGI Mouse Longevity benchmark.

Usage:
    python evaluate.py                    # effect + pairwise, test split
    python evaluate.py --task effect
    python evaluate.py --task pairwise
    python evaluate.py --think            # enable model thinking mode
    python evaluate.py --limit 10         # smoke test
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

from sklearn.metrics import accuracy_score, balanced_accuracy_score


DEFAULT_ENDPOINT = "https://sqrq2pj09htgequ0.us-east-2.aws.endpoints.huggingface.cloud/v1"
MODEL = "longevity-llm"
WORKERS = 6  # Stay <= 8 per hackathon guidance.

TASK_FILE_PREFIX = {
    "effect": "mgi_effect",
    "pairwise": "mgi_pairwise",
    "ternary": "mgi_ternary",  # Legacy support for older generated files.
}


def strip_think(raw: str) -> tuple[str | None, str]:
    """Split <think>...</think> from the final answer."""
    match = re.search(r"(?:<think>)?(.*?)</think>\s*", raw, flags=re.DOTALL)
    if match and "</think>" in raw:
        return match.group(1).strip(), raw[match.end() :].strip()
    return None, raw.strip()


def normalize_effect_label(label: str) -> str:
    label = label.lower()
    if label.startswith("inc"):
        return "Increased"
    if label.startswith("dec"):
        return "Decreased"
    return label


def parse_effect(text: str) -> str:
    """Extract Increased / Decreased from model output."""
    matches = re.findall(r"\b(Increased|Decrease[d]?|Increase[d]?)\b", text, flags=re.I)
    if matches:
        return normalize_effect_label(matches[-1])
    return text.strip()


def parse_ternary(text: str) -> str:
    """Extract Increased / Decreased / Not changed from legacy ternary output."""
    low = text.lower()
    if "not changed" in low or "unchanged" in low or "not change" in low:
        return "Not changed"
    matches = re.findall(r"\b(Increased|Decrease[d]?|Increase[d]?)\b", text, flags=re.I)
    if matches:
        return normalize_effect_label(matches[-1])
    return text.strip()


def parse_pairwise(text: str) -> str:
    """Extract the final standalone A or B from model output."""
    matches = re.findall(r"(?<![A-Za-z])([AB])(?![A-Za-z])", text.upper())
    return matches[-1] if matches else text.strip()


PARSERS = {
    "binary": parse_effect,
    "effect": parse_effect,
    "ternary": parse_ternary,
    "pairwise": parse_pairwise,
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


def run_eval(rows: list[dict], client, enable_thinking: bool, log_path: str) -> list[dict]:
    """Run rows concurrently, stream results to log_path, and return results."""
    results = []
    n_errors = 0
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


def report_metrics(results: list[dict], task: str) -> tuple[float, str, float]:
    """Print per-class breakdown and return score, metric name, baseline."""
    preds = [result["pred"] for result in results]
    golds = [result["gold"] for result in results]

    print(f"\n{'=' * 50}")
    print(f"Task: {task}  |  n={len(results)}")

    if task == "pairwise":
        score = accuracy_score(golds, preds)
        metric = "accuracy"
        baseline = 0.5
        print(f"Accuracy        : {score:.3f}")
        print(f"Random baseline : {baseline:.3f}")
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


def load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as file:
        return [json.loads(line) for line in file]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["effect", "pairwise", "ternary", "both"], default="both")
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--think", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    args = parser.parse_args()

    client = make_client(args.endpoint)
    os.makedirs("output/eval", exist_ok=True)
    think_tag = "think" if args.think else "nothink"
    tasks = ["effect", "pairwise"] if args.task == "both" else [args.task]

    for task in tasks:
        data_path = f"output/{TASK_FILE_PREFIX[task]}_{args.split}.jsonl"
        if not os.path.exists(data_path):
            print(f"File not found: {data_path} - skipping")
            continue

        rows = load_jsonl(data_path)
        if args.limit:
            rows = rows[: args.limit]

        log_path = f"output/eval/results_{task}_{args.split}_{think_tag}.jsonl"
        print(f"\n{'=' * 50}")
        print(f"Running {task} | split={args.split} | n={len(rows)} | thinking={args.think}")
        print(f"Logging -> {log_path}")
        print(f"{'=' * 50}")

        results = run_eval(rows, client, args.think, log_path)
        score, metric, baseline = report_metrics(results, task)

        summary = {
            "task": task,
            "split": args.split,
            "thinking": args.think,
            "n": len(results),
            "score": round(score, 4),
            "metric": metric,
            "baseline": round(baseline, 4),
        }
        summary_path = f"output/eval/summary_{task}_{args.split}_{think_tag}.json"
        with open(summary_path, "w", encoding="utf-8") as summary_file:
            json.dump(summary, summary_file, indent=2)
        print(f"Summary -> {summary_path}")


if __name__ == "__main__":
    main()
