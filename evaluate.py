"""
Evaluate Longevity-LLM on MGI Mouse Longevity benchmark.

Usage:
    python evaluate.py                   # both tasks, test split
    python evaluate.py --task ternary
    python evaluate.py --task pairwise
    python evaluate.py --think           # enable chain-of-thought
    python evaluate.py --limit 10        # smoke test
"""

import argparse
import json
import re
import time
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx
from openai import OpenAI
from sklearn.metrics import balanced_accuracy_score, accuracy_score

ENDPOINT = "https://sqrq2pj09htgequ0.us-east-2.aws.endpoints.huggingface.cloud/v1"
HF_TOKEN = os.environ.get("HF_TOKEN")
MODEL    = "longevity-llm"
WORKERS  = 6  # stay ≤ 8 per hackathon rules

if not HF_TOKEN:
    raise RuntimeError("Set HF_TOKEN in your environment before running evaluation.")

client = OpenAI(
    base_url=ENDPOINT,
    api_key=HF_TOKEN,
    http_client=httpx.Client(timeout=300),
)


# ── Parsing helpers ───────────────────────────────────────────────────────────

def strip_think(raw: str) -> tuple[str | None, str]:
    """Split <think>...</think> from the final answer."""
    m = re.search(r"(?:<think>)?(.*?)</think>\s*", raw, flags=re.DOTALL)
    if m and "</think>" in raw:
        return m.group(1).strip(), raw[m.end():].strip()
    return None, raw.strip()


def parse_ternary(text: str) -> str:
    """Extract Increased / Decreased / Not changed from model output."""
    low = text.lower()
    if "not changed" in low or "unchanged" in low or "not change" in low:
        return "Not changed"
    if "increased" in low or "increase" in low:
        return "Increased"
    if "decreased" in low or "decrease" in low:
        return "Decreased"
    return text


def parse_pairwise(text: str) -> str:
    """Extract A or B from model output."""
    m = re.search(r"\b([AB])\b", text.strip())
    return m.group(1) if m else text.strip()


PARSERS = {"ternary": parse_ternary, "pairwise": parse_pairwise}


# ── Single inference call ─────────────────────────────────────────────────────

def call_model(row: dict, enable_thinking: bool) -> dict:
    """Send one row to the model and return a scored result dict."""
    msgs = row["messages"][:-1]
    gold = row["messages"][-1]["content"].strip()
    fmt  = row["format"]

    t0 = time.time()
    resp = client.chat.completions.create(
        model=MODEL,
        messages=msgs,
        max_tokens=2000 if enable_thinking else 200,
        temperature=0.0,
        extra_body={"chat_template_kwargs": {"enable_thinking": enable_thinking}},
    )
    elapsed = time.time() - t0

    raw = resp.choices[0].message.content or ""
    think, answer = strip_think(raw)
    pred = PARSERS[fmt](answer)

    return {
        "lb_id":         row["lb_id"],
        "format":        fmt,
        "gold":          gold,
        "raw_response":  raw,
        "think":         think,
        "answer":        answer,
        "pred":          pred,
        "correct":       pred == gold,
        "elapsed_s":     round(elapsed, 2),
        "prompt_tokens": resp.usage.prompt_tokens,
        "total_tokens":  resp.usage.total_tokens,
        "metadata":      row.get("metadata", {}),
    }


# ── Run evaluation ────────────────────────────────────────────────────────────

def run_eval(rows: list, enable_thinking: bool, log_path: str) -> list:
    """Run all rows concurrently, stream results to log_path, return list."""
    results = []
    n_errors = 0
    with open(log_path, "w", encoding="utf-8") as flog:
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            futs = {ex.submit(call_model, r, enable_thinking): r for r in rows}
            done = 0
            for fut in as_completed(futs):
                done += 1
                try:
                    res = fut.result()
                    results.append(res)
                    flog.write(json.dumps(res) + "\n")
                    flog.flush()
                    mark = "✓" if res["correct"] else "✗"
                    print(
                        f"  [{done}/{len(rows)}] {mark}"
                        f"  pred={res['pred']!r:14s}"
                        f"  gold={res['gold']!r}"
                        f"  ({res['elapsed_s']}s)"
                    )
                except Exception as exc:  # noqa: BLE001
                    n_errors += 1
                    print(f"  [{done}/{len(rows)}] ERROR: {exc}")

    print(f"\n  Errors: {n_errors}/{len(rows)}")
    return results


# ── Metrics ───────────────────────────────────────────────────────────────────

def report_metrics(results: list, fmt: str) -> float:
    """Print per-class breakdown and return the headline score."""
    preds = [r["pred"] for r in results]
    golds = [r["gold"] for r in results]

    print(f"\n{'─'*50}")
    print(f"Task: {fmt}  |  n={len(results)}")

    if fmt == "ternary":
        score    = balanced_accuracy_score(golds, preds)
        baseline = 1 / 3
        print(f"Balanced accuracy : {score:.3f}")
        print(f"Random baseline   : {baseline:.3f}")
        for label in ["Increased", "Decreased", "Not changed"]:
            sub = [(p, g) for p, g in zip(preds, golds) if g == label]
            if sub:
                correct = sum(p == g for p, g in sub)
                print(f"  {label:12s}: {correct}/{len(sub)} = {correct/len(sub):.2f}")
    else:
        score    = accuracy_score(golds, preds)
        baseline = 0.5
        print(f"Accuracy          : {score:.3f}")
        print(f"Random baseline   : {baseline:.3f}")
        pred_dist = Counter(preds)
        print(f"  Pred dist: {dict(pred_dist)}")

    print(f"{'─'*50}")
    return score


# ── Load JSONL ────────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> list:
    """Load a JSONL file into a list of dicts."""
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task",  choices=["ternary", "pairwise", "both"], default="both")
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--think", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    os.makedirs("output/eval", exist_ok=True)
    think_tag = "think" if args.think else "nothink"
    task_list = ["ternary", "pairwise"] if args.task == "both" else [args.task]

    for fmt in task_list:
        data_path = f"output/mgi_{fmt}_{args.split}.jsonl"
        if not os.path.exists(data_path):
            print(f"File not found: {data_path} — skipping")
            continue

        rows = load_jsonl(data_path)
        if args.limit:
            rows = rows[: args.limit]

        log_path = f"output/eval/results_{fmt}_{args.split}_{think_tag}.jsonl"
        print(f"\n{'='*50}")
        print(f"Running {fmt} | split={args.split} | n={len(rows)} | thinking={args.think}")
        print(f"Logging → {log_path}")
        print(f"{'='*50}")

        results = run_eval(rows, args.think, log_path)
        score   = report_metrics(results, fmt)

        summary = {
            "task":     fmt,
            "split":    args.split,
            "thinking": args.think,
            "n":        len(results),
            "score":    round(score, 4),
            "metric":   "balanced_accuracy" if fmt == "ternary" else "accuracy",
            "baseline": round(1 / 3 if fmt == "ternary" else 0.5, 4),
        }
        summary_path = f"output/eval/summary_{fmt}_{args.split}_{think_tag}.json"
        with open(summary_path, "w", encoding="utf-8") as sf:
            json.dump(summary, sf, indent=2)
        print(f"Summary → {summary_path}")
