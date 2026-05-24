"""
compare.py — Side-by-side comparison of LongevityLLM, Gemini, Claude, and a
weighted-feature-similarity baseline on the benchmark tasks.

Weighted similarity baseline mirrors the scoring logic described in the project:
  feature weights: allele_type 40%, gene 30%, genetic_background 30%
  prediction: majority vote over the K most similar training examples.

Usage:
    python compare.py --task effect --split test --limit 30
    python compare.py --task ternary --split test
    python compare.py --task all --split test --limit 20
    python compare.py --task effect --providers longevity gemini  # skip claude
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ModuleNotFoundError:
    pass

import argparse

from longevity_benchmark.eval.runner import TASK_FILE_PREFIX

# ── constants ────────────────────────────────────────────────────────────────

ALL_PROVIDERS = ["longevity", "gemini", "claude"]

# Tasks where weighted-similarity baseline makes sense (classification only).
BASELINE_TASKS = {"effect", "ternary", "mcq"}

# Feature weights for the weighted-similarity predictor.
FEATURE_WEIGHTS = {
    "allele_type":        0.40,
    "gene":               0.30,
    "genetic_background": 0.30,
}

K_NEIGHBORS = 5

# Per-provider worker count. Gemini free tier is 10 RPM, so we use 2 workers
# to stay well under it and avoid silent 429-retry stalls inside the SDK.
PROVIDER_WORKERS = {
    "longevity": 6,
    "gemini":    2,
    "claude":    4,
}

# weighted-similarity baseline

def _parse_meta(raw) -> dict:
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except Exception:
            return {}
    return raw or {}


def _extract_features(row: dict) -> dict:
    meta = _parse_meta(row.get("metadata"))
    bg = meta.get("genetic_background", "")
    # Normalise background to the leading strain token (e.g. "C57BL/6J").
    bg_norm = bg.split()[0] if bg else ""
    return {
        "allele_type": meta.get("allele_type", "").strip().lower(),
        "gene":        meta.get("gene", "").strip().lower(),
        "genetic_background": bg_norm.strip().lower(),
    }


def _weighted_sim(a: dict, b: dict) -> float:
    score = 0.0
    for feat, w in FEATURE_WEIGHTS.items():
        av, bv = a.get(feat, ""), b.get(feat, "")
        if av and bv and av == bv:
            score += w
    return score


def _similarity_predict(test_feat: dict, train_rows: list[dict]) -> str:
    scored = []
    for row in train_rows:
        gold = row["messages"][-1]["content"].strip()
        sim  = _weighted_sim(test_feat, _extract_features(row))
        scored.append((sim, gold))
    scored.sort(key=lambda x: -x[0])
    votes = Counter(label for _, label in scored[:K_NEIGHBORS])
    return votes.most_common(1)[0][0]


def run_similarity_baseline(test_rows: list[dict], train_rows: list[dict]) -> list[dict]:
    results = []
    for row in test_rows:
        gold = row["messages"][-1]["content"].strip()
        pred = _similarity_predict(_extract_features(row), train_rows)
        results.append({
            "lb_id":   row["lb_id"],
            "gold":    gold,
            "pred":    pred,
            "correct": pred == gold,
        })
    return results


# LLM runners

def run_provider(
    test_rows: list[dict],
    provider:  str,
    enable_thinking: bool = False,
) -> tuple[str, list[dict]]:
    """Run one provider; return (label, results). Skips gracefully if key missing."""
    from longevity_benchmark.eval.runner import make_client, call_model, PROVIDER_CONFIGS

    try:
        client, model = make_client(provider)
    except RuntimeError as exc:
        print(f"  [{provider}] skipped — {exc}")
        return provider, []

    label = f"{provider}\n({model})"
    results: list[dict] = []
    n_errors = 0

    workers = PROVIDER_WORKERS.get(provider, 4)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {
            ex.submit(call_model, row, client, model, provider, enable_thinking): row
            for row in test_rows
        }
        done = 0
        for fut in as_completed(futs):
            done += 1
            try:
                results.append(fut.result())
            except Exception as exc:  # noqa: BLE001
                n_errors += 1
                print(f"  [{provider}] row error: {exc}")
            n_valid = len(results)
            acc = sum(r["correct"] for r in results) / n_valid if n_valid else 0.0
            print(
                f"  [{provider}] [{done}/{len(test_rows)}]  acc={acc:.1%}"
                f"  errors={n_errors}",
                end="\r", flush=True,
            )
    print()
    return label, results


# metrics

def _accuracy(results: list[dict]) -> float:
    if not results:
        return float("nan")
    return sum(r["correct"] for r in results) / len(results)


def _agreement(a: list[dict], b: list[dict]) -> float:
    """Fraction of rows where two methods predict the same label."""
    pred_a = {r["lb_id"]: r.get("pred") or r.get("answer") for r in a}
    pred_b = {r["lb_id"]: r.get("pred") or r.get("answer") for r in b}
    shared = [k for k in pred_a if k in pred_b]
    if not shared:
        return float("nan")
    return sum(pred_a[k] == pred_b[k] for k in shared) / len(shared)


# display

def _short_label(label: str) -> str:
    """Turn a multi-line label into a short column header."""
    return label.split("\n")[0]


def print_row_table(all_results: dict[str, list[dict]], test_rows: list[dict]) -> None:
    col_w = 16
    names  = list(all_results.keys())
    shorts = [_short_label(n) for n in names]

    header = f"{'#':<4} {'Gold':<14}" + "".join(f" {s:<{col_w}}" for s in shorts)
    sep    = "-" * len(header)
    print("\n" + sep)
    print(header)
    print(sep)

    gold_map = {row["lb_id"]: row["messages"][-1]["content"].strip() for row in test_rows}
    pred_maps = {
        name: {r["lb_id"]: (r.get("pred") or r.get("answer") or "?") for r in res}
        for name, res in all_results.items()
    }

    for i, row in enumerate(test_rows, 1):
        lid  = row["lb_id"]
        gold = gold_map[lid]
        line = f"{i:<4} {gold:<14}"
        for name in names:
            pred   = pred_maps[name].get(lid, "?")
            marker = "✓" if pred == gold else "✗"
            cell   = f"{marker} {pred}"
            line  += f" {cell:<{col_w}}"
        print(line)

    print(sep)


def annotate_compare_results(all_results: dict[str, list[dict]]) -> None:
    """Score traces in-place for every method whose results contain a `think` field."""
    from longevity_benchmark.reasoning_scorer import (
        load_allele_db, load_gene_db, load_mp_term_db, score_one,
    )
    gene_db, allele_db, mp_db = load_gene_db(), load_allele_db(), load_mp_term_db()
    for _, results in all_results.items():
        if not results or not any((r.get("think") or "").strip() for r in results):
            continue
        for r in results:
            ts = score_one(r, gene_db, allele_db, mp_db)
            r["trace_score"] = ts.aggregate
            subs = dict(ts.subscores)
            gv = ts.details.get("gene_validity") or {}
            av = ts.details.get("allele_validity") or {}
            mv = ts.details.get("mp_validity") or {}
            subs["_fab_genes"]   = gv.get("invalid", [])
            subs["_fab_alleles"] = av.get("invalid", [])
            subs["_invalid_mp"]  = sum(
                1 for d in mv.get("details", []) if d.get("status") == "invalid_id"
            )
            r["trace_subscores"] = subs


def print_trace_quality_table(all_results: dict[str, list[dict]]) -> None:
    rows_with_scores = {
        name: [r for r in res if r.get("trace_score") is not None]
        for name, res in all_results.items()
    }
    if not any(rows_with_scores.values()):
        return

    print("\n── Reasoning trace quality " + "─" * 33)
    print(f"  {'method':<22}  {'mean':>6}  {'cov':>5}  {'fab_gene':>8}  {'fab_allele':>10}  {'bad_mp':>6}")
    for name, _results in all_results.items():
        short  = _short_label(name)
        scored = rows_with_scores.get(name, [])
        if not scored:
            continue
        vals = [r["trace_score"] for r in scored]
        mean_score = sum(vals) / len(vals)
        cov = len(scored) / len(_results) if _results else 0.0
        fab_g  = sum(len((r.get("trace_subscores") or {}).get("_fab_genes",   [])) for r in scored)
        fab_a  = sum(len((r.get("trace_subscores") or {}).get("_fab_alleles", [])) for r in scored)
        bad_mp = sum((r.get("trace_subscores") or {}).get("_invalid_mp", 0) for r in scored)
        print(f"  {short:<22}  {mean_score:>6.3f}  {cov:>4.0%}  {fab_g:>8}  {fab_a:>10}  {bad_mp:>6}")
    print()


def print_summary_table(all_results: dict[str, list[dict]], n_total: int) -> None:
    print("\n── Accuracy summary " + "─" * 40)
    for name, results in all_results.items():
        short = _short_label(name)
        acc   = _accuracy(results)
        n     = len(results)
        bar_len = int(acc * 30) if acc == acc else 0
        bar   = "█" * bar_len + "░" * (30 - bar_len)
        print(f"  {short:<22}  {acc:>6.1%}  {n}/{n_total}  [{bar}]")

    print()
    names = list(all_results.keys())
    if len(names) < 2:
        return
    print("── Pairwise agreement " + "─" * 39)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            ag   = _agreement(all_results[a], all_results[b])
            sa, sb = _short_label(a), _short_label(b)
            print(f"  {sa:<18}  ↔  {sb:<18}  {ag:.1%}")
    print()


# per-provider disagreement with gold

def print_disagreement(all_results: dict[str, list[dict]], test_rows: list[dict]) -> None:
    gold_map = {row["lb_id"]: row["messages"][-1]["content"].strip() for row in test_rows}
    pred_maps = {
        name: {r["lb_id"]: (r.get("pred") or r.get("answer") or "?") for r in res}
        for name, res in all_results.items()
    }

    disagreed_rows = [
        row for row in test_rows
        if len({pm.get(row["lb_id"], "?") for pm in pred_maps.values()}) > 1
        or any(
            pm.get(row["lb_id"], "?") != gold_map[row["lb_id"]]
            for pm in pred_maps.values()
        )
    ]

    if not disagreed_rows:
        print("── All methods agreed on every row.\n")
        return

    print(f"── Rows with disagreement ({len(disagreed_rows)}/{len(test_rows)}) " + "─" * 20)
    col_w = 14
    names = list(all_results.keys())
    shorts = [_short_label(n) for n in names]
    header = f"  {'lb_id':<14} {'Gold':<14}" + "".join(f" {s:<{col_w}}" for s in shorts)
    print(header)
    for row in disagreed_rows:
        lid  = row["lb_id"]
        gold = gold_map[lid]
        line = f"  {lid:<14} {gold:<14}"
        for name in names:
            pred   = pred_maps[name].get(lid, "?")
            marker = "✓" if pred == gold else "✗"
            line  += f" {marker}{pred:<{col_w - 1}}"
        print(line)
    print()


# main

def run_task(
    task:        str,
    split:       str,
    input_dir:   Path,
    output_dir:  Path,
    providers:   list[str],
    limit:       int | None,
    no_baseline: bool,
    thinking:    bool,
) -> None:
    from longevity_benchmark.io import load_jsonl

    prefix     = TASK_FILE_PREFIX[task]
    test_path  = input_dir / f"{prefix}_{split}.jsonl"
    train_path = input_dir / f"{prefix}_train.jsonl"

    if not test_path.exists():
        print(f"[{task}] File not found: {test_path} — skipping\n")
        return

    test_rows = load_jsonl(test_path)
    if limit:
        test_rows = test_rows[:limit]

    print(f"\n{'=' * 60}")
    print(f"Task: {task}  |  split={split}  |  n={len(test_rows)}")
    print(f"{'=' * 60}")

    all_results: dict[str, list[dict]] = {}

    # 1. Weighted-similarity baseline
    if not no_baseline and task in BASELINE_TASKS and train_path.exists():
        print("Running weighted-similarity baseline …")
        train_rows = load_jsonl(train_path)
        t0 = time.time()
        baseline = run_similarity_baseline(test_rows, train_rows)
        print(f"  done in {time.time() - t0:.1f}s  acc={_accuracy(baseline):.1%}")
        all_results["similarity\n(weighted-feat)"] = baseline

    # 2. LLM providers (run sequentially to respect shared endpoint quota)
    for provider in providers:
        print(f"Running {provider} …")
        label, results = run_provider(test_rows, provider, enable_thinking=thinking)
        if results:
            all_results[label] = results

    if not all_results:
        print("  No results — check API keys.\n")
        return

    # 3. Score reasoning traces (only methods that ran with thinking will have non-empty traces)
    if thinking:
        print("Scoring reasoning traces (loading MGI/MP databases…) ", end="", flush=True)
        t0 = time.time()
        annotate_compare_results(all_results)
        print(f"done in {time.time() - t0:.1f}s")

    # 4. Display
    print_row_table(all_results, test_rows)
    print_summary_table(all_results, len(test_rows))
    if thinking:
        print_trace_quality_table(all_results)
    print_disagreement(all_results, test_rows)

    # 4. Save
    output_dir.mkdir(parents=True, exist_ok=True)
    provider_tag = "_".join(_short_label(n) for n in all_results)
    out_path = output_dir / f"compare_{task}_{split}_{provider_tag}.json"

    save_data = {
        "task": task, "split": split, "n": len(test_rows),
        "methods": {
            _short_label(name): {
                "accuracy": round(_accuracy(res), 4),
                "n": len(res),
                "rows": res,
            }
            for name, res in all_results.items()
        },
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)
    print(f"Saved → {out_path}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare LongevityLLM, Gemini, Claude, and a weighted-similarity baseline."
    )
    parser.add_argument(
        "--task", choices=[*list(TASK_FILE_PREFIX), "all"], default="effect",
        help="Task to evaluate. 'all' runs every task."
    )
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument(
        "--providers", nargs="+", choices=ALL_PROVIDERS, default=ALL_PROVIDERS,
        help="Which LLM providers to include."
    )
    parser.add_argument("--limit",       type=int,  default=None,  help="Cap rows per task (smoke test).")
    parser.add_argument("--no-baseline", action="store_true",      help="Skip the weighted-similarity baseline.")
    parser.add_argument("--think",       action="store_true",      help="Enable chain-of-thought (longevity & claude only).")
    parser.add_argument("--input-dir",   default="output")
    parser.add_argument("--output-dir",  default="output/compare")
    args = parser.parse_args()

    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    tasks      = list(TASK_FILE_PREFIX) if args.task == "all" else [args.task]

    for task in tasks:
        run_task(
            task=task,
            split=args.split,
            input_dir=input_dir,
            output_dir=output_dir,
            providers=args.providers,
            limit=args.limit,
            no_baseline=args.no_baseline,
            thinking=args.think,
        )


if __name__ == "__main__":
    main()
