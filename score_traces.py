"""
score_traces.py — Programmatic reasoning-trace scorer.

For a results JSONL file produced by ``evaluate.py --think`` (the trace must
be populated in the "think" field), compute per-row reasoning-quality
sub-scores, aggregate, and the correlation between trace quality and final
answer correctness.

Usage:
    # Step 1: produce a results file with thinking on
    python evaluate.py --provider longevity --task effect --think \\
        --eval-dir output/eval

    # Step 2: score the traces
    python score_traces.py \\
        --results output/eval/results_effect_test_longevity_longevity-llm_think.jsonl

    # Step 3 (optional): export per-row scores for offline analysis
    python score_traces.py \\
        --results output/eval/results_effect_test_longevity_longevity-llm_think.jsonl \\
        --out output/eval/trace_scores_effect.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from statistics import mean, pstdev

from longevity_benchmark.reasoning_scorer import (
    load_allele_db,
    load_gene_db,
    load_mp_term_db,
    score_results_file,
)


def _fmt(v: float | None, digits: int = 3) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "  —  "
    return f"{v:.{digits}f}"


def _point_biserial(values: list[float | None], labels: list[bool]) -> float | None:
    """Point-biserial correlation between a continuous score and a binary label."""
    paired = [(v, l) for v, l in zip(values, labels) if v is not None]
    if len(paired) < 5:
        return None
    xs = [v for v, _ in paired]
    ys = [1.0 if l else 0.0 for _, l in paired]
    n = len(xs)
    mx, my = mean(xs), mean(ys)
    sx, sy = pstdev(xs), pstdev(ys)
    if sx == 0 or sy == 0:
        return None
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / n
    return cov / (sx * sy)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument(
        "--results", required=True,
        help="Path to a results_*.jsonl file emitted by evaluate.py with --think.",
    )
    ap.add_argument(
        "--out", default=None,
        help="Optional path to write per-row scored output as JSONL.",
    )
    ap.add_argument(
        "--show", type=int, default=10,
        help="Show this many representative low- and high-scoring rows.",
    )
    args = ap.parse_args()

    results_path = Path(args.results)
    if not results_path.exists():
        raise SystemExit(f"Not found: {results_path}")

    print(f"Loading databases…")
    gene_db = load_gene_db()
    allele_db = load_allele_db()
    mp_db = load_mp_term_db()
    print(f"  genes={len(gene_db):,}   alleles={len(allele_db):,}   mp_terms={len(mp_db):,}")

    print(f"Scoring traces from {results_path.name} …")
    scored = score_results_file(results_path, gene_db, allele_db, mp_db)

    n = len(scored)
    if n == 0:
        print("No rows.")
        return

    has_trace = sum(1 for s in scored if s.details.get("trace_len_chars", 0) > 0)
    print(f"  rows={n}   rows_with_trace={has_trace}/{n}")

    # ── per-dimension stats ──────────────────────────────────────────────
    print("\n── Sub-score coverage and mean ─────────────────────────────")
    print(f"  {'dimension':<22}  {'coverage':>9}  {'mean(present)':>15}")
    dims = ["gene_validity", "allele_validity", "mp_validity",
            "answer_consistency", "prompt_grounding"]
    coverage_stats: dict[str, list] = {d: [] for d in dims}
    for s in scored:
        for d in dims:
            v = s.subscores.get(d)
            if v is not None:
                coverage_stats[d].append(v)
    for d in dims:
        vals = coverage_stats[d]
        cov = len(vals) / n if n else 0.0
        avg = mean(vals) if vals else None
        print(f"  {d:<22}  {cov:>8.1%}  {_fmt(avg):>15}")

    # ── aggregate stats ──────────────────────────────────────────────────
    agg_vals = [s.aggregate for s in scored if s.aggregate is not None]
    print(f"\n  aggregate (n={len(agg_vals)}/{n})  mean={_fmt(mean(agg_vals)) if agg_vals else '—'}"
          f"   std={_fmt(pstdev(agg_vals)) if len(agg_vals) > 1 else '—'}")

    # ── correlation between trace quality and final-answer correctness ──
    print("\n── Trace quality ↔ final-answer correctness ────────────────")
    print(f"  {'dimension':<22}  {'point-biserial r':>18}")
    correct_labels = [s.correct for s in scored]
    for d in dims:
        vals = [s.subscores.get(d) for s in scored]
        r = _point_biserial(vals, correct_labels)
        print(f"  {d:<22}  {_fmt(r):>18}")
    agg_vec = [s.aggregate for s in scored]
    r_agg = _point_biserial(agg_vec, correct_labels)
    print(f"  {'aggregate':<22}  {_fmt(r_agg):>18}")

    # ── hallucination counts ────────────────────────────────────────────
    n_invalid_gene = sum(
        len(s.details["gene_validity"].get("invalid", []))
        for s in scored if s.details.get("gene_validity")
    )
    n_invalid_allele = sum(
        len(s.details["allele_validity"].get("invalid", []))
        for s in scored if s.details.get("allele_validity")
    )
    n_invalid_mp = sum(
        sum(1 for d in s.details["mp_validity"].get("details", []) if d.get("status") == "invalid_id")
        for s in scored if s.details.get("mp_validity")
    )
    print("\n── Hallucination tally (raw counts across all traces) ──────")
    print(f"  fabricated genes (not in MGI marker list) : {n_invalid_gene}")
    print(f"  fabricated alleles (not in MGI allele DB) : {n_invalid_allele}")
    print(f"  invalid MP IDs                             : {n_invalid_mp}")

    # ── representative rows ─────────────────────────────────────────────
    show_n = args.show
    if show_n and agg_vals:
        with_agg = [s for s in scored if s.aggregate is not None]
        with_agg.sort(key=lambda s: s.aggregate)
        print(f"\n── Lowest-scoring traces (n={min(show_n, len(with_agg))}) ──")
        for s in with_agg[:show_n]:
            print(
                f"  {s.lb_id}  agg={_fmt(s.aggregate)}  correct={'✓' if s.correct else '✗'}"
                f"  ans={s.answer!r:<14}  invalid_genes={s.details['gene_validity'].get('invalid', []) if s.details.get('gene_validity') else []}"
            )
        print(f"\n── Highest-scoring traces (n={min(show_n, len(with_agg))}) ──")
        for s in with_agg[-show_n:]:
            print(
                f"  {s.lb_id}  agg={_fmt(s.aggregate)}  correct={'✓' if s.correct else '✗'}"
                f"  ans={s.answer!r:<14}"
            )

    # ── write per-row output ────────────────────────────────────────────
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            for s in scored:
                f.write(json.dumps(asdict(s), ensure_ascii=False) + "\n")
        print(f"\nPer-row scores → {out_path}")


if __name__ == "__main__":
    main()
