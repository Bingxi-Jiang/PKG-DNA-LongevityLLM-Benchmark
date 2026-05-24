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
        --out output/eval/trace_scores_mgi_mutation_lifespan_direction_binary.jsonl

    # Visualize every saved evaluate.py result as one grouped bar graph
    python score_traces.py --plot-all \\
        --eval-dir output/eval \\
        --plot-out output/eval/llm_task_performance.png

Evaluation result files are keyed by the CLI task alias, such as ``effect`` or
``ternary``. Dataset JSONL files use the longer descriptive names registered in
``longevity_benchmark.eval.runner.TASK_FILE_PREFIX``. Plot mode reads the
``summary_*.json`` and matching ``results_*.jsonl`` files emitted by
``evaluate.py``.
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


TASK_ORDER = ["effect", "mcq", "ternary", "set", "pairwise", "regression", "sex_effect"]

TASK_LABELS = {
    "effect": "MGI mutation\nlifespan direction\nbinary",
    "mcq": "MGI mutation\nlifespan effect\nMCQ",
    "ternary": "MGI mutation\nlifespan effect\nternary",
    "set": "MGI mutation\nMP term set\nset gen",
    "pairwise": "MGI mutation\nlonger-lived\npairwise",
    "regression": "MPD strain-sex\nmedian lifespan\nregression",
    "sex_effect": "MPD strain\nsex difference\nternary",
}


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


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _as_float(value: object) -> float | None:
    try:
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return None


def _parse_set_value(value: str) -> set[str]:
    if not value:
        return set()
    return {item.strip() for item in value.split(",") if item.strip()}


def _set_f1(gold: set[str], pred: set[str]) -> float:
    if not gold and not pred:
        return 1.0
    if not gold or not pred:
        return 0.0
    overlap = len(gold & pred)
    precision = overlap / len(pred)
    recall = overlap / len(gold)
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def _accuracy(rows: list[dict]) -> float:
    if not rows:
        return float("nan")
    return sum(bool(row.get("correct")) for row in rows) / len(rows)


def _balanced_accuracy(rows: list[dict]) -> float:
    by_label: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        by_label[str(row.get("gold", ""))].append(str(row.get("pred", "")) == str(row.get("gold", "")))
    if not by_label:
        return float("nan")
    return mean(sum(vals) / len(vals) for vals in by_label.values() if vals)


def _mean_set_f1(rows: list[dict]) -> float:
    if not rows:
        return float("nan")
    return mean(
        _set_f1(_parse_set_value(str(row.get("gold", ""))), _parse_set_value(str(row.get("pred", ""))))
        for row in rows
    )


def _regression_performance(rows: list[dict]) -> float:
    """Normalize regression MAE to a 0-1 higher-is-better plotting score."""
    pairs = [
        (gold, pred)
        for gold, pred in (
            (_as_float(row.get("gold")), _as_float(row.get("pred")))
            for row in rows
        )
        if gold is not None and pred is not None
    ]
    if not pairs:
        return float("nan")
    golds = [gold for gold, _ in pairs]
    gold_range = max(golds) - min(golds)
    if gold_range <= 0:
        return float("nan")
    mae = mean(abs(gold - pred) for gold, pred in pairs)
    return max(0.0, min(1.0, 1.0 - mae / gold_range))


def _result_rows_performance(task: str, rows: list[dict]) -> float:
    if task in {"effect", "ternary", "sex_effect"}:
        return _balanced_accuracy(rows)
    if task == "set":
        return _mean_set_f1(rows)
    if task == "regression":
        return _regression_performance(rows)
    return _accuracy(rows)


def _task_label(task: str, split: str, include_split: bool) -> str:
    label = TASK_LABELS.get(task, task)
    return f"{label}\n{split}" if include_split else label


def _summary_result_path(summary_path: Path, payload: dict) -> Path:
    raw_path = payload.get("results_path")
    if raw_path:
        result_path = Path(str(raw_path))
        if result_path.exists():
            return result_path
        return summary_path.with_name(result_path.name)
    return summary_path.with_name(summary_path.name.replace("summary_", "results_", 1)).with_suffix(".jsonl")


def _method_label(payload: dict) -> str:
    label = str(payload.get("model") or payload.get("provider") or "unknown-model")
    if payload.get("thinking"):
        return f"{label} (think)"
    return label


def _summary_value(task: str, payload: dict, result_path: Path) -> float:
    if result_path.exists():
        return _result_rows_performance(task, _read_jsonl(result_path))

    metric = str(payload.get("metric", ""))
    if metric == "mae_days":
        return float("nan")
    try:
        return float(payload.get("score"))
    except (TypeError, ValueError):
        return float("nan")


def _load_eval_records(
    eval_dir: Path,
    split: str,
) -> list[dict]:
    records_by_key = {}
    for path in sorted(eval_dir.glob("summary_*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        task = str(payload.get("task", ""))
        task_split = str(payload.get("split", ""))
        if not task or not task_split:
            continue
        if split != "all" and task_split != split:
            continue

        method = _method_label(payload)
        result_path = _summary_result_path(path, payload)
        value = _summary_value(task, payload, result_path)
        if math.isnan(value):
            continue
        key = (task, task_split, method)
        old = records_by_key.get(key)
        if old is None or path.stat().st_mtime > old["_path"].stat().st_mtime:
            records_by_key[key] = {
                "task": task,
                "split": task_split,
                "method": method,
                "value": value,
                "_path": path,
                "_result_path": result_path,
            }
    return list(records_by_key.values())


def plot_all_results(eval_dir: Path, out_path: Path, split: str) -> None:
    rows = _load_eval_records(eval_dir, split)
    if not rows:
        raise SystemExit(f"No plottable summary_*.json files found in {eval_dir} for split={split}.")

    rows = sorted(rows, key=lambda row: row["_path"].stat().st_mtime)

    include_split = split == "all"
    task_keys = sorted(
        {(row["task"], row["split"]) for row in rows},
        key=lambda item: (
            TASK_ORDER.index(item[0]) if item[0] in TASK_ORDER else len(TASK_ORDER),
            item[1],
            item[0],
        ),
    )
    methods = sorted({row["method"] for row in rows})
    values = {
        (row["task"], row["split"], row["method"]): row["value"]
        for row in rows
    }

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise SystemExit("Install matplotlib to use --plot-all: pip install matplotlib") from exc

    width = max(14.0, 2.25 * len(task_keys))
    fig, ax = plt.subplots(figsize=(width, 6.8))
    colors = plt.get_cmap("tab20").colors
    group_width = 0.82
    bar_width = group_width / max(1, len(methods))
    x_positions = list(range(len(task_keys)))

    for method_idx, method in enumerate(methods):
        xs = []
        ys = []
        for task_idx, (task, task_split) in enumerate(task_keys):
            value = values.get((task, task_split, method))
            if value is None:
                continue
            offset = (method_idx - (len(methods) - 1) / 2) * bar_width
            xs.append(x_positions[task_idx] + offset)
            ys.append(value)
        bars = ax.bar(
            xs,
            ys,
            width=bar_width * 0.92,
            label=method,
            color=colors[method_idx % len(colors)],
        )
        ax.bar_label(bars, labels=[f"{value:.2f}" for value in ys], fontsize=8, padding=2)

    ax.set_title("Model Performance Across Mouse Longevity Benchmark Tasks", pad=16)
    ax.set_ylabel("Normalized performance score (higher is better)")
    ax.set_ylim(0, 1.08)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        [_task_label(task, task_split, include_split) for task, task_split in task_keys],
        fontsize=9,
        rotation=18,
        ha="right",
    )
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if len(methods) <= 3:
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 1.03),
            ncol=len(methods),
            frameon=False,
        )
        layout_rect = (0, 0.08, 1, 0.94)
    else:
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
        layout_rect = (0, 0.08, 0.82, 1)
    fig.text(
        0.01,
        0.015,
        "Classification bars use the task metric. Regression is not accuracy; it is normalized as 1 - MAE / gold lifespan range.",
        fontsize=9,
    )
    fig.tight_layout(rect=layout_rect)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument(
        "--results",
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
    ap.add_argument(
        "--plot-all",
        action="store_true",
        help="Write one bar graph comparing all saved evaluate.py outputs.",
    )
    ap.add_argument(
        "--eval-dir",
        default="output/eval",
        help="Directory containing summary_*.json and results_*.jsonl files emitted by evaluate.py.",
    )
    ap.add_argument(
        "--plot-out",
        default="output/eval/llm_task_performance.png",
        help="Path for the single model-performance graph written by --plot-all.",
    )
    ap.add_argument(
        "--split",
        choices=["train", "test", "all"],
        default="test",
        help="Which split to visualize when using --plot-all.",
    )
    args = ap.parse_args()

    if args.plot_all:
        plot_all_results(
            Path(args.eval_dir),
            Path(args.plot_out),
            args.split,
        )
        return

    if not args.results:
        ap.error("--results is required unless --plot-all is used.")

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
            "answer_consistency", "prompt_grounding",
            "knowledge_extension", "pathway_consistency"]
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
