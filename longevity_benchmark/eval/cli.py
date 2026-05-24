"""CLI entry point for evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

from .runner import PROVIDER_CONFIGS, evaluate_task, make_client


ALL_TASKS = ["effect", "mcq", "ternary", "set", "pairwise", "regression", "sex_effect"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run LongevityBench evaluation against a model endpoint."
    )
    parser.add_argument(
        "--task",
        choices=[*ALL_TASKS, "all"],
        default="all",
    )
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--think", action="store_true", help="Enable chain-of-thought thinking (longevity and claude providers only).")
    parser.add_argument(
        "--score-traces", action="store_true",
        help="After evaluation, score each reasoning trace against MGI/MP databases "
             "(gene/allele/MP-term validity, answer consistency, prompt grounding). "
             "Requires --think to produce traces; the per-row JSONL and summary JSON "
             "are extended with trace_score / trace_subscores.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Cap the number of rows per task (useful for quick smoke tests).")
    parser.add_argument(
        "--provider",
        choices=list(PROVIDER_CONFIGS),
        default="longevity",
        help="Model provider: longevity (HuggingFace endpoint), gemini, or claude. Default: longevity.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Model name override. Defaults per provider: "
            + ", ".join(f"{p}={cfg['default_model']}" for p, cfg in PROVIDER_CONFIGS.items())
            + "."
        ),
    )
    parser.add_argument(
        "--endpoint",
        default=None,
        help="Override the provider's default API endpoint URL.",
    )
    parser.add_argument("--input-dir", default="output")
    parser.add_argument("--eval-dir", default="output/eval")
    args = parser.parse_args()

    client, model = make_client(args.provider, args.model, args.endpoint)
    Path(args.eval_dir).mkdir(parents=True, exist_ok=True)

    tasks = ALL_TASKS if args.task == "all" else [args.task]

    for task in tasks:
        evaluate_task(
            task=task,
            split=args.split,
            input_dir=args.input_dir,
            eval_dir=args.eval_dir,
            client=client,
            model=model,
            provider=args.provider,
            enable_thinking=args.think,
            limit=args.limit,
            score_traces=args.score_traces,
        )


if __name__ == "__main__":
    main()
