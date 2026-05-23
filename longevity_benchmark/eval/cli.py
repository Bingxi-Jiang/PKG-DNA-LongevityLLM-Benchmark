"""CLI entry point for evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

from .runner import DEFAULT_ENDPOINT, evaluate_task, make_client


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["effect", "pairwise", "ternary", "both"], default="both")
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--think", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--input-dir", default="output")
    parser.add_argument("--eval-dir", default="output/eval")
    args = parser.parse_args()

    client = make_client(args.endpoint)
    Path(args.eval_dir).mkdir(parents=True, exist_ok=True)
    tasks = ["effect", "pairwise"] if args.task == "both" else [args.task]

    for task in tasks:
        evaluate_task(
            task=task,
            split=args.split,
            input_dir=args.input_dir,
            eval_dir=args.eval_dir,
            client=client,
            enable_thinking=args.think,
            limit=args.limit,
        )


if __name__ == "__main__":
    main()
