"""Evaluate benchmark JSONL files against an OpenAI-compatible endpoint.

Thin CLI wrapper around ``longevity_benchmark.eval.cli``.
"""

from longevity_benchmark.eval.cli import main


if __name__ == "__main__":
    main()
