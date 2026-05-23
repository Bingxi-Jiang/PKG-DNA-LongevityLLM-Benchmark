"""Build MGI mouse longevity benchmark JSONL files.

Thin CLI wrapper around ``longevity_benchmark.build``.
"""

from longevity_benchmark.build import main


if __name__ == "__main__":
    main()
