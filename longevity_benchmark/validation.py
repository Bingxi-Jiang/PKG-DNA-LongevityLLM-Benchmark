"""In-memory validation checks for generated benchmark records."""

from __future__ import annotations

from collections import Counter

from .config import LEAKY_CONTEXT_RE


def count_leaky_context(records: list[dict]) -> int:
    hits = 0
    for record in records:
        user = record["messages"][1]["content"]
        bullets = [line.strip() for line in user.split("\n") if line.strip().startswith("- ")]
        hits += sum(1 for bullet in bullets if LEAKY_CONTEXT_RE.search(bullet))
    return hits


def effect_gene_overlap(effect_records: list[dict]) -> int:
    genes_by_split = {"train": set(), "test": set()}
    for record in effect_records:
        genes_by_split[record["split"]].add(record["metadata"]["gene"])
    return len(genes_by_split["train"] & genes_by_split["test"])


def pairwise_split_issues(pairwise_records: list[dict]) -> dict[str, Counter]:
    counters = {
        "bad_split": Counter(),
        "same_gene": Counter(),
        "labels": Counter(),
    }
    for record in pairwise_records:
        split = record["split"]
        metadata = record["metadata"]
        counters["labels"][(split, record["messages"][-1]["content"])] += 1
        if metadata["component_splits"] != [split, split]:
            counters["bad_split"][split] += 1
        if metadata["gene_A"] == metadata["gene_B"]:
            counters["same_gene"][split] += 1
    return counters


def print_sanity_checks(effect_records: list[dict], pairwise_records: list[dict]) -> None:
    all_records = effect_records + pairwise_records
    print("\n=== Sanity checks ===")
    print(f"leaky_context_hits: {count_leaky_context(all_records)} / {len(all_records)} prompts")
    print(f"effect_gene_overlap: {effect_gene_overlap(effect_records)}")

    issues = pairwise_split_issues(pairwise_records)
    for split in ["train", "test"]:
        label_counts = {
            label: count
            for (label_split, label), count in issues["labels"].items()
            if label_split == split
        }
        print(
            f"pairwise_{split}: labels={label_counts}, "
            f"bad_split={issues['bad_split'][split]}, "
            f"same_gene_pairs={issues['same_gene'][split]}"
        )
