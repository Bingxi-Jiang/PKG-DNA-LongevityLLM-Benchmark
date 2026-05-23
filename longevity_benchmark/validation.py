"""In-memory validation checks for generated benchmark records."""

from __future__ import annotations

from collections import Counter

from .config import LEAKY_CONTEXT_RE


def phenotype_bullets(user: str) -> list[str]:
    bullets = []
    in_phenotype_block = False
    for line in user.splitlines():
        stripped = line.strip()
        if stripped == "Other observed phenotypes:":
            in_phenotype_block = True
            continue
        if not in_phenotype_block:
            continue
        if not stripped:
            in_phenotype_block = False
            continue
        if stripped.startswith("- "):
            bullets.append(stripped)
            continue
        if not line.startswith(" "):
            in_phenotype_block = False
    return bullets


def count_leaky_context(records: list[dict]) -> int:
    hits = 0
    for record in records:
        user = record["messages"][1]["content"]
        bullets = phenotype_bullets(user)
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


def print_task_counts(name: str, records: list[dict]) -> None:
    for split in ["train", "test"]:
        split_records = [record for record in records if record["split"] == split]
        labels = Counter(
            record["messages"][-1]["content"] for record in records if record["split"] == split
        )
        if len(labels) > 12:
            print(f"{name}_{split}: n={len(split_records)}, unique_answers={len(labels)}")
        else:
            print(f"{name}_{split}: labels={dict(labels)}")


def print_sanity_checks(
    effect_records: list[dict],
    pairwise_records: list[dict],
    extra_records: dict[str, list[dict]] | None = None,
) -> None:
    extra_records = extra_records or {}
    all_records = effect_records + pairwise_records
    for records in extra_records.values():
        all_records.extend(records)
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

    for name, records in extra_records.items():
        print_task_counts(name, records)
