"""Train/test split and class balancing helpers."""

from __future__ import annotations

import random

import pandas as pd

from .config import BuildConfig


def choose_genes_for_target_rows(
    gene_counts: dict[str, int],
    target_rows: int,
) -> set[str]:
    """Choose genes whose row count is closest to a target without splitting genes."""
    states: dict[int, set[str]] = {0: set()}
    for gene, count in gene_counts.items():
        for current, subset in list(states.items()):
            new_total = current + count
            if new_total not in states:
                states[new_total] = subset | {gene}

    best_total = min(states, key=lambda total: (abs(total - target_rows), total < target_rows))
    return states[best_total]


def assign_gene_splits(
    df: pd.DataFrame,
    config: BuildConfig,
    rng: random.Random,
) -> pd.DataFrame:
    rare_labels = [
        label for label in ("Increased", "Inconclusive") if label in set(df["label"].unique())
    ]

    rare_train_genes: set[str] = set()
    rare_genes_all: set[str] = set()
    for label in rare_labels:
        label_df = df[df["label"] == label]
        gene_counts = label_df.groupby("marker_symbol").size().to_dict()
        candidate_counts = {
            gene: count for gene, count in gene_counts.items() if gene not in rare_genes_all
        }
        target_rows = max(1, int(round(len(label_df) * config.rare_label_train_fraction)))
        train_subset = choose_genes_for_target_rows(candidate_counts, target_rows)
        rare_train_genes |= train_subset
        rare_genes_all |= set(gene_counts)

    other_genes = sorted(set(df["marker_symbol"]) - rare_genes_all)
    rng.shuffle(other_genes)
    n_other_train = max(1, int(len(other_genes) * config.train_fraction))
    if len(other_genes) > 1:
        n_other_train = min(n_other_train, len(other_genes) - 1)

    train_genes = rare_train_genes | set(other_genes[:n_other_train])

    out = df.copy()
    out["split"] = out["marker_symbol"].apply(
        lambda gene: "train" if gene in train_genes else "test"
    )

    overlap = set(out.loc[out["split"] == "train", "marker_symbol"]) & set(
        out.loc[out["split"] == "test", "marker_symbol"]
    )
    assert not overlap, f"Gene leakage across splits: {sorted(overlap)[:5]}"
    return out


def balance_decreased_within_split(df: pd.DataFrame, config: BuildConfig) -> pd.DataFrame:
    kept = []
    for _, split_df in df.groupby("split"):
        inc = split_df[split_df["label"] == "Increased"]
        dec = split_df[split_df["label"] == "Decreased"]
        target_dec = max(
            len(inc) * config.negative_to_positive_ratio,
            config.min_decreased_per_split,
        )
        target_dec = min(len(dec), target_dec)

        dec_sample = dec.sample(n=target_dec, random_state=config.seed)
        kept.append(pd.concat([inc, dec_sample], ignore_index=True))

    out = pd.concat(kept, ignore_index=True)
    out = out.sample(frac=1.0, random_state=config.seed).reset_index(drop=True)

    print("\nDirectional rows after balancing:")
    for split, split_df in out.groupby("split"):
        print(f"{split}: {split_df['label'].value_counts().to_dict()}")

    return out


def balance_ternary_within_split(df: pd.DataFrame, config: BuildConfig) -> pd.DataFrame:
    """Keep all Increased and Inconclusive rows; downsample Decreased to a comparable level.

    Equal-class balancing would cap each split at the size of the rarest class (~11),
    which leaves the task below the 50-prompt minimum. balanced_accuracy already handles
    class imbalance, so we keep every rare-class row and trim only the majority class.
    """
    labels = ["Increased", "Decreased", "Inconclusive"]
    kept = []
    for split, split_df in df.groupby("split"):
        counts = split_df["label"].value_counts()
        missing = [label for label in labels if counts.get(label, 0) == 0]
        if missing:
            raise ValueError(f"Cannot build ternary split {split}; missing labels: {missing}")

        rare_max = max(counts["Increased"], counts["Inconclusive"])
        target_dec = min(counts["Decreased"], rare_max * config.ternary_majority_multiplier)

        split_parts = []
        for label in labels:
            label_df = split_df[split_df["label"] == label]
            if label == "Decreased":
                split_parts.append(label_df.sample(n=target_dec, random_state=config.seed))
            else:
                split_parts.append(label_df)
        kept.append(pd.concat(split_parts, ignore_index=True))

    out = pd.concat(kept, ignore_index=True)
    out = out.sample(frac=1.0, random_state=config.seed).reset_index(drop=True)

    print("\nTernary rows after balancing:")
    for split, split_df in out.groupby("split"):
        print(f"{split}: {split_df['label'].value_counts().to_dict()}")

    return out
