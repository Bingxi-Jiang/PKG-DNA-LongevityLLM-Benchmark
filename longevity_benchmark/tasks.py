"""Task builders that convert labeled rows into ChatML benchmark records."""

from __future__ import annotations

import random
from itertools import product

import pandas as pd

from .config import BuildConfig
from .prompts import describe_allele, metadata_for_row


MCQ_OPTIONS = [
    ("A", "Increased lifespan"),
    ("B", "Decreased lifespan"),
    ("C", "Lifespan-related annotation is non-directional or conflicting"),
    ("D", "No curated lifespan-related annotation"),
]

MCQ_ANSWER_BY_LABEL = {
    "Increased": "A",
    "Decreased": "B",
    "Inconclusive": "C",
}

STRICT_DIRECTIONAL_MP_OPTIONS = [
    ("MP:0001661", "extended life span"),
    ("MP:0011614", "slow aging"),
    ("MP:0002083", "premature death"),
    ("MP:0003786", "premature aging"),
]


def make_effect_records(
    df: pd.DataFrame,
    context_index: dict[tuple[str, str], list[str]],
    config: BuildConfig,
) -> list[dict]:
    records = []
    for row in df.to_dict("records"):
        desc = describe_allele(row, context_index)
        user = (
            "A researcher has engineered a mouse with the following genetic "
            "modification:\n\n"
            f"{desc}\n\n"
            "Based on curated mouse phenotype evidence, does this modification "
            "INCREASE or DECREASE lifespan compared to wild-type controls?\n\n"
            "Answer with exactly one phrase: Increased / Decreased"
        )
        records.append(
            {
                "lb_id": "LB-MGI-001",
                "display_name": "MGI Mouse Longevity / Directional Effect",
                "domain": "genetics",
                "format": "binary",
                "metric": "balanced_accuracy",
                "units": None,
                "task": config.task_description,
                "split": row["split"],
                "metadata": metadata_for_row(row),
                "messages": [
                    {"role": "system", "content": config.system_prompt},
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": row["label"]},
                ],
            }
        )
    return records


def make_mcq_records(
    df: pd.DataFrame,
    context_index: dict[tuple[str, str], list[str]],
    config: BuildConfig,
) -> list[dict]:
    records = []
    option_block = "\n".join(f"{letter}. {text}" for letter, text in MCQ_OPTIONS)
    for row in df.to_dict("records"):
        desc = describe_allele(row, context_index)
        user = (
            "A researcher has engineered a mouse with the following genetic "
            "modification:\n\n"
            f"{desc}\n\n"
            "Which option best describes the curated MGI lifespan effect for "
            "this modification?\n\n"
            f"{option_block}\n\n"
            "Answer with exactly one letter: A / B / C / D"
        )
        records.append(
            {
                "lb_id": "LB-MGI-003",
                "display_name": "MGI Mouse Longevity / Multiple Choice Effect",
                "domain": "genetics",
                "format": "mcq",
                "metric": "accuracy",
                "units": None,
                "task": config.task_description,
                "split": row["split"],
                "metadata": {
                    **metadata_for_row(row),
                    "options": {letter: text for letter, text in MCQ_OPTIONS},
                    "gold_label": row["label"],
                },
                "messages": [
                    {"role": "system", "content": config.system_prompt},
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": MCQ_ANSWER_BY_LABEL[row["label"]]},
                ],
            }
        )
    return records


def make_ternary_records(
    df: pd.DataFrame,
    context_index: dict[tuple[str, str], list[str]],
    config: BuildConfig,
) -> list[dict]:
    records = []
    for row in df.to_dict("records"):
        desc = describe_allele(row, context_index)
        user = (
            "A researcher has engineered a mouse with the following genetic "
            "modification:\n\n"
            f"{desc}\n\n"
            "Based on curated MGI lifespan or aging phenotype evidence, choose "
            "the best label:\n\n"
            "Increased = strict directional evidence for longer lifespan or slower aging\n"
            "Decreased = strict directional evidence for shorter lifespan or premature aging\n"
            "Inconclusive = lifespan-related evidence is broad, non-directional, or conflicting\n\n"
            "Answer with exactly one phrase: Increased / Decreased / Inconclusive"
        )
        records.append(
            {
                "lb_id": "LB-MGI-004",
                "display_name": "MGI Mouse Longevity / Ternary Inconclusive",
                "domain": "genetics",
                "format": "ternary",
                "metric": "balanced_accuracy",
                "units": None,
                "task": config.task_description,
                "split": row["split"],
                "metadata": metadata_for_row(row),
                "messages": [
                    {"role": "system", "content": config.system_prompt},
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": row["label"]},
                ],
            }
        )
    return records


def make_set_generation_records(
    df: pd.DataFrame,
    context_index: dict[tuple[str, str], list[str]],
    config: BuildConfig,
) -> list[dict]:
    records = []
    candidate_block = "\n".join(
        f"{mp_id} - {mp_name}" for mp_id, mp_name in STRICT_DIRECTIONAL_MP_OPTIONS
    )
    for row in df.to_dict("records"):
        desc = describe_allele(row, context_index)
        gold_terms = ",".join(sorted(row["gold_mp_terms"]))
        user = (
            "A researcher has engineered a mouse with the following genetic "
            "modification:\n\n"
            f"{desc}\n\n"
            "Candidate strict directional Mammalian Phenotype terms:\n"
            f"{candidate_block}\n\n"
            "Which candidate MP terms are curated for this genotype? Select all "
            "that apply.\n\n"
            "Answer with MP IDs only, comma-separated. If no candidate applies, "
            "answer None."
        )
        records.append(
            {
                "lb_id": "LB-MGI-005",
                "display_name": "MGI Mouse Longevity / Directional MP Term Set",
                "domain": "genetics",
                "format": "set_generation",
                "metric": "set_f1",
                "units": None,
                "task": config.task_description,
                "split": row["split"],
                "metadata": {
                    **metadata_for_row(row),
                    "candidate_mp_terms": [
                        {"mp_id": mp_id, "mp_name": mp_name}
                        for mp_id, mp_name in STRICT_DIRECTIONAL_MP_OPTIONS
                    ],
                },
                "messages": [
                    {"role": "system", "content": config.system_prompt},
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": gold_terms},
                ],
            }
        )
    return records


def sampled_pairs(
    inc_rows: list[dict],
    dec_rows: list[dict],
    max_pairs: int,
    rng: random.Random,
) -> list[tuple[dict, dict]]:
    all_pairs = [
        (inc, dec)
        for inc, dec in product(inc_rows, dec_rows)
        if inc["marker_symbol"] != dec["marker_symbol"]
    ]
    if not all_pairs:
        all_pairs = list(product(inc_rows, dec_rows))
    rng.shuffle(all_pairs)
    limit = min(len(all_pairs), max_pairs)
    if limit > 1 and limit % 2:
        limit -= 1
    return all_pairs[:limit]


def make_pairwise_records(
    df: pd.DataFrame,
    context_index: dict[tuple[str, str], list[str]],
    config: BuildConfig,
    rng: random.Random,
) -> list[dict]:
    records = []
    for split, split_df in df.groupby("split"):
        inc_rows = split_df[split_df["label"] == "Increased"].to_dict("records")
        dec_rows = split_df[split_df["label"] == "Decreased"].to_dict("records")
        max_pairs = config.max_pairwise_train if split == "train" else config.max_pairwise_test

        for idx, (inc, dec) in enumerate(sampled_pairs(inc_rows, dec_rows, max_pairs, rng)):
            if idx % 2 == 0:
                a, b = inc, dec
            else:
                a, b = dec, inc

            correct = "A" if a["label"] == "Increased" else "B"
            desc_a = describe_allele(a, context_index)
            desc_b = describe_allele(b, context_index)
            user = (
                "Two mouse strains carry different genetic modifications.\n\n"
                f"=== Mouse A ===\n{desc_a}\n\n"
                f"=== Mouse B ===\n{desc_b}\n\n"
                "Which mouse is expected to live LONGER?\n\n"
                "Answer with a single letter: A / B"
            )

            records.append(
                {
                    "lb_id": "LB-MGI-002",
                    "display_name": "MGI Mouse Longevity / Pairwise",
                    "domain": "genetics",
                    "format": "pairwise",
                    "metric": "accuracy",
                    "units": None,
                    "task": config.task_description,
                    "split": split,
                    "metadata": {
                        "allele_A": str(a["allele_symbol"]),
                        "gene_A": str(a["marker_symbol"]),
                        "label_A": str(a["label"]),
                        "allele_B": str(b["allele_symbol"]),
                        "gene_B": str(b["marker_symbol"]),
                        "label_B": str(b["label"]),
                        "component_splits": [str(a["split"]), str(b["split"])],
                        "pair_type": "Increased_vs_Decreased",
                    },
                    "messages": [
                        {"role": "system", "content": config.system_prompt},
                        {"role": "user", "content": user},
                        {"role": "assistant", "content": correct},
                    ],
                }
            )

    rng.shuffle(records)
    return records
