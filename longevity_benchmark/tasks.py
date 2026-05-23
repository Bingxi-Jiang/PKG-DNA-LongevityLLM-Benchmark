"""Task builders that convert labeled rows into ChatML benchmark records."""

from __future__ import annotations

import json
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
                "pool": "mgi_directional_effect",
                "display_name": "MGI Mouse Longevity / Directional Effect",
                "display_group": "MGI Mouse Longevity",
                "domain": "genetics",
                "format": "binary",
                "metric": "balanced_accuracy",
                "units": None,
                "task": config.task_description,
                "messages": [
                    {"role": "system", "content": config.system_prompt},
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": row["label"]},
                ],
                "has_reasoning": False,
                "metadata": json.dumps(metadata_for_row(row)),
                "split": row["split"],
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
                "pool": "mgi_mcq_effect",
                "display_name": "MGI Mouse Longevity / Multiple Choice Effect",
                "display_group": "MGI Mouse Longevity",
                "domain": "genetics",
                "format": "mcq",
                "metric": "accuracy",
                "units": None,
                "task": config.task_description,
                "messages": [
                    {"role": "system", "content": config.system_prompt},
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": MCQ_ANSWER_BY_LABEL[row["label"]]},
                ],
                "has_reasoning": False,
                "metadata": json.dumps({
                    **metadata_for_row(row),
                    "options": {letter: text for letter, text in MCQ_OPTIONS},
                    "gold_label": row["label"],
                }),
                "split": row["split"],
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
                "pool": "mgi_ternary_inconclusive",
                "display_name": "MGI Mouse Longevity / Ternary Inconclusive",
                "display_group": "MGI Mouse Longevity",
                "domain": "genetics",
                "format": "ternary",
                "metric": "balanced_accuracy",
                "units": None,
                "task": config.task_description,
                "messages": [
                    {"role": "system", "content": config.system_prompt},
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": row["label"]},
                ],
                "has_reasoning": False,
                "metadata": json.dumps(metadata_for_row(row)),
                "split": row["split"],
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
                "pool": "mgi_directional_mp_term_set",
                "display_name": "MGI Mouse Longevity / Directional MP Term Set",
                "display_group": "MGI Mouse Longevity",
                "domain": "genetics",
                "format": "set_generation",
                "metric": "set_f1",
                "units": None,
                "task": config.task_description,
                "messages": [
                    {"role": "system", "content": config.system_prompt},
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": gold_terms},
                ],
                "has_reasoning": False,
                "metadata": json.dumps({
                    **metadata_for_row(row),
                    "candidate_mp_terms": [
                        {"mp_id": mp_id, "mp_name": mp_name}
                        for mp_id, mp_name in STRICT_DIRECTIONAL_MP_OPTIONS
                    ],
                }),
                "split": row["split"],
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
                    "pool": "mgi_pairwise",
                    "display_name": "MGI Mouse Longevity / Pairwise",
                    "display_group": "MGI Mouse Longevity",
                    "domain": "genetics",
                    "format": "pairwise",
                    "metric": "accuracy",
                    "units": None,
                    "task": config.task_description,
                    "messages": [
                        {"role": "system", "content": config.system_prompt},
                        {"role": "user", "content": user},
                        {"role": "assistant", "content": correct},
                    ],
                    "has_reasoning": False,
                    "metadata": json.dumps({
                        "allele_A": str(a["allele_symbol"]),
                        "gene_A": str(a["marker_symbol"]),
                        "label_A": str(a["label"]),
                        "allele_B": str(b["allele_symbol"]),
                        "gene_B": str(b["marker_symbol"]),
                        "label_B": str(b["label"]),
                        "component_splits": [str(a["split"]), str(b["split"])],
                        "pair_type": "Increased_vs_Decreased",
                    }),
                    "split": split,
                }
            )

    rng.shuffle(records)
    return records


def make_mpd_lifespan_regression_records(df: pd.DataFrame, config: BuildConfig) -> list[dict]:
    records = []
    system = (
        "You are an expert in mouse genetics and aging biology. "
        "Answer concisely with the requested numeric value only."
    )
    for row in df.to_dict("records"):
        user = (
            "A colony observation lifespan study followed an inbred mouse strain.\n\n"
            f"Strain: {row['strain']}\n"
            f"Sex: {row['sex_label']}\n"
            "Panel: inbred\n"
            "Outcome: median lifespan\n\n"
            "Predict the median lifespan in days for this strain-sex group.\n\n"
            "Answer with a single number of days."
        )
        gold = str(int(row["median_lifespan_days"]))
        records.append(
            {
                "lb_id": "LB-MPD-006",
                "pool": "mpd_yuan2_lifespan_regression",
                "display_name": "MPD Yuan2 Mouse Lifespan / Regression",
                "display_group": "MPD Yuan2 Mouse Lifespan",
                "domain": "genetics",
                "format": "regression",
                "metric": "mae",
                "units": "days",
                "task": "Given an inbred mouse strain and sex, predict median lifespan in days.",
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": gold},
                ],
                "has_reasoning": False,
                "metadata": json.dumps({
                    "source": "MPD Yuan2",
                    "source_url": "https://phenome.jax.org/projects/Yuan2",
                    "measure_id": 23201,
                    "measure": "median lifespan",
                    "strain": str(row["strain"]),
                    "strainid": str(row["strainid"]),
                    "stocknum": str(row.get("stocknum", "")),
                    "sex": str(row["sex_label"]),
                    "gold_days": int(row["median_lifespan_days"]),
                }),
                "split": row["split"],
            }
        )
    return records


def make_mpd_sex_effect_records(df: pd.DataFrame, config: BuildConfig) -> list[dict]:
    records = []
    for row in df.to_dict("records"):
        user = (
            "A colony observation lifespan study followed male and female mice "
            "from the same inbred strain.\n\n"
            f"Strain: {row['strain']}\n"
            "Panel: inbred\n"
            "Outcome: lifespan in days\n\n"
            "Which sex had a statistically detectable longer lifespan, or was "
            "no significant sex difference detected?\n\n"
            "Answer with exactly one phrase: Female longer / Male longer / "
            "No significant difference"
        )
        records.append(
            {
                "lb_id": "LB-MPD-007",
                "pool": "mpd_yuan2_lifespan_sex_effect",
                "display_name": "MPD Yuan2 Mouse Lifespan / Sex Effect",
                "display_group": "MPD Yuan2 Mouse Lifespan",
                "domain": "genetics",
                "format": "ternary",
                "parser": "sex_effect",
                "metric": "accuracy",
                "units": None,
                "task": (
                    "Given an inbred mouse strain, identify whether females, males, "
                    "or neither sex had significantly longer lifespan."
                ),
                "messages": [
                    {"role": "system", "content": config.system_prompt},
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": row["label"]},
                ],
                "has_reasoning": False,
                "metadata": json.dumps({
                    "source": "MPD Yuan2",
                    "source_url": "https://phenome.jax.org/projects/Yuan2",
                    "measure_id": 23401,
                    "measure": "lifespan",
                    "strain": str(row["strain"]),
                    "strainid": str(row["strainid"]),
                    "stocknum": str(row["stocknum"]),
                    "female_n": int(row["female_n"]),
                    "male_n": int(row["male_n"]),
                    "female_median_days": float(row["female_median_days"]),
                    "male_median_days": float(row["male_median_days"]),
                    "p_value": float(row["p_value"]),
                    "alpha": 0.05,
                    "label_source": row["label_source"],
                }),
                "split": row["split"],
            }
        )
    return records
