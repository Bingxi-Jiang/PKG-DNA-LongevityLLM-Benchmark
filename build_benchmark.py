"""
Mouse Strain Longevity Benchmark Builder.

The benchmark uses curated MGI Mammalian Phenotype annotations to ask whether
an allele/background genotype has a directional effect on mouse lifespan.

Outputs:
    output/mgi_effect_train.jsonl
    output/mgi_effect_test.jsonl
    output/mgi_pairwise_train.jsonl
    output/mgi_pairwise_test.jsonl
"""

from __future__ import annotations

import json
import os
import random
import re
from collections import Counter, defaultdict
from itertools import product

import pandas as pd


SEED = 42
NEGATIVE_TO_POSITIVE_RATIO = 8
MIN_DECREASED_PER_SPLIT = 50
MAX_PAIRWISE_TRAIN = 500
MAX_PAIRWISE_TEST = 300

random.seed(SEED)
os.makedirs("output", exist_ok=True)


PHENO_PATH = "data/mgi/MGI_PhenoGenoMP.rpt"
ALLELE_PATH = "data/mgi/MGI_PhenotypicAllele.rpt"
MP_PATH = "data/mgi/VOC_MammalianPhenotype.rpt"

# Strictly directional terms only. Broader parents such as MP:0010768
# (mortality/aging) and MP:0010769 (abnormal survival) are intentionally not
# used as labels because they do not define the direction of lifespan change.
INCREASED_TERMS = {
    "MP:0001661",  # extended life span
    "MP:0011614",  # slow aging
}
DECREASED_TERMS = {
    "MP:0002083",  # premature death
    "MP:0003786",  # premature aging
}
DIRECTIONAL_TERMS = INCREASED_TERMS | DECREASED_TERMS
LABEL_BY_TERM = {term: "Increased" for term in INCREASED_TERMS}
LABEL_BY_TERM.update({term: "Decreased" for term in DECREASED_TERMS})

# Terms with these words are not shown as context phenotypes. They would leak
# the answer or turn the task into keyword matching rather than biology.
LEAKY_CONTEXT_RE = re.compile(
    r"life\s*span|lifespan|survival|death|lethal|lethality|mortality|"
    r"aging|ageing|senescence|viability|morbidity",
    flags=re.IGNORECASE,
)

SYSTEM = (
    "You are an expert in mouse genetics and aging biology. "
    "Answer concisely with exactly one of the provided options."
)

TASK_DESCRIPTION = (
    "Given an MGI mouse allele, its genetic background, and non-lifespan "
    "phenotype annotations, identify the direction of its effect on lifespan."
)


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    print("Loading MGI data...")
    pheno = pd.read_csv(
        PHENO_PATH,
        sep="\t",
        header=None,
        low_memory=False,
        names=[
            "allele_combo",
            "allele_symbol",
            "genetic_bg",
            "mp_term",
            "pubmed_id",
            "marker_id",
            "allele_id",
        ],
    )

    allele = pd.read_csv(
        ALLELE_PATH,
        sep="\t",
        header=None,
        comment="#",
        low_memory=False,
        names=[
            "allele_id",
            "allele_symbol_meta",
            "allele_name",
            "allele_type",
            "inheritance",
            "synonyms",
            "marker_id_meta",
            "marker_symbol",
            "refseq_id",
            "ensembl_id",
            "mp_ids",
            "mp_names",
            "marker_name",
        ],
    )

    mp_ont = pd.read_csv(
        MP_PATH,
        sep="\t",
        header=None,
        names=["mp_id", "mp_name", "mp_def"],
    )
    return pheno, allele, dict(zip(mp_ont["mp_id"], mp_ont["mp_name"]))


def clean_background(value: object) -> str:
    bg = "" if pd.isna(value) else str(value)
    return bg.replace("involves: ", "").replace("either: ", "").strip()


def is_leaky_context(mp_id: str, mp_name_by_id: dict[str, str]) -> bool:
    name = mp_name_by_id.get(mp_id, "")
    return bool(LEAKY_CONTEXT_RE.search(name))


def build_context_index(
    pheno: pd.DataFrame,
    mp_name_by_id: dict[str, str],
) -> dict[tuple[str, str], list[str]]:
    geno_to_mp = defaultdict(set)
    for row in pheno.itertuples(index=False):
        geno_to_mp[(row.allele_symbol, row.genetic_bg)].add(row.mp_term)

    context = {}
    for key, mp_ids in geno_to_mp.items():
        names = []
        for mp_id in sorted(mp_ids):
            if mp_id in DIRECTIONAL_TERMS:
                continue
            if is_leaky_context(mp_id, mp_name_by_id):
                continue
            name = mp_name_by_id.get(mp_id)
            if name:
                names.append(name)
        context[key] = names
    return context


def build_directional_rows(
    pheno: pd.DataFrame,
    allele: pd.DataFrame,
    mp_name_by_id: dict[str, str],
) -> pd.DataFrame:
    labeled = pheno[pheno["mp_term"].isin(DIRECTIONAL_TERMS)].copy()

    # Allele-symbol pipes denote multi-allele genotypes in MGI's flattened
    # report. Keep single allele/background examples for a cleaner mutation task.
    labeled = labeled[~labeled["allele_symbol"].str.contains(r"\|", na=False)]
    labeled["label"] = labeled["mp_term"].map(LABEL_BY_TERM)

    group_cols = ["allele_symbol", "genetic_bg"]
    grouped = (
        labeled.groupby(group_cols, dropna=False)
        .agg(
            labels=("label", lambda x: sorted(set(x))),
            gold_mp_terms=("mp_term", lambda x: sorted(set(x))),
            mgi_allele_ids=(
                "allele_id",
                lambda x: sorted({str(v) for v in x if pd.notna(v) and str(v)}),
            ),
            gold_pubmed_ids=(
                "pubmed_id",
                lambda x: sorted({str(v) for v in x if pd.notna(v) and str(v)}),
            ),
        )
        .reset_index()
    )
    grouped["n_labels"] = grouped["labels"].apply(len)

    conflicts = grouped[grouped["n_labels"] > 1]
    if not conflicts.empty:
        print(f"Dropping conflicting genotype labels: {len(conflicts)}")

    grouped = grouped[grouped["n_labels"] == 1].copy()
    grouped["label"] = grouped["labels"].apply(lambda values: values[0])
    grouped = grouped.drop(columns=["labels", "n_labels"])

    allele_meta = allele[
        [
            "allele_symbol_meta",
            "allele_name",
            "allele_type",
            "inheritance",
            "marker_symbol",
            "marker_name",
        ]
    ].copy()
    allele_meta = allele_meta.drop_duplicates(subset=["allele_symbol_meta"])

    df = grouped.merge(
        allele_meta,
        left_on="allele_symbol",
        right_on="allele_symbol_meta",
        how="left",
    )
    df = df.dropna(subset=["marker_symbol"]).copy()
    df = df.drop_duplicates(subset=["allele_symbol", "genetic_bg", "label"])

    df["gold_mp_names"] = df["gold_mp_terms"].apply(
        lambda terms: [mp_name_by_id.get(term, term) for term in terms]
    )

    print(f"Directional rows before balancing: {len(df)}")
    print(df["label"].value_counts().to_string())
    return df.reset_index(drop=True)


def assign_gene_splits(df: pd.DataFrame) -> pd.DataFrame:
    inc_genes = set(df.loc[df["label"] == "Increased", "marker_symbol"])
    other_genes = set(df["marker_symbol"]) - inc_genes

    inc_genes = list(inc_genes)
    other_genes = list(other_genes)
    random.shuffle(inc_genes)
    random.shuffle(other_genes)

    n_inc_train = max(1, int(len(inc_genes) * 0.7))
    if len(inc_genes) > 1:
        n_inc_train = min(n_inc_train, len(inc_genes) - 1)

    n_other_train = max(1, int(len(other_genes) * 0.7))
    if len(other_genes) > 1:
        n_other_train = min(n_other_train, len(other_genes) - 1)

    train_genes = set(inc_genes[:n_inc_train]) | set(other_genes[:n_other_train])
    test_genes = set(inc_genes[n_inc_train:]) | set(other_genes[n_other_train:])

    out = df.copy()
    out["split"] = out["marker_symbol"].apply(
        lambda gene: "train" if gene in train_genes else "test"
    )

    overlap = set(out.loc[out["split"] == "train", "marker_symbol"]) & set(
        out.loc[out["split"] == "test", "marker_symbol"]
    )
    assert not overlap, f"Gene leakage across splits: {sorted(overlap)[:5]}"

    return out


def balance_decreased_within_split(df: pd.DataFrame) -> pd.DataFrame:
    kept = []
    for split, split_df in df.groupby("split"):
        inc = split_df[split_df["label"] == "Increased"]
        dec = split_df[split_df["label"] == "Decreased"]
        target_dec = max(len(inc) * NEGATIVE_TO_POSITIVE_RATIO, MIN_DECREASED_PER_SPLIT)
        target_dec = min(len(dec), target_dec)

        dec_sample = dec.sample(n=target_dec, random_state=SEED)
        kept.append(pd.concat([inc, dec_sample], ignore_index=True))

    out = pd.concat(kept, ignore_index=True)
    out = out.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    print("\nDirectional rows after balancing:")
    for split, split_df in out.groupby("split"):
        print(f"{split}: {split_df['label'].value_counts().to_dict()}")

    return out


def context_for_row(row: dict, context_index: dict[tuple[str, str], list[str]]) -> list[str]:
    key = (row["allele_symbol"], row["genetic_bg"])
    return context_index.get(key, [])[:8]


def describe_allele(
    row: dict | pd.Series,
    context_index: dict[tuple[str, str], list[str]],
) -> str:
    row = row if isinstance(row, dict) else row.to_dict()
    allele_type = row["allele_type"] if pd.notna(row.get("allele_type")) else "unspecified"
    inheritance = row["inheritance"] if pd.notna(row.get("inheritance")) else "unspecified"
    gene_name = row["marker_name"] if pd.notna(row.get("marker_name")) else row["marker_symbol"]
    bg = clean_background(row.get("genetic_bg", ""))
    phenotypes = context_for_row(row, context_index)
    pheno_block = ""
    if phenotypes:
        pheno_block = "\nOther observed phenotypes:\n" + "\n".join(
            f"  - {name}" for name in phenotypes
        )

    return (
        f"Gene: {row['marker_symbol']} ({gene_name})\n"
        f"Allele: {row['allele_symbol']}\n"
        f"Allele type: {allele_type}\n"
        f"Mode of inheritance: {inheritance}\n"
        f"Genetic background: {bg}"
        f"{pheno_block}"
    )


def metadata_for_row(row: dict | pd.Series) -> dict:
    row = row if isinstance(row, dict) else row.to_dict()
    return {
        "mgi_allele_ids": list(row["mgi_allele_ids"]),
        "allele": str(row["allele_symbol"]),
        "gene": str(row["marker_symbol"]),
        "allele_type": str(row["allele_type"]),
        "genetic_background": str(row["genetic_bg"]),
        "label_source": "mgi_strict_directional_mp_term",
        "gold_mp_terms": list(row["gold_mp_terms"]),
        "gold_mp_names": list(row["gold_mp_names"]),
        "gold_pubmed_ids": list(row["gold_pubmed_ids"]),
    }


def make_effect_records(
    df: pd.DataFrame,
    context_index: dict[tuple[str, str], list[str]],
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
                "task": TASK_DESCRIPTION,
                "split": row["split"],
                "metadata": metadata_for_row(row),
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": row["label"]},
                ],
            }
        )
    return records


def sampled_pairs(inc_rows: list[dict], dec_rows: list[dict], max_pairs: int) -> list[tuple[dict, dict]]:
    all_pairs = [
        (inc, dec)
        for inc, dec in product(inc_rows, dec_rows)
        if inc["marker_symbol"] != dec["marker_symbol"]
    ]
    if not all_pairs:
        all_pairs = list(product(inc_rows, dec_rows))
    random.shuffle(all_pairs)
    return all_pairs[: min(len(all_pairs), max_pairs)]


def make_pairwise_records(
    df: pd.DataFrame,
    context_index: dict[tuple[str, str], list[str]],
) -> list[dict]:
    records = []
    for split, split_df in df.groupby("split"):
        inc_rows = split_df[split_df["label"] == "Increased"].to_dict("records")
        dec_rows = split_df[split_df["label"] == "Decreased"].to_dict("records")
        max_pairs = MAX_PAIRWISE_TRAIN if split == "train" else MAX_PAIRWISE_TEST

        for idx, (inc, dec) in enumerate(sampled_pairs(inc_rows, dec_rows, max_pairs)):
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
                    "task": TASK_DESCRIPTION,
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
                        {"role": "system", "content": SYSTEM},
                        {"role": "user", "content": user},
                        {"role": "assistant", "content": correct},
                    ],
                }
            )

    random.shuffle(records)
    return records


def save_records(records: list[dict], name: str) -> None:
    for split in ["train", "test"]:
        split_records = [record for record in records if record["split"] == split]
        path = f"output/{name}_{split}.jsonl"
        with open(path, "w", encoding="utf-8") as fh:
            for record in split_records:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")

        labels = Counter(record["messages"][-1]["content"] for record in split_records)
        print(f"  {path} - {len(split_records)} rows, labels: {dict(labels)}")


def main() -> None:
    pheno, allele, mp_name_by_id = load_inputs()
    context_index = build_context_index(pheno, mp_name_by_id)
    directional = build_directional_rows(pheno, allele, mp_name_by_id)
    directional = assign_gene_splits(directional)
    directional = balance_decreased_within_split(directional)

    effect_records = make_effect_records(directional, context_index)
    pairwise_records = make_pairwise_records(directional, context_index)

    print("\n=== Output ===")
    save_records(effect_records, "mgi_effect")
    save_records(pairwise_records, "mgi_pairwise")

    sample = effect_records[0]
    print("\n--- Sample prompt (effect) ---")
    print(sample["messages"][1]["content"])
    print(f"\nGold answer: {sample['messages'][-1]['content']}")


if __name__ == "__main__":
    main()
