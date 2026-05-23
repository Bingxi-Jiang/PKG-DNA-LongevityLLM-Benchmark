"""Prompt rendering and metadata helpers."""

from __future__ import annotations

import pandas as pd


def clean_background(value: object) -> str:
    bg = "" if pd.isna(value) else str(value)
    return bg.replace("involves: ", "").replace("either: ", "").strip()


def context_for_row(
    row: dict,
    context_index: dict[tuple[str, str], list[str]],
    max_phenotypes: int = 8,
) -> list[str]:
    key = (row["allele_symbol"], row["genetic_bg"])
    return context_index.get(key, [])[:max_phenotypes]


def describe_allele(
    row: dict | pd.Series,
    context_index: dict[tuple[str, str], list[str]],
) -> str:
    row = row if isinstance(row, dict) else row.to_dict()
    allele_type = row["allele_type"] if pd.notna(row.get("allele_type")) else "unspecified"
    inheritance = row["inheritance"] if pd.notna(row.get("inheritance")) else "unspecified"
    gene_name = row["marker_name"] if pd.notna(row.get("marker_name")) else row["marker_symbol"]
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
        f"Genetic background: {clean_background(row.get('genetic_bg', ''))}"
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
