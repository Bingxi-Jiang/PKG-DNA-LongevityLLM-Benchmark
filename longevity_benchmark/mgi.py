"""MGI-specific loading, label construction, and phenotype context handling."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import pandas as pd

from .config import BuildConfig, LEAKY_CONTEXT_RE, MGIPaths


@dataclass(frozen=True)
class MGITables:
    pheno: pd.DataFrame
    allele: pd.DataFrame
    mp_name_by_id: dict[str, str]


def load_mgi_tables(paths: MGIPaths) -> MGITables:
    print("Loading MGI data...")
    pheno = pd.read_csv(
        paths.pheno,
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
        paths.allele,
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

    mp_ontology = pd.read_csv(
        paths.mp_ontology,
        sep="\t",
        header=None,
        names=["mp_id", "mp_name", "mp_def"],
    )
    mp_name_by_id = dict(zip(mp_ontology["mp_id"], mp_ontology["mp_name"]))
    return MGITables(pheno=pheno, allele=allele, mp_name_by_id=mp_name_by_id)


def is_leaky_context(mp_id: str, mp_name_by_id: dict[str, str]) -> bool:
    return bool(LEAKY_CONTEXT_RE.search(mp_name_by_id.get(mp_id, "")))


def build_context_index(
    pheno: pd.DataFrame,
    mp_name_by_id: dict[str, str],
    config: BuildConfig,
) -> dict[tuple[str, str], list[str]]:
    geno_to_mp = defaultdict(set)
    for row in pheno.itertuples(index=False):
        geno_to_mp[(row.allele_symbol, row.genetic_bg)].add(row.mp_term)

    context = {}
    for key, mp_ids in geno_to_mp.items():
        names = []
        for mp_id in sorted(mp_ids):
            if mp_id in config.directional_terms:
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
    config: BuildConfig,
) -> pd.DataFrame:
    labeled = pheno[pheno["mp_term"].isin(config.directional_terms)].copy()

    # Allele-symbol pipes denote multi-allele genotypes in MGI's flattened
    # report. Keep single allele/background examples for a cleaner mutation task.
    labeled = labeled[~labeled["allele_symbol"].str.contains(r"\|", na=False)]
    labeled["label"] = labeled["mp_term"].map(config.label_by_term)

    grouped = (
        labeled.groupby(["allele_symbol", "genetic_bg"], dropna=False)
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


def build_inconclusive_rows(
    pheno: pd.DataFrame,
    allele: pd.DataFrame,
    mp_name_by_id: dict[str, str],
    config: BuildConfig,
) -> pd.DataFrame:
    """Build rows with lifespan evidence that is not a single strict direction."""
    single = pheno[~pheno["allele_symbol"].str.contains(r"\|", na=False)].copy()

    grouped = (
        single.groupby(["allele_symbol", "genetic_bg"], dropna=False)
        .agg(
            all_mp_terms=("mp_term", lambda x: sorted({str(v) for v in x if pd.notna(v)})),
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

    def reason_for_terms(terms: list[str]) -> str | None:
        strict_labels = {
            config.label_by_term[term] for term in terms if term in config.directional_terms
        }
        if len(strict_labels) > 1:
            return "conflicting_directional_terms"
        if not strict_labels and set(terms) & config.inconclusive_terms:
            return "broad_non_directional_survival_term"
        return None

    grouped["inconclusive_reason"] = grouped["all_mp_terms"].apply(reason_for_terms)
    grouped = grouped.dropna(subset=["inconclusive_reason"]).copy()
    grouped["label"] = "Inconclusive"
    grouped["label_source"] = grouped["inconclusive_reason"].apply(
        lambda reason: f"mgi_{reason}"
    )
    grouped["gold_mp_terms"] = grouped["all_mp_terms"].apply(
        lambda terms: [
            term
            for term in terms
            if term in config.directional_terms or term in config.inconclusive_terms
        ]
    )

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
    df = df.drop(columns=["all_mp_terms"])

    print(f"\nInconclusive rows before balancing: {len(df)}")
    if not df.empty:
        print(df["inconclusive_reason"].value_counts().to_string())
    return df.reset_index(drop=True)
