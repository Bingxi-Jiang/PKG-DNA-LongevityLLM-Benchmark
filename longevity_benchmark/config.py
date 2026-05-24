"""Configuration and label policy for benchmark generation."""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class MGIPaths:
    """Input paths for the MGI bulk reports."""

    pheno: str = "data/mgi/MGI_PhenoGenoMP.rpt"
    allele: str = "data/mgi/MGI_PhenotypicAllele.rpt"
    mp_ontology: str = "data/mgi/VOC_MammalianPhenotype.rpt"


@dataclass(frozen=True)
class MPDPaths:
    """Input paths for the MPD Yuan2 lifespan data files."""

    yuan2_strainmeans: str = "data/mpd/Yuan2_strainmeans.csv"
    yuan2_animals: str = "data/mpd/Yuan2_animal_lifespandays.csv"
    yuan2_measureinfo: str = "data/mpd/Yuan2_measureinfo.json"


@dataclass(frozen=True)
class BuildConfig:
    """Parameters that control benchmark construction."""

    paths: MGIPaths = field(default_factory=MGIPaths)
    mpd_paths: MPDPaths = field(default_factory=MPDPaths)
    output_dir: str = "output"
    seed: int = 42
    negative_to_positive_ratio: int = 8
    min_decreased_per_split: int = 50
    ternary_majority_multiplier: int = 2
    max_pairwise_train: int = 500
    max_pairwise_test: int = 300
    train_fraction: float = 0.7
    rare_label_train_fraction: float = 0.5
    increased_terms: frozenset[str] = frozenset(
        {
            "MP:0001661",  # extended life span
            "MP:0011614",  # slow aging
        }
    )
    decreased_terms: frozenset[str] = frozenset(
        {
            "MP:0002083",  # premature death
            "MP:0003786",  # premature aging
        }
    )
    inconclusive_terms: frozenset[str] = frozenset(
        {
            "MP:0010768",  # mortality/aging
            "MP:0010769",  # abnormal survival
        }
    )
    system_prompt: str = (
        "You are an expert in mouse genetics and aging biology. "
        "Answer concisely with exactly one of the provided options."
    )
    task_description: str = (
        "Given an MGI mouse allele, its genetic background, and non-lifespan "
        "phenotype annotations, identify the direction of its effect on lifespan."
    )

    @property
    def directional_terms(self) -> frozenset[str]:
        return self.increased_terms | self.decreased_terms

    @property
    def label_by_term(self) -> dict[str, str]:
        labels = {term: "Increased" for term in self.increased_terms}
        labels.update({term: "Decreased" for term in self.decreased_terms})
        return labels


LEAKY_CONTEXT_RE = re.compile(
    r"life\s*span|lifespan|survival|death|lethal|lethality|mortality|"
    r"aging|ageing|senescence|viability|morbidity",
    flags=re.IGNORECASE,
)
