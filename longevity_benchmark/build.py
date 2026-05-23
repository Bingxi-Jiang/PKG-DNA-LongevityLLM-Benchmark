"""CLI orchestration for benchmark generation."""

from __future__ import annotations

import argparse
import random
from dataclasses import replace
from pathlib import Path

import pandas as pd

from .config import BuildConfig, MGIPaths, MPDPaths
from .io import save_split_records
from .mgi import (
    build_context_index,
    build_directional_rows,
    build_inconclusive_rows,
    load_mgi_tables,
)
from .mpd import (
    assign_strain_splits,
    build_yuan2_sex_effect_rows,
    load_yuan2_animals,
    load_yuan2_strainmeans,
)
from .splits import (
    assign_gene_splits,
    balance_decreased_within_split,
    balance_ternary_within_split,
)
from .tasks import (
    make_effect_records,
    make_mpd_lifespan_regression_records,
    make_mpd_sex_effect_records,
    make_mcq_records,
    make_pairwise_records,
    make_set_generation_records,
    make_ternary_records,
)
from .validation import print_sanity_checks


def build_benchmark(config: BuildConfig) -> dict[str, list[dict]]:
    rng = random.Random(config.seed)
    ternary_rng = random.Random(config.seed + 101)
    mpd_rng = random.Random(config.seed + 202)
    tables = load_mgi_tables(config.paths)
    context_index = build_context_index(tables.pheno, tables.mp_name_by_id, config)
    directional = build_directional_rows(
        tables.pheno,
        tables.allele,
        tables.mp_name_by_id,
        config,
    )
    directional = assign_gene_splits(directional, config, rng)
    directional = balance_decreased_within_split(directional, config)

    effect_records = make_effect_records(directional, context_index, config)
    pairwise_records = make_pairwise_records(directional, context_index, config, rng)
    mcq_records = make_mcq_records(directional, context_index, config)
    set_records = make_set_generation_records(directional, context_index, config)

    inconclusive = build_inconclusive_rows(
        tables.pheno,
        tables.allele,
        tables.mp_name_by_id,
        config,
    )
    ternary_source = pd.concat(
        [directional.drop(columns=["split"]), inconclusive],
        ignore_index=True,
    )
    ternary_source = assign_gene_splits(ternary_source, config, ternary_rng)
    ternary_source = balance_ternary_within_split(ternary_source, config)
    ternary_records = make_ternary_records(ternary_source, context_index, config)

    yuan2_strainmeans = load_yuan2_strainmeans(config.mpd_paths)
    yuan2_regression_source = assign_strain_splits(yuan2_strainmeans, config, mpd_rng)
    mpd_regression_records = make_mpd_lifespan_regression_records(
        yuan2_regression_source,
        config,
    )

    yuan2_animals = load_yuan2_animals(config.mpd_paths)
    yuan2_sex_effect_source = build_yuan2_sex_effect_rows(yuan2_animals, config, mpd_rng)
    mpd_sex_effect_records = make_mpd_sex_effect_records(yuan2_sex_effect_source, config)

    return {
        "effect": effect_records,
        "pairwise": pairwise_records,
        "mcq": mcq_records,
        "ternary": ternary_records,
        "set": set_records,
        "regression": mpd_regression_records,
        "sex_effect": mpd_sex_effect_records,
    }


def write_benchmark(records_by_task: dict[str, list[dict]], output_dir: str) -> None:
    print("\n=== Output ===")
    save_split_records(records_by_task["effect"], output_dir, "mgi_effect")
    save_split_records(records_by_task["mcq"], output_dir, "mgi_mcq")
    save_split_records(records_by_task["ternary"], output_dir, "mgi_ternary")
    save_split_records(records_by_task["set"], output_dir, "mgi_set")
    save_split_records(records_by_task["pairwise"], output_dir, "mgi_pairwise")
    save_split_records(
        records_by_task["regression"],
        output_dir,
        "mpd_lifespan_regression",
    )
    save_split_records(records_by_task["sex_effect"], output_dir, "mpd_sex_effect")


def preview_first_effect(records_by_task: dict[str, list[dict]]) -> None:
    sample = records_by_task["effect"][0]
    print("\n--- Sample prompt (effect) ---")
    print(sample["messages"][1]["content"])
    print(f"\nGold answer: {sample['messages'][-1]['content']}")


def config_from_args(args: argparse.Namespace) -> BuildConfig:
    data_dir = Path(args.data_dir)
    paths = MGIPaths(
        pheno=str(data_dir / "MGI_PhenoGenoMP.rpt"),
        allele=str(data_dir / "MGI_PhenotypicAllele.rpt"),
        mp_ontology=str(data_dir / "VOC_MammalianPhenotype.rpt"),
    )
    mpd_data_dir = Path(args.mpd_data_dir)
    mpd_paths = MPDPaths(
        yuan2_strainmeans=str(mpd_data_dir / "Yuan2_strainmeans.csv"),
        yuan2_animals=str(mpd_data_dir / "Yuan2_animal_lifespandays.csv"),
        yuan2_measureinfo=str(mpd_data_dir / "Yuan2_measureinfo.json"),
    )
    return replace(
        BuildConfig(paths=paths),
        mpd_paths=mpd_paths,
        output_dir=args.output_dir,
        seed=args.seed,
        max_pairwise_train=args.max_pairwise_train,
        max_pairwise_test=args.max_pairwise_test,
        rare_label_train_fraction=args.rare_label_train_fraction,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/mgi")
    parser.add_argument("--mpd-data-dir", default="data/mpd")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-pairwise-train", type=int, default=500)
    parser.add_argument("--max-pairwise-test", type=int, default=300)
    parser.add_argument("--rare-label-train-fraction", type=float, default=0.5)
    parser.add_argument("--no-preview", action="store_true")
    args = parser.parse_args()

    config = config_from_args(args)
    records_by_task = build_benchmark(config)
    write_benchmark(records_by_task, config.output_dir)
    print_sanity_checks(
        records_by_task["effect"],
        records_by_task["pairwise"],
        {
            "mcq": records_by_task["mcq"],
            "ternary": records_by_task["ternary"],
            "set": records_by_task["set"],
            "regression": records_by_task["regression"],
            "sex_effect": records_by_task["sex_effect"],
        },
    )
    if not args.no_preview:
        preview_first_effect(records_by_task)


if __name__ == "__main__":
    main()
