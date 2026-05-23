"""MPD Yuan2 loading and derived lifespan benchmark labels."""

from __future__ import annotations

import math
import random
from collections import Counter

import pandas as pd

from .config import BuildConfig, MPDPaths


def load_yuan2_strainmeans(paths: MPDPaths) -> pd.DataFrame:
    df = pd.read_csv(paths.yuan2_strainmeans, dtype={"strainid": str})
    df = df[df["measnum"] == 23201].copy()
    df["mean"] = pd.to_numeric(df["mean"], errors="coerce")
    df = df.dropna(subset=["mean", "strain", "sex"]).copy()
    df["median_lifespan_days"] = df["mean"].round().astype(int)
    df["sex_label"] = df["sex"].map({"f": "female", "m": "male"})
    df = df.dropna(subset=["sex_label"]).copy()
    print(f"\nMPD Yuan2 regression rows before split: {len(df)}")
    return df.reset_index(drop=True)


def load_yuan2_animals(paths: MPDPaths) -> pd.DataFrame:
    df = pd.read_csv(paths.yuan2_animals, dtype={"stocknum": str, "strainid": str})
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value", "strain", "sex"]).copy()
    return df.reset_index(drop=True)


def assign_strain_splits(
    df: pd.DataFrame,
    config: BuildConfig,
    rng: random.Random,
) -> pd.DataFrame:
    strains = sorted(df["strain"].unique())
    rng.shuffle(strains)
    n_train = max(1, int(round(len(strains) * config.train_fraction)))
    if len(strains) > 1:
        n_train = min(n_train, len(strains) - 1)
    train_strains = set(strains[:n_train])

    out = df.copy()
    out["split"] = out["strain"].apply(lambda strain: "train" if strain in train_strains else "test")
    return out


def assign_stratified_label_splits(
    df: pd.DataFrame,
    config: BuildConfig,
    rng: random.Random,
) -> pd.DataFrame:
    kept = []
    for _, label_df in df.groupby("label"):
        rows = label_df.to_dict("records")
        rng.shuffle(rows)
        if len(rows) == 1:
            n_train = 0
        else:
            n_train = max(1, int(round(len(rows) * config.train_fraction)))
            n_train = min(n_train, len(rows) - 1)
        for idx, row in enumerate(rows):
            row["split"] = "train" if idx < n_train else "test"
        kept.extend(rows)

    out = pd.DataFrame(kept)
    out = out.sample(frac=1.0, random_state=config.seed).reset_index(drop=True)
    return out


def normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def ranks(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda pair: pair[1])
    out = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + 1 + j + 1) / 2
        for k in range(i, j + 1):
            out[indexed[k][0]] = avg_rank
        i = j + 1
    return out


def mann_whitney_u_pvalue(first: list[float], second: list[float]) -> float:
    """Two-sided Mann-Whitney U p-value using normal approximation with tie correction."""
    n1 = len(first)
    n2 = len(second)
    if n1 == 0 or n2 == 0:
        return math.nan

    combined = list(first) + list(second)
    ranked = ranks(combined)
    rank_sum_first = sum(ranked[:n1])
    u1 = rank_sum_first - n1 * (n1 + 1) / 2
    u2 = n1 * n2 - u1
    u = min(u1, u2)

    counts = Counter(combined)
    n = n1 + n2
    tie_sum = sum(count**3 - count for count in counts.values() if count > 1)
    tie_correction = 1 - tie_sum / (n**3 - n) if n > 1 else 1
    mean_u = n1 * n2 / 2
    sd_u = math.sqrt(n1 * n2 * (n + 1) * tie_correction / 12)
    if sd_u == 0:
        return 1.0

    z = max(0.0, abs(u - mean_u) - 0.5) / sd_u
    return max(0.0, min(1.0, 2 * (1 - normal_cdf(z))))


def build_yuan2_sex_effect_rows(
    animals: pd.DataFrame,
    config: BuildConfig,
    rng: random.Random,
    alpha: float = 0.05,
) -> pd.DataFrame:
    rows = []
    for strain, strain_df in animals.groupby("strain"):
        by_sex = {
            sex: sex_df["value"].astype(float).tolist()
            for sex, sex_df in strain_df.groupby("sex")
        }
        if "f" not in by_sex or "m" not in by_sex:
            continue

        female = by_sex["f"]
        male = by_sex["m"]
        female_median = float(pd.Series(female).median())
        male_median = float(pd.Series(male).median())
        p_value = mann_whitney_u_pvalue(female, male)
        if p_value >= alpha:
            label = "No significant difference"
        elif female_median > male_median:
            label = "Female longer"
        else:
            label = "Male longer"

        rows.append(
            {
                "strain": strain,
                "strainid": str(strain_df["strainid"].iloc[0]),
                "stocknum": str(strain_df["stocknum"].iloc[0]),
                "female_n": len(female),
                "male_n": len(male),
                "female_median_days": round(female_median, 1),
                "male_median_days": round(male_median, 1),
                "p_value": round(p_value, 6),
                "label": label,
                "label_source": "mpd_yuan2_mann_whitney_sex_effect",
            }
        )

    df = pd.DataFrame(rows)
    print("\nMPD Yuan2 sex-effect rows before split:")
    print(df["label"].value_counts().to_string())
    return assign_stratified_label_splits(df, config, rng)
