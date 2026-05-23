"""
Mouse Strain Longevity Benchmark Builder
Tasks: ternary classification + pairwise comparison
Split: stratified by gene prefix (train 70 / test 30)
"""

import json, random, os
import pandas as pd
from collections import defaultdict

random.seed(42)
os.makedirs("output", exist_ok=True)

# ── 1. Load MGI data ─────────────────────────────────────────────────────────

print("Loading MGI data...")
pheno = pd.read_csv("data/mgi/MGI_PhenoGenoMP.rpt", sep="\t", header=None, low_memory=False,
    names=["allele_combo","allele_symbol","genetic_bg","mp_term","pubmed_id","marker_id","allele_id"])

allele = pd.read_csv("data/mgi/MGI_PhenotypicAllele.rpt", sep="\t", header=None, comment="#", low_memory=False,
    names=["allele_id","allele_symbol","allele_name","allele_type","inheritance","synonyms",
           "marker_id","marker_symbol","refseq_id","ensembl_id","mp_ids","mp_names","marker_name"])

mp_ont = pd.read_csv("data/mgi/VOC_MammalianPhenotype.rpt", sep="\t", header=None,
    names=["mp_id","mp_name","mp_def"])
mp_dict = dict(zip(mp_ont["mp_id"], mp_ont["mp_name"]))

# ── 2. Define MP term sets ───────────────────────────────────────────────────

INCREASED_TERMS = {"MP:0001661", "MP:0011614"}   # extended life span, slow aging
DECREASED_TERMS = {"MP:0002083", "MP:0003786", "MP:0010769", "MP:0010768"}
ALL_LIFESPAN_TERMS = INCREASED_TERMS | DECREASED_TERMS
RANK = {"Increased": 2, "Not changed": 1, "Decreased": 0}

# ── 3. Build per-genotype phenotype index ────────────────────────────────────

geno_to_mp = defaultdict(set)
for _, row in pheno.iterrows():
    geno_to_mp[(row["allele_symbol"], row["genetic_bg"])].add(row["mp_term"])

allele_meta = allele[["allele_symbol","allele_name","allele_type","inheritance",
                       "marker_symbol","marker_name"]].copy()

# ── 4. Extract labeled Increased / Decreased entries ─────────────────────────

ls = pheno[pheno["mp_term"].isin(ALL_LIFESPAN_TERMS)].copy()
ls = ls[~ls["allele_symbol"].str.contains(r"\|", na=False)]
ls["label"] = ls["mp_term"].apply(
    lambda t: "Increased" if t in INCREASED_TERMS else "Decreased")
ls = ls.merge(allele_meta, on="allele_symbol", how="left")
ls = (ls.dropna(subset=["marker_symbol"])
        .drop_duplicates(subset=["allele_symbol","genetic_bg","label"])
        .reset_index(drop=True))

# ── 5. Mine "Not changed" entries ────────────────────────────────────────────
# Alleles with ≥ 3 distinct MP terms but NO lifespan terms

lifespan_alleles = set(ls["allele_symbol"])
pheno_counts = pheno.groupby("allele_symbol")["mp_term"].nunique()
candidates = [s for s in pheno_counts[pheno_counts >= 3].index
              if s not in lifespan_alleles and "|" not in str(s)]

# Target: ~4× the Increased count to enable balanced tasks
n_nc_target = len(ls[ls["label"]=="Increased"]) * 4
nc_syms = random.sample(candidates, min(len(candidates), max(n_nc_target, 100)))

nc_rows = (pheno[pheno["allele_symbol"].isin(nc_syms)]
           .drop_duplicates("allele_symbol")
           .merge(allele_meta, on="allele_symbol", how="left")
           .dropna(subset=["marker_symbol"])
           .drop_duplicates("allele_symbol")
           .reset_index(drop=True))
nc_rows["label"] = "Not changed"
nc_rows["genetic_bg"] = nc_rows["allele_symbol"].apply(
    lambda s: pheno[pheno["allele_symbol"]==s]["genetic_bg"].iloc[0])

KEEP = ["allele_symbol","allele_name","allele_type","inheritance",
        "marker_symbol","marker_name","genetic_bg","label"]
df = pd.concat([ls[KEEP], nc_rows[KEEP]], ignore_index=True)
df = df.drop_duplicates(subset=["allele_symbol","genetic_bg"]).reset_index(drop=True)

print(f"All entries: {len(df)}")
print(df["label"].value_counts().to_string())

# ── 6. Gene-level train/test split (no gene appears in both splits) ───────────
# Split all unique gene symbols globally 70/30, then assign every row by its gene.
# Genes for the rare "Increased" class are split separately to guarantee both
# splits contain at least some positive examples.

inc_genes  = list(df[df["label"]=="Increased"]["marker_symbol"].unique())
rest_genes = list(df[df["label"]!="Increased"]["marker_symbol"].unique())
# remove any gene that also has an Increased entry (already handled above)
rest_genes = [g for g in rest_genes if g not in inc_genes]

random.shuffle(inc_genes)
random.shuffle(rest_genes)

n_inc_train  = max(1, int(len(inc_genes)  * 0.7))
n_rest_train = max(1, int(len(rest_genes) * 0.7))

train_genes = set(inc_genes[:n_inc_train])  | set(rest_genes[:n_rest_train])
test_genes  = set(inc_genes[n_inc_train:])  | set(rest_genes[n_rest_train:])

df["split"] = df["marker_symbol"].apply(lambda g: "train" if g in train_genes else "test")

train_df = df[df["split"]=="train"]
test_df  = df[df["split"]=="test"]

# Sanity check: no gene in both splits
assert len(set(train_df["marker_symbol"]) & set(test_df["marker_symbol"])) == 0, \
    "Data leakage: same gene appears in train and test!"

print(f"\nTrain: {train_df['label'].value_counts().to_dict()}")
print(f"Test:  {test_df['label'].value_counts().to_dict()}")

# ── 7. Helpers ────────────────────────────────────────────────────────────────

def get_context(allele_sym, genetic_bg, max_p=8):
    mp_set = geno_to_mp.get((allele_sym, genetic_bg), set())
    return [mp_dict[t] for t in mp_set if t not in ALL_LIFESPAN_TERMS and t in mp_dict][:max_p]

def describe_allele(row):
    row = row if isinstance(row, dict) else row.to_dict()
    allele_type = row["allele_type"] if pd.notna(row.get("allele_type")) else "unspecified"
    inheritance = row["inheritance"] if pd.notna(row.get("inheritance")) else "unspecified"
    gene_name   = row["marker_name"] if pd.notna(row.get("marker_name")) else row["marker_symbol"]
    bg = str(row.get("genetic_bg","")).replace("involves: ","").replace("either: ","")
    ctx = get_context(row["allele_symbol"], row["genetic_bg"])
    pheno_block = ("\nOther observed phenotypes:\n" + "\n".join(f"  - {p}" for p in ctx)) if ctx else ""
    return (f"Gene: {row['marker_symbol']} ({gene_name})\n"
            f"Allele: {row['allele_symbol']}\n"
            f"Allele type: {allele_type}\n"
            f"Mode of inheritance: {inheritance}\n"
            f"Genetic background: {bg}"
            f"{pheno_block}")

SYSTEM = ("You are an expert in mouse genetics and aging biology. "
          "Answer concisely with exactly one of the provided options.")

# ── 8. Task A: Ternary (all data, report balanced accuracy) ──────────────────

task_a = []
for _, row in df.iterrows():
    desc = describe_allele(row)
    user = (f"A researcher has engineered a mouse with the following genetic modification:\n\n"
            f"{desc}\n\n"
            f"Does this modification INCREASE, DECREASE, or leave UNCHANGED the lifespan of "
            f"these mice compared to wild-type controls?\n\n"
            f"Answer with exactly one phrase: Increased / Decreased / Not changed")
    task_a.append({
        "lb_id": "LB-MGI-001",
        "display_name": "MGI Mouse Longevity / Ternary",
        "domain": "genetics", "format": "ternary", "metric": "balanced_accuracy",
        "split": row["split"],
        "metadata": {"allele": row["allele_symbol"], "gene": row["marker_symbol"],
                     "allele_type": str(row["allele_type"]), "bg": str(row["genetic_bg"])},
        "messages": [
            {"role": "system",    "content": SYSTEM},
            {"role": "user",      "content": user},
            {"role": "assistant", "content": row["label"]},
        ]
    })

print(f"\nTask A (Ternary) — total {len(task_a)}")

# ── 9. Task B: Pairwise (naturally balanced by construction) ──────────────────

def make_pairs(group_a_df, group_b_df, n_pairs):
    """Pair rows from two groups; the one with higher RANK is the correct answer."""
    pairs = []
    ga = group_a_df.to_dict("records")
    gb = group_b_df.to_dict("records")
    random.shuffle(ga); random.shuffle(gb)
    for i in range(n_pairs):
        a = ga[i % len(ga)]
        b = gb[i % len(gb)]
        # Randomly swap positions to prevent position bias
        if random.random() < 0.5:
            a, b = b, a
        correct = "A" if RANK[a["label"]] > RANK[b["label"]] else "B"
        split = "train" if (a.get("split")=="train" and b.get("split")=="train") else "test"
        desc_a = describe_allele(a)
        desc_b = describe_allele(b)
        user = (f"Two mouse strains carry different genetic modifications.\n\n"
                f"=== Mouse A ===\n{desc_a}\n\n"
                f"=== Mouse B ===\n{desc_b}\n\n"
                f"Which mouse is expected to live LONGER?\n\n"
                f"Answer with a single letter: A / B")
        pairs.append({
            "lb_id": "LB-MGI-002",
            "display_name": "MGI Mouse Longevity / Pairwise",
            "domain": "genetics", "format": "pairwise", "metric": "accuracy",
            "split": split,
            "metadata": {"allele_A": a["allele_symbol"], "label_A": a["label"],
                         "allele_B": b["allele_symbol"], "label_B": b["label"]},
            "messages": [
                {"role": "system",    "content": SYSTEM},
                {"role": "user",      "content": user},
                {"role": "assistant", "content": correct},
            ]
        })
    return pairs

inc_all = df[df["label"]=="Increased"].to_dict("records")
dec_all = df[df["label"]=="Decreased"].to_dict("records")
nc_all  = df[df["label"]=="Not changed"].to_dict("records")

task_b = []
task_b += make_pairs(df[df["label"]=="Increased"], df[df["label"]=="Decreased"],  80)
task_b += make_pairs(df[df["label"]=="Increased"], df[df["label"]=="Not changed"], 40)
task_b += make_pairs(df[df["label"]=="Not changed"],df[df["label"]=="Decreased"],  40)
random.shuffle(task_b)

print(f"Task B (Pairwise) — total {len(task_b)}")

# ── 10. Save ──────────────────────────────────────────────────────────────────

def save_task(task_list, name):
    for split in ["train", "test"]:
        recs = [r for r in task_list if r["split"] == split]
        if not recs:
            continue
        path = f"output/{name}_{split}.jsonl"
        with open(path, "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        labels = {}
        for r in recs:
            lbl = r["messages"][-1]["content"]
            labels[lbl] = labels.get(lbl, 0) + 1
        print(f"  {path} — {len(recs)} rows, labels: {labels}")

print("\n=== Output ===")
save_task(task_a, "mgi_ternary")
save_task(task_b, "mgi_pairwise")

# ── 11. Sample prompt preview ─────────────────────────────────────────────────

sample = task_a[0]
print("\n--- Sample prompt (ternary) ---")
print(sample["messages"][1]["content"])
print(f"\nGold answer: {sample['messages'][-1]['content']}")
