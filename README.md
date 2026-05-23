# MGI Mouse Longevity Benchmark

A benchmark for evaluating LLMs on predicting the effect of genetic mutations on mouse lifespan, built on top of [Mouse Genome Informatics (MGI)](https://www.informatics.jax.org/) phenotype annotations.

Part of the **LongevityLLM Hackathon** (Track 01, sponsored by Insilico Medicine).

---

## Overview

Most LLM aging benchmarks are human-centric. This task asks a model to reason about murine longevity: given a genetic modification (gene, allele type, inheritance mode, genetic background, and non-lifespan phenotypes), predict whether the mutation **increases**, **decreases**, or leaves **unchanged** the lifespan of the mouse strain.

Ground truth is derived from curated MGI Mammalian Phenotype (MP) annotations:

| Label | MP Terms |
|-------|----------|
| Increased | MP:0001661 (extended life span), MP:0011614 (slow aging) |
| Decreased | MP:0002083 (premature death), MP:0003786 (premature aging), MP:0010769 (abnormal survival), MP:0010768 (mortality/aging) |
| Not changed | Alleles with ≥3 phenotype annotations but no lifespan MP terms |

---

## Tasks

### LB-MGI-001 · Ternary Classification

Predict whether a mutation increases, decreases, or does not change lifespan.

- **Format**: ternary
- **Metric**: balanced accuracy (handles class imbalance)
- **Train**: 1,465 samples · **Test**: 629 samples
- **Class distribution**: Decreased (94%), Not changed (5%), Increased (1%)

### LB-MGI-002 · Pairwise Comparison

Given two mouse strains (A and B), identify which one lives longer.

- **Format**: pairwise
- **Metric**: accuracy
- **Train**: 81 pairs · **Test**: 79 pairs
- **Pair types**: Increased vs Decreased (50%), Increased vs Not changed (25%), Not changed vs Decreased (25%)

---

## Dataset

Built from three MGI bulk download files:

| File | Description |
|------|-------------|
| `MGI_PhenoGenoMP.rpt` | Genotype → MP term associations |
| `MGI_PhenotypicAllele.rpt` | Allele metadata (type, inheritance, gene) |
| `VOC_MammalianPhenotype.rpt` | MP ontology (term IDs → names) |

To regenerate the data files:

```bash
mkdir -p data/mgi
curl https://www.informatics.jax.org/downloads/reports/MGI_PhenoGenoMP.rpt     -o data/mgi/MGI_PhenoGenoMP.rpt
curl https://www.informatics.jax.org/downloads/reports/MGI_PhenotypicAllele.rpt -o data/mgi/MGI_PhenotypicAllele.rpt
curl https://www.informatics.jax.org/downloads/reports/VOC_MammalianPhenotype.rpt -o data/mgi/VOC_MammalianPhenotype.rpt
```

### Train/Test Split

Entries are split 70/30 within each label class (stratified) to ensure all three classes appear in both splits. This prevents data leakage while maintaining class representation.

---

## Prompt Format

Each prompt follows the ChatML format (`system` + `user` + `assistant`). The `assistant` message is the gold answer — strip it before sending to the model.

**Ternary example:**

```
[system]
You are an expert in mouse genetics and aging biology. Answer concisely with exactly one of the provided options.

[user]
A researcher has engineered a mouse with the following genetic modification:

Gene: Sirt1 (sirtuin 1)
Allele: Ccdc7a<Tg(Prnp-Sirt1)10Imai>
Allele type: Transgenic
Mode of inheritance: Inserted expressed sequence|Hypomorph
Genetic background: B6.Cg-Ccdc7a<Tg(Prnp-Sirt1)10Imai>
Other observed phenotypes:
  - increased energy expenditure
  - increased body temperature
  - hyperactivity

Does this modification INCREASE, DECREASE, or leave UNCHANGED the lifespan of these mice compared to wild-type controls?

Answer with exactly one phrase: Increased / Decreased / Not changed

[assistant]
Increased
```

---

## Quickstart

### Install dependencies

```bash
pip install openai pandas pyarrow
```

### Rebuild benchmark

```bash
python build_benchmark.py
# outputs: output/mgi_ternary_{train,test}.jsonl
#          output/mgi_pairwise_{train,test}.jsonl
```

### Run evaluation against Longevity-LLM

```python
import json, re
from openai import OpenAI

client = OpenAI(
    base_url="https://sqrq2pj09htgequ0.us-east-2.aws.endpoints.huggingface.cloud/v1",
    api_key="<your-hf-token>",
)

def run_task(jsonl_path):
    preds, golds = [], []
    with open(jsonl_path) as f:
        for line in f:
            row = json.loads(line)
            msgs = row["messages"][:-1]        # drop gold answer
            gold = row["messages"][-1]["content"].strip()
            r = client.chat.completions.create(
                model="longevity-llm",
                messages=msgs,
                max_tokens=50,
                temperature=0.0,
            )
            pred = r.choices[0].message.content.strip()
            preds.append(pred)
            golds.append(gold)
    return preds, golds

preds, golds = run_task("output/mgi_ternary_test.jsonl")
```

### Scoring

```python
from sklearn.metrics import balanced_accuracy_score, accuracy_score

# Ternary task
balanced_accuracy_score(golds, preds)

# Pairwise task
accuracy_score(golds, preds)
```

### Baseline (random)

| Task | Random baseline |
|------|----------------|
| Ternary | 33.3% balanced accuracy |
| Pairwise | 50.0% accuracy |

---

## File Structure

```
.
├── build_benchmark.py        # Dataset builder
├── README.md
├── .gitignore
├── data/                     # Raw MGI files (not committed)
│   └── mgi/
│       ├── MGI_PhenoGenoMP.rpt
│       ├── MGI_PhenotypicAllele.rpt
│       └── VOC_MammalianPhenotype.rpt
├── output/                   # Generated JSONL files (not committed)
│   ├── mgi_ternary_train.jsonl
│   ├── mgi_ternary_test.jsonl
│   ├── mgi_pairwise_train.jsonl
│   └── mgi_pairwise_test.jsonl
└── longebench/               # Original LongeBench dataset (HF submodule)
```

---

## Design Notes

**Retrieval resistance**: Prompts are constructed from raw MGI bulk download tables rather than PubMed abstracts. The specific allele symbol + background strain combinations are unlikely to appear verbatim in LLM training corpora.

**Class imbalance**: MGI is biased toward disease models (premature death), so `Decreased` dominates. Balanced accuracy is used as the primary metric; the pairwise task is constructed to be balanced by design.

**Leakage prevention**: The `Not changed` class is derived from absence of lifespan MP annotations — these are alleles with known, well-characterized phenotypes that were simply never shown to affect lifespan, not untested alleles.
