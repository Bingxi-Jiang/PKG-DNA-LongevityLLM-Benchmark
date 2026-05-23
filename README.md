# MGI Mouse Longevity Benchmark

A benchmark for evaluating whether LLMs can infer the directional effect of
mouse genetic mutations on lifespan from Mouse Genome Informatics (MGI)
phenotype annotations.

This was built for Track 01 of the LongevityLLM Hackathon.

## Overview

Most aging benchmarks are human-centric. This benchmark asks a model to reason
about murine genetics: given an allele, inheritance mode, genetic background,
and non-lifespan phenotype profile, predict whether the mutation increases or
decreases lifespan relative to wild-type controls.

The current benchmark intentionally avoids a "Not changed" label. In MGI,
absence of a lifespan phenotype annotation is not verifiable evidence that
lifespan was unchanged; it may simply mean the phenotype was not tested or not
curated. Keeping only directional MGI terms gives a cleaner ground truth.

## Ground Truth

Labels are derived from strict directional Mammalian Phenotype (MP) terms:

| Label | MP terms |
| --- | --- |
| Increased | `MP:0001661` extended life span; `MP:0011614` slow aging |
| Decreased | `MP:0002083` premature death; `MP:0003786` premature aging |

Broader or non-directional terms such as `MP:0010768` mortality/aging and
`MP:0010769` abnormal survival are excluded from labels.

Each output row includes provenance in `metadata`: allele, gene, genetic
background, gold MP terms/names, PubMed IDs, and label source.

## Tasks

### LB-MGI-001: Directional Effect

Predict whether the mutation increases or decreases lifespan.

| Property | Value |
| --- | --- |
| Format | Binary classification |
| Metric | Balanced accuracy |
| Train | 99 prompts |
| Test | 99 prompts |
| Train labels | 88 Decreased, 11 Increased |
| Test labels | 88 Decreased, 11 Increased |

The decreased class is downsampled within each split to control the natural MGI
imbalance while preserving all increased examples.

### LB-MGI-002: Pairwise Longevity

Given two genetically modified mouse strains, choose which one is expected to
live longer.

| Property | Value |
| --- | --- |
| Format | Pairwise comparison |
| Metric | Accuracy |
| Train | 500 prompts |
| Test | 300 prompts |
| Train answers | 250 A, 250 B |
| Test answers | 150 A, 150 B |

Pairwise examples are sampled only within the same split, so a test pair never
contains a training component.

## Leakage Controls

- Gene-level split: no `marker_symbol` appears in both train and test for the
  effect task.
- Pairwise split: pairs are generated from train-train or test-test components
  only; mixed train-test pairs are not allowed.
- Prompt context filtering: non-target phenotype context excludes terms whose
  names contain lifespan/survival/death/lethality/mortality/aging/senescence
  and related keywords.
- Conflict removal: genotype/background rows with both increased and decreased
  directional labels are dropped.
- Retrieval resistance: prompts are constructed from MGI bulk tables, not from
  PubMed abstracts or natural-language paper summaries.

Latest sanity checks after regeneration:

```text
effect gene overlap: 0
leaky context hits: 0 / 998 prompts
pairwise train bad_split: 0, labels: A=250, B=250
pairwise test bad_split: 0, labels: A=150, B=150
```

## Data Sources

The builder expects three MGI bulk download files:

| File | Description |
| --- | --- |
| `MGI_PhenoGenoMP.rpt` | Genotype to MP term associations |
| `MGI_PhenotypicAllele.rpt` | Allele metadata |
| `VOC_MammalianPhenotype.rpt` | MP ontology names/definitions |

To download fresh copies:

```bash
mkdir -p data/mgi
curl https://www.informatics.jax.org/downloads/reports/MGI_PhenoGenoMP.rpt -o data/mgi/MGI_PhenoGenoMP.rpt
curl https://www.informatics.jax.org/downloads/reports/MGI_PhenotypicAllele.rpt -o data/mgi/MGI_PhenotypicAllele.rpt
curl https://www.informatics.jax.org/downloads/reports/VOC_MammalianPhenotype.rpt -o data/mgi/VOC_MammalianPhenotype.rpt
```

## Prompt Format

Rows are JSONL records with ChatML-style messages. The final assistant message
is the gold answer and must be stripped before model inference.

Example:

```text
[system]
You are an expert in mouse genetics and aging biology. Answer concisely with exactly one of the provided options.

[user]
A researcher has engineered a mouse with the following genetic modification:

Gene: Mbp (myelin basic protein)
Allele: Mbp<jve>
Allele type: Chemically induced (ENU)
Mode of inheritance: Null/knockout
Genetic background: C57BL/6J
Other observed phenotypes:
  - tremors
  - abnormal corpus callosum morphology
  - decreased body weight
  - decreased locomotor activity

Based on curated mouse phenotype evidence, does this modification INCREASE or DECREASE lifespan compared to wild-type controls?

Answer with exactly one phrase: Increased / Decreased

[assistant]
Decreased
```

## Quickstart

Install dependencies:

```bash
pip install pandas scikit-learn openai httpx
```

Rebuild the benchmark:

```bash
python -B build_benchmark.py
```

Generated files:

```text
output/mgi_effect_train.jsonl
output/mgi_effect_test.jsonl
output/mgi_pairwise_train.jsonl
output/mgi_pairwise_test.jsonl
```

Run evaluation against the hosted Longevity-LLM endpoint:

```bash
set HF_TOKEN=<your-token>
python evaluate.py --task both --split test
```

Useful options:

```bash
python evaluate.py --task effect --limit 10
python evaluate.py --task pairwise --think
python evaluate.py --endpoint https://<endpoint>/v1
```

## Scoring

- Effect task: balanced accuracy, with random baseline 0.5.
- Pairwise task: accuracy, with random baseline 0.5.

The evaluator strips model thinking traces before parsing final answers.

## File Structure

```text
.
|-- build_benchmark.py
|-- evaluate.py
|-- README.md
|-- data/
|   `-- mgi/
|       |-- MGI_PhenoGenoMP.rpt
|       |-- MGI_PhenotypicAllele.rpt
|       `-- VOC_MammalianPhenotype.rpt
`-- output/
    |-- mgi_effect_train.jsonl
    |-- mgi_effect_test.jsonl
    |-- mgi_pairwise_train.jsonl
    `-- mgi_pairwise_test.jsonl
```

Legacy `mgi_ternary_*` files may exist in local output directories from older
builds, but they are not part of the current recommended benchmark.
