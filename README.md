# Mouse Longevity Benchmark

A benchmark for evaluating whether LLMs can reason about murine longevity from
Mouse Genome Informatics (MGI) phenotype annotations and Mouse Phenome Database
(MPD) lifespan measurements.

This was built for Track 01 of the LongevityLLM Hackathon.

## Overview

Most aging benchmarks are human-centric. This benchmark asks a model to reason
about murine genetics and strain biology: given an allele, inheritance mode,
genetic background, non-lifespan phenotype profile, strain, or sex, identify
curated lifespan effects or predict measured lifespan.

The benchmark intentionally avoids a "Not changed" label. In MGI, absence of a
lifespan phenotype annotation is not verifiable evidence that lifespan was
unchanged; it may simply mean the phenotype was not tested or not curated. The
ternary task therefore uses "Inconclusive" for broad, non-directional, or
conflicting lifespan-related annotations rather than treating missing evidence
as no effect.

## Ground Truth

Labels are derived from strict directional Mammalian Phenotype (MP) terms:

| Label | MP terms |
| --- | --- |
| Increased | `MP:0001661` extended life span; `MP:0011614` slow aging |
| Decreased | `MP:0002083` premature death; `MP:0003786` premature aging |

Broader or non-directional terms such as `MP:0010768` mortality/aging and
`MP:0010769` abnormal survival are excluded from strict directional labels.
They are used only for the ternary task's `Inconclusive` class.

Each output row includes provenance in `metadata`: allele, gene, genetic
background, gold MP terms/names, PubMed IDs, and label source.

MPD-derived tasks use Yuan2, an inbred-strain lifespan and survival-curve study
in the Mouse Phenome Database. Regression labels come from measure `23201`
median lifespan in days. Sex-effect labels are computed from animal-level
measure `23401` lifespan values using a two-sided Mann-Whitney U test at
alpha = 0.05.

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

### LB-MGI-003: Multiple Choice Effect

Choose the curated MGI lifespan-effect option from four choices.

| Property | Value |
| --- | --- |
| Format | Multiple choice question (MCQ) |
| Metric | Accuracy |
| Train | 99 prompts |
| Test | 99 prompts |
| Train answers | 88 B, 11 A |
| Test answers | 88 B, 11 A |

Options are fixed as: A increased lifespan, B decreased lifespan, C
non-directional/conflicting lifespan evidence, and D no curated
lifespan-related annotation. The current MCQ records are generated from the
strict directional rows, so gold answers are A or B.

### LB-MGI-004: Ternary Inconclusive

Predict whether MGI evidence supports increased lifespan, decreased lifespan,
or an inconclusive lifespan-related annotation.

| Property | Value |
| --- | --- |
| Format | Ternary classification |
| Metric | Balanced accuracy |
| Train | 33 prompts |
| Test | 33 prompts |
| Train labels | 11 Increased, 11 Decreased, 11 Inconclusive |
| Test labels | 11 Increased, 11 Decreased, 11 Inconclusive |

Inconclusive rows come only from MGI annotations that are explicitly
lifespan/aging-related but not a single strict direction: broad terms such as
`MP:0010768` mortality/aging and `MP:0010769` abnormal survival, plus genotype
rows with conflicting strict directional labels.

### LB-MGI-005: Directional MP Term Set

Generate the set of strict directional MP terms curated for the genotype.

| Property | Value |
| --- | --- |
| Format | Set generation |
| Metric | Mean set F1 |
| Train | 99 prompts |
| Test | 99 prompts |
| Candidate terms | 4 strict directional MP terms |

The gold answer is a comma-separated set of MP IDs, e.g. `MP:0002083` or
`MP:0002083,MP:0003786`.

### LB-MPD-006: Strain Lifespan Regression

Predict the median lifespan in days for a strain-sex group from MPD Yuan2.

| Property | Value |
| --- | --- |
| Format | Regression |
| Metric | MAE days |
| Train | 41 prompts |
| Test | 18 prompts |
| Units | Days |

Rows are split by strain so the same inbred strain does not appear in both
train and test.

### LB-MPD-007: Sex Effect With No Significant Difference

Predict whether females, males, or neither sex had significantly longer
lifespan for a Yuan2 inbred strain.

| Property | Value |
| --- | --- |
| Format | Ternary classification |
| Metric | Balanced accuracy |
| Train | 19 prompts |
| Test | 9 prompts |
| Test labels | 7 No significant difference, 1 Female longer, 1 Male longer |

This task is the strict no-effect-style addition: the "No significant
difference" class comes from tested animal-level lifespan data, not from absent
curation. Because only one strain has a female-longer result at alpha = 0.05,
that row is placed in the test split.

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
- MPD regression split: no Yuan2 strain appears in both train and test.
- MPD sex-effect labels are computed from animal-level lifespan values and
  keep measured medians and p-values in metadata only, not in prompts.

Latest sanity checks after regeneration:

```text
effect gene overlap: 0
leaky context hits: 0 / 1547 prompts
pairwise train bad_split: 0, labels: A=250, B=250
pairwise test bad_split: 0, labels: A=150, B=150
ternary train/test labels: 11 Increased, 11 Decreased, 11 Inconclusive
MPD regression train/test: 41 / 18
MPD sex-effect test labels: 7 No significant difference, 1 Female longer, 1 Male longer
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

The MPD-derived tasks expect three Yuan2 files:

| File | Description |
| --- | --- |
| `Yuan2_strainmeans.csv` | Strain-sex median lifespan and summary values |
| `Yuan2_animal_lifespandays.csv` | Animal-level lifespan in days |
| `Yuan2_measureinfo.json` | MPD Yuan2 measure metadata |

To download fresh copies:

```bash
mkdir -p data/mpd
curl "https://phenome.jax.org/api/pheno/strainmeans/Yuan2?csv=yes" -o data/mpd/Yuan2_strainmeans.csv
curl "https://phenome.jax.org/api/pheno/animalvals/23401?csv=yes" -o data/mpd/Yuan2_animal_lifespandays.csv
curl "https://phenome.jax.org/api/pheno/measureinfo/Yuan2" -o data/mpd/Yuan2_measureinfo.json
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
pip install pandas scikit-learn openai httpx matplotlib
```

Rebuild the benchmark:

```bash
python -B build_benchmark.py
```

The top-level script is a thin wrapper. The same builder can also be called as:

```bash
python -m longevity_benchmark.build --no-preview
```

Generated files:

```text
output/mgi_mutation_lifespan_direction_binary_train.jsonl
output/mgi_mutation_lifespan_direction_binary_test.jsonl
output/mgi_mutation_lifespan_effect_mcq_train.jsonl
output/mgi_mutation_lifespan_effect_mcq_test.jsonl
output/mgi_mutation_lifespan_effect_ternary_inconclusive_train.jsonl
output/mgi_mutation_lifespan_effect_ternary_inconclusive_test.jsonl
output/mgi_mutation_directional_lifespan_mp_terms_set_generation_train.jsonl
output/mgi_mutation_directional_lifespan_mp_terms_set_generation_test.jsonl
output/mgi_mutation_lifespan_longer_lived_pairwise_train.jsonl
output/mgi_mutation_lifespan_longer_lived_pairwise_test.jsonl
output/mpd_strain_sex_median_lifespan_days_regression_train.jsonl
output/mpd_strain_sex_median_lifespan_days_regression_test.jsonl
output/mpd_strain_lifespan_sex_difference_ternary_train.jsonl
output/mpd_strain_lifespan_sex_difference_ternary_test.jsonl
```

Output names follow `source_subject_prediction-target_format_split.jsonl`, so
the file name identifies both the task format and the biological question.

Copy `.env.example` to `.env` and fill in the API keys for the providers you
want to use:

```bash
cp .env.example .env
# then edit .env
```

Run evaluation against the hosted Longevity-LLM endpoint:

```bash
HF_TOKEN=<your-token> python evaluate.py --task all --split test
```

Run against Google Gemini:

```bash
GEMINI_API_KEY=<your-key> python evaluate.py --provider gemini --task all --split test
```

Run against Anthropic Claude:

```bash
ANTHROPIC_API_KEY=<your-key> python evaluate.py --provider claude --task all --split test
```

Run against OpenAI:

```bash
OPENAI_API_KEY=<your-key> python evaluate.py --provider openai --task all --split test
```

Run every provider's default model from one command:

```bash
python evaluate.py --providers all --task all --split test
```

Run selected providers' default models:

```bash
python evaluate.py --providers longevity gemini openai --task effect --limit 20
```

Run explicit model names:

```bash
python evaluate.py --models openai:gpt-5.5 claude:claude-sonnet-4-6 --task all
```

Run multiple models from one provider:

```bash
python evaluate.py --provider openai --models gpt-5.5 gpt-5.1 --task effect
```

Results land in separate files per provider, e.g.
`output/eval/results_effect_test_claude_claude-sonnet-4-6_nothink.jsonl`.
Complete existing `results_*.jsonl` + `summary_*.json` pairs are skipped by
default so repeated runs do not spend API calls again. Add `--rerun` to force
a fresh run.

## Model Providers

| Provider flag | Default model | API key env var |
| --- | --- | --- |
| `longevity` (default) | `longevity-llm` | `HF_TOKEN` |
| `gemini` | `gemini-3.5-flash` | `GEMINI_API_KEY` or `GOOGLE_API_KEY` |
| `claude` | `claude-sonnet-4-6` | `ANTHROPIC_API_KEY` |
| `openai` | `gpt-5.5` | `OPENAI_API_KEY` |

Override the model with `--model <name>`, e.g.:

```bash
python evaluate.py --provider claude --model claude-opus-4-7 --think --task mcq
```

Task aliases accepted by `--task` are `effect`, `mcq`, `ternary`, `set`,
`pairwise`, `regression`, `sex_effect`, and `all`.

Thinking (`--think`) is supported for `longevity` (via `chat_template_kwargs`)
and `claude` (via Anthropic extended thinking). It is a no-op for `gemini`
and `openai`.

Useful options:

```bash
# Quick smoke test — 10 rows per task, longevity provider
python evaluate.py --task effect  --limit 10
python evaluate.py --task mcq     --limit 10
python evaluate.py --task ternary --limit 10
python evaluate.py --task set     --limit 10
python evaluate.py --task pairwise --think
python evaluate.py --task regression  --limit 10
python evaluate.py --task sex_effect  --limit 10

# Override endpoint URL
# Only valid when evaluating one provider/model.
python evaluate.py --endpoint https://<endpoint>/v1

# Custom input/output dirs
python evaluate.py --input-dir output --eval-dir output/eval

# Gemini with a specific model, smoke test
python evaluate.py --provider gemini --model gemini-2.0-flash --limit 10

# Claude Opus with chain-of-thought thinking
python evaluate.py --provider claude --model claude-opus-4-7 --think --limit 10

# OpenAI with the default GPT-5.5 model
python evaluate.py --provider openai --limit 10

# Re-run even if matching output files already exist
python evaluate.py --provider openai --task effect --rerun
```

## Scoring

- Effect task: balanced accuracy, with random baseline 0.5.
- MCQ task: accuracy, with random baseline 0.25.
- Ternary task: balanced accuracy, with random baseline 0.333.
- Set-generation task: mean per-row set F1.
- Pairwise task: accuracy, with random baseline 0.5.
- Regression task: mean absolute error and RMSE in days.
- Sex-effect task: balanced accuracy over Female longer, Male longer, and No
  significant difference.

For regression, the per-row `correct` field is only exact numeric-string
matching and should not be interpreted as accuracy. The regression metric is
MAE days, where lower is better.

The evaluator strips model thinking traces before parsing final answers.

## Multi-Model Evaluation Output

`evaluate.py` is the single entry point for one model, selected providers, or
explicit multi-model batches. Each task/model pair writes:

- `results_{task}_{split}_{provider}_{model}_{think}.jsonl` with per-row model
  outputs, parsed predictions, correctness, timing, token counts, and metadata.
- `summary_{task}_{split}_{provider}_{model}_{think}.json` with task, provider,
  model, metric, score, baseline, row count, and optional trace-quality stats.

At the end of a multi-model run, `evaluate.py` prints a compact batch summary
across all task/model pairs. `score_traces.py --plot-all` reads these saved
evaluation summaries/results directly, so no separate comparison script is
needed.

## Reasoning Trace Scoring

When `--think` is enabled, the model emits a chain-of-thought trace before
its answer. `reasoning_scorer` evaluates the trace's biological correctness
against the actual MGI/MP ontology databases — no LLM-in-the-loop required.

Five sub-scores per trace (each in `[0, 1]`):

| Sub-score | What it checks | Source of truth |
| --- | --- | --- |
| `allele_validity` | Allele symbols cited (e.g. `Sirt6<tm1Caid>`) exist in MGI | `MGI_PhenotypicAllele.rpt` (~129k alleles) |
| `gene_validity` | Gene symbols cited in explicit gene contexts ("the X gene", "X knockout") are real MGI markers | MGI marker symbols (~35k genes) |
| `mp_validity` | Cited `MP:NNNNNNN` IDs exist and their claimed names match the ontology (Jaccard ≥ 0.4) | `VOC_MammalianPhenotype.rpt` (~15k terms) |
| `answer_consistency` | Directional language at the end of the trace aligns with the final classification answer | Built-in direction vocab |
| `prompt_grounding` | Entities cited in the trace (gene/allele/strain) appear in the prompt's metadata, not in unrelated biology | Row metadata |

The aggregate is a weighted mean of present sub-scores (missing dimensions
are excluded rather than scored 0, so a trace with no MP-term mentions is
not penalised for it).

### Integration with evaluation

Add `--score-traces` to `evaluate.py` to compute trace quality alongside
accuracy:

```bash
python evaluate.py --provider longevity --task ternary --think --score-traces
```

The per-row `results_*.jsonl` is extended with `trace_score` and
`trace_subscores`, and the `summary_*.json` gets a `trace_quality` block
with the global mean and hallucination counts.

### Multi-model trace scoring

`evaluate.py --think --score-traces` scores traces for every selected
provider/model that emits them:

```bash
python evaluate.py --task ternary --providers longevity claude --think --score-traces --limit 30
```

```text
── Reasoning trace quality ─────────────────────
  Trace quality: mean=0.867  fabricated_genes=0  fabricated_alleles=0  invalid_mp_ids=0
```

The summary JSON for each model receives a `trace_quality` block with the same
aggregate values.

### Standalone scoring (for already-saved results)

```bash
python score_traces.py \
  --results output/eval/results_ternary_test_longevity_longevity-llm_think.jsonl \
  --out output/eval/trace_scores_mgi_mutation_lifespan_effect_ternary_inconclusive.jsonl
```

This prints sub-score means, point-biserial correlation between each
sub-score and final-answer correctness, fabrication tallies, and the
lowest- and highest-scoring traces for manual review.
Evaluation result files use the short CLI task alias, while dataset JSONL files
use the longer descriptive names.

To visualize all saved `evaluate.py` outputs in one graph:

```bash
python score_traces.py --plot-all \
  --eval-dir output/eval \
  --plot-out output/eval/llm_task_performance.png
```

This writes a single grouped bar chart comparing model names saved by
`evaluate.py` across tasks. Classification tasks use their task metric, while
regression is not accuracy; it is normalized as `1 - MAE / gold lifespan
range` so higher bars are better across the whole figure.

## File Structure

```text
.
|-- build_benchmark.py
|-- evaluate.py
|-- README.md
|-- longevity_benchmark/
|   |-- build.py              # build orchestration and CLI args
|   |-- config.py             # label policy, paths, sampling parameters
|   |-- io.py                 # JSONL read/write helpers
|   |-- mgi.py                # MGI table loading and label extraction
|   |-- mpd.py                # MPD Yuan2 loading and derived labels
|   |-- prompts.py            # allele descriptions and metadata
|   |-- splits.py             # leakage-aware split and balancing
|   |-- tasks.py              # task record builders
|   |-- validation.py         # generated-record sanity checks
|   `-- eval/
|       |-- cli.py            # evaluation CLI
|       |-- parsing.py        # model-output parsers
|       `-- runner.py         # endpoint calls, logging, metrics
|-- data/
|   |-- mgi/
|   |   |-- MGI_PhenoGenoMP.rpt
|   |   |-- MGI_PhenotypicAllele.rpt
|   |   `-- VOC_MammalianPhenotype.rpt
|   `-- mpd/
|       |-- Yuan2_strainmeans.csv
|       |-- Yuan2_animal_lifespandays.csv
|       `-- Yuan2_measureinfo.json
`-- output/
    |-- mgi_mutation_lifespan_direction_binary_train.jsonl
    |-- mgi_mutation_lifespan_direction_binary_test.jsonl
    |-- mgi_mutation_lifespan_effect_mcq_train.jsonl
    |-- mgi_mutation_lifespan_effect_mcq_test.jsonl
    |-- mgi_mutation_lifespan_effect_ternary_inconclusive_train.jsonl
    |-- mgi_mutation_lifespan_effect_ternary_inconclusive_test.jsonl
    |-- mgi_mutation_directional_lifespan_mp_terms_set_generation_train.jsonl
    |-- mgi_mutation_directional_lifespan_mp_terms_set_generation_test.jsonl
    |-- mgi_mutation_lifespan_longer_lived_pairwise_train.jsonl
    |-- mgi_mutation_lifespan_longer_lived_pairwise_test.jsonl
    |-- mpd_strain_sex_median_lifespan_days_regression_train.jsonl
    |-- mpd_strain_sex_median_lifespan_days_regression_test.jsonl
    |-- mpd_strain_lifespan_sex_difference_ternary_train.jsonl
    `-- mpd_strain_lifespan_sex_difference_ternary_test.jsonl
```

The current `mgi_mutation_lifespan_effect_ternary_inconclusive_*` files use an
`Inconclusive` label, not a `Not changed` label.

## Extending

The code is split so new prototype tasks can be added without editing the full
pipeline:

1. Add task-specific record generation in `longevity_benchmark/tasks.py` or a
   new module under `longevity_benchmark/tasks/` if it grows.
2. Reuse `mgi.py` for source tables, `splits.py` for leakage-aware cohorts, and
   `prompts.py` for allele descriptions.
3. Register the new record list in `longevity_benchmark/build.py` and save it
   with `save_split_records`.
4. Add an output parser or metric branch under `longevity_benchmark/eval/` if
   the answer format is not already registered.

The intended shape is: data extraction -> split/balance -> task record builder
-> JSONL writer -> evaluator parser/metric.
