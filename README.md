# Mouse Longevity Benchmark

A benchmark for testing whether LLMs can answer mouse longevity questions from
Mouse Genome Informatics (MGI) phenotype annotations and Mouse Phenome Database
(MPD) lifespan measurements.

The benchmark covers genetics, strain biology, lifespan direction, MP term
generation, pairwise longevity, regression, and sex effects. It also includes an
offline reasoning scorer that checks whether a model's rationale is biologically
grounded, not just whether the final answer is correct.

## Quickstart

Install dependencies:

```bash
pip install pandas scikit-learn openai httpx matplotlib
```

Copy credentials:

```bash
cp .env.example .env
# then fill in the API keys you want to use
```

Build the benchmark JSONL files:

```bash
python -B build_benchmark.py
```

Run all default provider models on the test split:

```bash
python evaluate.py --providers all --task all --split test
```

Run a small reasoning smoke test:

```bash
python evaluate.py --providers all --task effect --split test --limit 10 --think --score-traces --rerun
```

## Providers

| Provider | Default model | API key |
| --- | --- | --- |
| `longevity` | `longevity-llm` | `HF_TOKEN` |
| `gemini` | `gemini-3.5-flash` | `GEMINI_API_KEY` or `GOOGLE_API_KEY` |
| `claude` | `claude-sonnet-4-6` | `ANTHROPIC_API_KEY` |
| `openai` | `gpt-5.5` | `OPENAI_API_KEY` |

Run one provider:

```bash
python evaluate.py --provider openai --task ternary --limit 10
```

Run selected providers:

```bash
python evaluate.py --providers longevity gemini openai --task effect
```

Run explicit models:

```bash
python evaluate.py --models openai:gpt-5.5 claude:claude-sonnet-4-6 --task all
```

Run multiple models from one provider:

```bash
python evaluate.py --provider openai --models gpt-5.5 gpt-5.1 --task effect
```

Existing complete `results_*.jsonl` + `summary_*.json` pairs are skipped by
default to avoid repeat API calls. Add `--rerun` to force a fresh run.

## Tasks

| Alias | Task | Format | Metric |
| --- | --- | --- | --- |
| `effect` | Mutation lifespan direction | Binary classification | Balanced accuracy |
| `mcq` | Mutation lifespan effect | Multiple choice | Accuracy |
| `ternary` | Increased / decreased / inconclusive | Ternary classification | Balanced accuracy |
| `set` | Directional MP term set | Set generation | Mean set F1 |
| `pairwise` | Longer-lived mutation pair | A/B choice | Accuracy |
| `regression` | MPD strain-sex median lifespan | Numeric days | MAE days |
| `sex_effect` | Female / male / no significant difference | Ternary classification | Balanced accuracy |
| `all` | Run every task | - | - |

Regression note: the per-row `correct` field is only exact numeric-string
matching. Use `mae_days` as the real regression metric; lower is better.

## Evaluation Output

Evaluation writes one result file and one summary file per task/model:

```text
output/eval/results_{task}_{split}_{provider}_{model}_{think}.jsonl
output/eval/summary_{task}_{split}_{provider}_{model}_{think}.json
```

Result rows include raw model output, parsed prediction, gold answer,
correctness, token counts, timing, and metadata. Summary files include metric,
score, baseline, row count, and optional reasoning quality.

## Reasoning Scoring

Add `--think --score-traces` during evaluation:

```bash
python evaluate.py --providers all --task ternary --split test --think --score-traces
```

`--think` emits a scorable rationale before the final answer:

- `longevity` uses its native thinking mode.
- Other providers are prompted to output a visible `<think>...</think>` block.

The scorer checks seven dimensions:

| Sub-score | Meaning |
| --- | --- |
| `gene_validity` | Gene mentions are valid MGI marker symbols |
| `allele_validity` | Allele mentions exist in MGI |
| `mp_validity` | MP IDs exist and claimed names match the ontology |
| `answer_consistency` | Reasoning direction matches the final answer |
| `prompt_grounding` | Cited entities are grounded in the prompt metadata |
| `knowledge_extension` | Valid citations go beyond simply echoing prompt entities |
| `pathway_consistency` | Gene-specific pathway keywords match known longevity biology |

Scores are stored back into:

- `trace_score` and `trace_subscores` on each result row
- `trace_quality` in the summary JSON

## Score Existing Reasoning Runs

If you already have `*_think.jsonl` outputs, score them without re-running any
model:

```bash
python score_traces.py --providers all --task all --split test
```

Target a provider or model:

```bash
python score_traces.py --provider claude --task effect --split test
python score_traces.py --models openai:gpt-5.5 claude:claude-sonnet-4-6 --task ternary
```

Recompute existing trace scores:

```bash
python score_traces.py --providers all --task all --split test --rerun
```

Inspect one file in detail:

```bash
python score_traces.py \
  --results output/eval/results_effect_test_longevity_longevity-llm_think.jsonl \
  --show 10
```

Important: check `trace_cov` in the batch summary. A low coverage means the
model did not produce a parseable reasoning trace for many rows, so the
reasoning mean is not representative of the full task.

## Plot Results

Plot saved evaluation summaries across models and tasks:

```bash
python score_traces.py --plot-all \
  --eval-dir output/eval \
  --plot-out output/eval/llm_task_performance.png
```

Classification bars use the task metric. Regression is normalized as
`1 - MAE / gold lifespan range`, so higher bars are better in the plot.

## Data

The builder expects these source files:

```text
data/mgi/MGI_PhenoGenoMP.rpt
data/mgi/MGI_PhenotypicAllele.rpt
data/mgi/VOC_MammalianPhenotype.rpt
data/mpd/Yuan2_strainmeans.csv
data/mpd/Yuan2_animal_lifespandays.csv
data/mpd/Yuan2_measureinfo.json
```

Download fresh copies:

```bash
mkdir -p data/mgi data/mpd
curl https://www.informatics.jax.org/downloads/reports/MGI_PhenoGenoMP.rpt -o data/mgi/MGI_PhenoGenoMP.rpt
curl https://www.informatics.jax.org/downloads/reports/MGI_PhenotypicAllele.rpt -o data/mgi/MGI_PhenotypicAllele.rpt
curl https://www.informatics.jax.org/downloads/reports/VOC_MammalianPhenotype.rpt -o data/mgi/VOC_MammalianPhenotype.rpt
curl "https://phenome.jax.org/api/pheno/strainmeans/Yuan2?csv=yes" -o data/mpd/Yuan2_strainmeans.csv
curl "https://phenome.jax.org/api/pheno/animalvals/23401?csv=yes" -o data/mpd/Yuan2_animal_lifespandays.csv
curl "https://phenome.jax.org/api/pheno/measureinfo/Yuan2" -o data/mpd/Yuan2_measureinfo.json
```

## Ground Truth

MGI directional labels use strict Mammalian Phenotype terms:

| Label | MP terms |
| --- | --- |
| Increased | `MP:0001661` extended life span; `MP:0011614` slow aging |
| Decreased | `MP:0002083` premature death; `MP:0003786` premature aging |

Broader lifespan/aging terms are not treated as "not changed"; they are used
only for the ternary task's `Inconclusive` class. MPD regression and sex-effect
tasks come from Yuan2 lifespan measurements.

## Repository Layout

```text
build_benchmark.py          Build benchmark JSONL files
evaluate.py                 Run one or more model evaluations
score_traces.py             Score reasoning traces and plot saved results
longevity_benchmark/
  build.py                  Build orchestration
  tasks.py                  Task record builders
  mgi.py, mpd.py            Data loading and label extraction
  prompts.py                Prompt construction
  splits.py                 Leakage-aware splits
  eval/
    cli.py                  Evaluation CLI
    runner.py               Provider calls, output, metrics
    parsing.py              Output parsers
  reasoning_scorer.py       Offline biological reasoning scorer
```
