"""Evaluation runner for OpenAI-compatible chat endpoints."""

from __future__ import annotations

import json
import os
import time

try:
    from dotenv import load_dotenv
    load_dotenv()
except ModuleNotFoundError:
    pass
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from sklearn.metrics import accuracy_score, balanced_accuracy_score

from ..io import load_jsonl
from .parsing import PARSERS, strip_think


DEFAULT_ENDPOINT = "https://swchnq0ekc3scmqw.us-east-2.aws.endpoints.huggingface.cloud/v1"

PROVIDER_CONFIGS: dict[str, dict] = {
    "longevity": {
        "endpoint": DEFAULT_ENDPOINT,
        "default_model": "longevity-llm",
        "api_key_env": "HF_TOKEN",
    },
    "gemini": {
        "endpoint": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "default_model": "gemini-3.5-flash",
        "api_key_env": "GEMINI_API_KEY",
    },
    "claude": {
        "endpoint": "https://api.anthropic.com/v1/",
        "default_model": "claude-sonnet-4-6",
        "api_key_env": "ANTHROPIC_API_KEY",
    },
    "openai": {
        "endpoint": "https://api.openai.com/v1",
        "default_model": "gpt-5.5",
        "api_key_env": "OPENAI_API_KEY",
    },
}

WORKERS = 6  # Stay <= 8 per hackathon guidance.

TASK_FILE_PREFIX = {
    "effect": "mgi_mutation_lifespan_direction_binary",
    "mcq": "mgi_mutation_lifespan_effect_mcq",
    "ternary": "mgi_mutation_lifespan_effect_ternary_inconclusive",
    "set": "mgi_mutation_directional_lifespan_mp_terms_set_generation",
    "pairwise": "mgi_mutation_lifespan_longer_lived_pairwise",
    "regression": "mpd_strain_sex_median_lifespan_days_regression",
    "sex_effect": "mpd_strain_lifespan_sex_difference_ternary",
}


def resolve_model(provider: str, model: str | None = None) -> str:
    """Return the explicit model or the provider default without creating a client."""
    if provider not in PROVIDER_CONFIGS:
        raise ValueError(f"Unknown provider '{provider}'. Choose from: {list(PROVIDER_CONFIGS)}")
    return model or PROVIDER_CONFIGS[provider]["default_model"]


def model_slug(model: str) -> str:
    return model.replace("/", "-").replace(".", "-").replace(":", "-")


def eval_output_paths(
    eval_dir: str | Path,
    task: str,
    split: str,
    provider: str,
    model: str,
    enable_thinking: bool,
) -> tuple[Path, Path]:
    think_tag = "think" if enable_thinking else "nothink"
    file_tag = f"{provider}_{model_slug(model)}_{think_tag}"
    eval_path = Path(eval_dir)
    return (
        eval_path / f"results_{task}_{split}_{file_tag}.jsonl",
        eval_path / f"summary_{task}_{split}_{file_tag}.json",
    )


def _jsonl_row_count(path: Path) -> int:
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def _existing_complete_summary(
    summary_path: Path,
    log_path: Path,
    expected_n: int,
) -> dict | None:
    if not summary_path.exists() or not log_path.exists():
        return None
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary_n = int(summary.get("n", 0))
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    if summary_n < expected_n:
        return None
    try:
        if _jsonl_row_count(log_path) < expected_n:
            return None
    except OSError:
        return None
    return summary


def make_client(provider: str, model: str | None = None, endpoint: str | None = None) -> tuple:
    """Return (client, resolved_model_name) for the given provider."""
    try:
        import httpx
        from openai import OpenAI
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Install evaluation dependencies first: pip install openai httpx scikit-learn"
        ) from exc

    resolved_model = resolve_model(provider, model)
    cfg = PROVIDER_CONFIGS[provider]
    resolved_endpoint = endpoint or cfg["endpoint"]

    if provider == "longevity":
        api_key = os.environ.get("HF_TOKEN")
        if not api_key:
            raise RuntimeError("Set HF_TOKEN in your environment before running evaluation.")
        client = OpenAI(
            base_url=resolved_endpoint,
            api_key=api_key,
            http_client=httpx.Client(timeout=300),
        )

    elif provider == "gemini":
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Set GEMINI_API_KEY or GOOGLE_API_KEY in your environment before running evaluation."
            )
        client = OpenAI(
            base_url=resolved_endpoint,
            api_key=api_key,
            http_client=httpx.Client(timeout=300),
        )

    elif provider == "claude":
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Set ANTHROPIC_API_KEY in your environment before running evaluation."
            )
        client = OpenAI(
            base_url=resolved_endpoint,
            api_key=api_key,
            default_headers={"anthropic-version": "2023-06-01"},
            http_client=httpx.Client(timeout=300),
        )

    else:  # openai
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Set OPENAI_API_KEY in your environment before running evaluation."
            )
        client = OpenAI(
            base_url=resolved_endpoint,
            api_key=api_key,
            http_client=httpx.Client(timeout=300),
        )

    return client, resolved_model


def _with_visible_reasoning_request(messages: list[dict]) -> list[dict]:
    """Ask providers without native trace support to emit a scorable rationale."""
    augmented = [dict(message) for message in messages]
    instruction = (
        "\n\nFor this reasoning-scored run, override the final-answer-only output "
        "format with exactly two parts:\n"
        "<think>Brief evidence-grounded biological rationale. Mention the gene, "
        "allele/strain, relevant phenotype or pathway evidence, and why it supports "
        "the final answer.</think>\n"
        "Then output only the final answer in the exact format requested above."
    )
    augmented[-1]["content"] = str(augmented[-1].get("content", "")) + instruction
    return augmented


def call_model(row: dict, client, model: str, provider: str, enable_thinking: bool) -> dict:
    """Send one row to the model and return a scored result dict."""
    messages = row["messages"][:-1]
    if enable_thinking and provider != "longevity":
        messages = _with_visible_reasoning_request(messages)
    gold = row["messages"][-1]["content"].strip()
    fmt = row["format"]
    parser_key = row.get("parser", fmt)
    parser = PARSERS.get(parser_key)
    if parser is None:
        raise ValueError(f"No parser registered for format/parser: {fmt}/{parser_key}")

    # Longevity has provider-native thinking support. Other providers get a
    # visible <think>...</think> rationale prompt above so the scorer can read it.
    thinking_active = enable_thinking and provider == "longevity"
    if thinking_active:
        max_tokens = 2000
    elif enable_thinking:
        max_tokens = 2000
    elif provider == "openai":
        # OpenAI reasoning models count hidden reasoning tokens against the
        # completion budget, so keep enough room for a short visible answer.
        max_tokens = 2000
    elif provider == "gemini":
        # Gemini 2.x+ runs server-side reasoning that consumes the output
        # budget before any visible text; 200 tokens truncates mid-thought.
        max_tokens = 2000
    else:
        max_tokens = 200

    create_kwargs: dict = dict(
        model=model,
        messages=messages,
        temperature=0.0,
    )

    if provider == "openai":
        create_kwargs["max_completion_tokens"] = max_tokens
        if model == PROVIDER_CONFIGS["openai"]["default_model"]:
            create_kwargs["reasoning_effort"] = "none"
    else:
        create_kwargs["max_tokens"] = max_tokens

    if provider == "longevity":
        create_kwargs["extra_body"] = {
            "chat_template_kwargs": {"enable_thinking": enable_thinking}
        }
    elif provider == "gemini":
        create_kwargs["extra_body"] = {
            "extra_body": {
                "google": {"thinking_config": {"thinking_budget": 0}}
            }
        }
    start = time.time()
    response = client.chat.completions.create(**create_kwargs)
    elapsed = time.time() - start

    raw = response.choices[0].message.content or ""
    think, answer = strip_think(raw)

    # Some OpenAI-compatible endpoints may expose provider reasoning separately;
    # prefer it if present, otherwise use the visible <think> block.
    if provider == "claude" and enable_thinking:
        reasoning = getattr(response.choices[0].message, "reasoning_content", None)
        if reasoning:
            think = reasoning

    pred = parser(answer)

    return {
        "lb_id": row["lb_id"],
        "format": fmt,
        "provider": provider,
        "model": model,
        "gold": gold,
        "raw_response": raw,
        "think": think,
        "answer": answer,
        "pred": pred,
        "correct": pred == gold,
        "elapsed_s": round(elapsed, 2),
        "prompt_tokens": response.usage.prompt_tokens,
        "total_tokens": response.usage.total_tokens,
        "metadata": row.get("metadata"),
    }


def run_eval(
    rows: list[dict],
    client,
    model: str,
    provider: str,
    task: str,
    enable_thinking: bool,
    log_path: str | Path,
) -> list[dict]:
    """Run rows concurrently, stream results to log_path, and return results."""
    results = []
    n_errors = 0
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as log_file:
        with ThreadPoolExecutor(max_workers=WORKERS) as executor:
            futures = {
                executor.submit(call_model, row, client, model, provider, enable_thinking): row
                for row in rows
            }
            done = 0
            for future in as_completed(futures):
                done += 1
                try:
                    result = future.result()
                    results.append(result)
                    log_file.write(json.dumps(result, ensure_ascii=False) + "\n")
                    log_file.flush()
                except Exception as exc:  # noqa: BLE001
                    n_errors += 1
                    print(f"  [{done}/{len(rows)}] ERROR: {exc}")
                    continue
                print(_progress_line(done, len(rows), results, task, n_errors), end="\r", flush=True)

    print()
    return results


def _progress_line(
    done: int,
    total: int,
    results: list[dict],
    task: str,
    n_errors: int,
) -> str:
    if task == "regression":
        pairs = [
            (parse_float(str(result.get("gold", ""))), parse_float(str(result.get("pred", ""))))
            for result in results
        ]
        valid_pairs = [(gold, pred) for gold, pred in pairs if gold is not None and pred is not None]
        if valid_pairs:
            mae = sum(abs(gold - pred) for gold, pred in valid_pairs) / len(valid_pairs)
            metric_text = f"mae={mae:.1f}d  valid={len(valid_pairs)}/{len(results)}"
        else:
            metric_text = f"mae=n/a  valid=0/{len(results)}"
    elif task == "set":
        row_scores = [
            set_f1(
                parse_set_value(str(result.get("gold", ""))),
                parse_set_value(str(result.get("pred", ""))),
            )
            for result in results
        ]
        mean_f1 = sum(row_scores) / len(row_scores) if row_scores else 0.0
        exact = sum(result["correct"] for result in results)
        metric_text = f"set_f1={mean_f1:.1%}  exact={exact}/{len(results)}"
    else:
        correct = sum(result["correct"] for result in results)
        acc = correct / len(results)
        metric_text = f"acc={acc:.1%}  correct={correct}/{len(results)}"
    return f"  [{done}/{total}]  {metric_text}  api_errors={n_errors}"


def parse_set_value(value: str) -> set[str]:
    if not value:
        return set()
    return {item.strip() for item in value.split(",") if item.strip()}


def set_f1(gold: set[str], pred: set[str]) -> float:
    if not gold and not pred:
        return 1.0
    if not gold or not pred:
        return 0.0
    overlap = len(gold & pred)
    precision = overlap / len(pred)
    recall = overlap / len(gold)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def parse_float(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None


def _round_metric(value: float, digits: int = 4) -> float | None:
    return round(value, digits) if value == value else None


def report_metrics(results: list[dict], task: str) -> tuple[float, str, float]:
    """Print per-class breakdown and return score, metric name, baseline."""
    if not results:
        metric_by_task = {
            "pairwise": "accuracy",
            "mcq": "accuracy",
            "set": "mean_set_f1",
            "regression": "mae_days",
        }
        metric = metric_by_task.get(task, "balanced_accuracy")
        baseline_by_task = {
            "pairwise": 0.5,
            "mcq": 0.25,
            "set": 0.0,
            "regression": 0.0,
        }
        baseline = baseline_by_task.get(task, float("nan"))
        print(f"\n{'=' * 50}")
        print(f"Task: {task}  |  n=0")
        print("No successful model responses; metric is unavailable.")
        print(f"{'=' * 50}")
        return float("nan"), metric, baseline

    preds = [result["pred"] for result in results]
    golds = [result["gold"] for result in results]

    print(f"\n{'=' * 50}")
    print(f"Task: {task}  |  n={len(results)}")

    if task in {"pairwise", "mcq"}:
        score = accuracy_score(golds, preds)
        metric = "accuracy"
        baseline = 0.5
        if task == "mcq":
            baseline = 0.25
        print(f"Accuracy        : {score:.3f}")
        print(f"Random baseline : {baseline:.3f}")
        print(f"Pred dist       : {dict(Counter(preds))}")
    elif task == "set":
        row_scores = [
            set_f1(parse_set_value(gold), parse_set_value(pred))
            for gold, pred in zip(golds, preds)
        ]
        score = sum(row_scores) / len(row_scores) if row_scores else 0.0
        metric = "mean_set_f1"
        baseline = 0.0
        exact = sum(gold == pred for gold, pred in zip(golds, preds))
        print(f"Mean set F1     : {score:.3f}")
        exact_rate = exact / len(results) if results else 0.0
        print(f"Exact match     : {exact}/{len(results)} = {exact_rate:.2f}")
        print(f"Pred dist       : {dict(Counter(preds))}")
    elif task == "regression":
        pairs = [
            (parse_float(gold), parse_float(pred))
            for gold, pred in zip(golds, preds)
        ]
        valid_pairs = [(gold, pred) for gold, pred in pairs if gold is not None and pred is not None]
        errors = [abs(gold - pred) for gold, pred in valid_pairs]
        squared_errors = [(gold - pred) ** 2 for gold, pred in valid_pairs]
        score = sum(errors) / len(errors) if errors else float("nan")
        rmse = (sum(squared_errors) / len(squared_errors)) ** 0.5 if squared_errors else float("nan")
        metric = "mae_days"
        baseline = 0.0
        print(f"MAE days        : {score:.2f}")
        print(f"RMSE days       : {rmse:.2f}")
        print(f"Invalid preds   : {len(results) - len(valid_pairs)}/{len(results)}")
    else:
        score = balanced_accuracy_score(golds, preds)
        metric = "balanced_accuracy"
        labels = sorted(set(golds))
        baseline = 1 / len(labels)
        print(f"Balanced accuracy : {score:.3f}")
        print(f"Random baseline   : {baseline:.3f}")
        for label in labels:
            sub = [(pred, gold) for pred, gold in zip(preds, golds) if gold == label]
            correct = sum(pred == gold for pred, gold in sub)
            print(f"  {label:12s}: {correct}/{len(sub)} = {correct / len(sub):.2f}")

    print(f"{'=' * 50}")
    return score, metric, baseline


def evaluate_task(
    task: str,
    split: str,
    input_dir: str,
    eval_dir: str,
    client,
    model: str,
    provider: str,
    enable_thinking: bool,
    limit: int | None,
    score_traces: bool = False,
    skip_existing: bool = True,
) -> dict | None:
    data_path = Path(input_dir) / f"{TASK_FILE_PREFIX[task]}_{split}.jsonl"
    if not data_path.exists():
        print(f"File not found: {data_path} - skipping")
        return None

    rows = load_jsonl(data_path)
    if limit:
        rows = rows[:limit]

    log_path, summary_path = eval_output_paths(
        eval_dir,
        task,
        split,
        provider,
        model,
        enable_thinking,
    )

    if skip_existing:
        existing = _existing_complete_summary(summary_path, log_path, len(rows))
        if existing is not None:
            if score_traces and enable_thinking and existing.get("trace_quality") is None:
                from ..reasoning_scorer import annotate_results_inplace
                saved_results = load_jsonl(log_path)
                trace_summary = annotate_results_inplace(saved_results)
                with open(log_path, "w", encoding="utf-8") as log_file:
                    for result in saved_results:
                        log_file.write(json.dumps(result, ensure_ascii=False) + "\n")
                existing["trace_quality"] = trace_summary
                existing["results_path"] = str(log_path)
                with open(summary_path, "w", encoding="utf-8") as summary_file:
                    json.dump(existing, summary_file, indent=2)
                existing["status"] = "scored_existing"
                existing["summary_path"] = str(summary_path)
                print(
                    f"Scored existing traces for {task} | split={split} | "
                    f"provider={provider} | model={model} -> {summary_path}"
                )
                return existing
            existing["status"] = "skipped"
            existing["results_path"] = str(log_path)
            existing["summary_path"] = str(summary_path)
            print(
                f"Skipping {task} | split={split} | provider={provider} | model={model} "
                f"| existing n={existing.get('n')} -> {summary_path}"
            )
            return existing

    if client is None:
        raise RuntimeError(
            f"No client available for {provider}:{model}; cannot run missing evaluation."
        )

    print(f"\n{'=' * 50}")
    print(f"Running {task} | split={split} | provider={provider} | model={model} | n={len(rows)} | thinking={enable_thinking}")
    print(f"Logging -> {log_path}")
    print(f"{'=' * 50}")

    results = run_eval(rows, client, model, provider, task, enable_thinking, log_path)
    score, metric, baseline = report_metrics(results, task)

    summary = {
        "task": task,
        "split": split,
        "provider": provider,
        "model": model,
        "thinking": enable_thinking,
        "n": len(results),
        "score": _round_metric(score),
        "metric": metric,
        "baseline": _round_metric(baseline),
        "results_path": str(log_path),
    }

    # Optional reasoning-trace scoring (only useful when thinking was enabled).
    if score_traces:
        if not enable_thinking:
            print("Note: --score-traces was set but --think was off; skipping trace scoring.")
        else:
            from ..reasoning_scorer import annotate_results_inplace
            trace_summary = annotate_results_inplace(results)
            # Rewrite the log file with annotated rows so the per-row trace_score is persisted.
            with open(log_path, "w", encoding="utf-8") as log_file:
                for result in results:
                    log_file.write(json.dumps(result, ensure_ascii=False) + "\n")
            mean_ts = trace_summary["mean_trace_score"]
            print(
                f"  Trace quality: mean={mean_ts:.3f}  "
                f"fabricated_genes={trace_summary['fabricated_gene_mentions']}  "
                f"fabricated_alleles={trace_summary['fabricated_allele_mentions']}  "
                f"invalid_mp_ids={trace_summary['invalid_mp_ids']}"
                if mean_ts is not None else
                "  Trace quality: (no scorable traces)"
            )
            summary["trace_quality"] = trace_summary

    with open(summary_path, "w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2)
    print(f"Summary -> {summary_path}")
    summary["summary_path"] = str(summary_path)
    summary["status"] = "ran"
    return summary
