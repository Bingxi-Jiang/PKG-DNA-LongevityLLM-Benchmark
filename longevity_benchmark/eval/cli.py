"""CLI entry point for evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

from .runner import PROVIDER_CONFIGS, evaluate_task, make_client, resolve_model


ALL_TASKS = ["effect", "mcq", "ternary", "set", "pairwise", "regression", "sex_effect"]


def _expand_providers(values: list[str] | None, parser: argparse.ArgumentParser) -> list[str] | None:
    if values is None:
        return None
    if "all" in values:
        if len(values) > 1:
            parser.error("--providers all cannot be combined with other provider names.")
        return list(PROVIDER_CONFIGS)
    return values


def _dedupe_specs(specs: list[tuple[str, str | None]]) -> list[tuple[str, str | None]]:
    seen = set()
    deduped = []
    for provider, model in specs:
        key = (provider, model)
        if key not in seen:
            seen.add(key)
            deduped.append(key)
    return deduped


def _parse_model_specs(args: argparse.Namespace, parser: argparse.ArgumentParser) -> list[tuple[str, str | None]]:
    providers = _expand_providers(args.providers, parser)
    if args.model and args.models:
        parser.error("Use either --model or --models, not both.")

    if args.models:
        specs = []
        fallback_provider = None
        bare_specs = [spec for spec in args.models if ":" not in spec]
        if providers and bare_specs:
            if len(providers) != 1:
                parser.error("Bare --models entries require exactly one --provider/--providers value.")
            fallback_provider = providers[0]
        else:
            fallback_provider = args.provider

        for raw_spec in args.models:
            if ":" in raw_spec:
                provider, model = raw_spec.split(":", 1)
                if not provider or not model:
                    parser.error(f"Invalid --models entry: {raw_spec!r}. Use provider:model.")
            else:
                provider, model = fallback_provider, raw_spec
            if provider not in PROVIDER_CONFIGS:
                parser.error(
                    f"Unknown provider {provider!r}. Choose from: {', '.join(PROVIDER_CONFIGS)}."
                )
            specs.append((provider, model))
        return _dedupe_specs(specs)

    if providers:
        if args.model and len(providers) > 1:
            parser.error("--model can only be used with one provider. Use --models provider:model ... instead.")
        return _dedupe_specs([(provider, args.model) for provider in providers])

    return [(args.provider, args.model)]


def _print_batch_summary(summaries: list[dict]) -> None:
    if not summaries:
        return
    print("\nEvaluation batch summary")
    print(f"  {'task':<12} {'model':<28} {'metric':<20} {'value':>9} {'direction':>9} {'n':>6} {'status':>9}")
    for summary in summaries:
        model = str(summary.get("model", ""))
        if len(model) > 28:
            model = model[:25] + "..."
        metric = str(summary.get("metric", ""))
        score = summary.get("score")
        score_text = f"{score:.4f}" if isinstance(score, (int, float)) else str(score)
        direction = "lower" if metric == "mae_days" else "higher"
        print(
            f"  {str(summary.get('task', '')):<12} "
            f"{model:<28} "
            f"{metric:<20} "
            f"{score_text:>9} "
            f"{direction:>9} "
            f"{int(summary.get('n', 0)):>6} "
            f"{str(summary.get('status', '')):>9}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run LongevityBench evaluation for one or more models."
    )
    parser.add_argument(
        "--task",
        choices=[*ALL_TASKS, "all"],
        default="all",
    )
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument(
        "--think",
        action="store_true",
        help=(
            "Emit a scorable reasoning trace. Longevity uses native thinking; "
            "other providers are prompted to produce a visible <think>...</think> rationale."
        ),
    )
    parser.add_argument(
        "--score-traces", action="store_true",
        help="After evaluation, score each reasoning trace against MGI/MP databases "
             "(gene/allele/MP-term validity, answer consistency, prompt grounding). "
             "Requires --think to produce traces; the per-row JSONL and summary JSON "
             "are extended with trace_score / trace_subscores.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Cap the number of rows per task (useful for quick smoke tests).")
    parser.add_argument(
        "--provider",
        choices=list(PROVIDER_CONFIGS),
        default="longevity",
        help=(
            "Single-provider mode: "
            + ", ".join(PROVIDER_CONFIGS)
            + ". Default: longevity."
        ),
    )
    parser.add_argument(
        "--providers",
        nargs="+",
        choices=[*list(PROVIDER_CONFIGS), "all"],
        default=None,
        help="Evaluate multiple providers with their default model. Use 'all' for every provider.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model override for single-provider mode.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        metavar="PROVIDER:MODEL",
        help=(
            "Evaluate explicit models, e.g. openai:gpt-5.5 claude:claude-sonnet-4-6. "
            "Bare model names use --provider when exactly one provider is selected."
        ),
    )
    parser.add_argument(
        "--endpoint",
        default=None,
        help="Override the provider's default API endpoint URL. Only valid for one model spec.",
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Re-run even when a complete results/summary pair already exists.",
    )
    parser.add_argument("--input-dir", default="output")
    parser.add_argument("--eval-dir", default="output/eval")
    args = parser.parse_args()

    model_specs = _parse_model_specs(args, parser)
    if args.endpoint and len(model_specs) != 1:
        parser.error("--endpoint can only be used when evaluating one provider/model.")

    Path(args.eval_dir).mkdir(parents=True, exist_ok=True)

    tasks = ALL_TASKS if args.task == "all" else [args.task]
    summaries: list[dict] = []
    client_cache = {}

    print("Default models: " + ", ".join(
        f"{provider}={cfg['default_model']}" for provider, cfg in PROVIDER_CONFIGS.items()
    ))

    for provider, model_override in model_specs:
        model = resolve_model(provider, model_override)
        client = None
        client_unavailable = False

        for task in tasks:
            try:
                summary = evaluate_task(
                    task=task,
                    split=args.split,
                    input_dir=args.input_dir,
                    eval_dir=args.eval_dir,
                    client=client,
                    model=model,
                    provider=provider,
                    enable_thinking=args.think,
                    limit=args.limit,
                    score_traces=args.score_traces,
                    skip_existing=not args.rerun,
                )
            except RuntimeError as exc:
                if "No client available" not in str(exc) or client_unavailable:
                    raise
                cache_key = (provider, args.endpoint)
                if cache_key not in client_cache:
                    try:
                        client_cache[cache_key], _ = make_client(provider, model, args.endpoint)
                    except RuntimeError as client_exc:
                        print(f"Skipping missing runs for {provider}:{model} - {client_exc}")
                        client_unavailable = True
                        break
                client = client_cache[cache_key]
                summary = evaluate_task(
                    task=task,
                    split=args.split,
                    input_dir=args.input_dir,
                    eval_dir=args.eval_dir,
                    client=client,
                    model=model,
                    provider=provider,
                    enable_thinking=args.think,
                    limit=args.limit,
                    score_traces=args.score_traces,
                    skip_existing=not args.rerun,
                )
            if summary is not None:
                summaries.append(summary)

    _print_batch_summary(summaries)


if __name__ == "__main__":
    main()
