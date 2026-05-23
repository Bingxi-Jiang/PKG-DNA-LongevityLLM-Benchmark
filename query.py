#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys
import urllib.request
import urllib.error

try:
    from dotenv import load_dotenv
    load_dotenv()
except ModuleNotFoundError:
    pass

HF_TOKEN = os.environ.get("HF_TOKEN", "")
ENDPOINT = "https://swchnq0ekc3scmqw.us-east-2.aws.endpoints.huggingface.cloud"


def post(path: str, payload: dict, timeout: float) -> str:
    req = urllib.request.Request(
        ENDPOINT.rstrip("/") + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {HF_TOKEN}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        sys.stderr.write(f"HTTP {e.code} on {path}: {e.read().decode('utf-8', errors='replace')}\n")
        sys.exit(1)


def ask(query: str, max_new_tokens: int, temperature: float, raw: bool, mode: str, think: bool, out, json_out: bool, timeout: float) -> None:
    if mode == "chat":
        path = "/v1/chat/completions"
        payload = {
            "model": "longevity-llm",
            "messages": [{"role": "user", "content": query}],
            "max_tokens": max_new_tokens,
            "temperature": temperature,
            "chat_template_kwargs": {"enable_thinking": think},
        }
    elif mode == "completions":
        path = "/v1/completions"
        payload = {
            "model": "longevity-llm",
            "prompt": query,
            "max_tokens": max_new_tokens,
            "temperature": temperature,
        }
    elif mode == "generate":
        path = "/generate"
        payload = {
            "inputs": query,
            "parameters": {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "return_full_text": False,
            },
        }
    else:
        sys.stderr.write(f"Unknown mode: {mode}\n")
        sys.exit(2)

    body = post(path, payload, timeout)

    if raw:
        print(body, file=out)
        return

    data = json.loads(body)
    reasoning = None
    finish_reason = None
    usage = data.get("usage") if isinstance(data, dict) else None
    model_name = data.get("model") if isinstance(data, dict) else None
    if mode == "chat":
        choice = data["choices"][0]
        finish_reason = choice.get("finish_reason")
        msg = choice["message"]
        content = msg.get("content") or ""
        reasoning = msg.get("reasoning_content")
        if reasoning is None:
            m = re.search(r"(?:<think>)?(.*?)</think>", content, flags=re.DOTALL)
            if m and "</think>" in content:
                reasoning = m.group(1).strip()
                content = content[m.end():].lstrip()
    elif mode == "completions":
        choice = data["choices"][0]
        finish_reason = choice.get("finish_reason")
        content = choice["text"]
    elif mode == "generate":
        if isinstance(data, list) and data and "generated_text" in data[0]:
            content = data[0]["generated_text"]
            finish_reason = data[0].get("details", {}).get("finish_reason")
        else:
            content = json.dumps(data, indent=2)

    if json_out:
        record = {
            "query": query,
            "think": reasoning,
            "response": content,
            "finish_reason": finish_reason,
            "model": model_name,
            "usage": usage,
        }
        json.dump(record, out, indent=2, ensure_ascii=False)
        out.write("\n")
    else:
        if reasoning:
            print("=== THINKING ===", file=out)
            print(reasoning, file=out)
            print("=== ANSWER ===", file=out)
        print(content, file=out)
        if finish_reason or usage:
            print("", file=out)
            print(f"[finish_reason={finish_reason}  usage={usage}]", file=out)


def main() -> None:
    p = argparse.ArgumentParser(description="Send a query to the HF Inference Endpoint.")
    p.add_argument("-q", "--query", required=True, help="Prompt to send.")
    p.add_argument("-n", "--max-new-tokens", type=int, default=256)
    p.add_argument("-t", "--temperature", type=float, default=0.7)
    p.add_argument(
        "-m", "--mode",
        choices=["chat", "completions", "generate"],
        default="chat",
        help="API shape. Default: chat (OpenAI-compatible /v1/chat/completions).",
    )
    p.add_argument("--raw", action="store_true", help="Print raw JSON response.")
    p.add_argument(
        "--timeout", type=float, default=900.0,
        help="HTTP read timeout in seconds (default: 900).",
    )
    p.add_argument(
        "-o", "--output",
        help="Write output to this file instead of stdout.",
    )
    p.add_argument(
        "-a", "--append", action="store_true",
        help="With -o, append to the file instead of overwriting.",
    )
    p.add_argument(
        "--think", dest="think", action="store_true",
        help="Enable thinking mode (chat mode only). Default: off.",
    )
    p.add_argument("--no-think", dest="think", action="store_false")
    p.set_defaults(think=False)
    args = p.parse_args()

    if not HF_TOKEN:
        sys.stderr.write("HF_TOKEN not set. Add it to .env or export it in your shell.\n")
        sys.exit(2)

    json_out = bool(args.output) and not args.raw
    if args.output:
        with open(args.output, "a" if args.append else "w") as f:
            ask(args.query, args.max_new_tokens, args.temperature, args.raw, args.mode, args.think, f, json_out, args.timeout)
    else:
        ask(args.query, args.max_new_tokens, args.temperature, args.raw, args.mode, args.think, sys.stdout, False, args.timeout)


if __name__ == "__main__":
    main()
