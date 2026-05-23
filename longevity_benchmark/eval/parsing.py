"""Model-output parsing utilities."""

from __future__ import annotations

import re


def strip_think(raw: str) -> tuple[str | None, str]:
    """Split <think>...</think> from the final answer."""
    match = re.search(r"(?:<think>)?(.*?)</think>\s*", raw, flags=re.DOTALL)
    if match and "</think>" in raw:
        return match.group(1).strip(), raw[match.end() :].strip()
    return None, raw.strip()


def normalize_effect_label(label: str) -> str:
    label = label.lower()
    if label.startswith("inc"):
        return "Increased"
    if label.startswith("dec"):
        return "Decreased"
    return label


def parse_effect(text: str) -> str:
    """Extract Increased / Decreased from model output."""
    matches = re.findall(r"\b(Increased|Decrease[d]?|Increase[d]?)\b", text, flags=re.I)
    if matches:
        return normalize_effect_label(matches[-1])
    return text.strip()


def parse_ternary(text: str) -> str:
    """Extract Increased / Decreased / Not changed from legacy ternary output."""
    low = text.lower()
    if "not changed" in low or "unchanged" in low or "not change" in low:
        return "Not changed"
    matches = re.findall(r"\b(Increased|Decrease[d]?|Increase[d]?)\b", text, flags=re.I)
    if matches:
        return normalize_effect_label(matches[-1])
    return text.strip()


def parse_pairwise(text: str) -> str:
    """Extract the final standalone A or B from model output."""
    matches = re.findall(r"(?<![A-Za-z])([AB])(?![A-Za-z])", text.upper())
    return matches[-1] if matches else text.strip()


PARSERS = {
    "binary": parse_effect,
    "effect": parse_effect,
    "ternary": parse_ternary,
    "pairwise": parse_pairwise,
}
