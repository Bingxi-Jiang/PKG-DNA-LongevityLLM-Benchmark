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
    """Extract Increased / Decreased / Inconclusive from model output."""
    low = text.lower()
    inconclusive_hits = [
        "inconclusive",
        "cannot determine",
        "can't determine",
        "uncertain",
        "non-directional",
        "nondirectional",
        "conflicting",
        "not changed",
        "unchanged",
        "not change",
    ]
    if any(hit in low for hit in inconclusive_hits):
        return "Inconclusive"
    matches = re.findall(r"\b(Increased|Decrease[d]?|Increase[d]?)\b", text, flags=re.I)
    if matches:
        return normalize_effect_label(matches[-1])
    return text.strip()


def parse_pairwise(text: str) -> str:
    """Extract the final standalone A or B from model output."""
    matches = re.findall(r"(?<![A-Za-z])([AB])(?![A-Za-z])", text.upper())
    return matches[-1] if matches else text.strip()


def parse_mcq(text: str) -> str:
    """Extract a multiple-choice option A-D."""
    matches = re.findall(r"(?<![A-Za-z])([ABCD])(?![A-Za-z])", text.upper())
    if matches:
        return matches[-1]

    low = text.lower()
    if "increase" in low or "longer" in low or "extended" in low or "slow aging" in low:
        return "A"
    if "decrease" in low or "shorter" in low or "premature" in low:
        return "B"
    if "inconclusive" in low or "non-directional" in low or "conflicting" in low:
        return "C"
    if "no curated" in low or "no lifespan" in low or "none" in low:
        return "D"
    return text.strip()


def parse_set_generation(text: str) -> str:
    """Extract a canonical comma-separated set of MP IDs."""
    if re.search(r"\b(None|No candidate|No terms?)\b", text, flags=re.I):
        return ""
    matches = re.findall(r"MP:\d{7}", text.upper())
    return ",".join(sorted(set(matches))) if matches else text.strip()


PARSERS = {
    "binary": parse_effect,
    "effect": parse_effect,
    "ternary": parse_ternary,
    "pairwise": parse_pairwise,
    "mcq": parse_mcq,
    "set_generation": parse_set_generation,
    "set": parse_set_generation,
}
