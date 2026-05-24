"""
Programmatic scorer for LLM reasoning traces on the LongevityBench tasks.

The score is a weighted combination of five sub-scores that are cheap to
compute (no LLM-in-the-loop), hard to game by surface-level reformatting,
and grounded against the actual MGI/MP ontology databases rather than
regex heuristics:

  1. allele_validity    — Allele symbols cited in the trace (e.g. "Sirt6<tm1Caid>")
                          must exist in MGI's allele table.
  2. gene_validity      — Gene symbols cited in explicit gene contexts
                          ("the X gene", "X knockout", "X-deficient", "X<tm…>")
                          must be valid MGI markers.
  3. mp_term_validity   — Every MP: identifier cited must (a) exist in the
                          ontology and (b) match its claimed name within a
                          fuzzy token-overlap threshold.
  4. answer_consistency — The directional language at the END of the trace
                          must align with the final classification answer.
                          Catches cases where the trace reasons toward one
                          direction and emits the opposite token.
  5. prompt_grounding   — The dominant entities cited in the trace (gene,
                          allele, MP terms) must overlap with the entities
                          actually present in the prompt's metadata.

Missing dimensions are excluded from the average rather than scored 0, so
an MCQ trace that mentions no MP terms is not penalised for it. Each
sub-score is in [0, 1]. The aggregate is in [0, 1].
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

# DB loaders

_GENE_DB: set[str] | None = None
_GENE_DB_CI: set[str] | None = None  # case-insensitive lookup
_ALLELE_DB: set[str] | None = None
_MP_DB: dict[str, str] | None = None
_STRAIN_DB: set[str] | None = None


def load_gene_db(path: str | Path = "data/mgi/MGI_PhenotypicAllele.rpt") -> set[str]:
    """Load the MGI marker symbol set (column 8 of the allele report)."""
    global _GENE_DB, _GENE_DB_CI
    if _GENE_DB is not None:
        return _GENE_DB
    genes: set[str] = set()
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) >= 8 and cols[7]:
                genes.add(cols[7].strip())
    _GENE_DB = genes
    _GENE_DB_CI = {g.lower() for g in genes}
    return genes


def load_allele_db(path: str | Path = "data/mgi/MGI_PhenotypicAllele.rpt") -> set[str]:
    """Load the MGI allele symbol set (column 2)."""
    global _ALLELE_DB
    if _ALLELE_DB is not None:
        return _ALLELE_DB
    alleles: set[str] = set()
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) >= 2 and cols[1]:
                alleles.add(cols[1].strip())
    _ALLELE_DB = alleles
    return alleles


def load_mp_term_db(path: str | Path = "data/mgi/VOC_MammalianPhenotype.rpt") -> dict[str, str]:
    """Load MP ID → MP name."""
    global _MP_DB
    if _MP_DB is not None:
        return _MP_DB
    terms: dict[str, str] = {}
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            cols = line.rstrip("\n").split("\t")
            if len(cols) >= 2 and cols[0].startswith("MP:"):
                terms[cols[0].strip()] = cols[1].strip()
    _MP_DB = terms
    return terms


def load_strain_db(path: str | Path = "data/mpd/Yuan2_strainmeans.csv") -> set[str]:
    """Load the MPD strain names from Yuan2 (column 'strain')."""
    global _STRAIN_DB
    if _STRAIN_DB is not None:
        return _STRAIN_DB
    strains: set[str] = set()
    p = Path(path)
    if p.exists():
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)
            for row in reader:
                s = (row.get("strain") or "").strip()
                if s:
                    strains.add(s)
    _STRAIN_DB = strains
    return strains


# extractors

# Gene-context patterns: words that strongly imply the adjacent token is a gene symbol.
_GENE_BEFORE = re.compile(
    r"\b(?:gene|allele|mutation|knockout|knock-?out|null|deficient|deletion|"
    r"knock-?in|transgene|mutant|overexpress(?:ion|ing|ed)?|loss\s+of|"
    r"hypomorph(?:ic)?)\s+([A-Za-z][A-Za-z0-9-]{1,14})\b",
    re.IGNORECASE,
)
_GENE_AFTER = re.compile(
    r"\b([A-Za-z][A-Za-z0-9-]{1,14})\s+"
    r"(?:gene|allele|knockout|knock-?out|mutant|null|deficient|deletion|"
    r"knock-?in|transgene|hypomorph(?:ic)?)\b",
    re.IGNORECASE,
)
# Allele-notation like "Sirt6<tm1Caid>" or "Sirt6^tm1Caid"
_ALLELE_NOTATION = re.compile(
    r"\b([A-Za-z][A-Za-z0-9-]{1,14})(?:<[^>]+>|\^[A-Za-z0-9]+)"
)

_MP_TERM_RE = re.compile(r"MP:\d{7}", re.IGNORECASE)
_MP_TERM_WITH_NAME = re.compile(
    r"(MP:\d{7})\s*[:\-\(\s]*([a-z][^\.,;\n\)<>]*)",
    re.IGNORECASE,
)

_STRAIN_RE = re.compile(
    r"\b("
    r"C57BL/[0-9][A-Za-z]*|"     # C57BL/6J, C57BL/6N
    r"BALB/c[A-Za-z]*|"           # BALB/cJ
    r"DBA/[12][A-Za-z]*|"
    r"129[A-Za-z0-9/-]+|"         # 129S6, 129/Sv etc.
    r"FVB/[A-Za-z0-9]+|"
    r"CBA/[A-Za-z0-9]+|"
    r"NOD/[A-Za-z0-9]+"
    r")\b"
)

# Common English / biomedical capitalised tokens to never treat as gene candidates.
_TOKEN_STOPLIST = {
    # Biology abbreviations that look gene-like but aren't gene symbols
    "DNA", "RNA", "mRNA", "tRNA", "rRNA", "miRNA", "lncRNA", "snoRNA",
    "ATP", "GTP", "ADP", "AMP", "cAMP", "NAD", "NADH", "FAD",
    "MGI", "MP", "PCR", "qPCR", "ELISA", "WB", "IHC", "FACS",
    "WT", "KO", "Tg", "Cre", "loxP", "GFP", "RFP", "YFP", "BFP",
    "ChIP", "RIP", "CLIP", "GWAS", "QTL", "SNP", "CRISPR",
    "USA", "UK", "EU", "PhD", "MD", "AAV", "PBS",
    # Common English words that get caught by "<word> gene/allele/…" pattern
    "the", "a", "an", "this", "that", "these", "those", "its", "their",
    "his", "her", "our", "your", "my", "any", "all", "some", "each",
    "every", "no", "one", "two", "three", "four", "five",
    "is", "are", "was", "were", "be", "been", "being", "has", "have", "had",
    "do", "does", "did", "will", "would", "can", "could", "may", "might",
    "shall", "should", "must", "of", "for", "in", "on", "at", "to", "from",
    "by", "with", "as", "and", "or", "but", "not", "so", "if", "then",
    "than", "into", "onto", "out", "up", "down", "over", "under",
    # Biology nouns/verbs that commonly precede or follow "gene"/"allele"
    "gene", "genes", "allele", "alleles", "mutation", "mutations",
    "knockout", "knockouts", "mutant", "mutants", "deletion", "deletions",
    "null", "deficient", "deficiency", "transgene", "expression",
    "function", "functions", "encodes", "encoding", "encoded",
    "protein", "proteins", "phenotype", "phenotypes", "pathway", "pathways",
    "study", "studies", "research", "report", "reports", "data",
    "mice", "mouse", "animal", "animals", "model", "models",
    "result", "results", "effect", "effects", "evidence",
    "wild", "type", "control", "controls", "background",
    "increase", "decrease", "increased", "decreased", "increases", "decreases",
    "reduce", "reduced", "reduces", "elevate", "elevated",
    "show", "shows", "showed", "demonstrate", "demonstrates",
    "suggest", "suggests", "indicate", "indicates", "imply",
    "lifespan", "longevity", "survival", "aging", "ageing", "death",
    "given", "such", "same", "other", "another", "both", "either",
    "neither", "more", "most", "less", "least", "many", "much",
    "very", "well", "only", "also", "however", "therefore",
}
_TOKEN_STOPLIST_LC = {t.lower() for t in _TOKEN_STOPLIST}

# Trigger words that indicate the trace is making a directional claim near the end.
_DIRECTION_INCREASE = re.compile(
    r"(?:increased?|extended?|longer|prolong(?:ed|s)?|enhanced?|"
    r"slow(?:ed|er)?\s+aging|longevity)\b",
    re.IGNORECASE,
)
_DIRECTION_DECREASE = re.compile(
    r"(?:decreased?|shorter|reduced?|premature\s+(?:death|aging)|"
    r"shortened?|accelerat(?:ed|es)\s+aging|early\s+(?:death|mortality)|"
    r"lethal(?:ity)?)\b",
    re.IGNORECASE,
)


# Point-substitution mutation notation like "A315T" / "P301S" / "R175H"
_POINT_MUTATION_RE = re.compile(r"^[A-Z]\d{1,5}[A-Z]$")


def _looks_gene_shaped(tok: str) -> bool:
    """Heuristic filter: only let through tokens shaped like real gene symbols."""
    if len(tok) < 2:
        return False
    if _POINT_MUTATION_RE.match(tok):
        return False          # rules out A315T, P301S, …
    has_digit = any(c.isdigit() for c in tok)
    has_upper = any(c.isupper() for c in tok)
    has_lower = any(c.islower() for c in tok)
    # mouse style: first uppercase + at least one lowercase  (Bub1b, Sirt6, Foxo3)
    # human style: all uppercase, length 2-7                (SIRT6, TP53)
    # plus anything containing a digit is gene-likely        (p53, Tert)
    if has_digit:
        return True
    if has_upper and has_lower and tok[0].isupper():
        return True
    if tok.isupper() and 2 <= len(tok) <= 7:
        return True
    return False


def extract_gene_mentions(trace: str) -> list[str]:
    """Return gene-symbol-shaped tokens cited in an explicit gene context."""
    hits = []
    for pat in (_GENE_BEFORE, _GENE_AFTER, _ALLELE_NOTATION):
        for m in pat.finditer(trace):
            tok = m.group(1)
            if tok.lower() in _TOKEN_STOPLIST_LC:
                continue
            if not _looks_gene_shaped(tok):
                continue
            hits.append(tok)
    return hits


def extract_mp_terms(trace: str) -> list[tuple[str, str]]:
    """Return (mp_id, claimed_name_near_it) pairs. claimed_name may be ''. """
    pairs = []
    seen = set()
    for m in _MP_TERM_WITH_NAME.finditer(trace):
        mp_id = m.group(1).upper()
        claimed = (m.group(2) or "").strip().lower()
        if mp_id in seen:
            continue
        seen.add(mp_id)
        pairs.append((mp_id, claimed))
    # MP IDs that appear without an adjacent name
    for m in _MP_TERM_RE.finditer(trace):
        mp_id = m.group(0).upper()
        if mp_id not in seen:
            seen.add(mp_id)
            pairs.append((mp_id, ""))
    return pairs


def extract_strain_mentions(trace: str) -> list[str]:
    return [m.group(1) for m in _STRAIN_RE.finditer(trace)]


def extract_allele_notation(trace: str) -> list[str]:
    """Return full allele-shaped tokens like 'Sirt6<tm1Caid>'."""
    pat = re.compile(r"\b[A-Za-z][A-Za-z0-9-]{1,14}<[^>]+>")
    return [m.group(0) for m in pat.finditer(trace)]


# per-dimension scoring

def _safe_ratio(num: int, den: int) -> float:
    return num / den if den else float("nan")


def _token_overlap(a: str, b: str) -> float:
    """Jaccard overlap of lowercase content tokens between two short strings."""
    if not a or not b:
        return 0.0
    ta = set(re.findall(r"[a-z]+", a.lower()))
    tb = set(re.findall(r"[a-z]+", b.lower()))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def score_gene_validity(trace: str, gene_db: set[str]) -> tuple[float | None, dict]:
    mentions = extract_gene_mentions(trace)
    if not mentions:
        return None, {"n": 0, "valid": [], "invalid": []}
    db_ci = {g.lower() for g in gene_db}
    valid, invalid = [], []
    for tok in mentions:
        if tok.lower() in db_ci or tok in gene_db:
            valid.append(tok)
        else:
            invalid.append(tok)
    return _safe_ratio(len(valid), len(mentions)), {
        "n": len(mentions),
        "valid": valid,
        "invalid": invalid,
    }


def score_allele_validity(trace: str, allele_db: set[str]) -> tuple[float | None, dict]:
    mentions = extract_allele_notation(trace)
    if not mentions:
        return None, {"n": 0, "valid": [], "invalid": []}
    valid = [a for a in mentions if a in allele_db]
    invalid = [a for a in mentions if a not in allele_db]
    return _safe_ratio(len(valid), len(mentions)), {
        "n": len(mentions),
        "valid": valid,
        "invalid": invalid,
    }


def score_mp_validity(trace: str, mp_db: dict[str, str]) -> tuple[float | None, dict]:
    pairs = extract_mp_terms(trace)
    if not pairs:
        return None, {"n": 0, "details": []}
    details = []
    score_acc = 0.0
    for mp_id, claimed in pairs:
        entry = {"id": mp_id, "claimed_name": claimed}
        if mp_id not in mp_db:
            entry["status"] = "invalid_id"
            entry["score"] = 0.0
        elif not claimed:
            entry["status"] = "id_only"
            entry["true_name"] = mp_db[mp_id]
            entry["score"] = 1.0  # ID is valid; no name to mis-state
        else:
            true_name = mp_db[mp_id].lower()
            overlap = _token_overlap(claimed, true_name)
            entry["status"] = "named"
            entry["true_name"] = true_name
            entry["token_overlap"] = round(overlap, 3)
            entry["score"] = 1.0 if overlap >= 0.4 else overlap
        score_acc += entry["score"]
        details.append(entry)
    return score_acc / len(pairs), {"n": len(pairs), "details": details}


def score_answer_consistency(
    trace: str, answer: str, task: str
) -> tuple[float | None, dict]:
    """Compare directional cues in the trace's tail with the final answer."""
    if task not in {"effect", "ternary", "mcq", "pairwise"}:
        return None, {"reason": "task_not_directional"}
    tail = trace[-600:] if trace else ""
    if not tail.strip():
        return None, {"reason": "empty_trace"}
    n_inc = len(_DIRECTION_INCREASE.findall(tail))
    n_dec = len(_DIRECTION_DECREASE.findall(tail))
    ans = answer.strip().lower()

    if task == "effect":
        if ans.startswith("inc"):
            implied = "increase"
        elif ans.startswith("dec"):
            implied = "decrease"
        else:
            return None, {"reason": f"unrecognised_answer:{answer}"}
    elif task == "ternary":
        if ans.startswith("inc"):
            implied = "increase"
        elif ans.startswith("dec"):
            implied = "decrease"
        else:
            implied = "neutral"
    elif task == "mcq":
        if ans.upper().startswith("A"):
            implied = "increase"
        elif ans.upper().startswith("B"):
            implied = "decrease"
        else:
            implied = "neutral"
    else:  # pairwise
        return None, {"reason": "pairwise_uses_AB"}

    if implied == "neutral":
        # No directional commitment — give credit only if neither direction dominates.
        score = 1.0 if abs(n_inc - n_dec) <= 1 else 0.5
    elif implied == "increase":
        if n_inc == 0 and n_dec == 0:
            return None, {"reason": "no_directional_cues"}
        score = n_inc / (n_inc + n_dec)
    else:  # decrease
        if n_inc == 0 and n_dec == 0:
            return None, {"reason": "no_directional_cues"}
        score = n_dec / (n_inc + n_dec)

    return score, {"implied": implied, "n_inc": n_inc, "n_dec": n_dec}


def score_prompt_grounding(
    trace: str, prompt_text: str, metadata: dict | str | None
) -> tuple[float | None, dict]:
    """Fraction of explicit entities in the trace that also appear in the prompt context."""
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except Exception:
            metadata = {}
    metadata = metadata or {}

    prompt_lower = (prompt_text or "").lower()
    expected_gene = (metadata.get("gene") or "").lower()
    expected_allele = (metadata.get("allele") or "").lower()
    expected_bg = (metadata.get("genetic_background") or "").lower()

    gene_mentions = [g.lower() for g in extract_gene_mentions(trace)]
    allele_mentions = [a.lower() for a in extract_allele_notation(trace)]
    strain_mentions = [s.lower() for s in extract_strain_mentions(trace)]

    total = 0
    grounded = 0
    for g in gene_mentions:
        total += 1
        if g == expected_gene or g in prompt_lower:
            grounded += 1
    for a in allele_mentions:
        total += 1
        if a in expected_allele or a in prompt_lower:
            grounded += 1
    for s in strain_mentions:
        total += 1
        if s in expected_bg or s in prompt_lower:
            grounded += 1

    if total == 0:
        return None, {"n": 0}
    return grounded / total, {"n": total, "grounded": grounded}


# aggregate

WEIGHTS = {
    "allele_validity":    0.20,
    "gene_validity":      0.25,
    "mp_validity":        0.20,
    "answer_consistency": 0.20,
    "prompt_grounding":   0.15,
}


@dataclass
class TraceScore:
    lb_id: str
    task: str
    correct: bool
    answer: str
    aggregate: float | None
    subscores: dict = field(default_factory=dict)
    details:   dict = field(default_factory=dict)


def score_one(
    result: dict,
    gene_db:   set[str],
    allele_db: set[str],
    mp_db:     dict[str, str],
) -> TraceScore:
    trace = (result.get("think") or "").strip()
    answer = result.get("answer") or result.get("pred") or ""
    task = result.get("format") or result.get("task", "")
    # Map benchmark "format" → scoring task
    fmt = (result.get("format") or "").lower()
    if fmt in {"binary"}:
        task = "effect"
    elif fmt == "ternary":
        task = "ternary"
    elif fmt == "mcq":
        task = "mcq"
    elif fmt == "pairwise":
        task = "pairwise"

    prompt_text = ""
    # The prompt_text is not stored on the result; pass via metadata.row_text if available.
    prompt_text = result.get("prompt_text") or ""

    g_score, g_det = score_gene_validity(trace, gene_db)
    a_score, a_det = score_allele_validity(trace, allele_db)
    m_score, m_det = score_mp_validity(trace, mp_db)
    c_score, c_det = score_answer_consistency(trace, answer, task)
    p_score, p_det = score_prompt_grounding(trace, prompt_text, result.get("metadata"))

    subscores = {
        "gene_validity":      g_score,
        "allele_validity":    a_score,
        "mp_validity":        m_score,
        "answer_consistency": c_score,
        "prompt_grounding":   p_score,
    }
    weighted_sum = 0.0
    weight_total = 0.0
    for name, val in subscores.items():
        if val is None:
            continue
        w = WEIGHTS[name]
        weighted_sum += val * w
        weight_total += w
    aggregate = (weighted_sum / weight_total) if weight_total > 0 else None

    return TraceScore(
        lb_id=result.get("lb_id", ""),
        task=task,
        correct=bool(result.get("correct", False)),
        answer=answer,
        aggregate=aggregate,
        subscores=subscores,
        details={
            "gene_validity":      g_det,
            "allele_validity":    a_det,
            "mp_validity":        m_det,
            "answer_consistency": c_det,
            "prompt_grounding":   p_det,
            "trace_len_chars":    len(trace),
        },
    )


def score_results_file(
    results_path: str | Path,
    gene_db: set[str] | None = None,
    allele_db: set[str] | None = None,
    mp_db: dict[str, str] | None = None,
) -> list[TraceScore]:
    """Score every row of a runner.py-emitted results JSONL."""
    gene_db   = gene_db   if gene_db   is not None else load_gene_db()
    allele_db = allele_db if allele_db is not None else load_allele_db()
    mp_db     = mp_db     if mp_db     is not None else load_mp_term_db()
    scores: list[TraceScore] = []
    with open(results_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                result = json.loads(line)
            except json.JSONDecodeError:
                continue
            scores.append(score_one(result, gene_db, allele_db, mp_db))
    return scores


def annotate_results_inplace(
    results: list[dict],
    gene_db: set[str] | None = None,
    allele_db: set[str] | None = None,
    mp_db: dict[str, str] | None = None,
) -> dict:
    """
    Score each result in-place: every dict gets `trace_score` (aggregate) and
    `trace_subscores`. Returns a small summary dict with mean/std/coverage and
    fabrication counts that the caller can attach to its summary JSON.
    """
    if not results:
        return {"n": 0, "rows_with_trace": 0}

    gene_db   = gene_db   if gene_db   is not None else load_gene_db()
    allele_db = allele_db if allele_db is not None else load_allele_db()
    mp_db     = mp_db     if mp_db     is not None else load_mp_term_db()

    rows_with_trace = 0
    agg_vals: list[float] = []
    subscore_vals: dict[str, list[float]] = {
        "gene_validity": [], "allele_validity": [], "mp_validity": [],
        "answer_consistency": [], "prompt_grounding": [],
    }
    n_invalid_gene = n_invalid_allele = n_invalid_mp = 0

    for r in results:
        ts = score_one(r, gene_db, allele_db, mp_db)
        r["trace_score"] = ts.aggregate
        r["trace_subscores"] = ts.subscores
        if ts.details.get("trace_len_chars", 0) > 0:
            rows_with_trace += 1
        if ts.aggregate is not None:
            agg_vals.append(ts.aggregate)
        for k, v in ts.subscores.items():
            if v is not None:
                subscore_vals[k].append(v)
        gv = ts.details.get("gene_validity") or {}
        av = ts.details.get("allele_validity") or {}
        mv = ts.details.get("mp_validity") or {}
        n_invalid_gene += len(gv.get("invalid", []))
        n_invalid_allele += len(av.get("invalid", []))
        n_invalid_mp += sum(
            1 for d in mv.get("details", []) if d.get("status") == "invalid_id"
        )

    n = len(results)
    summary = {
        "n": n,
        "rows_with_trace": rows_with_trace,
        "mean_trace_score": (sum(agg_vals) / len(agg_vals)) if agg_vals else None,
        "subscore_means": {
            k: (sum(v) / len(v)) if v else None for k, v in subscore_vals.items()
        },
        "subscore_coverage": {k: len(v) / n for k, v in subscore_vals.items()},
        "fabricated_gene_mentions": n_invalid_gene,
        "fabricated_allele_mentions": n_invalid_allele,
        "invalid_mp_ids": n_invalid_mp,
    }
    return summary
