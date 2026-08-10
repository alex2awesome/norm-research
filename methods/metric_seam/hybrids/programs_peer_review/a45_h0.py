"""a45 hybrid: Dataset provenance, composition, and representativeness.

Construct: ~1.0 = abstract names datasets with source/provenance (e.g. "we curate X from Y");
~0.5 = datasets named but no provenance; ~0.0 = no dataset named. Most relevant for empirical/
data-driven papers; for theory papers no dataset is expected (handled by applicability).

INPUT = abstract only. Code sees: count of named datasets, whether provenance/source language
accompanies them, and (tier-2 catalog lookup, if ops.datasets_known is provided) whether the named
datasets are real/canonical. Code CANNOT see dataset composition/representativeness in depth from
an abstract — that needs the methods section (out of scope here).
"""
import re

LLM_FIELDS = {
    "dataset_names": (
        "Comma-separated datasets named in the abstract, each WITH its stated source/provenance "
        "if given (e.g. 'ImageNet (image benchmark)', 'a clinical corpus from 3 hospitals'). "
        "Answer NONE if no dataset is named."
    ),
}

_NONE = {"none", "n/a", "na", "not stated", "not mentioned", "unknown", "null", ""}
_PROVENANCE_REG = re.compile(r"\b(from|collected|curated|sourced|drawn from|based on|comprising|consisting of|public|benchmark|corpus|dataset)\b", re.I)


def _parse_datasets(val):
    """Each comma-separated item: (name, has_provenance)."""
    if not isinstance(val, str) or val.strip().lower().strip(". ") in _NONE:
        return []
    out = []
    for part in re.split(r"[,;]", val):
        p = part.strip()
        if not p or p.lower() in _NONE:
            continue
        has_prov = bool(_PROVENANCE_REG.search(p))
        # strip provenance clause to get the bare name for grounding
        name = re.split(r"\s*(?:from|collected|curated|sourced|drawn from|based on|comprising|consisting of)\b", p, maxsplit=1, flags=re.I)[0].strip()
        name = name.strip("()\"' ")
        out.append((name or p, has_prov))
    return out


def _code_score(text, extracted, ops):
    t = (text or "").lower()
    ds = _parse_datasets(extracted.get("dataset_names"))
    if not ds:
        return 0.0
    # grounding: name must appear in abstract (drop hallucinated)
    grounded = [(n, p) for (n, p) in ds if n.lower() in t]
    if not grounded:
        return 0.1
    n = len(grounded)
    n_prov = sum(1 for _, p in grounded if p)
    # tier-2: catalog lookup (how many named datasets are known/canonical)
    n_known = 0
    if ops and hasattr(ops, "datasets_known"):
        try:
            known = ops.datasets_known
            n_known = sum(1 for nm, _ in grounded if nm.lower() in known)
        except Exception:
            n_known = 0

    breadth = min(0.4, 0.2 * n)                      # up to 0.4 for >=2 datasets
    prov = min(0.3, 0.15 * n_prov)                    # up to 0.3 for provenance
    catalog = min(0.3, 0.15 * n_known)                # up to 0.3 for known datasets
    return max(0.0, min(1.0, breadth + prov + catalog))


def _llm_score(extracted):
    ds = _parse_datasets(extracted.get("dataset_names"))
    if not ds:
        return 0.1
    n = len(ds)
    n_prov = sum(1 for _, p in ds if p)
    return min(1.0, 0.5 * (1 if n else 0) + 0.1 * (n - 1) + 0.15 * n_prov)


def score(text: str, extracted: dict, ops) -> float:
    try:
        t = ops.normalize(text) if (text and ops) else (text or "")
        return max(0.0, min(1.0, 0.6 * _code_score(t, extracted, ops) + 0.4 * _llm_score(extracted)))
    except Exception:
        return 0.5
