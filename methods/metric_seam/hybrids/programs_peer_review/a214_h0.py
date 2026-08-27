"""a214 hybrid: Methods, data/code transparency and reproducibility readiness.

Construct: ~1.0 = abstract states code AND data/models will be made available (ideally with a
URL); ~0.5 = partial (code OR data, or a vague "will release" with no URL); ~0.0 = no
reproducibility statement. This was the STRONGEST single signal in the peer-review V/A baseline
(best A-rubric AUC 0.77, best V feature `v_kw_code`), so it is the highest-value code-metric target.

INPUT = abstract only. Code sees: presence + validity of a repository URL (regex), the count of
named datasets, and whether a code/data release is explicitly claimed.
Code CANNOT see (tier-2/3): whether the URL resolves to a real repo with content (needs a fetch),
whether the released code actually reproduces the stated results (tier-3 exec = out-of-scope ceiling).
The URL-existence fetch is wired as a tier-2 hook (ops.url_exists if provided); absent it, h0 falls
back to URL-format validity only.
"""
import re

LLM_FIELDS = {
    "github_url": (
        "The GitHub or other repository URL stated in the abstract, verbatim. Answer NONE if no "
        "repository URL is stated."
    ),
    "dataset_names": (
        "Comma-separated datasets named in the abstract (e.g. ImageNet, GLUE, MATH, a custom "
        "clinical corpus). Answer NONE if no dataset is named."
    ),
    "release_claim": (
        "One word describing the code/models release statement: 'code' if code will be released, "
        "'data' if data/models will be released, 'both' if both, 'no' if explicitly unavailable, "
        "'unsaid' if the abstract says nothing."
    ),
}

_NONE = {"none", "n/a", "na", "not stated", "not mentioned", "unknown", "null", ""}
_URL_RE = re.compile(r"https?://(?:www\.)?(?:github|gitlab|bitbucket|huggingface|zenodo|figshare)\S+|github\.com/\S+", re.I)


def _split_names(val):
    if not isinstance(val, str) or val.strip().lower().strip(". ") in _NONE:
        return []
    return [p.strip() for p in re.split(r"[,;]", val) if p.strip() and p.strip().lower() not in _NONE]


def _url_in_text(url, text):
    return isinstance(url, str) and bool(url.strip()) and url.strip().lower() in (text or "").lower()


def _code_score(text, extracted, ops):
    t = text or ""
    url = extracted.get("github_url")
    # validity: URL must (a) parse as a repo URL and (b) actually appear in the abstract (grounding)
    url_valid = bool(_URL_RE.search(url)) if isinstance(url, str) else False
    url_grounded = _url_in_text(url, t) if url_valid else False
    # tier-2: does it resolve? (only if ops provides url_exists; else assume valid-format only)
    url_resolves = True
    if url_grounded and ops and hasattr(ops, "url_exists"):
        try:
            url_resolves = bool(ops.url_exists(url.strip()))
        except Exception:
            url_resolves = True

    n_data = len(_split_names(extracted.get("dataset_names")))
    rel = (extracted.get("release_claim") or "").strip().lower()

    # URL contributes up to 0.5 (format 0.3, grounded 0.1, resolves 0.1)
    url_part = 0.3 * (1 if url_valid else 0) + 0.1 * (1 if url_grounded else 0) + 0.1 * (1 if url_resolves else 0)
    # datasets up to 0.25
    data_part = min(0.25, 0.12 * n_data) if n_data else 0.0
    # release claim up to 0.25
    rel_part = {"both": 0.25, "code": 0.18, "data": 0.18, "no": 0.0, "unsaid": 0.0}.get(rel, 0.0)

    s = url_part + data_part + rel_part
    return max(0.0, min(1.0, s))


def _llm_score(extracted):
    rel = (extracted.get("release_claim") or "").strip().lower()
    return {"both": 0.9, "code": 0.65, "data": 0.6, "no": 0.05, "unsaid": 0.2}.get(rel, 0.2)


def score(text: str, extracted: dict, ops) -> float:
    try:
        t = ops.normalize(text) if (text and ops) else (text or "")
        return max(0.0, min(1.0, 0.65 * _code_score(t, extracted, ops) + 0.35 * _llm_score(extracted)))
    except Exception:
        return 0.5
