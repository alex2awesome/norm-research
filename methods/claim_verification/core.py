"""Claim-verification core: extract claims -> build evidence pool -> localize-then-verify ->
doc-level metrics. Backend = any OpenAI-compatible server (70B/Gemma) via guided JSON.

Doc-level metrics derived (the journalism V-bank):
  n_claims, claim_support_rate (FULL / all), claim_partial_rate,
  n_supported_full, evidence-type mix (frac data_statistic / named_source_quote /
  document_official / expert_attribution / assertion_only),
  n_named_sources, n_anon_sources, n_doc_sources, n_org_sources, self_ref_only.

Controls (controls.py): shuffled-evidence placebo + length stratification.
"""
import json, re, hashlib, os, urllib.request
from concurrent.futures import ThreadPoolExecutor
from .prompts import CLAIM_EXTRACT, LOCALIZE_VERIFY, SOURCE_CENSUS

def _post(base_url, model, prompt, max_tokens=700, temperature=0.0, timeout=120, schema=None):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": temperature}
    if schema is not None:
        body["response_format"] = {"type": "json_object"}
    req = urllib.request.Request(base_url.rstrip("/") + "/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read())
    return out["choices"][0]["message"]["content"]

def _parse_json(raw):
    m = re.search(r"\{[\s\S]*\}", raw or "")
    if not m: return None
    try: return json.loads(m.group(0))
    except Exception:
        try: return json.loads(m.group(0).replace("'", '"'))
        except Exception: return None

class Cache:
    def __init__(self, path):
        self.path = path; self.d = {}
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.exists(path):
            for ln in open(path):
                try: r = json.loads(ln); self.d[r["k"]] = r["v"]
                except Exception: pass
        self.f = open(path, "a")
    def get(self, k): return self.d.get(k)
    def put(self, k, v):
        self.d[k] = v
        self.f.write(json.dumps({"k": k, "v": v}) + "\n"); self.f.flush()

def _key(*parts):
    return hashlib.sha1("||".join(str(p)[:4000] for p in parts).encode()).hexdigest()

def _sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z\"'“(])", text) if len(s.strip()) > 20]

def split_head_body(text, head_paras=2):
    """Head = title/dateline + first ~2 paragraphs; body = the rest.
    Many PR texts are SINGLE-LINE (no newlines) -> fall back to sentence windows:
    head = first 3 sentences, body = 3-sentence windows over the rest."""
    paras = [p.strip() for p in re.split(r"\n\s*\n|\n(?=[A-Z])", text) if len(p.strip()) > 30]
    if len(paras) >= head_paras + 2:
        return "\n".join(paras[:head_paras]), paras[head_paras:]
    # newline-poor doc: sentence-window fallback
    sents = _sentences(text)
    if len(sents) <= 4:
        return (sents[0] if sents else text[:400]), sents[1:]
    head = " ".join(sents[:3])
    body = [" ".join(sents[i:i + 3]) for i in range(3, len(sents), 3)]
    return head, body

def make_passages(body_paras, max_passages=12, max_chars=600):
    ps = [p[:max_chars] for p in body_paras[:max_passages]]
    return ps

def extract_claims(head, cfg, cache):
    k = _key("claims", cfg["model"], head)
    hit = cache.get(k)
    if hit is not None: return hit
    raw = _post(cfg["base_url"], cfg["model"],
                CLAIM_EXTRACT.format(doc_kind=cfg["doc_kind"], head=head[:3000],
                                     max_claims=cfg.get("max_claims", 5)))
    obj = _parse_json(raw) or {}
    claims = [c for c in obj.get("claims", []) if isinstance(c, dict) and c.get("claim")][:cfg.get("max_claims", 5)]
    cache.put(k, claims)
    return claims

def verify_claim(claim, passages, cfg, cache):
    ptxt = "\n".join(f"[{i}] {p}" for i, p in enumerate(passages))
    k = _key("verify", cfg["model"], claim, ptxt[:2000])
    hit = cache.get(k)
    if hit is not None: return hit
    raw = _post(cfg["base_url"], cfg["model"],
                LOCALIZE_VERIFY.format(doc_kind=cfg["doc_kind"], claim=claim, passages=ptxt))
    obj = _parse_json(raw) or {}
    verdict = str(obj.get("verdict", "NONE")).upper()
    if verdict not in ("FULL", "PARTIAL", "NONE"): verdict = "NONE"
    # verbatim-grounding check (patents discipline): span must appear in a passage
    span = (obj.get("span") or "").strip()
    grounded = bool(span) and any(span[:120].lower() in p.lower() for p in passages)
    if verdict != "NONE" and not grounded:
        verdict = "PARTIAL" if verdict == "FULL" else verdict  # demote ungrounded FULL
        obj["ungrounded"] = True
    out = {"verdict": verdict, "evidence_type": obj.get("evidence_type"),
           "span": span[:200], "grounded": grounded, "reason": (obj.get("reason") or "")[:200]}
    cache.put(k, out)
    return out

def source_census(body, cfg, cache):
    k = _key("census", cfg["model"], body[:3000])
    hit = cache.get(k)
    if hit is not None: return hit
    raw = _post(cfg["base_url"], cfg["model"],
                SOURCE_CENSUS.format(doc_kind=cfg["doc_kind"], body=body[:6000]))
    obj = _parse_json(raw) or {}
    out = {kk: obj.get(kk) for kk in ("named_person_sources", "anonymous_sources",
                                      "document_sources", "organization_sources",
                                      "self_reference_only")}
    cache.put(k, out)
    return out

EVIDENCE_TYPES = ["data_statistic", "named_source_quote", "document_official",
                  "expert_attribution", "assertion_only"]

def doc_metrics(text, cfg, cache, evidence_pool=None):
    """Full pipeline for one document. evidence_pool overrides the body passages
    (used by the shuffled-evidence placebo)."""
    head, body_paras = split_head_body(text)
    passages = evidence_pool if evidence_pool is not None else make_passages(body_paras)
    claims = extract_claims(head, cfg, cache)
    verdicts = [verify_claim(c["claim"], passages, cfg, cache) for c in claims] if passages else []
    census = source_census("\n".join(body_paras)[:6000], cfg, cache)
    n = len(verdicts)
    full = sum(1 for v in verdicts if v["verdict"] == "FULL")
    part = sum(1 for v in verdicts if v["verdict"] == "PARTIAL")
    m = {"n_claims": float(n),
         "claim_support_rate": (full / n) if n else float("nan"),
         "claim_partial_rate": (part / n) if n else float("nan"),
         "n_supported_full": float(full)}
    for et in EVIDENCE_TYPES:
        m[f"ev_{et}"] = (sum(1 for v in verdicts if v.get("evidence_type") == et) / n) if n else float("nan")
    for kk in ("named_person_sources", "anonymous_sources", "document_sources", "organization_sources"):
        try: m[f"n_{kk}"] = float(census.get(kk) or 0)
        except Exception: m[f"n_{kk}"] = float("nan")
    m["self_ref_only"] = 1.0 if census.get("self_reference_only") else 0.0
    return m, {"claims": claims, "verdicts": verdicts, "census": census}

def run_docs(df, text_col, cfg, cache_path, workers=16, evidence_shuffle=False, seed=0):
    """Score a dataframe of documents. evidence_shuffle=True runs the placebo:
    each doc's claims verified against ANOTHER doc's passages (must collapse)."""
    import numpy as np
    cache = Cache(cache_path)
    texts = df[text_col].fillna("").tolist()
    pools = [None] * len(texts)
    if evidence_shuffle:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(texts))
        # derangement-ish: shift by 1 where fixed point
        perm = [(p if p != i else (p + 1) % len(texts)) for i, p in enumerate(perm)]
        pools = [make_passages(split_head_body(texts[j])[1]) for j in perm]
    def work(i):
        try:
            return doc_metrics(texts[i], cfg, cache, evidence_pool=pools[i])[0]
        except Exception as e:
            return {"error": str(e)[:100]}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(work, range(len(texts))))
