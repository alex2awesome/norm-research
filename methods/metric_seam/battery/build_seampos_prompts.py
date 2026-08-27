"""SEAM-POS (E5 aperture) prompt builder — pre-reg note §1.3, PR v2 certified 12.

Conditions per certified criterion:
  apdigest  — field question asked of a CODE-BUILT digest nu(x): first/last 3 sentences,
              doc stats, top-5 TF-IDF sentences vs criterion name+description (all E1 ops)
  aphead / apmid / aptail — field question asked of one document third (positional aperture)
  ccl       — LLM aggregator: criterion + code-extracted signals (3 flavor scores + the
              certified Gemma field values), no document; outputs SCORE: 0-10

-> outputs/metric_seam_pilot/v2/seampos_prompts.jsonl
"""
import importlib.util, json, math, pathlib, re
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parents[3]
BASE = ROOT / "outputs/metric_seam_pilot"
HYB = ROOT / "methods/metric_seam/hybrids/programs_v2"

CERTIFIED = ["a103", "a104", "a112", "a2", "a28", "a42",
             "a65", "a66", "a75", "a76", "a87", "a97"]

FIELD_T = """From the {view} below, {instruction}

<{tag}>
{body}
</{tag}>

Reply with ONE short line (max 20 words): the answer only, or NONE if absent."""

CCL_T = """You are scoring one document on a quality criterion WITHOUT seeing the document. \
You only have code-extracted signals about it.

Criterion: {name} — {description}

Signals:
{signals}

Based only on these signals, estimate the score an expert judge would give the document \
on this criterion. Reply with exactly one line: SCORE: <integer 0-10>"""

_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")
_TOK = re.compile(r"[a-z]{3,}")

STOP = set("the and for with that this from are was were has have had not you your our "
           "their its his her they them will would can could should about into over more "
           "than when where which while been being also only such other some any all".split())


def sentences(text):
    return [s.strip() for s in _SENT.split(text) if s.strip()]


def digest(text, query):
    sents = sentences(text)
    head, tail = sents[:3], sents[-3:] if len(sents) > 3 else []
    qtok = [t for t in _TOK.findall(query.lower()) if t not in STOP]
    # idf over this doc's sentences
    df = Counter()
    stoks = []
    for s in sents:
        toks = set(t for t in _TOK.findall(s.lower()) if t not in STOP)
        stoks.append(toks)
        for t in toks:
            df[t] += 1
    n = max(1, len(sents))
    scored = sorted(
        ((sum(math.log(n / df[t]) for t in set(qtok) & toks), i) for i, toks in enumerate(stoks)),
        reverse=True)
    top = [sents[i] for sc, i in scored[:5] if sc > 0]
    parts = ["[FIRST SENTENCES]"] + head
    if tail:
        parts += ["[LAST SENTENCES]"] + tail
    if top:
        parts += ["[SENTENCES MOST RELEVANT TO THE CRITERION (TF-IDF)]"] + top
    parts += ["[DOCUMENT STATS]",
              f"characters={len(text)} sentences={len(sents)} "
              f"paragraphs={text.count(chr(10)*2)+1} "
              f"quote_marks={text.count(chr(34))} "
              f"digits={sum(c.isdigit() for c in text)} "
              f"pct_uppercase={sum(c.isupper() for c in text)/max(1,len(text)):.3f}"]
    return "\n".join(parts)[:6000]


def thirds(text):
    sents = sentences(text)
    if len(sents) < 6:
        k = max(1, len(text) // 3)
        return text[:k], text[k:2 * k], text[2 * k:]
    a, b = len(sents) // 3, 2 * len(sents) // 3
    return " ".join(sents[:a]), " ".join(sents[a:b]), " ".join(sents[b:])


def main():
    items = json.load(open(BASE / "v1/items_v1.json"))
    aspects = {a["aspect_id"]: a for a in
               json.load(open(ROOT / "runs/validity_full/v2/press_releases/aspects.json"))}
    code = json.load(open(BASE / "v2/code_scores_v2.json"))
    fraw = {}
    for line in open(BASE / "v2/field_results_v2.jsonl"):
        r = json.loads(line)
        fraw[(r["aspect_id"], r["datapoint_id"])] = (r.get("raw") or "").strip()

    out = BASE / "v2/seampos_prompts.jsonl"
    n = 0
    with open(out, "w") as f:
        for aid in CERTIFIED:
            prog = HYB / f"{aid}_h0.py"
            spec = importlib.util.spec_from_file_location(prog.stem, prog)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            fields = dict(list((getattr(mod, "LLM_FIELDS", {}) or {}).items())[:2])
            a = aspects[aid]
            query = f"{a['name']}. {a['description']}"
            for it in items:
                d, text = it["datapoint_id"], it["ctext"]
                if fields:
                    dg = digest(text, query)
                    h, m, t3 = thirds(text)
                    views = {
                        "apdigest": ("structured code-extracted digest of a document",
                                     "document_digest", dg),
                        "aphead": ("excerpt below (the FIRST third of a document)",
                                   "document_excerpt", h[:8000]),
                        "apmid": ("excerpt below (the MIDDLE third of a document; you do "
                                  "not see its start or end)", "document_excerpt", m[:8000]),
                        "aptail": ("excerpt below (the LAST third of a document)",
                                   "document_excerpt", t3[:8000]),
                    }
                    for fn, ins in fields.items():
                        for cond, (view, tag, body) in views.items():
                            f.write(json.dumps({
                                "channel": "field",
                                "aspect_id": f"{aid}.{cond}__{fn}",
                                "datapoint_id": d,
                                "prompt": FIELD_T.format(view=view, instruction=ins,
                                                         tag=tag, body=body)}) + "\n")
                            n += 1
                sig = []
                for fl in ["v0_keyword", "v1_structure", "v2_holistic"]:
                    v = (code.get(f"{aid}_{fl}") or {}).get(d)
                    sig.append(f"code flavor {fl}: "
                               f"{'NA' if v is None else format(v, '.3f')}")
                for fn in fields:
                    rv = fraw.get((f"{aid}__{fn}", d), "")
                    sig.append(f"extracted field {fn}: {rv[:80] or 'NONE'}")
                f.write(json.dumps({
                    "channel": "ccl", "aspect_id": f"{aid}.ccl", "datapoint_id": d,
                    "prompt": CCL_T.format(name=a["name"], description=a["description"],
                                           signals="\n".join(sig))}) + "\n")
                n += 1
    print(f"wrote {n} prompts -> {out}")


if __name__ == "__main__":
    main()
