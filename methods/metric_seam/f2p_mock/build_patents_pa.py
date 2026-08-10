"""Build the patents_pa seam task: applications that ALREADY have prior-art measurements.

Items = 250 applications (balanced on final outcome, seed 0) drawn from the ~8.3k apps present
in BOTH patents_final_outcome_cpc_balanced_with_rejections_with_retrievals_v3.csv.gz (document
text) and option3_claims_gemma_scale.jsonl (per-claim K=8 retrieved refs + Gemma disclosure
verdicts). NOTHING is retrieved or verified here — the precomputed disclosure records become
the evidence op's payload (pa_features.json). All apps in this pool got a §102 rejection at
some point (option3 construction), so `judgement` = FINAL outcome after prosecution; it goes
into items.json meta ONLY (anchor — never into pa_features). Gold/rejection fields (label,
rejection_type, gold_docs, is_gold, gold_disclose, rejected_*) are stripped from the payload.

Emits ONE prompts.jsonl: 8 judge aspects x 2 passes + scope + LLM fields for the 4 prior-art
hybrids in programs_pa/ (channel="field", aspect_id="<aid>__<field>").

Usage (sk3): /lfs/skampere3/0/alexspan/miniconda3/bin/python build_patents_pa.py
"""
import importlib.util, json, pathlib, random

R = pathlib.Path("/lfs/skampere3/0/alexspan/norm-research")
D = R / "datasets/patents"
OUT = R / "outputs/metric_seam_pilot/tasks/patents_pa"
PROGS = pathlib.Path(__file__).parent / "programs_pa"
HEAD, TAIL, N_ITEMS = 5000, 2500, 250

ASPECTS = ["a26", "a34", "a60", "a35", "a22", "a25", "a36", "a16"]
ROLE = ("an experienced patent examiner", "patent application document")

T1 = """You are {role} grading ONE {doctype} against ONE criterion.

Criterion: {name}
Definition: {description}

Document (may be truncated; '[...]' marks an elided middle):
<document>
{text}
</document>

Grade how well the document satisfies this criterion on a 0-10 integer scale \
(0 = completely fails, 10 = exemplary). If the criterion genuinely does not apply to this \
kind of document, answer NA. Reply with exactly one line in the form:
SCORE: <integer 0-10 or NA>"""

T2 = """As {role}, evaluate the document below on a single quality criterion, ignoring all \
other qualities.

The criterion — {name}: {description}

Document ('[...]' = elided middle):
<document>
{text}
</document>

Give an integer 0-10 (0 = criterion badly violated or absent, 10 = criterion fully \
exemplified). If this criterion simply cannot apply to a document of this kind, answer NA. \
Your entire reply must be one line:
SCORE: <integer 0-10 or NA>"""

TSCOPE = """Look at the following text scraped from the web.

<text>
{text}
</text>

Is this a genuine, substantive {doctype} (as opposed to navigation chrome, an empty stub, \
an unrelated page, or a different document type)? Reply with exactly one line:
SCORE: <integer 0-10>  (0 = clearly NOT, 10 = clearly a substantive {doctype})"""

TFIELD = """From the document below, {instruction}

<document>
{text}
</document>

Reply with ONE short line (max 20 words): the answer only, or NONE if absent."""


def canonical(text):
    if len(text) <= HEAD + TAIL + 500:
        return text
    return text[:HEAD] + "\n[...]\n" + text[-TAIL:]


def load_option3():
    """app_id -> payload (gold/label fields STRIPPED).

    KNOWN CAVEATS (Codex audit 2026-07-10; behavior unchanged to keep existing WS3 artifacts
    reproducible — apply on any REBUILD):
    - "claims" are option3 ROWS: ~12.2% exact-duplicate claim-element rows inflate n_claims and
      weight the mean/frac summaries by extraction multiplicity — dedup on (claim_num, element).
    - candidate sets were built with the examiner-cited gold doc force-included for targeted
      claims; stripping is_gold does not make candidate construction label-free (oracle-injection
      caveat in datasets/patents/2026-07-10__evidence_aware_judge_ws3.md).
    """
    apps = {}
    for line in open(D / "processed/option3_claims_gemma_scale.jsonl"):
        r = json.loads(line)
        a = apps.setdefault(str(r["app_id"]), {"claims": []})
        refs = [{"doc_id": q.get("doc_id", ""), "discloses": bool(q.get("discloses")),
                 "vreason": (q.get("vreason") or "")[:200],
                 "span_head": ((q.get("spans") or [""])[0] or "")[:300]}
                for q in (r.get("refs") or [])]
        a["claims"].append({"claim_num": r.get("claim_num"),
                            "element_head": (r.get("element") or "")[:300],
                            "n_refs": int(r.get("n_refs") or len(refs)),
                            "n_disclose": int(r.get("n_disclose") or 0),
                            "refs": refs})
    for a in apps.values():
        fracs = [c["n_disclose"] / c["n_refs"] for c in a["claims"] if c["n_refs"]]
        a["n_claims"] = len(a["claims"])
        a["frac_claims_any_disclose"] = round(
            sum(1 for c in a["claims"] if c["n_disclose"] > 0) / max(1, len(a["claims"])), 4)
        a["mean_frac_disclose"] = round(sum(fracs) / max(1, len(fracs)), 4)
        a["max_frac_disclose"] = round(max(fracs) if fracs else 0.0, 4)
    return apps


def main():
    import pandas as pd
    OUT.mkdir(parents=True, exist_ok=True)
    o3 = load_option3()
    print(f"option3 apps: {len(o3)}")

    cols = ["text", "app_id", "judgement"] + [f"top{i}_score" for i in range(1, 6)]
    rows = {}
    for chunk in pd.read_csv(
            D / "patents_final_outcome_cpc_balanced_with_rejections_with_retrievals_v3.csv.gz",
            usecols=cols, chunksize=50000):
        m = chunk[chunk["app_id"].astype(str).isin(o3.keys())]
        for r in m.itertuples(index=False):
            a = str(r.app_id)
            if a not in rows and isinstance(r.text, str) and len(r.text) >= 600:
                rows[a] = r
    print(f"pool apps with text: {len(rows)}")

    rng = random.Random(0)
    by_lab = {0: [], 1: []}
    for a in sorted(rows):
        by_lab[int(rows[a].judgement)].append(a)
    picked = []
    for lab in (0, 1):
        rng.shuffle(by_lab[lab])
        picked += by_lab[lab][:N_ITEMS // 2]

    items, feats = [], {}
    for a in picked:
        r = rows[a]
        dpid = f"pa{a}"
        items.append({"datapoint_id": dpid, "text": r.text, "ctext": canonical(r.text),
                      "app_id": a, "judgement": int(r.judgement)})
        pay = dict(o3[a])
        scores = [getattr(r, f"top{i}_score") for i in range(1, 6)]
        pay["retrieval_top_scores"] = [round(float(s), 4) for s in scores
                                       if s == s and s is not None]
        feats[dpid] = pay
    json.dump(items, open(OUT / "items.json", "w"))
    json.dump(feats, open(OUT / "pa_features.json", "w"))
    json.dump(ASPECTS, open(OUT / "aspects_used.json", "w"))
    print(f"items: {len(items)}  (judgement balance "
          f"{sum(x['judgement'] for x in items)}/{len(items)})")

    fields = {}
    for prog in sorted(PROGS.glob("*_h0.py")):
        spec = importlib.util.spec_from_file_location(prog.stem, prog)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        fields[prog.stem.split("_")[0]] = dict(mod.LLM_FIELDS)
    print("field sets:", {k: list(v) for k, v in fields.items()})

    aspects = {x["aspect_id"]: x for x in
               json.load(open(R / "runs/validity_full/v2/patents/aspects.json"))}
    role, doctype = ROLE
    n = 0
    with open(OUT / "prompts.jsonl", "w") as f:
        def emit(ch, aid, dpid, prompt):
            nonlocal n
            f.write(json.dumps({"channel": ch, "aspect_id": aid,
                                "datapoint_id": dpid, "prompt": prompt}) + "\n")
            n += 1
        for aid in ASPECTS:
            asp = aspects[aid]
            for it in items:
                emit("pass1", aid, it["datapoint_id"],
                     T1.format(role=role, doctype=doctype, name=asp["name"],
                               description=asp["description"], text=it["ctext"]))
                emit("pass2", aid, it["datapoint_id"],
                     T2.format(role=role, doctype=doctype, name=asp["name"],
                               description=asp["description"], text=it["ctext"]))
        for it in items:
            emit("scope", "scope", it["datapoint_id"],
                 TSCOPE.format(doctype=doctype, text=it["ctext"]))
        for aid, fset in fields.items():
            for fname, instr in fset.items():
                for it in items:
                    emit("field", f"{aid}__{fname}", it["datapoint_id"],
                         TFIELD.format(instruction=instr, text=it["ctext"]))
    print(f"{n} prompts -> {OUT / 'prompts.jsonl'}")


if __name__ == "__main__":
    main()
