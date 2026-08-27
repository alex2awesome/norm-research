#!/usr/bin/env python
"""Same-family decompression authoring (z×a fam arms; user 2026-07-09: "asking larger models
within the same family how they fill in tacitness").

For --author in {llama70b, qwen25-72b}: author explanation (130-180w) + dossier (360-450w,
DEFINITION / WHAT COUNTS / CONTRAST EXEMPLARS / BOUNDARY CASES in order) for every slate
metric, matching the v1 Sonnet-agent gates byte-for-byte so build validation carries over.
Planted metrics must contain the rule sentence verbatim in BOTH texts. Output rows mirror
zxa_authoring/agent_*.json: {task, name, explanation, dossier, author}.

Offline batch vLLM (never an HTTP server). Resume-safe: skips metrics already valid on disk.
"""
import argparse, json, os, re, sys

B = "/lfs/skampere3/0/alexspan"
OM = f"{B}/outputs/osl_multi"
sys.path.insert(0, f"{B}/norm-research")

LABELS = ["DEFINITION", "WHAT COUNTS", "CONTRAST EXEMPLARS", "BOUNDARY CASES"]
TASK_DESC = {
    "humor": "judging short humorous texts (jokes, bits, comedic writing)",
    "creative_writing": "judging short fiction and creative prose",
    "peer_review": "judging scientific peer-review reports",
    "math": "judging mathematics Q&A answers (Math StackExchange style)",
}


def wc(s):
    return len(s.split())


def clean(o):
    o = re.sub(r"<think>.*?</think>", "", o, flags=re.S)
    o = o.strip()
    # drop a preamble line ("Here is the explanation:" etc.)
    lines = o.split("\n")
    if lines and re.match(r"^(here|sure|certainly|below)\b.*[:.]$", lines[0].strip(), re.I):
        o = "\n".join(lines[1:]).strip()
    o = re.sub(r"^\*\*(.+?)\*\*:?\s*$", r"\1", o, flags=re.M)  # unbold bare heading lines
    return o.strip()


def ok_expl(o, rule=None):
    o = clean(o)
    return 130 <= wc(o) <= 180 and (rule is None or rule in o)


def ok_doss(o, rule=None):
    o = clean(o)
    if not (360 <= wc(o) <= 450):
        return False
    pos = [o.find(L) for L in LABELS]
    if any(p < 0 for p in pos) or pos != sorted(pos):
        return False
    return rule is None or rule in o


def prompts_for(m):
    name, rubric, task = m["name"], m["rubric"], m["task"]
    planted = m["class"] == "PLANTED"
    rule = rubric.strip() if planted else None
    ctx = TASK_DESC[task]
    base = (f"You are writing evaluator guidance for the criterion \"{name}\" used when "
            f"{ctx}. The criterion's existing definition:\n\"{rubric}\"\n\n")
    verb = (f"\nCRITICAL: the following rule sentence must appear VERBATIM (character-for-"
            f"character) somewhere in your text:\n\"{rule}\"\n") if planted else ""
    p_expl = (base +
              "Write a single-paragraph EXPLANATION of this criterion in your own words: what "
              "it means, what to look for in a text, and how to decide yes vs no. Make it "
              "operational, not promotional. STRICT LENGTH: between 135 and 175 words — anything under 135 words will be REJECTED, so aim for about 155 words." + verb +
              "\nOutput ONLY the paragraph, no title, no preamble.")
    p_doss = (base +
              "Write a judging DOSSIER for this criterion with EXACTLY these four section "
              "labels, in this order, each label on its own line exactly as written:\n"
              "DEFINITION\nWHAT COUNTS\nCONTRAST EXEMPLARS\nBOUNDARY CASES\n\n"
              "DEFINITION: precise statement of the construct. WHAT COUNTS: concrete markers "
              "that satisfy it and common near-misses that do not. CONTRAST EXEMPLARS: 2-3 "
              "SHORT invented one-line examples that DO satisfy it and 2-3 that do NOT "
              "(clearly marked). BOUNDARY CASES: the hard calls and the tiebreak rule. "
              "STRICT TOTAL LENGTH: between 370 and 440 words across all sections — aim for about 405 words; under 370 will be REJECTED." + verb +
              "\nOutput ONLY the dossier text starting with the line DEFINITION, no preamble.")
    return p_expl, p_doss, rule


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--author", required=True, choices=["llama70b", "qwen25-72b"])
    ap.add_argument("--fake", action="store_true")
    a = ap.parse_args()
    slate = json.load(open(f"{OM}/zxa_slate_v1.json"))
    outdir = f"{OM}/zxa_authoring_fam"
    os.makedirs(outdir, exist_ok=True)
    out_fp = f"{outdir}/{a.author}.json"
    done = {}
    if os.path.exists(out_fp):
        for r in json.load(open(out_fp)):
            if ok_expl(r["explanation"], r.get("rule")) and ok_doss(r["dossier"], r.get("rule")):
                done[(r["task"], r["name"])] = r
    todo = [m for m in slate if (m["task"], m["name"]) not in done]
    print(f"[fam-author {a.author}] slate {len(slate)}, already valid {len(done)}, todo {len(todo)}")
    if not todo:
        return

    from methods.metric_implementer.experiments.osl_sweep import EXECUTORS
    from methods.metric_implementer import config as cfgmod
    from methods.metric_implementer.vllm_backend import make_judge_backend
    model, _, _ = EXECUTORS[a.author]
    cfg0 = cfgmod.ImplementerConfig()
    cfg0.max_retries = 6
    if a.fake:
        cfg0.vllm_fake = True
    ex = make_judge_backend(model, cfg0, temperature=0.7)

    pe, pd, rules = [], [], []
    for m in todo:
        p1, p2, rule = prompts_for(m)
        pe.append(p1); pd.append(p2); rules.append(rule)
    # per-index validators via closure trick: validate batches per unique rule group
    outs_e = [None] * len(todo)
    outs_d = [None] * len(todo)
    for is_doss, prompts, outs in ((False, pe, outs_e), (True, pd, outs_d)):
        # group indices by rule so a single validate closure applies to the whole subbatch
        by_rule = {}
        for i, r in enumerate(rules):
            by_rule.setdefault(r, []).append(i)
        for rule, idxs in by_rule.items():
            f = (lambda o, _r=rule: ok_doss(o, _r)) if is_doss else (lambda o, _r=rule: ok_expl(o, _r))
            res = ex.generate_batch([prompts[i] for i in idxs], system=None,
                                    max_tokens=1100 if is_doss else 450, validate=f)
            for j, i in enumerate(idxs):
                outs[i] = clean(res[j])

        # repair pass: word-count misses get targeted expand/trim seeded with own text
        # (fresh resampling alone fails: qwen25-72b lands ~110-125w vs 130w floor 64/72 times)
        for _round in range(2):
            bad_by_rule, rp = {}, {}
            for i in range(len(todo)):
                if outs[i] is None:
                    continue
                cur = clean(outs[i])
                if (ok_doss(cur, rules[i]) if is_doss else ok_expl(cur, rules[i])):
                    continue
                if is_doss:
                    pos = [cur.find(L) for L in LABELS]
                    if any(p < 0 for p in pos) or pos != sorted(pos):
                        continue  # structural miss: repair has no anchor, leave to rerun
                n = wc(cur)
                verb = (f"\nCRITICAL: the following rule sentence must appear VERBATIM "
                        f"(character-for-character) somewhere in your text:\n\"{rules[i]}\"\n"
                        ) if rules[i] else ""
                if is_doss:
                    tgt = ("expand it to between 380 and 430 words by adding concrete markers "
                           "in WHAT COUNTS and one more contrast exemplar per side"
                           if n < 360 else
                           "shorten it to between 380 and 430 words without dropping any section")
                    rp[i] = (f"The judging dossier below is {n} words, which is the wrong "
                             f"length. Rewrite it: {tgt}. Keep EXACTLY these four section "
                             "labels, in this order, each on its own line: DEFINITION, "
                             "WHAT COUNTS, CONTRAST EXEMPLARS, BOUNDARY CASES." + verb +
                             "\n\nDossier:\n" + cur +
                             "\n\nOutput ONLY the rewritten dossier starting with the line "
                             "DEFINITION, no preamble.")
                else:
                    tgt = ("expand it to between 140 and 170 words by adding concrete "
                           "operational detail: what markers to look for and how to decide "
                           "borderline cases. Keep the existing content"
                           if n < 130 else "shorten it to between 140 and 170 words")
                    rp[i] = (f"The evaluator explanation below is {n} words, which is the "
                             f"wrong length. Rewrite it: {tgt}." + verb +
                             "\n\nExplanation:\n" + cur +
                             "\n\nOutput ONLY the rewritten single paragraph, no title, "
                             "no preamble.")
                bad_by_rule.setdefault(rules[i], []).append(i)
            if not rp:
                break
            print(f"[fam-author] repair round {_round + 1} "
                  f"({'doss' if is_doss else 'expl'}): {len(rp)} rows")
            for rule, idxs in bad_by_rule.items():
                f = (lambda o, _r=rule: ok_doss(o, _r)) if is_doss else \
                    (lambda o, _r=rule: ok_expl(o, _r))
                res = ex.generate_batch([rp[i] for i in idxs], system=None,
                                        max_tokens=1100 if is_doss else 500, validate=f)
                for j, i in enumerate(idxs):
                    cand = clean(res[j])
                    if (ok_doss(cand, rule) if is_doss else ok_expl(cand, rule)):
                        outs[i] = cand

    rows = list(done.values())
    n_ok = 0
    for m, oe, od, rule in zip(todo, outs_e, outs_d, rules):
        valid = ok_expl(oe, rule) and ok_doss(od, rule)
        n_ok += valid
        rows.append({"task": m["task"], "name": m["name"], "explanation": oe,
                     "dossier": od, "author": a.author, "rule": rule, "valid": bool(valid)})
    json.dump(rows, open(out_fp, "w"), indent=1)
    print(f"[fam-author {a.author}] wrote {out_fp}: {len(rows)} rows, "
          f"{n_ok}/{len(todo)} new valid (invalid rows kept flagged for retry-on-rerun)")


if __name__ == "__main__":
    main()
