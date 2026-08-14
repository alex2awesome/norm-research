"""GLM-5.2 matched prompt-judge trial (2026-08-14, user-directed, runs on sk3 CPU + z.ai).

Question: does RECONSTRUCTION-SELECTING the prompt form FOR the GLM judge lift mention-AUC
above the July judge-swap grid (grand AUC: 8B .574 / g4 .568 / 70B .573 / GLM .529 —
glm_g1_summary.json), or is the sparse mention label the binding ceiling?

Design (peer-review, the grid task):
  A. For each trial metric, score ALL candidate forms (form_idx 1..11 from
     peer_forms_manifest) with GLM over the first N_PROBES probe texts, plus form 0
     (the canonical rubric) as the metric's own-verdict target M_i.
  B. Per metric, select argmax_form i_binary(median-binarized form sig, median-binarized
     M_i sig) — the form whose GLM behavior best reconstructs the metric's GLM verdict.
     (Form 0 excluded from candidacy: it IS the target.)
  C. Score each selected form on the full y-covered corpus with GLM; AUC vs positive-only
     mention y. Compare against the grid's GLM row (same docs, form 0, glm_g1 results).

Prompts replicate the glm_g1 template verbatim (SCORE: N, 0-10). API via
api_field_runner_patient.py (zai_anthropic, resumable). Subcommands:
  build_a | select | build_c | analyze
"""
import json
import re
import sys
from pathlib import Path

MD = Path("/lfs/skampere3/0/alexspan/mention_auc")
N_PROBES = 100
MODEL_TAG = "glm52"

TEMPLATE = """Text:
{text}

You are evaluating the above text on ONE specific criterion.
Criterion:
{rubric}

How well does the text satisfy the criterion? Reply with exactly "SCORE: N" where N is an integer 0-10 (0 = not at all, 10 = fully)."""


def trial_metrics():
    g = json.load(open(MD / "glm_g1_metrics.json"))
    mids = list(dict.fromkeys(list(g.get("top8", [])) + list(g.get("controls", []))))
    return [m.split("__")[0] if "__" in m else m for m in mids]


def forms_by_metric():
    man = json.load(open(MD / "peer_forms_manifest.json"))
    out = {}
    for e in man:
        out.setdefault(e["metric_id"], {})[e["form_idx"]] = e["rubric"]
    return out


def probes():
    rows = [json.loads(l) for l in open(MD / "peer_probe_texts.jsonl")]
    return rows[:N_PROBES]


def build_a():
    mids, fbm, prb = trial_metrics(), forms_by_metric(), probes()
    n = 0
    with open(MD / f"{MODEL_TAG}_trial_a_prompts.jsonl", "w") as f:
        for mid in mids:
            forms = fbm.get(mid, {})
            for fi, rubric in sorted(forms.items()):        # 0 = M_i target, 1.. = candidates
                for r in prb:
                    did = r.get("probe_id") or r.get("id")
                    f.write(json.dumps({"channel": "formsig", "aspect_id": f"{mid}__{fi}",
                                        "datapoint_id": did,
                                        "prompt": TEMPLATE.format(text=r["text"][:6000],
                                                                  rubric=rubric)}) + "\n")
                    n += 1
    print(f"phase A: {n} prompts, {len(mids)} metrics -> {MODEL_TAG}_trial_a_prompts.jsonl")


def _scores(path):
    out = {}
    for line in open(path):
        try:
            r = json.loads(line)
        except Exception:
            continue
        s = r.get("score")
        if s is None:
            m = re.search(r"SCORE:\s*(\d+)", str(r.get("raw", "")))
            s = float(m.group(1)) if m else None
        if s is not None:
            out.setdefault(r["aspect_id"], {})[r["datapoint_id"]] = float(s)
    return out


def _ibin(sig, tgt):
    import numpy as np
    ids = sorted(set(sig) & set(tgt))
    if len(ids) < 30:
        return None
    a = np.array([sig[i] for i in ids]); b = np.array([tgt[i] for i in ids])
    ab = (a >= np.median(a)).astype(int); bb = (b >= np.median(b)).astype(int)
    p = np.zeros((2, 2))
    for x, y in zip(ab, bb):
        p[x, y] += 1
    p /= p.sum()
    mi = 0.0
    import math
    for x in (0, 1):
        for y in (0, 1):
            if p[x, y] > 0:
                mi += p[x, y] * math.log2(p[x, y] / (p[x].sum() * p[:, y].sum()))
    return mi


def select():
    S = _scores(MD / f"{MODEL_TAG}_trial_a_results.jsonl")
    sel = {}
    for mid in trial_metrics():
        tgt = S.get(f"{mid}__0")
        if not tgt:
            print(f"{mid}: no M_i target scores, skipped"); continue
        best = None
        for aid, sig in S.items():
            if not aid.startswith(f"{mid}__") or aid.endswith("__0"):
                continue
            mi = _ibin(sig, tgt)
            if mi is not None and (best is None or mi > best[1]):
                best = (aid, mi)
        if best:
            sel[mid] = {"form": best[0], "i_binary": round(best[1], 4)}
            print(mid, "->", best[0], "i_binary", round(best[1], 4))
    json.dump(sel, open(MD / f"{MODEL_TAG}_trial_selected.json", "w"), indent=1)


def build_c():
    sel = json.load(open(MD / f"{MODEL_TAG}_trial_selected.json"))
    fbm = forms_by_metric()
    y = json.load(open(MD / "peer_y_pos.json"))
    texts = {}
    for line in open(MD / "peer_paper_texts.jsonl"):
        r = json.loads(line)
        texts[r.get("paper_id") or r.get("id")] = r["text"]
    p8 = json.load(open(MD / "peer_p_scores.json"))
    ids = [i for i in p8["post_ids"] if i in texts]
    n = 0
    with open(MD / f"{MODEL_TAG}_trial_c_prompts.jsonl", "w") as f:
        for mid, s in sel.items():
            fi = int(s["form"].split("__")[1])
            rubric = fbm[mid][fi]
            for did in ids:
                f.write(json.dumps({"channel": "corpus", "aspect_id": s["form"],
                                    "datapoint_id": did,
                                    "prompt": TEMPLATE.format(text=texts[did][:6000],
                                                              rubric=rubric)}) + "\n")
                n += 1
    print(f"phase C: {n} prompts over {len(ids)} docs x {len(sel)} selected forms")


def analyze():
    import numpy as np

    def auc(y, p):
        o = np.argsort(p); r = np.empty(len(p)); r[o] = np.arange(1, len(p) + 1)
        n1 = y.sum(); n0 = len(y) - n1
        return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else None

    S = _scores(MD / f"{MODEL_TAG}_trial_c_results.jsonl")
    ypos = json.load(open(MD / "peer_y_pos.json"))
    out = {}
    for aid, sc in S.items():
        mid = aid.split("__")[0]
        ids = sorted(sc)
        p = np.array([sc[i] for i in ids])
        yv = np.array([1 if mid in set(ypos.get(i, [])) else 0 for i in ids])
        if yv.sum() < 10:
            continue
        out[mid] = {"form": aid, "n": len(ids), "n_pos": int(yv.sum()),
                    "auc_matched_glm": round(auc(yv, p), 4)}
    json.dump(out, open(MD / f"{MODEL_TAG}_trial_result.json", "w"), indent=1)
    if out:
        import statistics as st
        print("matched-GLM mention-AUC: n=%d median %.3f mean %.3f"
              % (len(out), st.median(v["auc_matched_glm"] for v in out.values()),
                 st.mean(v["auc_matched_glm"] for v in out.values())))
        print("grid comparison (glm_g1_summary grand_auc): 8B .574 / g4 .568 / 70B .573 / GLM .529")


if __name__ == "__main__":
    {"build_a": build_a, "select": select, "build_c": build_c,
     "analyze": analyze}[sys.argv[1]]()
