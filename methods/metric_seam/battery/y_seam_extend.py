"""y-prediction seam: retarget the code<->language seam from the LLM JUDGE (m) to the REAL
binary OUTCOME y (items.json 'judgement'). Unlike the m-seam, y is ground truth (no judge-
reliability ceiling) and the readout is AUC on a binary label, not Spearman.

  G-tier (language->y): one Gemma prompt per doc predicting the favorable outcome, AUC vs y.
  trivial baseline   : char-length AUC vs y (the classic length confound floor).
  (V-tier code->y and dense ceiling added separately.)

  python3 y_seam_extend.py build <task>            -> y_seam_<task>_prompts.jsonl (TEST rows)
  python3 y_seam_extend.py eval  <task> <results>  -> prints AUC_G vs y, char-len AUC, seam gap
"""
import json, sys, pathlib

HERE = pathlib.Path(__file__).resolve().parent
ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "methods/metric_seam/battery"))
import battery_common as bc
for t, pd in {"legal_title_vii": "programs_legal", "peer_review": "programs_peer",
              "patents_pa": "programs_pa", "code_review": "programs_cr"}.items():
    bc.PROGDIR.setdefault(t, pd)
BASE = ROOT / "outputs/metric_seam_pilot"
OUT = HERE / "y_seam"; OUT.mkdir(exist_ok=True)

# task -> (domain noun, favorable-outcome question). y=1 is the favorable class per items.json.
TASKS = {
    "legal_title_vii": ("U.S. Title VII employment-discrimination court opinion",
                        "the court will rule in FAVOR OF THE PLAINTIFF (find discrimination / deny the defense)"),
    "peer_review": ("academic paper (with its peer reviews)",
                    "this paper will be ACCEPTED at the venue"),
    "patents_pa": ("patent application",
                   "this application will be GRANTED (not abandoned/rejected)"),
    "code_review": ("GitHub pull request (with its code review)",
                    "this pull request will be MERGED (not closed unmerged)"),
}
MARK = "<<<DOCUMENT>>>"; FOOT = "\n\nReply with exactly one line: SCORE: <integer 0-10>"
TMPL = ("You are forecasting the real-world outcome of a single {noun}.\n\n"
        "Question: How likely is it that {q}?\n\nDocument:\n{m}\n\n"
        "Give your probability as an integer from 0 (certain the favorable outcome will NOT happen) "
        "to 10 (certain it WILL). Base it only on the document.")


def items_with_y(task):
    """datapoint_id -> (ctext, judgement) from the task items.json."""
    d = "v2" if task == "press_releases" else f"tasks/{task}"
    lst = json.load(open(BASE / d / "items.json"))
    return {x["datapoint_id"]: (x.get("ctext", x.get("text", "")), x.get("judgement")) for x in lst}


def auc(scores, labels):
    """Rank AUC (Mann-Whitney); labels binary 0/1."""
    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = [s for s, l in zip(scores, labels) if l == 0]
    if not pos or not neg:
        return float("nan"), 0, 0
    # rank-sum
    allv = sorted(scores)
    import bisect
    from collections import Counter
    cnt = Counter(scores)
    ranks = {}
    i = 0
    for v in sorted(cnt):
        n = cnt[v]; avg = (i + 1 + i + n) / 2.0
        ranks[v] = avg; i += n
    r_pos = sum(ranks[s] for s in pos)
    u = r_pos - len(pos) * (len(pos) + 1) / 2.0
    return u / (len(pos) * len(neg)), len(pos), len(neg)


def cmd_build(task):
    ctx = bc.load_ctx(task)
    iy = items_with_y(task)
    noun, q = TASKS[task]
    test = sorted(ctx["test"])
    outp = OUT / f"y_seam_{task}_prompts.jsonl"
    body = TMPL.format(noun=noun, q=q, m=MARK)
    n = 0
    with open(outp, "w") as f:
        for dp in test:
            txt = iy.get(dp, ("", None))[0][:6000]
            f.write(json.dumps({"channel": "field", "aspect_id": f"{task}.Y.final",
                                "datapoint_id": dp, "prompt": body.replace(MARK, txt) + FOOT}) + "\n")
            n += 1
    print(f"{task}: {n} test rows -> {outp}")


def cmd_eval(task, results):
    ctx = bc.load_ctx(task)
    iy = items_with_y(task)
    test = set(ctx["test"])
    g = {}
    for line in open(results):
        r = json.loads(line)
        if r.get("aspect_id", "").endswith(".Y.final") and isinstance(r.get("score"), int):
            g[r["datapoint_id"]] = r["score"]
    sel = [d for d in test if d in g and iy.get(d, ("", None))[1] in (0, 1)]
    y = [iy[d][1] for d in sel]
    gy = [g[d] for d in sel]
    cl = [len(iy[d][0]) for d in sel]
    auc_g, npos, nneg = auc(gy, y)
    auc_len, _, _ = auc(cl, y)
    auc_g = max(auc_g, 1 - auc_g); auc_len2 = max(auc_len, 1 - auc_len)  # orientation-free
    print(f"{task}: n={len(sel)} (y=1:{npos} y=0:{nneg}) | AUC(language->y)={auc_g:.3f} | "
          f"AUC(char-len->y)={auc_len2:.3f} | lift over length={auc_g-auc_len2:+.3f}")
    res = dict(task=task, n=len(sel), n_pos=npos, n_neg=nneg,
               auc_language_y=round(auc_g, 4), auc_charlen_y=round(auc_len2, 4))
    json.dump(res, open(OUT / f"y_seam_{task}_final.json", "w"), indent=1)
    return res


if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] not in ("build", "eval"):
        print("usage: y_seam_extend.py build <task> | eval <task> <results>"); sys.exit(1)
    (cmd_build if sys.argv[1] == "build" else lambda t: cmd_eval(t, sys.argv[3]))(sys.argv[2])
