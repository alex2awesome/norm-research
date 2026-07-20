"""y-GEPA seam, stage 1: GEPA-optimize a single forecast prompt DIRECTLY against the real
outcome y (user-specified design, 2026-07-19).

Unlike the m-seam GEPA (judge reconstruction, never label-aware), this is the VAT side:
y is ground truth, so the proposer IS shown TRAIN labels (misranked exemplars), exactly as
the manual VAT trains on labels. The TEST split is frozen and touched only by `final`.

  python3 y_gepa.py init  <task>            -> y_gepa_state_<task>.json (seed = y_seam prompt)
  python3 y_gepa.py build <task> <round>    -> y_gepa/<task>_round<r>_prompts.jsonl (TRAIN)
  python3 y_gepa.py ingest <task> <round> <results.jsonl>   (train AUC -> history/best)
  python3 y_gepa.py propose <task> <round>  (GLM-5.2 revision from AUC history + labeled errors)
  python3 y_gepa.py final <task>            -> TEST prompts from argmax-train prompt
  python3 y_gepa.py eval <task> <results>   -> held-out y-AUC + length-stratified

Stage 2 (unit decomposition / compile) lives in y_gepa_units.py.
"""
import json, pathlib, sys, urllib.request

HERE = pathlib.Path(__file__).resolve().parent
ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "methods/metric_seam/battery"))
import battery_common as bc
from y_seam_extend import items_with_y, auc, TASKS as YTASKS
from y_seam_vtier import stratified_auc

bc.PROGDIR.update({"legal_title_vii": "programs_legal", "peer_review": "programs_peer",
                   "patents_pa": "programs_pa", "code_review": "programs_code_review"})
OUT = HERE / "y_gepa"; OUT.mkdir(exist_ok=True)

MARK = "<<<DOCUMENT>>>"
FOOT = "\n\nReply with exactly one line: SCORE: <integer 0-10>"
MAXCHARS = 6000

KEY_PATHS = ["~/.z-ai-api-key-alexander-spangher.txt", "~/.z-ai-api-key-spangher.txt",
             "~/.z-ai-api-key.txt"]
ZAI_URL = "https://api.z.ai/api/anthropic/v1/messages"


def state_path(task):
    return OUT / f"y_gepa_state_{task}.json"


def seed_prompt(task):
    noun, q = YTASKS[task]
    return (f"You are forecasting the real-world outcome of a single {noun}.\n\n"
            f"Question: How likely is it that {q}?\n\nDocument:\n{MARK}\n\n"
            "Give your probability as an integer from 0 (certain the favorable outcome will "
            "NOT happen) to 10 (certain it WILL). Base it only on the document.")


def render(body, text):
    t = text[:MAXCHARS] + ("\n...[truncated]" if len(text) > MAXCHARS else "")
    if MARK not in body:
        body = body.rstrip() + "\n\nDocument:\n" + MARK
    return body.replace(MARK, t) + FOOT


def glm_call(system, user, model="glm-5.2", max_tokens=4000):
    import os
    key = None
    for p in KEY_PATHS:
        fp = pathlib.Path(os.path.expanduser(p))
        if fp.exists():
            key = fp.read_text().strip(); break
    if not key:
        raise FileNotFoundError("no z.ai key")
    body = json.dumps({"model": model, "max_tokens": max_tokens, "system": system,
                       "messages": [{"role": "user", "content": user}]}).encode()
    req = urllib.request.Request(ZAI_URL, data=body, headers={
        "x-api-key": key, "anthropic-version": "2023-06-01",
        "content-type": "application/json"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=240) as r:
                resp = json.loads(r.read())
            return "".join(b.get("text", "") for b in resp.get("content", []))
        except Exception as e:
            if attempt == 2:
                raise
            import time; time.sleep(4 * (attempt + 1))


def cmd_init(task):
    sp = state_path(task)
    if sp.exists():
        print(f"REFUSING: {sp} exists"); sys.exit(1)
    json.dump({"task": task, "round": 0, "prompt": seed_prompt(task),
               "history": [], "best": None}, open(sp, "w"), indent=1)
    print(f"-> {sp} (round 0, seed prompt)")


def cmd_build(task, rd):
    st = json.load(open(state_path(task)))
    assert st["round"] == rd, f"state at round {st['round']}, not {rd}"
    ctx = bc.load_ctx(task)
    iy = items_with_y(task)
    outp = OUT / f"{task}_round{rd}_prompts.jsonl"
    n = 0
    with open(outp, "w") as f:
        for d in sorted(ctx["train"]):
            if iy.get(d, ("", None))[1] not in (0, 1):
                continue
            f.write(json.dumps({"channel": "field", "aspect_id": f"{task}.YG.g{rd}",
                                "datapoint_id": d,
                                "prompt": render(st["prompt"], iy[d][0])}) + "\n")
            n += 1
    print(f"wrote {n} TRAIN rows -> {outp}")


def cmd_ingest(task, rd, results):
    st = json.load(open(state_path(task)))
    iy = items_with_y(task)
    col = {}
    for line in open(results):
        r = json.loads(line)
        if r.get("aspect_id") == f"{task}.YG.g{rd}" and isinstance(r.get("score"), int):
            col[r["datapoint_id"]] = r["score"]
    ids = [d for d in col if iy.get(d, ("", None))[1] in (0, 1)]
    y = [iy[d][1] for d in ids]; s = [col[d] for d in ids]
    a, npos, nneg = auc(s, y)
    entry = {"round": rd, "train_auc": round(a, 4), "n": len(ids), "prompt": st["prompt"]}
    st["history"].append(entry)
    if st["best"] is None or a > st["best"]["train_auc"]:
        st["best"] = entry
    json.dump(st, open(state_path(task), "w"), indent=1)
    # labeled error exemplars for the proposer (train only, by design label-aware)
    scored = sorted(ids, key=lambda d: col[d])
    fp = [d for d in reversed(scored) if iy[d][1] == 0][:5]   # high score, y=0
    fn = [d for d in scored if iy[d][1] == 1][:5]             # low score, y=1
    json.dump({"round": rd, "train_auc": round(a, 4),
               "false_pos": [{"score": col[d], "y": 0, "snippet": iy[d][0][:600]} for d in fp],
               "false_neg": [{"score": col[d], "y": 1, "snippet": iy[d][0][:600]} for d in fn]},
              open(OUT / f"{task}_round{rd}_feedback.json", "w"), indent=1)
    print(f"round {rd}: TRAIN AUC={a:.4f} (n={len(ids)}, {npos}/{nneg}) | "
          f"best={st['best']['train_auc']:.4f} (round {st['best']['round']})")


def cmd_propose(task, rd):
    st = json.load(open(state_path(task)))
    fb = json.load(open(OUT / f"{task}_round{rd}_feedback.json"))
    noun, q = YTASKS[task]
    hist = "\n".join(f"  round {h['round']}: train AUC {h['train_auc']}" for h in st["history"])
    ex = "\n\n".join(
        [f"[predicted favorable (score {e['score']}) but outcome was UNFAVORABLE]\n{e['snippet']}"
         for e in fb["false_pos"][:3]] +
        [f"[predicted unfavorable (score {e['score']}) but outcome was FAVORABLE]\n{e['snippet']}"
         for e in fb["false_neg"][:3]])
    system = ("You are optimizing a forecasting prompt. The scorer model reads the prompt with a "
              "document substituted at <<<DOCUMENT>>> and answers SCORE: 0-10 (probability of "
              "the favorable outcome). Your revision must: keep the literal marker <<<DOCUMENT>>> "
              "exactly once; keep the 0-10 integer output convention (do NOT restate the reply "
              "format, it is appended automatically); be a COMPLETE replacement prompt. "
              "Improve ranking (AUC), not calibration. Return ONLY the new prompt text.")
    user = (f"Task: forecast whether {q} for a {noun}.\n\nCurrent prompt:\n---\n{st['prompt']}\n---\n\n"
            f"Dev history (higher AUC better):\n{hist}\n\n"
            f"Hardest misranked TRAIN examples (with true outcomes):\n{ex}\n\n"
            "Write the improved prompt. Consider: what document facts actually predict the outcome "
            "(use the misranked examples), what the current prompt overweights, and add concrete "
            "guidance the scorer can apply. Return only the prompt text.")
    new = glm_call(system, user).strip()
    if MARK not in new or len(new) < 100:
        print("proposal invalid (marker missing/too short) — keeping current prompt")
    else:
        st["prompt"] = new
    st["round"] = rd + 1
    json.dump(st, open(state_path(task), "w"), indent=1)
    print(f"-> round {rd+1} prompt ({len(st['prompt'])} chars)")


def cmd_final(task):
    st = json.load(open(state_path(task)))
    best = st["best"]
    ctx = bc.load_ctx(task)
    iy = items_with_y(task)
    outp = OUT / f"{task}_final_prompts.jsonl"
    n = 0
    with open(outp, "w") as f:
        for d in sorted(ctx["test"]):
            if iy.get(d, ("", None))[1] not in (0, 1):
                continue
            f.write(json.dumps({"channel": "field", "aspect_id": f"{task}.YG.final",
                                "datapoint_id": d,
                                "prompt": render(best["prompt"], iy[d][0])}) + "\n")
            n += 1
    json.dump({"frozen_round": best["round"], "train_auc": best["train_auc"],
               "prompt": best["prompt"]}, open(OUT / f"{task}_final_frozen.json", "w"), indent=1)
    print(f"froze round {best['round']} (train AUC {best['train_auc']}); {n} TEST rows -> {outp}")


def cmd_eval(task, results):
    iy = items_with_y(task)
    ctx = bc.load_ctx(task)
    test = set(ctx["test"])
    col = {}
    for line in open(results):
        r = json.loads(line)
        if r.get("aspect_id") == f"{task}.YG.final" and isinstance(r.get("score"), int):
            col[r["datapoint_id"]] = r["score"]
    ids = [d for d in test if d in col and iy.get(d, ("", None))[1] in (0, 1)]
    y = [iy[d][1] for d in ids]; s = [col[d] for d in ids]
    lens = [len(iy[d][0]) for d in ids]
    a, npos, nneg = auc(s, y)
    sa = stratified_auc(s, y, lens)
    print(f"{task} y-GEPA held-out: AUC={a:.4f} len-strat={sa:.4f} (n={len(ids)}, {npos}/{nneg})")
    json.dump({"task": task, "auc_test": round(a, 4), "auc_len_strat": round(sa, 4),
               "n_test": len(ids)}, open(OUT / f"{task}_final_eval.json", "w"), indent=1)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "init":
        cmd_init(sys.argv[2])
    elif cmd == "build":
        cmd_build(sys.argv[2], int(sys.argv[3]))
    elif cmd == "ingest":
        cmd_ingest(sys.argv[2], int(sys.argv[3]), sys.argv[4])
    elif cmd == "propose":
        cmd_propose(sys.argv[2], int(sys.argv[3]))
    elif cmd == "final":
        cmd_final(sys.argv[2])
    elif cmd == "eval":
        cmd_eval(sys.argv[2], sys.argv[3])
