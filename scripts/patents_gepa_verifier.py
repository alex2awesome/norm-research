#!/usr/bin/env python3
"""GEPA on the patents disclosure verifier — reconstruction-only.

Optimizes the PROMPT of the (weak, promptable) Gemma disclosure verifier
  (element E, prior-art span S) -> discloses? {yes,no}
to maximize agreement with a STRONG-judge (GLM-5.2) disclosure gold on a dev set of (E,S) pairs.
The objective is the disclosure CONSTRUCT (strong->weak distillation), never the fell OUTCOME —
so the metric stays label-free (Y is used only in the evaluate-only recovery readout).

Reviser = GLM-4.7 (reflects on Gemma-vs-gold disagreements, rewrites the prompt). Target = Gemma-4
(local vLLM). Reference = GLM-5.2. Answers: is the disclosure verdict prompt-improvable, and does
improved fidelity raise recovery I(M;Y)?

Stages (run ON sk3, HOME=/lfs):
  builddev  (CPU)         : balanced (E,S) dev set from localize results -> dev_{train,test}.jsonl
  goldlabel (GLM-5.2 API) : strong-judge disclosure gold for every dev pair
  gepa      (Gemma GPU + GLM-4.7 API): R rounds of reflective prompt mutation; writes gepa_prompts.json
"""
import argparse, hashlib, json, os, re, sys, time, urllib.request

BASE = "/lfs/skampere3/0/alexspan/norm-research"
PROC = f"{BASE}/datasets/patents/processed"
RESULTS = f"{PROC}/localize_results_scale_gemma.jsonl"
OUTD = f"{BASE}/outputs/patents_gepa"
GEMMA4 = "/lfs/skampere3/0/shared_hf_cache/models--google--gemma-4-31b-it/snapshots/3548789868c5356dbf307c98e6f609007b82b3eb"
# key #1 was 529-overloaded 2026-07-08; key #2 (alexander) live -> default to it, toggle to #1 on fail
KEY_FILE2 = "/lfs/skampere3/0/alexspan/.z-ai-api-key.txt"
KEY_FILE = "/lfs/skampere3/0/alexspan/.z-ai-api-key-alexander-spangher.txt"

# seed = the hand-tuned calibrated verifier; {el}/{span} are the required placeholders
SEED_PROMPT = (
    "You are a strict USPTO examiner judging anticipation under 35 U.S.C. 102. Decide whether the "
    "prior-art PASSAGE discloses the specific claim LIMITATION. Substance counts, not exact wording, "
    "but the passage must actually teach the limitation (not merely the same general field).\n\n"
    "LIMITATION:\n{el}\n\nPRIOR-ART PASSAGE:\n{span}\n\n"
    'Reply with ONE JSON object: {{"discloses": true|false, "reason": "<=15 words"}}')

_OBJ = re.compile(r"\{[\s\S]*\}")
# field-first parse: survives outputs whose closing brace was truncated by max_tokens
_DISC = re.compile(r'"discloses"\s*:\s*(true|false)', re.I)


def parse_disc(raw):
    m = _DISC.search(raw or "")
    if m:
        return m.group(1).lower() == "true"
    m = _OBJ.search(raw or "")
    if not m:
        return None
    for fix in (lambda s: s, lambda s: re.sub(r",\s*}", "}", s)):
        try:
            o = json.loads(fix(m.group(0)))
            v = o.get("discloses")
            if isinstance(v, str):
                v = v.strip().lower() in ("true", "yes", "1")
            return bool(v) if v is not None else None
        except Exception:
            continue
    return None


# ---------------- GLM (z.ai Anthropic endpoint, 0 GPU) ----------------
def glm(messages, model="glm-4.7", system=None, max_tokens=1500, temp=0.0, key_file=KEY_FILE):
    # toggle across both quota keys on 529/overload (z.ai transient capacity); prefer the passed key
    keyfiles = [key_file] + [k for k in (KEY_FILE2, KEY_FILE) if k != key_file]
    body = {"model": model, "max_tokens": max_tokens, "temperature": temp, "messages": messages}
    if system:
        body["system"] = system
    data = json.dumps(body).encode()
    last = None
    for attempt in range(10):
        kf = keyfiles[attempt % len(keyfiles)]
        try:
            key = open(kf).read().strip()
            req = urllib.request.Request(
                "https://api.z.ai/api/anthropic/v1/messages", data=data,
                headers={"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"})
            with urllib.request.urlopen(req, timeout=120) as r:
                o = json.loads(r.read())
            return "".join(b.get("text", "") for b in o.get("content", []))
        except Exception as e:
            last = e
            time.sleep(min(60, 5 * (attempt + 1)))  # linear backoff to 60s; alternate keys
    raise last


DISCLOSE_JUDGE_SYS = (
    "You are a senior USPTO examiner applying 35 U.S.C. 102 anticipation strictly. Given a claim "
    "LIMITATION and a prior-art PASSAGE, decide whether the passage DISCLOSES that specific "
    "limitation in substance (not merely the same technical field, and not a different feature). "
    'Answer ONLY: {"discloses": true|false}.')


# ---------------- stages ----------------
def cmd_builddev(a):
    os.makedirs(OUTD, exist_ok=True)
    rows = []
    for ln in open(RESULTS):
        r = json.loads(ln)
        sp = r.get("spans") or []
        if isinstance(sp, list):
            sp = " ".join(s for s in sp if isinstance(s, str))
        el = (r.get("element") or "").strip()
        if len(el) > 15 and sp.strip() and el != "(no per-element breakdown)":
            rows.append({"uid": r["uid"], "label": r["label"], "element": el[:600],
                         "span": sp[:1200]})
    # sort by APP hash when available (Codex #10): same-app rows stay adjacent, so the
    # train/test slice boundary crosses at most one application instead of scattering apps.
    def gkey(r):
        return hashlib.md5(str(r.get("app_id") or r["uid"]).encode()).hexdigest()
    pos = sorted((r for r in rows if r["label"] == "pos"), key=gkey)
    neg = sorted((r for r in rows if r["label"] == "neg"), key=gkey)
    ntr, nte = a.n_train // 2, a.n_test // 2
    train = pos[:ntr] + neg[:ntr]
    test = pos[ntr:ntr + nte] + neg[ntr:ntr + nte]
    for name, data in (("dev_train", train), ("dev_test", test)):
        with open(f"{OUTD}/{name}.jsonl", "w") as fh:
            for r in data:
                fh.write(json.dumps(r) + "\n")
    print(f"[builddev] train {len(train)} (pos {ntr}/neg {ntr}) test {len(test)} -> {OUTD}", flush=True)


def cmd_goldlabel(a):
    """Sequential + RESUMABLE (checkpoint each label): robust to z.ai 529 capacity throttling.
    Re-run until it prints GOLDLABEL_DONE; already-labeled uids are skipped."""
    for split in ("dev_train", "dev_test"):
        rows = [json.loads(l) for l in open(f"{OUTD}/{split}.jsonl")]
        gpath = f"{OUTD}/{split}_gold.jsonl"
        done = {}
        if os.path.exists(gpath):
            for ln in open(gpath):
                try:
                    r = json.loads(ln); done[r["uid"]] = r
                except Exception:
                    pass
        todo = [r for r in rows if r["uid"] not in done]
        print(f"[goldlabel] {split}: {len(done)} done, {len(todo)} to label", flush=True)
        with open(gpath, "a", buffering=1) as fh:
            for i, r in enumerate(todo):
                msg = [{"role": "user", "content": f"LIMITATION:\n{r['element']}\n\nPRIOR-ART PASSAGE:\n{r['span']}\n\nDoes the passage disclose this limitation? Reply ONLY the JSON."}]
                out = glm(msg, model="glm-5", system=DISCLOSE_JUDGE_SYS, max_tokens=30, key_file=a.key_file)
                r["gold"] = parse_disc(out)
                fh.write(json.dumps(r) + "\n"); os.fsync(fh.fileno())
                if (i + 1) % 50 == 0:
                    print(f"[goldlabel]   {split} {i+1}/{len(todo)}", flush=True)
        allrows = [json.loads(l) for l in open(gpath)]
        n_ok = sum(1 for r in allrows if r.get("gold") is not None)
        pos = sum(1 for r in allrows if r.get("gold"))
        print(f"[goldlabel] {split}: {n_ok}/{len(allrows)} labeled, gold-disclose rate {pos/max(1,n_ok):.2f}", flush=True)
    print("GOLDLABEL_DONE", flush=True)


def gemma_verify(llm, sp_params, prompt, rows):
    from vllm import SamplingParams  # noqa
    convs = [[{"role": "user", "content": prompt.format(el=r["element"], span=r["span"])}] for r in rows]
    outs = llm.chat(convs, sp_params)
    return [parse_disc(o.outputs[0].text) for o in outs]


def agree(preds, rows):
    """None (unparseable) counts as WRONG — a verdict that can't be parsed is a failed verdict.
    (v1 silently dropped Nones; a prompt that answered only 8% of pairs scored 0.929 on the
    answerable subset while being degenerate on the pool. Coverage is now printed too.)"""
    p = [(pr, r["gold"]) for pr, r in zip(preds, rows) if r.get("gold") is not None]
    if not p:
        return 0.0, 0.0, 0
    cov = sum(1 for a, _ in p if a is not None) / len(p)
    acc = sum(1 for a, b in p if a == b) / len(p)
    # balanced: mean of per-class accuracy (gold-yes recall + gold-no recall)/2; None fails both
    yes = [(a, b) for a, b in p if b]
    no = [(a, b) for a, b in p if not b]
    ry = (sum(1 for a, b in yes if a is True) / len(yes)) if yes else 0.0
    rn = (sum(1 for a, b in no if a is False) / len(no)) if no else 0.0
    print(f"    [agree] n={len(p)} coverage={cov:.3f}", flush=True)
    return acc, (ry + rn) / 2, len(p)


REVISE_SYS = (
    "You improve the PROMPT of a weak model that judges patent-claim anticipation. You are given "
    "the current prompt and cases where the weak model DISAGREED with an expert examiner. Rewrite "
    "the prompt so the weak model matches the expert better. Keep it a strict but fair 35 U.S.C. 102 "
    "disclosure judgment. The prompt MUST keep the literal placeholders {el} and {span} and MUST end "
    "by requiring a JSON object {{\"discloses\": true|false, \"reason\": \"<=15 words\"}} with the "
    "reason capped at 15 words. Return ONLY the new prompt text, nothing else.")


def cmd_gepa(a):
    train = [json.loads(l) for l in open(f"{OUTD}/dev_train_gold.jsonl") if json.loads(l).get("gold") is not None]
    test = [json.loads(l) for l in open(f"{OUTD}/dev_test_gold.jsonl") if json.loads(l).get("gold") is not None]
    print(f"[gepa] train {len(train)} test {len(test)} (both gold-labeled)", flush=True)

    from vllm import LLM, SamplingParams
    llm = LLM(model=GEMMA4, dtype="bfloat16", gpu_memory_utilization=0.85, max_model_len=4096,
              enable_prefix_caching=True, trust_remote_code=True)
    # 200 tokens: room for a full JSON object even with a wordy reason (60 truncated mid-JSON in v1)
    sp = SamplingParams(temperature=0.0, max_tokens=200)

    history = []
    best = {"prompt": SEED_PROMPT, "round": 0}
    # seed eval
    tr_pred = gemma_verify(llm, sp, SEED_PROMPT, train)
    tr_acc, tr_bal, _ = agree(tr_pred, train)
    te_pred = gemma_verify(llm, sp, SEED_PROMPT, test)
    te_acc, te_bal, _ = agree(te_pred, test)
    best.update({"train_bal": tr_bal, "test_bal": te_bal, "test_acc": te_acc})
    history.append({"round": 0, "op": "seed", "train_bal": tr_bal, "test_bal": te_bal, "test_acc": te_acc})
    print(f"[gepa] R0 seed: train_bal={tr_bal:.3f} test_bal={te_bal:.3f} test_acc={te_acc:.3f}", flush=True)

    cur_prompt, cur_pred = SEED_PROMPT, tr_pred
    for rnd in range(1, a.rounds + 1):
        # reflect on disagreements
        dis = [(r, pr) for r, pr in zip(train, cur_pred)
               if pr is not None and r.get("gold") is not None and pr != r["gold"]][:10]
        if not dis:
            print(f"[gepa] R{rnd}: no disagreements, stopping", flush=True)
            break
        cases = "\n\n".join(
            f"CASE {i+1}: weak model said discloses={pr}, expert said discloses={r['gold']}\n"
            f"  LIMITATION: {r['element'][:300]}\n  PASSAGE: {r['span'][:400]}"
            for i, (r, pr) in enumerate(dis))
        msg = [{"role": "user", "content":
                f"CURRENT PROMPT:\n-----\n{cur_prompt}\n-----\n\nDISAGREEMENT CASES:\n{cases}\n\n"
                "Rewrite the prompt to fix these misreadings. Return ONLY the new prompt."}]
        new_prompt = glm(msg, model="glm-4.7", system=REVISE_SYS, max_tokens=1200, key_file=a.key_file).strip()
        new_prompt = re.sub(r"^```[a-z]*|```$", "", new_prompt).strip()
        if "{el}" not in new_prompt or "{span}" not in new_prompt or "discloses" not in new_prompt:
            print(f"[gepa] R{rnd}: reviser dropped placeholders/format, skipping", flush=True)
            history.append({"round": rnd, "op": "revise", "status": "invalid"})
            continue
        np_tr = gemma_verify(llm, sp, new_prompt, train)
        na, nbal, _ = agree(np_tr, train)
        accepted = nbal > best["train_bal"] + 1e-4
        rec = {"round": rnd, "op": "revise", "train_bal": nbal, "accepted": accepted}
        print(f"[gepa] R{rnd}: train_bal={nbal:.3f} (best {best['train_bal']:.3f}) "
              f"{'ACCEPT' if accepted else 'reject'}", flush=True)
        if accepted:
            np_te = gemma_verify(llm, sp, new_prompt, test)
            tea, tebal, _ = agree(np_te, test)
            rec.update({"test_bal": tebal, "test_acc": tea})
            best = {"prompt": new_prompt, "round": rnd, "train_bal": nbal, "test_bal": tebal, "test_acc": tea}
            cur_prompt, cur_pred = new_prompt, np_tr
            print(f"[gepa]     -> test_bal={tebal:.3f} test_acc={tea:.3f}", flush=True)
        history.append(rec)

    json.dump({"seed_prompt": SEED_PROMPT, "best": best, "history": history},
              open(f"{OUTD}/{a.out}", "w"), indent=1)
    seed_te = history[0]["test_bal"]
    print(f"\n[gepa] FIDELITY: seed test_bal={seed_te:.3f} -> best test_bal={best['test_bal']:.3f} "
          f"(round {best['round']}, delta {best['test_bal'] - seed_te:+.3f})", flush=True)
    print("GEPA_DONE", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("builddev"); b.add_argument("--n-train", type=int, default=250, dest="n_train")
    b.add_argument("--n-test", type=int, default=150, dest="n_test")
    g = sub.add_parser("goldlabel"); g.add_argument("--key-file", default=KEY_FILE, dest="key_file")
    p = sub.add_parser("gepa"); p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--key-file", default=KEY_FILE, dest="key_file")
    p.add_argument("--out", default="gepa_prompts.json")
    a = ap.parse_args()
    {"builddev": cmd_builddev, "goldlabel": cmd_goldlabel, "gepa": cmd_gepa}[a.cmd](a)


if __name__ == "__main__":
    main()
