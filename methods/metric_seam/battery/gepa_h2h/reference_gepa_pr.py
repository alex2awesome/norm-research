#!/usr/bin/env python
"""GEPA loop for press_releases norm-extraction.

3 composable subcommands (single GPU can cycle; mutator is an API call, 0 GPU):
  gen    : Gemma runs (cfg, few_shot, mode) over the 30-pair eval slice -> full parsed dump.
  judge  : Qwen scores each signal {faithful, valid} vs its source article -> scores + coverage/precision.
  mutate : GLM (zai_anthropic subscription API) revises cfg semantic fields from low-scoring examples.
  run    : driver. round 0 = gen+judge(baseline); each round = mutate -> gen -> judge -> keep-if-better.

OBJECTIVE (label-free distillation): maximize COVERAGE on signal-rich (positive) pairs
(fraction of marker-rich articles where >=1 faithful+valid normative signal is extracted),
s.t. validity-precision >= FLOOR (don't degenerate into firehose). Per user: true positives
matter; missing them is the failure mode. Mutator sees coverage + precision + concrete failures.
"""
import sys, os, json, re, time, importlib.util, statistics as st, urllib.request, subprocess

ROOT = "/lfs/skampere3/0/alexspan/scripts/llama_norm_extraction"
_DATADIR = "/lfs/skampere3/0/alexspan/data"
GEMMA = "/lfs/skampere3/0/shared_hf_cache/models--google--gemma-4-31b-it/snapshots/3548789868c5356dbf307c98e6f609007b82b3eb"
QWEN = "/lfs/skampere3/0/shared_hf_cache/models--Qwen--Qwen3.5-122B-A10B-FP8/snapshots/a099dee70ccfcd8d5dda56aaa0b60cb8ecadabc9"
KEYFILES = ["/lfs/skampere3/0/alexspan/.z-ai-api-key-alexander-spangher.txt", "/lfs/skampere3/0/alexspan/.z-ai-api-key.txt"]  # try working key first, fall back on 429
KEYFILE = KEYFILES[1]  # back-compat
ZAI_URL = "https://api.z.ai/api/anthropic/v1/messages"
MAXCHARS = 6000
FLOOR = 0.50          # validity-precision floor; reject mutants below it
MIN_EXAMPLES = 8      # signals to show GLM as failures
GLM_TEMPERATURE = 0.8
OBJECTIVE = os.environ.get("GEPA_OBJECTIVE", "f1")   # "f1" (cov×prec balanced) or "yield" (max faithful+valid count)
YIELD_FLOOR = 0.35    # looser precision floor for yield objective (judge filters; volume is the point)

# ---- corpus selection (GEPA_CORPUS env; press_releases is default) ----
# input_field: the record field holding the source text to extract from / judge against.
#   "article_text" => press_releases dual-text (article vs pr) handled via `mode`.
#   any other => single-text corpus, `mode` ignored, that field is the input.
CORPUS = os.environ.get("GEPA_CORPUS", "press_releases")
CORPORA = {
    "press_releases": dict(data="press_releases", eval="eval_pairs.jsonl",
        corpus="pr_article_pairs_full.jsonl", cfg="press_releases_full_v3.json",
        fewshot="press_releases_full_v3_minimal.json", input_field="article_text",
        id="pair_id", domain="news articles about corporate press releases"),
    "legaladvice_uk": dict(data="legaladvice_uk", eval="gepa_eval_pairs.jsonl",
        corpus="input.jsonl", cfg="legaladvice_uk.json",
        fewshot="legaladvice_uk_v2_examples.json", input_field="text",
        id="unit_id", domain="UK legal-advice forum comments (r/LegalAdviceUK style)"),
    "humor_multi": dict(data="humor/standup_multi", eval="gepa_eval_pairs.jsonl",
        corpus_path="humor/standup_multi/input_v2_no_ast.jsonl", cfg="humor_multi_v2.json",
        fewshot="humor_multi_v2_examples.json", input_field="text",
        id="thread_id", domain="comedy-craft feedback threads (r/StandUpWorkshop etc.)"),
    "wp_comments": dict(data="creative_writing/wp_comments", eval="gepa_eval_pairs.jsonl",
        corpus_path="creative_writing/wp_comments/input.jsonl", cfg="wp_comments.json",
        fewshot="wp_comments_examples.json", input_field="text",
        id="unit_id", domain="r/WritingPrompts story-feedback comment chains"),
    "math_se": dict(data="math_se", eval="gepa_eval_pairs.jsonl",
        corpus="input.jsonl", cfg="math_se.json",
        fewshot="math_se_examples.json", input_field="text",
        id="unit_id", domain="Math StackExchange answers/comments"),
    "crse": dict(data="crse", eval="gepa_eval_pairs.jsonl",
        corpus="input.jsonl", cfg="crse.json",
        fewshot="crse_examples.json", input_field="text",
        id="unit_id", domain="CodeReview.StackExchange answers"),
    "code_review": dict(data="code_review", eval="gepa_eval_pairs.jsonl",
        corpus="input.jsonl", cfg="code_review.json",
        fewshot="code_review_examples.json", input_field="text",
        id="unit_id", domain="GitHub pull-request review comments"),
    "peer_review": dict(data="peer_review", eval="gepa_eval_pairs.jsonl",
        corpus="input.jsonl", cfg="peer_review.json",
        fewshot="peer_review_examples.json", input_field="text",
        id="unit_id", domain="academic peer reviews of papers"),
    "law_se": dict(data="law_se", eval="gepa_eval_pairs.jsonl",
        corpus="input.jsonl", cfg="law_se.json",
        fewshot="law_se_examples.json", input_field="text",
        id="unit_id", domain="Law StackExchange comments and answers"),
    "reddit_supremecourt": dict(data="reddit_supremecourt", eval="gepa_eval_pairs.jsonl",
        corpus="input.jsonl", cfg="reddit_supremecourt.json",
        fewshot="reddit_supremecourt_examples.json", input_field="text",
        id="unit_id", domain="r/supremecourt commentary on Supreme Court cases"),
    "litbench": dict(data="litbench_rationales", eval="gepa_eval_pairs.jsonl",
        corpus="input.jsonl", cfg="litbench.json",
        fewshot="litbench_examples.json", input_field="text",
        id="unit_id", domain="literary criticism rationales comparing creative writing quality"),
    "nc_public_comments": dict(data="nc_public_comments", eval="gepa_eval_pairs.jsonl",
        corpus="input.jsonl", cfg="nc_public_comments.json",
        fewshot="nc_public_comments_examples.json", input_field="text",
        id="unit_id", domain="public comments on proposed federal regulations (notice-and-comment)"),
    "courtlistener": dict(data="courtlistener_opinions", eval="gepa_eval_pairs.jsonl",
        corpus="input.jsonl", cfg="courtlistener.json",
        fewshot="courtlistener_examples.json", input_field="text",
        id="unit_id", domain="federal court opinions (Title VII, Social Security, etc.)"),
    "aops_forum": dict(data="aops_forum", eval="gepa_eval_pairs.jsonl",
        corpus="input.jsonl", cfg="aops_forum.json",
        fewshot="aops_forum_examples.json", input_field="text",
        id="unit_id", domain="Art of Problem Solving math competition forum posts"),
    "competition_editorials": dict(data="competition_editorials", eval="gepa_eval_pairs.jsonl",
        corpus="input.jsonl", cfg="competition_editorials.json",
        fewshot="competition_editorials_examples.json", input_field="text",
        id="unit_id", domain="competitive programming problem editorials (LeetCode, Codeforces)"),
    "mathlib": dict(data="mathlib", eval="gepa_eval_pairs.jsonl",
        corpus="input.jsonl", cfg="mathlib.json",
        fewshot="mathlib_examples.json", input_field="text",
        id="unit_id", domain="Mathlib Lean 4 PR review comments"),
"bva_opinions": dict(data="bva_opinions", eval="gepa_eval_pairs.jsonl",
        corpus="input.jsonl", cfg="bva_opinions.json",
        fewshot="bva_opinions_examples.json", input_field="text",
        id="unit_id", domain="Board of Veterans Appeals decisions on veterans' benefits claims"),
    "cavc_decisions": dict(data="cavc_decisions", eval="gepa_eval_pairs.jsonl",
        corpus="input.jsonl", cfg="cavc_decisions.json",
        fewshot="cavc_decisions_examples.json", input_field="text",
        id="unit_id", domain="Court of Appeals for Veterans Claims reviewing BVA decisions"),
    "nlrb_decisions": dict(data="nlrb_decisions", eval="gepa_eval_pairs.jsonl",
        corpus="input.jsonl", cfg="nlrb_decisions.json",
        fewshot="nlrb_decisions_examples.json", input_field="text",
        id="unit_id", domain="National Labor Relations Board unfair labor practice decisions"),
    "dol_arb": dict(data="dol_arb", eval="gepa_eval_pairs.jsonl",
        corpus="input.jsonl", cfg="dol_arb.json",
        fewshot="dol_arb_examples.json", input_field="text",
        id="unit_id", domain="Department of Labor arbitration awards in labor disputes"),
    "ptab_fwd": dict(data="ptab_fwd", eval="gepa_eval_pairs.jsonl",
        corpus="input.jsonl", cfg="ptab_fwd.json",
        fewshot="ptab_fwd_examples.json", input_field="text",
        id="unit_id", domain="Patent Trial and Appeal Board ex parte appeal decisions"),
    "ttab_inter_partes": dict(data="ttab_inter_partes", eval="gepa_eval_pairs.jsonl",
        corpus="input.jsonl", cfg="ttab_inter_partes.json",
        fewshot="ttab_inter_partes_examples.json", input_field="text",
        id="unit_id", domain="Trademark Trial and Appeal Board inter partes decisions"),
    "notice_and_comment": dict(data="notice_and_comment", eval="gepa_eval_pairs.jsonl",
        corpus="input.jsonl", cfg="notice_and_comment.json",
        fewshot="notice_and_comment_examples.json", input_field="text",
        id="unit_id", domain="federal agency responses to public comments on proposed rules"),
}
_C = CORPORA[CORPUS]
DATA = _DATADIR + "/" + _C["data"]
EVAL = DATA + "/" + _C["eval"]
CORPUS_FILE = (_DATADIR + "/" + _C["corpus_path"]) if "corpus_path" in _C else (DATA + "/" + _C["corpus"])
CFG_PATH = ROOT + "/configs/" + _C["cfg"]
FEWSHOT_PATH = ROOT + "/few_shots/" + _C["fewshot"]
INPUT_FIELD = _C["input_field"]
ID_FIELD = _C["id"]
DOMAIN = _C["domain"]

# ---- import runner helpers (build_prompt / parse_output / template / fmt_worked_examples) ----
spec = importlib.util.spec_from_file_location("rsb", ROOT + "/run_sk3_batch.py")
rsb = importlib.util.module_from_spec(spec); spec.loader.exec_module(rsb)
TEMPLATE = rsb.load_template()


def load_eval():
    return [json.loads(l) for l in open(EVAL)]


def build_input(r, mode):
    if INPUT_FIELD == "article_text":   # press_releases dual-text structure
        if mode == "article_only":
            t = "NEWS ARTICLE:\n" + r.get("article_text", "")
        elif mode == "article_first":
            t = "NEWS ARTICLE:\n" + r.get("article_text", "") + "\n\n--- PRESS RELEASE (context):\n" + r.get("pr_text", "")
        else:
            t = r.get("text", "")
    else:                                # single-text corpora
        t = r.get(INPUT_FIELD, "") or r.get("text", "")
    return (t[:MAXCHARS] + "\n...[truncated]") if len(t) > MAXCHARS else t


def article_for_judge(r, mode):
    if INPUT_FIELD == "article_text":
        t = r.get("article_text", "") if mode != "combined" else r.get("text", "")
    else:
        t = r.get(INPUT_FIELD, "") or r.get("text", "")
    return t[:MAXCHARS]


def parse_verdicts(raw):
    """Extract {i: verdict} from a judge (GLM/Qwen) response. Strips <think>, finds the signals JSON.
    stdlib json first (judge output is clean JSON), json_repair only as an optional fallback so this
    works in any env (gemma4 lacks json_repair)."""
    raw = re.sub(r"<think>.*?</think>", "", raw or "", flags=re.S)
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return {}
    blob = m.group(0)
    obj = None
    try:
        obj = json.loads(blob)
    except Exception:
        try:
            import json_repair
            obj = json_repair.loads(blob)
        except Exception:
            obj = None
    sigs = obj.get("signals") if isinstance(obj, dict) else None
    if not isinstance(sigs, list):
        return {}
    out = {}
    for v in sigs:
        if isinstance(v, dict) and "i" in v:
            try:
                i = int(v["i"]) if not isinstance(v["i"], int) else v["i"]
                out[i] = v
            except (ValueError, TypeError):
                pass  # skip verdicts with non-int "i"
    return out


def flatten_signals(parsed):
    """-> list of {passage_text, signal_text, signal_type, polarity} from a parsed record."""
    out = []
    for p in (parsed or {}).get("passages", []) or []:
        pt = p.get("passage_text", "")
        for s in p.get("signals", []) or []:
            out.append({"passage_text": pt, "signal_text": s.get("signal_text", ""),
                        "signal_type": s.get("signal_type", ""), "polarity": s.get("polarity", "")})
    return out


# ================= GEN (Gemma) =================
def cmd_gen(cfg_path, fewshot_path, mode, out_path):
    rows = load_eval()
    cfg = json.load(open(cfg_path))
    fs = json.load(open(fewshot_path))
    worked = rsb.fmt_worked_examples(fs)
    print("=== loading Gemma-4 GPU " + os.environ.get("CUDA_VISIBLE_DEVICES", "?") + " ===", flush=True)
    from vllm import LLM, SamplingParams
    llm = LLM(model=GEMMA, gpu_memory_utilization=float(os.environ.get("GPU_MEM_UTIL", "0.90")),
              max_model_len=16384, dtype="bfloat16", trust_remote_code=True)
    msgs = []
    for r in rows:
        text = build_input(r, mode)
        msgs.append([{"role": "user", "content": rsb.build_prompt(TEMPLATE, cfg, "", worked, r[ID_FIELD], text)}])
    outs = llm.chat(msgs, SamplingParams(temperature=0.0, max_tokens=2500))
    dump = []
    n_ok = 0
    for r, o in zip(rows, outs):
        raw = o.outputs[0].text if o.outputs else ""
        parsed, err = rsb.parse_output(raw)
        if parsed:
            n_ok += 1
        sigs = flatten_signals(parsed)
        dump.append({ID_FIELD: r[ID_FIELD], "pos": r.get("is_positive", r.get("pos", False)), "mode": mode, "ok": bool(parsed),
                     "err": err, "n_sig": len(sigs), "signals": sigs})
    open(out_path, "w").write("\n".join(json.dumps(d, ensure_ascii=True) for d in dump))
    print("GEN_DONE ok=" + str(n_ok) + "/" + str(len(rows)) + " sigs=" + str(sum(d["n_sig"] for d in dump))
          + " -> " + out_path, flush=True)


# ================= GEN CORPUS (Gemma, full deploy — resumable, chunked) =================
def cmd_gen_corpus(cfg_path, fewshot_path, in_path, out_path, chunk=2000):
    rows = [json.loads(l) for l in open(in_path)]
    cfg = json.load(open(cfg_path))
    fs = json.load(open(fewshot_path))
    worked = rsb.fmt_worked_examples(fs)
    # resume: collect already-done ID_FIELDs
    done = set()
    if os.path.exists(out_path):
        for l in open(out_path):
            try:
                done.add(json.loads(l)[ID_FIELD])
            except Exception:
                pass
    todo = [r for r in rows if r.get(ID_FIELD) not in done]
    print("=== deploy Gemma-4 GPU " + os.environ.get("CUDA_VISIBLE_DEVICES", "?") +
          " | total=" + str(len(rows)) + " done=" + str(len(done)) + " todo=" + str(len(todo)) + " ===", flush=True)
    from vllm import LLM, SamplingParams
    llm = LLM(model=GEMMA, gpu_memory_utilization=float(os.environ.get("GPU_MEM_UTIL", "0.90")),
              max_model_len=16384, dtype="bfloat16", trust_remote_code=True)
    sp = SamplingParams(temperature=0.0, max_tokens=2500)
    mode = "article_only"
    fout = open(out_path, "a")
    n_ok = 0; n_sig = 0
    for i in range(0, len(todo), chunk):
        batch = todo[i:i + chunk]
        msgs = [[{"role": "user", "content": rsb.build_prompt(TEMPLATE, cfg, "", worked, r[ID_FIELD], build_input(r, mode))}] for r in batch]
        outs = llm.chat(msgs, sp)
        for r, o in zip(batch, outs):
            raw = o.outputs[0].text if o.outputs else ""
            parsed, err = rsb.parse_output(raw)
            sigs = flatten_signals(parsed)
            if parsed:
                n_ok += 1
            n_sig += len(sigs)
            fout.write(json.dumps({ID_FIELD: r[ID_FIELD], "ok": bool(parsed), "err": err,
                                   "n_sig": len(sigs), "signals": sigs}) + "\n")
        fout.flush()
        print("  chunk " + str(i // chunk + 1) + ": " + str(min(i + chunk, len(todo))) + "/" +
              str(len(todo)) + "  (running ok=" + str(n_ok) + " sigs=" + str(n_sig) + ")", flush=True)
    fout.close()
    print("DEPLOY_DONE ok=" + str(n_ok) + "/" + str(len(todo)) + " sigs=" + str(n_sig) + " -> " + out_path, flush=True)


# ================= JUDGE (Qwen, label-free) =================
JUDGE_SYS = ("You are a strict evaluator for a DISTANT-SUPERVISION norm-extraction pipeline on news articles "
             "about corporate press releases. The pipeline extracts 'signals' -- short phrases that express a "
             "NORMATIVE JUDGMENT about how a company COMMUNICATED (e.g. non-disclosure/refusal to comment, "
             "defensive framing, third-party contestation, conspicuous omissions, suspicious timing, claim "
             "discrepancies, novelty/first-time disclosures, independent sourcing). You score each extracted signal "
             "on two axes.\n\n"
             "FAITHFUL = 1 iff the signal (and its passage) is actually GROUNDED IN the article below -- the article "
             "really says/supports this. Hallucinated or unsupported claims -> 0.\n"
             "VALID = 1 iff the signal is a genuine NORMATIVE/CRAFT judgment about the company's communication, NOT a "
             "plain reported fact, not fresh news content, not neutral description of what happened, not boilerplate. "
             "A bare fact ('revenue rose 12%') is INVALID; a normative reading ('declined to disclose the figure') is VALID.\n\n"
             "Be strict. Respond with ONLY a JSON object: "
             '{"signals":[{"i":<int>,"faithful":0|1,"valid":0|1,"reason":"<=15 words"}]}')


# per-corpus judge override: if judge_sys/<CORPUS>.txt exists, it replaces the default JUDGE_SYS above
_JSYS_FILE = ROOT + "/judge_sys/" + CORPUS + ".txt"
if os.path.exists(_JSYS_FILE):
    JUDGE_SYS = open(_JSYS_FILE).read().strip()


def cmd_judge(gen_path, mode, score_path):
    rows = {r[ID_FIELD]: r for r in load_eval()}
    recs = [json.loads(l) for l in open(gen_path)]
    os.environ.setdefault("VLLM_USE_FLASHINFER_MOE_FP8", "0")
    os.environ.setdefault("HF_HOME", "/lfs/skampere3/0/shared_hf_cache")
    print("=== loading Qwen-122B-A10B-FP8 judge GPU " + os.environ.get("CUDA_VISIBLE_DEVICES", "?") + " ===", flush=True)
    from vllm import LLM, SamplingParams
    llm = LLM(model=QWEN, dtype="auto",
              gpu_memory_utilization=float(os.environ.get("GPU_MEM_UTIL", "0.90")),
              trust_remote_code=True,
              limit_mm_per_prompt={"image": 0, "video": 0, "audio": 0},
              max_model_len=16384)
    sp = SamplingParams(temperature=0.0, max_tokens=1500)
    prompts, idx = [], []
    for rec in recs:
        r = rows.get(rec[ID_FIELD])
        if not r or not rec["signals"]:
            continue
        art = article_for_judge(r, mode)
        lines = ["ARTICLE:", art, "", "Extracted signals to score:"]
        for i, s in enumerate(rec["signals"]):
            lines.append("[" + str(i) + "] signal: " + s["signal_text"] + "  | passage: " + (s["passage_text"] or "")[:300])
        prompts.append("\n".join(lines))
        idx.append(rec)
    convos = [[{"role": "system", "content": JUDGE_SYS}, {"role": "user", "content": p}] for p in prompts]
    outs = llm.chat(convos, sp, chat_template_kwargs={"enable_thinking": False}, use_tqdm=False)
    # merge scores back
    by_pid = {rec[ID_FIELD]: dict(rec, scored=[]) for rec in recs}
    raws = []
    for rec, o in zip(idx, outs):
        raw = o.outputs[0].text if o.outputs else ""
        raws.append({ID_FIELD: rec[ID_FIELD], "raw": raw[:4000]})
        verdicts = parse_verdicts(raw)
        scored = []
        for i, s in enumerate(rec["signals"]):
            v = verdicts.get(i, {})
            scored.append({"signal_text": s["signal_text"], "passage_text": (s["passage_text"] or "")[:200],
                           "faithful": int(bool(v.get("faithful"))), "valid": int(bool(v.get("valid"))),
                           "reason": str(v.get("reason") or "")[:120]})
        by_pid[rec[ID_FIELD]]["scored"] = scored
    final = list(by_pid.values())
    open(score_path, "w").write("\n".join(json.dumps(d, ensure_ascii=True) for d in final))
    open(score_path + ".raw.jsonl", "w").write("\n".join(json.dumps(d, ensure_ascii=True) for d in raws))
    # aggregate
    n_pos = sum(1 for d in final if d["pos"])
    hits = sum(1 for d in final if d["pos"] and any(s["faithful"] and s["valid"] for s in d["scored"]))
    tot = sum(len(d["scored"]) for d in final)
    good = sum(1 for d in final for s in d["scored"] if s["faithful"] and s["valid"])
    agg = {"coverage": round(hits / max(n_pos, 1), 3), "precision": round(good / max(tot, 1), 3),
           "volume": round(tot / max(len(final), 1), 2), "n_pos": n_pos, "n_sig": tot, "n_good": good,
           "pos_hit": hits}
    print("JUDGE_DONE " + json.dumps(agg), flush=True)
    return agg


# ================= JUDGE CORPUS (Qwen over full deploy dump -> anchor pool, resumable+chunked) =================
def cmd_judge_corpus(gen_path, corpus_path, score_path, anchors_path, chunk=1000):
    recs = [json.loads(l) for l in open(gen_path)]
    art = {}
    for l in open(corpus_path):
        d = json.loads(l)
        art[d[ID_FIELD]] = d.get(INPUT_FIELD, "") or d.get("article_text", "") or d.get("text", "")
    os.environ.setdefault("VLLM_USE_FLASHINFER_MOE_FP8", "0")
    os.environ.setdefault("HF_HOME", "/lfs/skampere3/0/shared_hf_cache")
    done = set()
    if os.path.exists(score_path):
        for l in open(score_path):
            try:
                done.add(json.loads(l)[ID_FIELD])
            except Exception:
                pass
    todo = [r for r in recs if r.get(ID_FIELD) not in done and r.get("signals")]
    n_sig = sum(1 for r in recs if r.get("signals"))
    print("=== judge_corpus Qwen GPU " + os.environ.get("CUDA_VISIBLE_DEVICES", "?") +
          " | recs=" + str(len(recs)) + " with_sig=" + str(n_sig) +
          " done=" + str(len(done)) + " todo=" + str(len(todo)) + " ===", flush=True)
    from vllm import LLM, SamplingParams
    llm = LLM(model=QWEN, dtype="auto", gpu_memory_utilization=float(os.environ.get("GPU_MEM_UTIL", "0.90")),
              trust_remote_code=True, limit_mm_per_prompt={"image": 0, "video": 0, "audio": 0}, max_model_len=16384)
    sp = SamplingParams(temperature=0.0, max_tokens=1500)
    fscore = open(score_path, "a")
    fanch = open(anchors_path, "a")
    for i in range(0, len(todo), chunk):
        batch = todo[i:i + chunk]
        def _clean(s):
            """Strip lone surrogates that crash HF tokenizers."""
            return s.encode("utf-8", errors="replace").decode("utf-8") if s else ""
        convos = []
        for rec in batch:
            a = _clean((art.get(rec[ID_FIELD]) or "")[:MAXCHARS])
            lines = ["ARTICLE:", a, "", "Extracted signals to score:"]
            for j, s in enumerate(rec["signals"]):
                lines.append("[" + str(j) + "] signal: " + _clean(str(s.get("signal_text") or "")) +
                             "  | passage: " + _clean(str(s.get("passage_text") or ""))[:300])
            convos.append([{"role": "system", "content": JUDGE_SYS},
                           {"role": "user", "content": "\n".join(lines)}])
        outs = llm.chat(convos, sp, chat_template_kwargs={"enable_thinking": False}, use_tqdm=False)
        for rec, o in zip(batch, outs):
            raw = o.outputs[0].text if o.outputs else ""
            verdicts = parse_verdicts(raw)
            scored = []
            for j, s in enumerate(rec["signals"]):
                v = verdicts.get(j, {})
                f = int(bool(v.get("faithful")))
                vv = int(bool(v.get("valid")))
                sc = {"signal_text": s["signal_text"], "passage_text": str(s.get("passage_text") or "")[:200],
                      "faithful": f, "valid": vv, "reason": str(v.get("reason") or "")[:120]}
                scored.append(sc)
                if f and vv:
                    fanch.write(json.dumps({ID_FIELD: rec[ID_FIELD], **sc}, ensure_ascii=True) + "\n")
            fscore.write(json.dumps({ID_FIELD: rec[ID_FIELD], "n_sig": len(scored),
                                     "scored": scored}, ensure_ascii=True) + "\n")
        fscore.flush()
        fanch.flush()
        print("  chunk " + str(i // chunk + 1) + ": " + str(min(i + chunk, len(todo))) + "/" +
              str(len(todo)) + "  anchors=" + str(sum(1 for _ in open(anchors_path))), flush=True)
    fscore.close()
    fanch.close()
    print("JUDGE_CORPUS_DONE -> " + score_path + " + " + anchors_path, flush=True)


# ================= MUTATE (GLM via zai_anthropic subscription API, 0 GPU) =================
def glm_call(prompt, system, max_tokens, temperature, retries=4):
    body = {"model": "glm-5.2", "max_tokens": max_tokens, "messages": [{"role": "user", "content": prompt}]}
    if system:
        body["system"] = system
    body["temperature"] = temperature
    last = ""
    for a in range(retries):
        key = None
        for kf in KEYFILES:
            try:
                key = open(kf).read().strip()
                break
            except Exception:
                continue
        if not key:
            print("  glm: no key file found", flush=True); return last
        req = urllib.request.Request(ZAI_URL, data=json.dumps(body).encode(), method="POST",
                                     headers={"x-api-key": key, "anthropic-version": "2023-06-01",
                                              "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=150) as r:
                obj = json.loads(r.read().decode())
            if obj.get("type") == "error" or "error" in obj:
                raise RuntimeError(str(obj.get("error") or obj))
            c = obj.get("content") or []
            last = c[0]["text"] if c and isinstance(c[0], dict) else ""
            return last
        except urllib.error.HTTPError as e:
            msg = "HTTP " + str(e.code)
            # rotate keys on 429 (quota); also try the OTHER key this attempt
            if e.code == 429 and len(KEYFILES) > 1:
                KEYFILES.append(KEYFILES.pop(0))  # rotate to back so next attempt tries the other key
            print("  glm retry " + str(a) + ": " + msg + " (rotated keys)", flush=True)
            time.sleep(3.0 * (a + 1))
        except Exception as e:
            print("  glm retry " + str(a) + ": " + str(e)[:120], flush=True)
            time.sleep(2.5 * (a + 1))
    return last


MUT_SYS = ("You are a prompt-optimization engine (GEPA) for a norm-extraction system on " + DOMAIN +
           ". You revise the SEMANTIC fields of an extraction prompt to fix concrete "
           "failure modes, keeping what works and changing only what is broken. Never invent rubric IDs. "
           "Respond with ONLY a JSON object.")


def cmd_mutate(base_cfg_path, score_path, gen_path, new_cfg_path, eval_path=None):
    cfg = json.load(open(base_cfg_path))
    recs = [json.loads(l) for l in open(score_path)]
    # collect failures: signals that are unfaithful OR invalid, prefer ones with reasons
    fails = []
    tot = 0
    for d in recs:
        tot += len(d["scored"])
        for s in d["scored"]:
            if not (s["faithful"] and s["valid"]):
                fails.append(s)
    good = sum(1 for d in recs for s in d["scored"] if s["faithful"] and s["valid"])
    prec = round(good / max(tot, 1), 3)
    agg_cov = st.mean([1.0 if any(s["faithful"] and s["valid"] for s in d["scored"]) else 0.0
                       for d in recs if d["pos"]]) if any(d["pos"] for d in recs) else 0.0
    # ---- COLD-START: extractor returned ZERO signals. Show GLM the positive eval
    # texts it should have mined, instead of an empty failure block. ----
    cold = (tot == 0)
    pos_texts = []
    if cold and eval_path and os.path.exists(eval_path):
        try:
            _ev = [json.loads(l) for l in open(eval_path)]
        except Exception:
            _ev = []
        pos_texts = [d.get("text", "") for d in _ev if (d.get("pos") or d.get("is_positive")) and d.get("text", "")]
        pos_texts = [t for t in pos_texts if t.strip()][:8]
    if cold and pos_texts:
        print("  COLD-START: 0 signals extracted; showing GLM %d positive texts" % len(pos_texts), flush=True)

    fails = fails[:24]
    if cold and pos_texts:
        failblock = "\n".join("POSITIVE-TEXT-%d (extractor returned NOTHING from this): %s" % (i + 1, t[:900])
                              for i, t in enumerate(pos_texts)) or "(none)"
    else:
        failblock = "\n".join(
            "- signal: {sig} || passage: {pas} || faithful={f} valid={v} || reason: {r}".format(
                sig=f["signal_text"][:140], pas=f["passage_text"][:140],
                f=f["faithful"], v=f["valid"], r=f["reason"])
            for f in fails[:MIN_EXAMPLES]) or "(none)"
    if cold:
        goal_text = ("GOAL: raise COVERAGE from 0. The extractor returned ZERO normative signals from the POSITIVE texts below. "
                     "These texts DO contain genuine advisory/corrective/evaluative norms -- the prompt is failing to surface them. "
                     "Rewrite role/inline_evidence_example/polarity_hint so that Gemma-4 EXTRACTS concrete normative signals (short quoted "
                     "phrases + their polarity) from texts like these. Cast a wide net; do not raise the bar so high that nothing qualifies. "
                     "Match the few-shot output schema exactly (passages[].signals[]).\n")
    elif OBJECTIVE == "yield":
        goal_text = ("GOAL: maximize YIELD -- the COUNT of faithful+valid anchors (currently " + str(good) + " of " + str(tot) + " signals). "
                     "Cast a WIDE NET: extract MORE candidate normative signals from each text so more genuine advisory/corrective norms are captured. "
                     "Do NOT over-prune or raise the extraction bar too high -- signals later filtered as factual/non-normative are an acceptable cost. "
                     "Keep validity-precision at or above " + str(YIELD_FLOOR) + " (currently " + str(prec) + "); below that the firehose hurts more than the volume helps.\n")
    elif prec < FLOOR:
        goal_text = ("GOAL: validity-PRECISION is BELOW the floor (" + str(prec) + " < " + str(FLOOR) + "), so your PRIMARY aim is to RAISE precision. "
                     "The failure examples are being extracted but are NOT valid normative judgments -- they are factual recitations of law/rules, bare citations, "
                     "or neutral factual statements rather than advisory/corrective norms. Sharpen the prompt to extract ONLY genuine ADVISORY/CORRECTIVE norms "
                     "(what the reader should DO, deadlines to act on, errors in their understanding flagged) and STOP extracting bare statements of fact or citations. "
                     "Hold coverage near its current " + str(round(agg_cov, 3)) + ".\n")
    else:
        goal_text = ("GOAL: raise COVERAGE on signal-rich examples WITHOUT dropping validity-precision below " + str(FLOOR) + " (currently " + str(prec) + ").\n")
    prompt = (
        "CURRENT PROMPT semantic fields (the rest of the config is structural, do not change):\n"
        "role: " + cfg.get("role", "") + "\n"
        "inline_evidence_example: " + cfg.get("inline_evidence_example", "") + "\n"
        "polarity_hint: " + cfg.get("polarity_hint", "") + "\n\n"
        "EVAL (" + DOMAIN + "): coverage(positives with >=1 faithful+valid signal)=" + str(round(agg_cov, 3)) + ", validity-precision=" + str(prec) + ".\n\n"
        "FAILURE EXAMPLES (extracted signals the judge rejected as unfaithful or not-a-valid-normative-judgment):\n" + failblock + "\n\n"
        + goal_text +
        "Address the specific failures above. Prefer SHARP positive guidance (what TO extract) over empty-by-default rhetoric. "
        "Output JSON with keys role, inline_evidence_example, polarity_hint (rewrite any/all of them).")
    print("=== GLM mutate (zai_anthropic) ===", flush=True)
    out = glm_call(prompt, MUT_SYS, max_tokens=3200, temperature=GLM_TEMPERATURE)
    out2 = out.strip()
    out2 = re.sub(r"^```[a-zA-Z]*\s*\n?", "", out2)   # strip leading ```json fence
    out2 = re.sub(r"\n?```\s*$", "", out2)             # strip trailing fence
    i = out2.find("{")
    if i < 0:
        print("MUTATE_FAIL no JSON; raw head: " + out[:200], flush=True)
        return False
    _blob = out2[i:]
    try:
        new = json.loads(_blob)
    except Exception:
        try:
            import json_repair
            new = json_repair.loads(_blob)   # tolerates trailing truncation
        except Exception:
            new = None
    if not isinstance(new, dict) or "role" not in new:
        print("MUTATE_FAIL bad JSON; raw head: " + out[:300], flush=True)
        return False
    cfg2 = dict(cfg)
    for k in ("role", "inline_evidence_example", "polarity_hint"):
        if isinstance(new.get(k), str) and new[k].strip():
            cfg2[k] = new[k].strip()
    cfg2["task"] = cfg.get("task", "x") + "_mut"
    json.dump(cfg2, open(new_cfg_path, "w"), indent=2, ensure_ascii=True)
    print("MUTATE_DONE -> " + new_cfg_path + "  (changed: " +
          ", ".join(k for k in ("role", "inline_evidence_example", "polarity_hint") if cfg2[k] != cfg.get(k, "")) + ")", flush=True)
    return True



# ================= JUDGE (GLM-5.2 via subscription API, 0 GPU) =================
# Same I/O contract as cmd_judge -> writes score file consumed by read_agg/cmd_mutate.
# Used for the GEPA EVAL loop only (small eval set). Bulk deploy still uses cmd_judge_corpus (Qwen, GPU).
def cmd_judge_glm(gen_path, mode, score_path, workers=8):
    rows = {r[ID_FIELD]: r for r in load_eval()}
    recs = [json.loads(l) for l in open(gen_path)]
    jobs = []
    for rec in recs:
        r = rows.get(rec[ID_FIELD])
        if not r or not rec.get("signals"):
            continue
        art = article_for_judge(r, mode)
        lines = ["ARTICLE:", art, "", "Extracted signals to score:"]
        for i, sg in enumerate(rec["signals"]):
            lines.append("[" + str(i) + "] signal: " + str(sg.get("signal_text") or "") +
                         "  | passage: " + str(sg.get("passage_text") or "")[:300])
        jobs.append((rec, "\n".join(lines)))
    print("=== GLM-5.2 judge (API, 0 GPU): " + str(len(jobs)) + " recs with signals ===", flush=True)
    raw_map = {}
    from concurrent.futures import ThreadPoolExecutor
    def _do(job):
        rec, prompt = job
        raw = glm_call(prompt, JUDGE_SYS, max_tokens=1500, temperature=0.0)
        return rec[ID_FIELD], (raw or "")
    if jobs:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for pid, raw in ex.map(_do, jobs):
                raw_map[pid] = raw
    by_pid = {rec[ID_FIELD]: dict(rec, scored=[]) for rec in recs}
    raws = []
    for rec in recs:
        if not rec.get("signals"):
            continue
        raw = raw_map.get(rec[ID_FIELD], "")
        verdicts = parse_verdicts(raw)
        scored = []
        for i, sg in enumerate(rec["signals"]):
            v = verdicts.get(i, {})
            scored.append({"signal_text": sg["signal_text"], "passage_text": (sg.get("passage_text") or "")[:200],
                           "faithful": int(bool(v.get("faithful"))), "valid": int(bool(v.get("valid"))),
                           "reason": str(v.get("reason") or "")[:120]})
        by_pid[rec[ID_FIELD]]["scored"] = scored
        raws.append({ID_FIELD: rec[ID_FIELD], "raw": raw[:4000]})
    final = list(by_pid.values())
    open(score_path, "w").write("\n".join(json.dumps(d, ensure_ascii=True) for d in final))
    open(score_path + ".raw.jsonl", "w").write("\n".join(json.dumps(d, ensure_ascii=True) for d in raws))
    n_pos = sum(1 for d in final if d["pos"])
    hits = sum(1 for d in final if d["pos"] and any(sg["faithful"] and sg["valid"] for sg in d["scored"]))
    tot = sum(len(d["scored"]) for d in final)
    good = sum(1 for d in final for sg in d["scored"] if sg["faithful"] and sg["valid"])
    agg = {"coverage": round(hits / max(n_pos, 1), 3), "precision": round(good / max(tot, 1), 3),
           "volume": round(tot / max(len(final), 1), 2), "n_pos": n_pos, "n_sig": tot,
           "n_good": good, "pos_hit": hits, "judge": "glm-5.2"}
    print("JUDGE_DONE " + json.dumps(agg), flush=True)
    return agg

# ================= RUN (driver) =================
def _drain_gpu(min_free_gib=165, timeout=150):
    """vLLM subprocesses don't release GPU memory synchronously on exit. Between cmd_run's
    alternating gen (Gemma) and judge (Qwen) steps, the prior EngineCore's allocation can linger
    and starve the next init (observed: 70 GiB residue -> Qwen needs 160 GiB -> ValueError).
    Poll CUDA_VISIBLE_DEVICES free mem until it's effectively drained (or timeout), then proceed."""
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    t0 = time.time(); last = -1
    try:
        while time.time() - t0 < timeout:
            out = subprocess.run(["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.free",
                                  "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip()
            last = int(out.split()[0]) if out else 0
            if last >= min_free_gib:
                break
            time.sleep(4)
        print("  [gpu-drain gpu" + str(gpu) + "] free=" + str(last) + "GiB after " +
              str(int(time.time() - t0)) + "s", flush=True)
    except Exception as e:
        print("  [gpu-drain] skip: " + str(e), flush=True)


def cmd_run(base_cfg, fewshot, mode, rounds, workdir, gpu):
    os.makedirs(workdir, exist_ok=True)
    os.environ["GEPA_CORPUS"] = mode  # force gen/judge subprocesses to load THIS corpus globals
    gpy = "/lfs/skampere3/0/alexspan/envs/gemma4/bin/python"
    qpy = "/lfs/skampere3/0/alexspan/envs/ai_usage/bin/python"
    me = os.path.abspath(__file__)
    # round 0: baseline
    r0_cfg, r0_fs = workdir + "/round0.json", fewshot
    json.dump(json.load(open(base_cfg)), open(r0_cfg, "w"), indent=2)

    def step(cmd):
        _drain_gpu()   # wait for prior vLLM subprocess to release GPU mem (teardown lag)
        print("\n>>> " + " ".join(cmd), flush=True)
        r = subprocess.run(cmd, capture_output=True, text=True)
        print(r.stdout, flush=True)
        if r.returncode != 0:
            print("STDERR:\n" + r.stderr[-2000:], flush=True)
        return r.returncode
    # baseline gen + judge
    step([gpy, me, "gen", r0_cfg, r0_fs, mode, workdir + "/gen0.jsonl"])
    cmd_judge_glm(workdir + "/gen0.jsonl", mode, workdir + "/score0.jsonl")  # GLM-5.2 judge (0 GPU)
    agg0 = read_agg(workdir + "/score0.jsonl")
    best = {"cov": agg0["coverage"], "prec": agg0["precision"], "n_good": agg0["n_good"], "cfg": r0_cfg, "agg": agg0}
    print("ROUND0 coverage=" + str(agg0["coverage"]) + " precision=" + str(agg0["precision"]), flush=True)
    for rd in range(1, int(rounds) + 1):
        prev_score = workdir + "/score" + str(rd - 1) + ".jsonl"
        prev_gen = workdir + "/gen" + str(rd - 1) + ".jsonl"
        new_cfg = workdir + "/round" + str(rd) + ".json"
        _eval_path = os.path.join(os.path.dirname(workdir.rstrip("/")), "gepa_eval_pairs.jsonl")
        ok = cmd_mutate(best["cfg"], prev_score, prev_gen, new_cfg, eval_path=_eval_path)  # in-proc (API, no GPU)
        if not ok:
            print("round " + str(rd) + " mutate failed, stopping", flush=True); break
        step([gpy, me, "gen", new_cfg, r0_fs, mode, workdir + "/gen" + str(rd) + ".jsonl"])
        cmd_judge_glm(workdir + "/gen" + str(rd) + ".jsonl", mode, workdir + "/score" + str(rd) + ".jsonl")  # GLM-5.2 judge
        agg = read_agg(workdir + "/score" + str(rd) + ".jsonl")
        print("ROUND" + str(rd) + " coverage=" + str(agg["coverage"]) + " precision=" + str(agg["precision"]), flush=True)
        # Keep: precision >= FLOOR AND F1(cov,prec) improves. F1 is symmetric, so it handles both
        # coverage-limited corpora (press_releases: raise recall) and precision-limited ones
        # (legaladvice_uk: raise specificity) without favoring one axis.
        f1 = lambda c, p: 0.0 if (c + p) == 0 else round(2 * c * p / (c + p), 3)
        if OBJECTIVE == "yield":
            better = agg["precision"] >= YIELD_FLOOR and agg["n_good"] > best["n_good"]
        else:
            better = agg["precision"] >= FLOOR and f1(agg["coverage"], agg["precision"]) > f1(best["cov"], best["prec"])
        if better:
            best = {"cov": agg["coverage"], "prec": agg["precision"], "n_good": agg["n_good"], "cfg": new_cfg, "agg": agg}
            print("  -> NEW BEST", flush=True)
        else:
            print("  -> not better (cov " + str(agg["coverage"]) + " vs " + str(best["cov"]) + ", prec " + str(agg["precision"]) + " vs floor " + str(FLOOR) + ")", flush=True)
    json.dump(best, open(workdir + "/best.json", "w"), indent=2)
    print("\n=== BEST cfg=" + best["cfg"] + " coverage=" + str(best["cov"]) + " precision=" + str(best.get("prec")), flush=True)


def read_agg(score_path):
    recs = [json.loads(l) for l in open(score_path)]
    n_pos = sum(1 for d in recs if d["pos"])
    hits = sum(1 for d in recs if d["pos"] and any(s["faithful"] and s["valid"] for s in d["scored"]))
    tot = sum(len(d["scored"]) for d in recs)
    good = sum(1 for d in recs for s in d["scored"] if s["faithful"] and s["valid"])
    return {"coverage": round(hits / max(n_pos, 1), 3), "precision": round(good / max(tot, 1), 3),
            "volume": round(tot / max(len(recs), 1), 2), "n_pos": n_pos, "n_sig": tot, "n_good": good, "pos_hit": hits}


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "gen":
        cmd_gen(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
    elif cmd == "judge":
        cmd_judge(sys.argv[2], sys.argv[3], sys.argv[4])
    elif cmd == "mutate":
        ok = cmd_mutate(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
        sys.exit(0 if ok else 1)
    elif cmd == "gen_corpus":
        cmd_gen_corpus(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
    elif cmd == "judge_corpus":
        cmd_judge_corpus(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
    elif cmd == "run":
        cmd_run(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7])
    else:
        print("usage: gepa_pr.py {gen|judge|mutate|run} ...")
