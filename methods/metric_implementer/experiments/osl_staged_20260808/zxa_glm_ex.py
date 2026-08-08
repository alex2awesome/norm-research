"""z-x-a EXEMPLAR arms on the GLM subscription ladder (user 2026-08-07: score the EXEMPLAR
arms with GLM-5.2 as a frontier receiver -> "do examples become useful beyond definitions in
the limit of model capacity"). Analog of zxa_glm.py (same backend/prompt/YES-NO conventions)
but reads freeze_zxa_ex_<task>_v1.json (exemplar freeze: exemplars_authored/_mm,
exemplars/_mm, def_exemplars) instead of the base freeze_zxa_<task>_v1.json (name/definition/
explanation/dossier/... — already scored, never re-touched here).

Budget discipline (hard rules from the job spec):
  - deterministic 150/300-probe subset: keep probe j iff
    md5("glmlimit|"+probe_text) % 2 == 0 (recorded in the sidecar json).
  - SMOKE (40 calls, mixed arms) -> token/call projection -> HARD CAP (env ZXAEX_TOKEN_CAP,
    default 35,000,000 prompt+completion tokens) enforced via a sidecar-persisted running
    counter (llm.stats deltas), checked BEFORE each rubric's batch is issued.
  - ANCHOR CHECK: all PLANTED-base rows (every arm) scored FIRST; balanced accuracy vs
    code-checkable ground truth (methods.metric_implementer.experiments.osl_sweep.planted_metrics)
    reported per arm before the non-planted majority is scored.
  - Resume-safe: progressive JSONL of (rubric_key, probe_idx, score); npz written at the end
    (and on every checkpoint) with m_bar aligned to the FULL 300-probe grid (NaN elsewhere).

Usage: zxa_glm_ex.py <model> <short> <task> <key_file> [phase]
  phase: all (default) | smoke | anchor | main | finalize
"""
import hashlib
import json
import os
import sys
import time

import numpy as np

B = "/lfs/skampere3/0/alexspan"
sys.path.insert(0, f"{B}/norm-research")
from methods.metric_implementer.backends import LLMBackend
from methods.metric_implementer import config as cfgmod
from methods.metric_implementer.experiments.run_real_test import _load_texts
from methods.metric_implementer.experiments import alpha_probe as ap
from methods.metric_implementer.experiments.osl_sweep import planted_metrics

MODEL, SHORT, TASK, KEYF = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
PHASE = sys.argv[5] if len(sys.argv) > 5 else "all"
os.environ["ZAI_KEY_FILE"] = KEYF

TOKEN_CAP = int(os.environ.get("ZXAEX_TOKEN_CAP", "35000000"))
N_PROBE_TARGET = int(os.environ.get("ZXAEX_N_PROBES", "0"))  # 0 = use full deterministic-kept set
ARM_PRIORITY = ["exemplars_authored", "exemplars_authored_mm", "exemplars", "exemplars_mm",
                "def_exemplars"]
SKIP_ARMS = set(os.environ.get("ZXAEX_SKIP_ARMS", "").split(",")) - {""}
SMOKE_N = 40

cfg0 = cfgmod.ImplementerConfig()
cfg0.backend = "zai_anthropic"
cfg0.llm_concurrency = min(int(os.environ.get("ZXA_CONC", "8")), 8)  # spec: never >8 in flight
llm = LLMBackend(MODEL, role="osl_probe", cfg=cfg0, temperature=0.0)


def yn(t):
    return (t or "").strip().upper()


def balanced(pred, ref, min_per=8):
    m = np.isfinite(pred) & np.isfinite(ref)
    if m.sum() < 30:
        return float("nan")
    p, r = pred[m], ref[m]
    accs = [float(np.mean(p[r == v] == v)) for v in (0, 1) if (r == v).sum() >= min_per]
    return float(np.mean(accs)) if len(accs) == 2 else float("nan")


# ---------------------------------------------------------------------------------------- setup
frz = json.load(open(f"{B}/outputs/osl_multi/freeze_zxa_ex_{TASK}_v1.json"))
meta = frz["meta"]
task_cfg = meta["task"]
cfg = cfgmod.apply_task_preset(cfgmod.ImplementerConfig(), task_cfg)
n_probes = int(meta["n_probes"])
texts, _ = _load_texts(task_cfg, 60 + n_probes, cfg)
probes = texts[60: 60 + n_probes]
mxc = cfg.max_text_chars
assert len(probes) == n_probes, f"expected {n_probes} probes, got {len(probes)}"

kept_idx = [j for j, t in enumerate(probes)
           if int(hashlib.md5(("glmlimit|" + t).encode()).hexdigest(), 16) % 2 == 0]
if N_PROBE_TARGET and N_PROBE_TARGET < len(kept_idx):
    kept_idx = kept_idx[:N_PROBE_TARGET]   # budget-shrink lever #3 (reduce probes), stable prefix
print(f"[{SHORT}|{TASK}] probe universe {len(probes)} -> deterministic kept {len(kept_idx)}",
      flush=True)

truth = {m["name"]: np.asarray(m["truth"], float)
        for m in planted_metrics([t[:mxc] for t in probes], int(meta["k_med_words"]))}

metrics = [m for m in frz["metrics"]
          if m["zxa"]["arm"] in ARM_PRIORITY and m["zxa"]["arm"] not in SKIP_ARMS]


def sortkey(m):
    is_planted = m["zxa"]["class"] == "PLANTED"
    return (0 if is_planted else 1, ARM_PRIORITY.index(m["zxa"]["arm"]), m["zxa"]["base"])


metrics.sort(key=sortkey)
n_planted = sum(1 for m in metrics if m["zxa"]["class"] == "PLANTED")
print(f"[{SHORT}|{TASK}] {len(metrics)} rubrics ({n_planted} planted-anchor, "
      f"{len(metrics) - n_planted} substantive) x up to {len(kept_idx)} probes", flush=True)

JSONL = f"{B}/outputs/osl_multi/mbar_zxaglmex_{TASK}_{SHORT}.jsonl"
OUT = f"{B}/outputs/osl_multi/mbar_zxaglmex_{TASK}_{SHORT}.npz"
SIDECAR = f"{B}/outputs/osl_multi/mbar_zxaglmex_{TASK}_{SHORT}.sidecar.json"

# ---------------------------------------------------------------------------------------- resume
have = {}   # rubric_key -> {probe_idx: score}
if os.path.exists(JSONL):
    for line in open(JSONL):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        have.setdefault(rec["rubric_key"], {})[int(rec["probe_idx"])] = rec["score"]
    n_have_pairs = sum(len(v) for v in have.values())
    print(f"[{SHORT}|{TASK}] resume: {len(have)} rubrics touched, {n_have_pairs} (rubric,probe) "
          f"pairs already scored", flush=True)

tok_state = {"prompt_tokens": 0, "completion_tokens": 0, "n_calls": 0}
if os.path.exists(SIDECAR):
    try:
        tok_state.update(json.load(open(SIDECAR)).get("token_state", {}))
    except Exception:
        pass


def total_tokens():
    return tok_state["prompt_tokens"] + tok_state["completion_tokens"]


def avg_tokens_per_call():
    return total_tokens() / tok_state["n_calls"] if tok_state["n_calls"] else None


def save_sidecar(extra=None):
    obj = {"task": TASK, "model": MODEL, "short": SHORT, "kept_idx": kept_idx,
          "n_probes_total": n_probes, "probe_window": meta.get("probe_window"),
          "token_state": tok_state, "token_cap": TOKEN_CAP,
          "avg_tokens_per_call": avg_tokens_per_call(), "updated": time.time()}
    if extra:
        obj.update(extra)
    json.dump(obj, open(SIDECAR, "w"), indent=1)


jf = open(JSONL, "a")


def score_batch(key, idxs):
    """Score `key` rubric's rubric text against probes[idxs]; append jsonl lines; update tok_state."""
    m = key_to_metric[key]
    prompts = [ap._YESNO_TEXTFIRST.format(text=probes[j][:mxc], rubric=m["rubric"]) for j in idxs]
    before = (llm.stats.prompt_tokens, llm.stats.completion_tokens, llm.stats.n_calls)
    outs = llm.generate_batch(prompts, max_tokens=8,
                              validate=lambda t: yn(t)[:3] in ("YES", "NO", "NO."))
    d_p = llm.stats.prompt_tokens - before[0]
    d_c = llm.stats.completion_tokens - before[1]
    d_n = llm.stats.n_calls - before[2]
    tok_state["prompt_tokens"] += d_p
    tok_state["completion_tokens"] += d_c
    tok_state["n_calls"] += d_n
    for j, o in zip(idxs, outs):
        s = 1.0 if yn(o).startswith("Y") else (0.0 if yn(o).startswith("N") else float("nan"))
        jf.write(json.dumps({"rubric_key": key, "probe_idx": j, "score": s}) + "\n")
        have.setdefault(key, {})[j] = s
    jf.flush()
    return d_p, d_c, d_n


key_to_metric = {m["name"]: m for m in metrics}

# ---------------------------------------------------------------------------------------- SMOKE
if PHASE in ("all", "smoke") and tok_state["n_calls"] == 0:
    # mixed-arm smoke: same base ("PLANTED-digit", present in all 5 arms), 8 probes/arm = 40 calls.
    smoke_base = "PLANTED-digit"
    smoke_keys = [f"{smoke_base}||{a}" for a in ARM_PRIORITY if f"{smoke_base}||{a}" in key_to_metric]
    n_each = SMOKE_N // max(len(smoke_keys), 1)
    tot_p = tot_c = tot_n = 0
    for key in smoke_keys:
        idxs = [j for j in kept_idx[:n_each] if j not in have.get(key, {})]
        if not idxs:
            continue
        dp, dc, dn = score_batch(key, idxs)
        tot_p += dp; tot_c += dc; tot_n += dn
        print(f"[SMOKE] {key[:60]:60s} n={dn} in/call={dp / max(dn,1):.0f} out/call={dc / max(dn,1):.0f}",
              flush=True)
    save_sidecar()
    if tot_n:
        avg_in, avg_out = tot_p / tot_n, tot_c / tot_n
        avg_tot = avg_in + avg_out
        remaining_pairs = sum(len(kept_idx) for m in metrics) - sum(len(v) for v in have.values())
        # remaining (rubric,probe) work at full scope, net of what smoke already covered
        n_total_pairs = len(metrics) * len(kept_idx)
        n_done_pairs = sum(len(v) for v in have.values())
        n_left = n_total_pairs - n_done_pairs
        projected_remaining = n_left * avg_tot
        projected_total = total_tokens() + projected_remaining
        print(f"[SMOKE] avg in/call={avg_in:.0f} out/call={avg_out:.0f} tot/call={avg_tot:.0f}",
              flush=True)
        print(f"[SMOKE] scope: {len(metrics)} rubrics x {len(kept_idx)} probes = {n_total_pairs} "
              f"pairs; done={n_done_pairs} left={n_left}", flush=True)
        print(f"[SMOKE] PROJECTED TOTAL TOKENS (this task, this model) = {projected_total:,.0f} "
              f"(cap={TOKEN_CAP:,})", flush=True)
        save_sidecar({"smoke": {"avg_in": avg_in, "avg_out": avg_out, "avg_tot": avg_tot,
                                "projected_total_this_task": projected_total,
                                "n_total_pairs_this_task": n_total_pairs}})
    if PHASE == "smoke":
        sys.exit(0)

def anchor_report():
    """PLANTED-base balanced accuracy per arm, over whatever's in `have` so far. Returns
    (mean_bal_acc_over_arms, per_arm_dict)."""
    per_arm = {}
    for arm in ARM_PRIORITY:
        accs = []
        for base in ("PLANTED-length-long", "PLANTED-question", "PLANTED-digit", "PLANTED-quote",
                    "PLANTED-length-short"):
            key = f"{base}||{arm}"
            if key not in have or base not in truth:
                continue
            row = np.full(n_probes, np.nan)
            for j, s in have[key].items():
                row[j] = s
            b = balanced(row, truth[base])
            if np.isfinite(b):
                accs.append(b)
        if accs:
            per_arm[arm] = float(np.mean(accs))
    overall = float(np.mean(list(per_arm.values()))) if per_arm else float("nan")
    return overall, per_arm


ANCHOR_MIN = float(os.environ.get("ZXAEX_ANCHOR_MIN", "0.85"))
planted_end = sum(1 for m in metrics if m["zxa"]["class"] == "PLANTED")

# ---------------------------------------------------------------------------------------- MAIN LOOP
aborted = False
anchor_checked = False
for mi, m in enumerate(metrics):
    key = m["name"]
    todo = [j for j in kept_idx if j not in have.get(key, {})]
    if todo:
        avgc = avg_tokens_per_call()
        if avgc is not None:
            projected_after = total_tokens() + avgc * len(todo)
            if projected_after > TOKEN_CAP:
                print(f"[{SHORT}|{TASK}] ABORT before '{key[:60]}': projected {projected_after:,.0f} "
                      f"> cap {TOKEN_CAP:,}. Saved partial results.", flush=True)
                aborted = True
                break
        dp, dc, dn = score_batch(key, todo)
        row = have[key]
        yes_rate = np.nanmean([v for v in row.values()]) if row else float("nan")
        print(f"[{SHORT}|{TASK}] {mi + 1}/{len(metrics)} {key[:56]:56s} n={dn} "
              f"yes={yes_rate:.2f} tot_tok={total_tokens():,.0f}", flush=True)
        if (mi + 1) % 5 == 0:
            save_sidecar()

    # ANCHOR CHECK: once every PLANTED row has been attempted (all 5 arms x 5 code-checkable
    # bases), report bal-acc per arm as a diagnostic. The MANDATORY gate (standing rule: verify
    # the instrument before burning the rest of the budget) is on the pre-existing "definition"
    # arm's planted balanced accuracy, which is verified UP FRONT (zero-cost, from the
    # already-scored mbar_zxaglm_<task>_<short>.npz) before this job is ever launched -- for
    # humor/glm-5.2 that was 0.932, comfortably >= 0.85, so the run proceeds.
    # NOTE: this per-arm exemplar readout is NOT symmetrically gate-worthy: the two mismatch
    # arms (exemplars_authored_mm / exemplars_mm) are a deliberate placebo -- swapped-in
    # examples from a DIFFERENT base while the true base name/rule is still shown -- so
    # collapse-toward-chance on THOSE is the CORRECT signature of a working instrument (content
    # matters), not a failure. Only the clean arms (exemplars_authored, exemplars, def_exemplars)
    # are informative about instrument health, and even those are expected to sit below the
    # definition-arm bar for PLANTED/structural rules specifically (bare exemplars can't convey
    # an exact numeric threshold the way an explicit rule statement does) -- that gap is close to
    # the actual research question, not noise to gate on.
    if not anchor_checked and (mi + 1) == planted_end and PHASE != "smoke":
        anchor_checked = True
        overall, per_arm = anchor_report()
        clean_mean = float(np.mean([per_arm[a] for a in
                                    ("exemplars_authored", "exemplars", "def_exemplars")
                                    if a in per_arm])) if per_arm else float("nan")
        print(f"[{SHORT}|{TASK}] === ANCHOR CHECK (planted, exemplar arms; diagnostic only): "
              f"overall={overall:.3f} clean_arms_mean={clean_mean:.3f} (mandatory gate already "
              f"passed pre-launch on the definition arm) === "
              + " ".join(f"{a}={b:.3f}" for a, b in per_arm.items()), flush=True)
        save_sidecar({"anchor_overall": overall, "anchor_per_arm": per_arm,
                     "anchor_clean_arms_mean": clean_mean, "anchor_is_diagnostic_not_gate": True})

save_sidecar({"aborted": aborted, "phase_completed": PHASE})
print(f"[{SHORT}|{TASK}] loop done (aborted={aborted}). total_tokens={total_tokens():,.0f} "
      f"n_calls={tok_state['n_calls']}", flush=True)

if PHASE == "smoke":
    sys.exit(0)

# ---------------------------------------------------------------------------------------- FINALIZE -> npz
names, kinds, MB = [], [], []
for m in metrics:
    key = m["name"]
    row = np.full(n_probes, np.nan, dtype=float)
    for j, s in have.get(key, {}).items():
        row[j] = s
    if np.all(np.isnan(row)):
        continue   # never attempted (e.g. dropped by an abort) -> not written
    names.append(key)
    kinds.append(m["kind"])
    MB.append(row)

arr = np.array(MB, float) if MB else np.zeros((0, n_probes))
np.savez(OUT, executor=SHORT, family="glm", params=0, hard_readout=1,
        names=np.array(names, dtype=object), kinds=np.array(kinds, dtype=object),
        m_bar=arr, per_form=arr[:, None, :] if len(MB) else np.zeros((0, 1, n_probes)),
        kept_idx=np.array(kept_idx, dtype=int))
print(f"[{SHORT}|{TASK}] FINALIZED -> {OUT} ({len(names)} rows x {n_probes} probes, "
      f"{len(kept_idx)} non-NaN cols)", flush=True)

# planted-arm anchor readout (diagnostic; always printed at finalize)
overall, per_arm = anchor_report()
print(f"\n[{SHORT}|{TASK}] === PLANTED anchor balanced-accuracy by arm (final): overall={overall:.3f} "
     f"=== " + " ".join(f"{a}={b:.3f}" for a, b in per_arm.items()), flush=True)
save_sidecar({"aborted": aborted, "finalized": True, "anchor_overall": overall,
             "anchor_per_arm": per_arm})
