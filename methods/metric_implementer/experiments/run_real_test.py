"""run_real_test — the END-TO-END real test of the within-class certificate pipeline.

Exercises the NEW components together on real metrics: the atomic-unit Ω architecture (harvest →
per-item scoring → orthogonalization), the small-Ω EXACT certificate (small_omega_brute_force), and
the large-Ω FALLBACK (large_omega). Role split (prompt-optimality theory):

  * **Target model X** (vLLM resident, --target-model): GEPA-optimizes its prompt, then scores every
    train+test datapoint under it. X is the executor E under study.
  * **Strong model** (API, --reconstructor-backend/--reconstructor-model, default GLM-4.6 on z.ai):
    the GEPA reviser that iterates X's prompt AND the reconstructor that induces m̂ from X's labels.
    GLM articulates; X executes — recovery R = I(M_X; exec(M̂_GLM)), no holistic anchor.

Per task, two phases:
  A. GEPA-optimize X's prompt   — `optimizer.improve` with `make_roles_mixed(judge=X-vLLM, reviser=GLM)`.
     Populates a per-task registry with X's optimized prompt (the no-anchor target metric).
  B. Certificate on X's output  — harvest Ω from A's lineage → score each criterion per-item with X
     → orthogonalize (atomic-unit filter) → write Ω rubric → `omega_certificate` (auto: exact K≤15,
     fallback K>15). Reconstruction inside recon_channel uses GLM (induce), X (re-execute).

sk3 (1 GPU per task, sequential):  pick a free GPU; HOME=/lfs; VLLM_GPU_MEM_UTIL modest.
  $PY -m methods.metric_implementer.experiments.run_real_test \
      --tasks math,creative-writing,code-review,peer-review \
      --target-model meta-llama/Llama-3.1-8B-Instruct --reconstructor-backend zai_anthropic \
      --reconstructor-model glm-5 --n-items 60 --budget 4 --rounds 3

Laptop dry-run (no GPU, no API): add --dry-run (FakeVLLM for X, a mock GLM) to exercise orchestration.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile

import numpy as np

from .. import config as cfgmod
from ..artifact import MetricArtifact
from ..backends import make_roles_mixed, LLMBackend
from ..optimizer import improve
from ..registry import Registry
from ..vllm_backend import make_judge_backend
from .harvest_gepa_omega import _dedup, _criteria, _parse_body
from .omega_certificate import OmegaCertificate
from .real_gamma import _YESNO, _signal, _median_split
from .orthogonalize import (orthogonalization_filter, submodular_tail_bound,
                            adversarial_saturation)
from .small_omega_brute_force import _compile


def _mock_glm():
    """A tiny stand-in GLM for --dry-run: returns a fixed rule so induce_* has something to parse.
    Same interface as LLMBackend.generate_batch (List[str] -> List[str]). No API call."""
    from ..backends import CallStats
    class _Mock:
        model = "mock-glm"
        def __init__(self): self.stats = CallStats()
        def generate_batch(self, prompts, system=None, max_tokens=600, validate=None,
                           temperature=None, seed=0):
            self.stats.n_calls += 1
            return ["A high-quality example is clear, complete, and well-structured." for _ in prompts]
        def generate(self, prompt, system=None, max_tokens=600, validate=None, temperature=None):
            self.stats.n_calls += 1
            return "A high-quality example is clear, complete, and well-structured."
        def score_binary(self, prompts, system=None, pos="YES", neg="NO"):
            self.stats.n_calls += 1
            return [0.5] * len(prompts)
    return _Mock()


def phase_a_gepa(task, texts, ids, target_model, strong_model, strong_backend, *, rounds, fake,
                 seed_body=None, registry_dir=None):
    """Phase A: GEPA-optimize X's prompt with GLM as reviser, X as judge. Returns the registry.

    ``seed_body``: a WARM START — if given (e.g. a liked p̂ loaded from a prior run's npz), GEPA
    refines THAT prompt instead of the bland "is this high-quality?" seed. This avoids the leniency
    drift: the bland seed has nothing to anchor on, so failures route to MECHANIZE/DECOMPOSE and the
    prompt collapses to a clamped sum (low T_prose). Starting from a detailed, already-discriminative
    p̂ lets the discrimination boost (w_disc) push spread instead of re-mechanizing from scratch.

    ``registry_dir``: a PERSISTENT named dir for the GEPA lineage (the default; falls back to a
    tempfile only if None). The lineage — every evolved version's rubric body, with parent/operator/
    round provenance — is the OPTIMIZATION-RELEVANCE source for the candidate pool and the missing-
    impact defense, and is re-harvestable by `harvest_gepa_omega`/`mine_gepa_omega`. The old
    `tempfile.mkdtemp` orphaned it in /tmp (no re-harvest, no crash recovery, no cross-run
    accumulation); a named dir fixes all three."""
    from ..backends import Roles
    cfg = cfgmod.apply_task_preset(cfgmod.ImplementerConfig(), task)
    cfg.vllm_fake = fake
    if not registry_dir:                                  # legacy ephemeral fallback
        registry_dir = tempfile.mkdtemp(prefix=f"gepa_{task}_")
    os.makedirs(registry_dir, exist_ok=True)
    registry = Registry(registry_dir)
    print(f"[{task} A] GEPA registry (persistent lineage): {registry_dir}")
    judge = make_judge_backend(target_model, cfg, cfg.judge_temperature)   # X, vLLM
    if strong_backend == "mock":
        # dry-run: no API key — construct Roles directly with the mock GLM for all strong roles
        mock = _mock_glm()
        roles = Roles(judge=judge, reviser=mock, reconstructor=mock,
                      acceptance_reconstructor=mock, generator=mock,
                      acceptance_generator=mock, grader=mock)
    else:
        roles = make_roles_mixed(judge, strong_model=strong_model, strong_backend=strong_backend,
                                 base_cfg=cfg)
    body = seed_body or f"Is this a high-quality {task} item? Answer YES or NO."
    seed = MetricArtifact(metric_id=f"{task}__quality", kind="prompt",
                          name=f"{task} quality", description=f"overall quality of a {task} item",
                          body=body)
    summary = improve(seed, texts, roles, cfg, registry, rounds=rounds, data_ids=ids,
                      caps=cfgmod.BudgetCaps(instruction_tokens=400, n_fewshots=1,
                                             optimizer_rounds=rounds),
                      log=lambda *a, **k: None)
    print(f"[{task} A] GEPA done: accepted={summary['accepted']} "
          f"seed={summary['seed_fidelity_acceptance']:.3f} → best={summary['best_fidelity_acceptance']:.3f}")
    return registry, cfg


def _prose_prompt_from_registry(registry):
    """Extract the prose prompt p̂ — the best (acceptance-highest) evolved prompt body from the GEPA
    run. The optimizer emits {operator, rubric(prose), rationale}; we take the rubric field, which is
    the natural-language prompt X executes. Falls back to the seed body if no version parses."""
    import glob
    from ..backends import parse_json_obj
    bodies = []
    for vf in sorted(glob.glob(f"{registry.root}/metrics/*/versions/v*__prompt.json")):
        try:
            body = json.load(open(vf)).get("body", "")
        except Exception:
            continue
        if not body:
            continue
        # the body may be a JSON {"rubric": "..."} (optimizer output) or raw prose (seed)
        obj = parse_json_obj(body) if isinstance(body, str) else None
        prose = (obj.get("rubric") if isinstance(obj, dict) else None) or body
        if isinstance(prose, str) and len(prose) > 20:
            bodies.append(prose)
    if not bodies:
        return None
    return bodies[-1]            # last = highest-version (GEPA runs append in order)


def _decompose_prose(strong_backend, prose, noun, max_k):
    """GLM decomposes the prose prompt p̂ into ≤max_k atomic criteria (the Ω). Reuses real_gamma's
    decomposer (LLM split-into-atomic-checks). This is the SEPARATE decomposition pass — Phase A's
    prose prompt stays prose; Ω is a post-hoc decomposition for the certificate."""
    from .real_gamma import _decompose
    return _decompose(strong_backend, prose, noun, max_k)


def _free_generate(backend, task, existing, noun, k, *, pos_ex=None, neg_ex=None):
    """GLM proposes NOVEL distinct criteria the pool is MISSING — source (d), the only Ω source not
    anchored to the executor's own prompt. Modeled on real_gamma._decompose but OPEN-ENDED + failure-
    informed (a light GEPA): the existing pool is the exclusion set, and when `pos_ex`/`neg_ex` are
    supplied (items a prior prompt got wrong) the prompt asks for criteria that would score those cases
    correctly. Returns ≤k criterion strings. temp 0.7 for novelty (vs _decompose's 0.3)."""
    import re as _re
    excl = "\n".join(f"- {c}" for c in (existing or [])[:40]) or "(none yet)"
    fail_blk = ""
    if pos_ex or neg_ex:
        pp = "\n".join(f"- {x}" for x in (pos_ex or [])[:6]) or "(none)"
        nn = "\n".join(f"- {x}" for x in (neg_ex or [])[:6]) or "(none)"
        fail_blk = ("\nSome current criteria MISJUDGE these cases — propose criteria that would score "
                    f"them CORRECTLY:\n  SHOULD endorse but currently miss:\n{pp}\n"
                    f"  SHOULD reject but currently endorse:\n{nn}\n")
    sys = f"You propose NOVEL, independently-checkable evaluation criteria for {task} items."
    prompt = (f"Task: evaluating {task} items ({noun}).\n\nExisting criteria (do NOT repeat or lightly "
              f"rephrase these):\n{excl}\n{fail_blk}\nPropose up to {k} NOVEL, DISTINCT, independently-"
              f"checkable YES/NO criteria the list is MISSING — each one positive sentence on a single "
              f"dimension the {noun} could be judged on, NOT already covered above. One criterion per "
              f"line, no numbering, no preamble.")
    raw = backend.generate(prompt, system=sys, max_tokens=500, temperature=0.7)
    lines = [_re.sub(r"^[\-\*\d\.\)\s]+", "", ln).strip() for ln in (raw or "").splitlines()]
    return [ln for ln in lines if len(ln) > 10][:k]


def _score_criteria_signals(backend, crits, texts, max_chars):
    """Score each mined criterion per-item with X → (N_items, n_criteria) binary signals X_e, AND a
    per-criterion SPREAD report (mean/std/base-rate/collapsed). A criterion whose P(YES) barely varies
    (std < 0.02) is NON-discriminative — a lenient executor makes every criterion induce the SAME
    near-constant partition, so orthogonalization collapses Ω (feedback_check_judge_score_distribution).
    The spread report is the diagnostic that separates under-mining (few candidates) from
    degeneracy (constant scores) as the cause of Ω-collapse."""
    signals, spread = [], []
    for c in crits:
        rub = f"Judge whether the item meets this criterion:\n- {c}"
        py = np.asarray(_signal(backend, rub, texts, max_chars), float)
        bin_ = _median_split(py).astype(float)
        signals.append(bin_)
        spread.append({"mean": float(np.mean(py)), "std": float(np.std(py)),
                       "base_rate": float(np.mean(bin_)),
                       "collapsed": bool(np.std(py) < 0.02)})
    S = np.column_stack(signals) if signals else np.zeros((len(texts), 0))
    return S, spread


def _candidate_pool(registry, prose, recon_be, noun, pool_max, deep_k, *, task=None,
                    scoped_criteria=None, free_gen_k=0):
    """The rich candidate pool R_pool (theory §6.5 step 1) = UNION of complementary sources:

      (a) GEPA lineage — criteria harvested from EVERY evolved version's rubric. OPTIMIZATION-RELEVANT
          (what moved X's acceptance) but model-relative and narrow — alone collapses Ω to ~3-6.
      (b) deep GLM decomposition of the final prose p̂ — a small supplement.
      (c) corpus coverage — the curated cluster hierarchy (mine_clusters R1+L0, ~thousands, task-level),
          OR, when `scoped_criteria` is given, the family's OWN children (R2-level certificate).
      (d) GLM free-generation of NOVEL criteria (`free_gen_k>0`) — the only source not anchored to X's
          prompt; failure-informed via the existing pool (light GEPA).

    (a)+(b)+(d) anchor in what is causally relevant to THIS executor X; (c) guarantees coverage of the
    normative universe (or the family). Behavioral CMI-orthogonalization downstream SELECTS which are
    discriminative on X's labels. `scoped_criteria` (R2 cert) replaces the whole-task corpus."""
    import glob
    from .mine_clusters import mine as _mine_cluster_pool
    pool = []
    try:                                            # (a) lineage harvest (structured criteria)
        for vf in sorted(glob.glob(f"{registry.root}/metrics/*/versions/v*__prompt.json")):
            try:
                obj = _parse_body(json.load(open(vf)).get("body"))
            except Exception:
                obj = None
            if obj:
                _criteria(obj, pool)
    except Exception:
        pass
    if recon_be is not None and prose:              # (b) deep decomposition of the final prose
        try:
            pool.extend(_decompose_prose(recon_be, prose, noun, deep_k))
        except Exception:
            pass
    n_ab = len(pool)
    if scoped_criteria:                             # (c) R2 cert: family's own children (scoped)
        pool.extend([c for c in scoped_criteria if isinstance(c, str) and len(c) > 4])
        c_tag = f"scoped={len(scoped_criteria)}"
    elif task:                                      # (c) task-level: curated cluster hierarchy
        try:
            pool.extend(_mine_cluster_pool(task, levels=("R1", "L0")))
        except Exception as e:
            print(f"  [_candidate_pool] cluster mine failed for {task}: {e}")
        c_tag = "cluster"
    else:
        c_tag = "none"
    n_abc = len(pool)
    if free_gen_k and recon_be is not None:         # (d) GLM free-generation of novel criteria
        try:
            pool.extend(_free_generate(recon_be, task, pool, noun, free_gen_k))
        except Exception as e:
            print(f"  [_candidate_pool] free-gen failed: {e}")
    n_d = len(pool) - n_abc
    pool = _dedup([c for c in pool if isinstance(c, str) and len(c) > 12]) or []
    print(f"  [_candidate_pool] (a+b)={n_ab} {c_tag}={n_abc - n_ab} freegen={n_d} "
          f"→ {len(pool)} dedup'd (capped to {pool_max})")
    return pool[:pool_max]


def _missing_impact(omega_out, X_omega, texts, K_omega, *, delta=0.005):
    """The missing-IMPACT defense (theory §6.7c): bounds how much an UNSEEN criterion could MOVE
    recovery — "could a criterion we didn't find change the answer?" NOT missing-mass (Good–Turing,
    dropped). (1) submodular tail-bound: max_{e∉Ω}Δ(e|Ω) ≤ smallest greedy gain, given tail-γ≈1;
    (2) adversarial saturation: I(probe;M|X_Ω)≈0 for side-channel probes (length is the canonical one).
    CLAIMABLE iff certified_bound < δ AND probes saturate. Needs |Ω|≥3 for the tail to even begin
    decaying (realistically ~8-25 to cross δ)."""
    if X_omega.shape[1] < 3:
        return {"claimable": False, "K_omega": int(K_omega),
                "reason": "|Ω|<3 — greedy-gain tail cannot decay; keep mining (need ~8-25)"}
    d = np.load(omega_out, allow_pickle=True)        # M (target) was persisted by OmegaCertificate
    M_mi = np.asarray(d["M"], int)
    tb = submodular_tail_bound(M_mi, X_omega)
    cb = float(tb.get("certified_bound", float("nan")))
    loo = float(tb.get("tail_bound", float("nan")))
    gains = [round(float(g), 4) for g in tb.get("marginal_gains", []) if np.isfinite(g)]
    L = np.array([len(t) for t in texts])
    probes = {"length_tophalf": (L >= np.median(L)).astype(int)}
    sat = {n: bool(adversarial_saturation(M_mi, X_omega, p[:, None]).get("saturated"))
           for n, p in probes.items()}
    claim = bool(np.isfinite(cb) and cb < delta and all(sat.values()))
    return {"K_omega": int(K_omega), "delta_tol": delta,
            "tail_bound_certified": cb, "tail_bound_loo": loo, "greedy_gains": gains,
            "saturation": sat, "claimable": claim,
            "note": "missing-IMPACT: max unseen criterion marginal ≤ certified_bound (given tail-γ≈1); "
                    "claimable iff certified_bound < δ AND side-channel probes saturate."}


def _save_missing_impact_diag(omega_out, crits, omega, spread, filt, mi_defense):
    """Persist the Ω-collapse / missing-impact diagnostics so a run tells us WHY Ω is the size it is."""
    json.dump({"n_candidates_mined": len(crits), "n_collapsed_signals": sum(s["collapsed"] for s in spread),
               "orth_kept": filt.get("kept"), "n_kept": filt.get("n_kept"),
               "n_dropped": filt.get("n_candidates", 0) - filt.get("n_kept", 0),
               "per_criterion_spread": spread, "omega": omega, "missing_impact": mi_defense},
              open(f"{omega_out}.missing_impact.json", "w"), indent=2)


def phase_b_certificate(task, registry, cfg, target_model, strong_model, strong_backend, *,
                        n_items, budget, large_k, fake, omega_out, max_k=9, compiler="conjunction",
                        cmi_thresh=0.02, deep_k=25, pool_max=60, scoped_criteria=None, free_gen_k=0):
    """Phase B: rich candidate pool → orthogonalize → certify → missing-impact defense.

    R_pool (§6.5) = criteria from the FULL GEPA lineage + a deep decomposition of p̂ (the old path
    decomposed only the final prompt → Ω collapsed to 2-6 → missing-impact unmeasurable). Then the
    within-class certificate, then the missing-IMPACT defense (§6.7c). Per-criterion signal spread is
    logged to diagnose leniency-collapse (a constant-P(YES) executor makes all criteria induce the same
    partition → Ω collapses)."""
    prose = _prose_prompt_from_registry(registry)
    if not prose:
        print(f"[{task} B] no prose prompt p̂ found in registry; skipping.")
        return {"task": task, "error": "no_prose_prompt"}
    print(f"[{task} B] prose p̂ ({len(prose)} chars): {prose[:100]!r}")

    # ---- rich candidate pool: lineage harvest + deep decomposition ----
    if strong_backend == "mock":
        crits = (list(scoped_criteria)[:pool_max] if scoped_criteria else
                 ["the item is clear and well-organized", "the item is complete and accurate",
                  "the item uses appropriate style", "the item is engaging", "the item is accurate"])
        recon_be = None
    else:
        rcfg = cfgmod.ImplementerConfig(); rcfg.backend = strong_backend
        recon_be = LLMBackend(strong_model, "reconstructor", rcfg)
        crits = _candidate_pool(registry, prose, recon_be, cfg.item_noun, pool_max, deep_k, task=task,
                                scoped_criteria=scoped_criteria, free_gen_k=free_gen_k)
    if len(crits) < 2:
        print(f"[{task} B] only {len(crits)} candidates mined — need ≥2; skipping.")
        return {"task": task, "error": "too_few_criteria", "n_candidates": len(crits), "prose_len": len(prose)}
    print(f"[{task} B] mined {len(crits)} candidate criteria (lineage + deep decomposition)")

    texts, _ = _load_texts(task, n_items, cfg)
    if len(texts) < 20:
        print(f"[{task} B] only {len(texts)} items loadable; skipping.")
        return {"task": task, "error": "too_few_items", "n_items": len(texts)}

    backend = make_judge_backend(target_model, cfg)   # X, vLLM — scores the criteria
    signals, spread = _score_criteria_signals(backend, crits, texts, cfg.max_text_chars)
    n_collapsed = sum(1 for s in spread if s["collapsed"])
    print(f"[{task} B] signal spread: {n_collapsed}/{len(crits)} criteria collapsed (std<0.02 → "
          f"non-discriminative); base-rates {[round(s['base_rate'], 2) for s in spread[:8]]}")
    filt = orthogonalization_filter(signals, cmi_thresh=cmi_thresh)
    omega = [crits[i] for i in filt["kept"]]
    print(f"[{task} B] orthogonalized {len(crits)} → {len(omega)} atomic units "
          f"(dropped {filt['n_candidates'] - filt['n_kept']} redundant; cmi_thresh={cmi_thresh})")
    if len(omega) < 2:
        print(f"[{task} B] orthogonalization collapsed to {len(omega)} — pool too redundant or signals "
              f"degenerate; skipping certificate.")
        _save_missing_impact_diag(omega_out, crits, omega, spread, filt, None)
        return {"task": task, "error": "omega_collapsed", "n_omega": len(omega),
                "n_candidates": len(crits), "n_collapsed": n_collapsed,
                "crits": crits, "signals": signals}

    rub_file = f"{omega_out}.rubric.txt"
    with open(rub_file, "w") as f:
        for c in omega:
            f.write(f"- {c}\n")
    pool = os.path.join(_TRIAL_DIR, _TRIAL_POOLS.get(task, f"pool_{task.replace('-', '_')}.jsonl.gz"))
    if not os.path.exists(pool):                     # fall back to competitive-code trial pool
        pool = os.path.join(_TRIAL_DIR, "pool_competitive_code.jsonl.gz")
    cert = OmegaCertificate(rubric_file=rub_file, pool=pool, text_col=cfg.text_column,
                            task=task, model=target_model, n_items=n_items, budget=budget,
                            large_k=large_k, fake=fake, out=omega_out,
                            compiler=compiler, prose_prompt=prose)  # ← p̂ so I(M, M_ω) is measured
    result = cert.run()
    decomp = result.get("decomposition", {}) or {}
    print(f"[{task} B] {result['mode']}: K={result['K']} subsets={result['subsets_scored']} "
          f"M_ω base-rate={result['M_base_rate']:.2f} | "
          f"I(M,M_ω)={decomp.get('I_M_Momega', float('nan')):.3f} "
          f"[T_prose={decomp.get('T_prose', float('nan')):.3f}, T_ω={decomp.get('T_omega', float('nan')):.3f}]")

    # ---- missing-IMPACT defense (§6.7c): can an unseen criterion change the answer? ----
    mi = _missing_impact(omega_out, signals[:, filt["kept"]], texts, len(omega))
    _save_missing_impact_diag(omega_out, crits, omega, spread, filt, mi)
    print(f"[{task} B] missing-impact: K_ω={mi.get('K_omega')} "
          f"tail_bound(certified)={mi.get('tail_bound_certified')} "
          f"saturation={mi.get('saturation')} claimable={mi.get('claimable')}")
    result["missing_impact"] = mi
    result["crits"] = crits
    result["signals"] = signals
    return result


# Local trial pools (laptop-safe; the full manifest points at sk3 dataset paths). Used as a fallback
# when a task's manifest corpus isn't on disk (e.g. dry-run off-sk3) and as the certificate's pool.
_TRIAL_POOLS = {
    "code-review": "pool_competitive_code.jsonl.gz",
    "creative-writing": "pool_creative_writing.jsonl.gz",
    "law": "pool_law.jsonl.gz",
}
_TRIAL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "trial")


def _load_texts(task, n_items, cfg):
    """Load up to n_items texts for `task`. Try the manifest corpus first (sk3 paths); fall back to
    the local trial pool so the driver runs off-sk3 (dry-run). Returns (texts, ids)."""
    # env-gated custom probe pool (Leg A full-article arms 2026-07-09): file is JSONL rows
    # {"id":..., "text":...}; caller slicing ([60:60+n]) applies unchanged, so pools ship
    # with 60 padding rows up front. No effect unless OSL_PROBES_FILE is set.
    if os.environ.get("OSL_PROBES_FILE"):
        import json as _json
        rows = [_json.loads(l) for l in open(os.environ["OSL_PROBES_FILE"])]
        return [r["text"] for r in rows][:n_items], [str(r["id"]) for r in rows][:n_items]
    from ..manifest import full_manifest, load_corpus
    entry = next((e for e in full_manifest().datasets if e.task == task), None)
    if entry is not None:
        try:
            texts, ids = load_corpus(entry, n_items, seed=7)
            if texts:
                return texts, ids
        except Exception:
            pass
    # trial-pool fallback
    import gzip, json
    fn = _TRIAL_POOLS.get(task)
    if fn and os.path.exists(os.path.join(_TRIAL_DIR, fn)):
        col = cfg.text_column
        rows = [json.loads(l) for l in gzip.open(os.path.join(_TRIAL_DIR, fn), "rt")][: n_items + 50]
        texts = [str(r.get(col, "")) for r in rows if str(r.get(col, "")).strip()][:n_items]
        return texts, [str(i) for i in range(len(texts))]
    return [], []


def main(argv=None):
    import sys
    ap = argparse.ArgumentParser(prog="run_real_test", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", default="math,creative-writing,code-review,peer-review")
    ap.add_argument("--target-model", default="meta-llama/Llama-3.1-8B-Instruct",
                    help="executor X (vLLM resident) — the target model under study")
    ap.add_argument("--reconstructor-backend", default="zai_anthropic",
                    choices=["zai_anthropic", "zai", "openrouter", "mock"],
                    help="API backend for the strong model (reviser+reconstructor). "
                         "'zai_anthropic' = glm-5 via z.ai subscription (free); 'mock' = dry-run.")
    ap.add_argument("--reconstructor-model", default="glm-5")
    ap.add_argument("--n-items", type=int, default=60)
    ap.add_argument("--budget", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=3, help="GEPA rounds in Phase A")
    ap.add_argument("--large-k", type=int, default=15, help="exact-vs-fallback threshold for omega_certificate")
    ap.add_argument("--force-fallback", action="store_true",
                    help="force the large-Ω fallback (set --large-k below the real K) to test that path")
    ap.add_argument("--warm-start-npz", default=None,
                    help="path to a prior run's omega_certificate npz; its saved prose_prompt p̂ is "
                         "used as the GEPA SEED (warm start) so the loop refines a liked, already-"
                         "discriminative prompt instead of the bland seed (avoids leniency drift).")
    ap.add_argument("--compiler", default="conjunction",
                    choices=["conjunction", "weighted_sum", "prose_join"],
                    help="C(Ω) framing for the certificate's all-criteria verdict (form axis).")
    ap.add_argument("--cmi-thresh", type=float, default=0.02,
                    help="orthogonalization CMI threshold; RAISE to keep fewer (cleaner) units, LOWER to "
                         "keep more (grow Ω toward the ~8-25 needed for missing-impact).")
    ap.add_argument("--deep-k", type=int, default=25, help="GLM decomposition depth of p̂ (candidate pool).")
    ap.add_argument("--pool-max", type=int, default=60, help="cap on the mined candidate pool size.")
    ap.add_argument("--out-dir", default="/tmp/real_test")
    ap.add_argument("--registry-dir", default=None,
                    help="persistent GEPA-lineage dir; default {out-dir}/{task}.registry. Saves "
                         "every evolved prompt body (parent/operator/round provenance) so the "
                         "candidate pool's GEPA source survives the run and is re-harvestable.")
    ap.add_argument("--dry-run", action="store_true", help="FakeVLLM for X + mock GLM (no GPU, no API)")
    a = ap.parse_args(argv)
    os.makedirs(a.out_dir, exist_ok=True)
    fake = a.dry_run
    sb = "mock" if a.dry_run else a.reconstructor_backend
    large_k = 2 if a.force_fallback else a.large_k        # force-fallback: any real K (>2) triggers it
    warm_body = None
    if a.warm_start_npz:
        wd = np.load(a.warm_start_npz, allow_pickle=True)
        warm_body = str(wd["prose_prompt"])
        print(f"[warm-start] seeding GEPA from saved p̂ ({len(warm_body)} chars) in {a.warm_start_npz}")

    from ..manifest import full_manifest
    man = full_manifest()
    by_task = {e.task: e for e in man.datasets}
    summary = []
    for task in [t.strip() for t in a.tasks.split(",") if t.strip()]:
        print(f"\n{'='*70}\nTASK: {task}\n{'='*70}")
        entry = by_task.get(task)
        cfg = cfgmod.apply_task_preset(cfgmod.ImplementerConfig(), task)
        texts, ids = _load_texts(task, a.n_items, cfg)
        if len(texts) < 20:
            print(f"  SKIP {task}: only {len(texts)} items loadable locally "
                  f"(manifest corpus may live on sk3)"); continue
        if len(texts) < 20:
            print(f"  SKIP {task}: only {len(texts)} items"); continue
        try:
            registry, cfg = phase_a_gepa(task, texts, ids, a.target_model, a.reconstructor_model,
                                         sb, rounds=a.rounds, fake=fake, seed_body=warm_body,
                                         registry_dir=(a.registry_dir or os.path.join(a.out_dir, f"{task}.registry")))
            result = phase_b_certificate(task, registry, cfg, a.target_model, a.reconstructor_model,
                                         sb, n_items=a.n_items, budget=a.budget, large_k=large_k,
                                         fake=fake, compiler=a.compiler,
                                         cmi_thresh=a.cmi_thresh, deep_k=a.deep_k, pool_max=a.pool_max,
                                         omega_out=os.path.join(a.out_dir, f"{task}.npz"))
            summary.append({"task": task, **result})
        except Exception as e:
            import traceback
            print(f"  FAIL {task}: {e}\n{traceback.format_exc()}")
            summary.append({"task": task, "error": str(e)})
    json.dump(summary, open(os.path.join(a.out_dir, "summary.json"), "w"), indent=2, default=str)
    print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    for s in summary:
        c = s.get("certificate", {}) if "certificate" in s else s
        if s.get("error"):
            print(f"  {s['task']:>18}: ERROR ({s['error']})")
        else:
            cert = c.get("certificate", c)
            print(f"  {s['task']:>18}: {s.get('mode','?')} K={s.get('K','?')} "
                  f"greedy/OPT={cert.get('greedy_R')}/{cert.get('OPT_R','') if 'OPT_R' in cert else cert.get('global_R','?')}")


if __name__ == "__main__":
    main()
