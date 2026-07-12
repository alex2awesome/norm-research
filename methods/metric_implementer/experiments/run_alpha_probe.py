"""§12.1a ALPHA-PROBE at the METRIC level — α_i per R2 cluster (NOT per task). Theory §12.1a (per §1).

The unit of analysis is ONE metric M_i (one R2 cluster); we optimize prompts to tag metrics, so α is
measured per metric: for each M_i, breadth-sample its Ω_i (atomic criteria) and estimate that metric's
B_E dimensionality. [[feedback_alpha_probe_is_metric_level]] — task-level α≈0.9 is the WRONG level
(non-discriminating); this is the per-metric read where simple metrics α_i<0.5 and complex ones α_i→1.

Ω_i per metric = ``children`` (r2_children, the cluster's atomic backbone) + ``gepa_criteria`` (mined
from M_i's GEPA prompt versions WHEN a registry artifact exists — deferred otherwise; GEPA Phase A is
GLM-gated) + within-metric free-gen from K diverse proposer families (the frozen-iid capture stream).
All scored by the ONE frozen executor over the TASK probe items. The driver ALSO computes M_i (the
metric's OWN verdict, ``metric_verdict`` on the cluster merged_description) and saves it in each
per-metric checkpoint — the anchor-free reconstruction target for the §12.3 value census (never Y).

EFFICIENCY: the vLLM executor loads ONCE and is reused across all clusters. CMI is skipped per-metric
(compute_cmi=False) — slow O(n²) and a τ-irrelevant diagnostic; recompute on a checkpoint if wanted.

Example (sk3, executor resident on 1 GPU):
    HOME=/lfs/... python -m methods.metric_implementer.experiments.run_alpha_probe \\
        --task creative-writing --r2-bucket general --n-metrics 30 --no-glm
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
from typing import Dict

import numpy as np

from .. import config as cfgmod
from ..backends import LLMBackend
from ..vllm_backend import make_judge_backend
from . import alpha_probe as ap
from .mine_clusters import (r1_groups, r1_children, r2_groups, r2_children,
                            r3_groups, r3_children)
from .run_real_test import _load_texts, _parse_body, _criteria

# Diverse model FAMILIES (genuinely different families — what makes multi-list capture-recapture
# informative). CLI-overridable via --families (JSON name->{"model","backend"}); drop GLM with --no-glm.
DEFAULT_FAMILIES: Dict[str, Dict[str, str]] = {
    "glm":     {"model": "glm-4.7", "backend": "zai_anthropic"},          # GLM family (quota-gated)
    "qwen":    {"model": "qwen/qwen-2.5-72b-instruct", "backend": "openrouter"},   # Qwen family
    "llama":   {"model": "meta-llama/llama-3.3-70b-instruct", "backend": "openrouter"},  # Llama
    "haiku":   {"model": "anthropic/claude-haiku-4.5", "backend": "openrouter"},   # Anthropic family
}


class _MockProposer:
    """Dry-run proposer: returns canned criteria so the plumbing runs with no API/GPU."""
    def __init__(self, name, seed=0):
        self.name = name
        self._rng = np.random.default_rng(abs(hash(name)) % 2**31 + seed)

    def generate(self, prompt, system=None, max_tokens=600, validate=None, temperature=None):
        n = 3 + int(self._rng.integers(0, 3))
        return "\n".join(f"The item demonstrates quality aspect {self.name}-{i} "
                         f"with score {int(self._rng.integers(1,9))}" for i in range(n))


def _build_families(a, base_cfg) -> Dict[str, object]:
    """Construct the K proposer backends (HTTP LLMBackends, or _MockProposer in dry-run)."""
    if getattr(a, "dry_run", False):
        return {name: _MockProposer(name) for name in ("A", "B", "C", "D")}
    fams = json.loads(a.families) if a.families else copy.deepcopy(DEFAULT_FAMILIES)
    if a.no_glm and "glm" in fams:
        del fams["glm"]
    if len(fams) < 2:
        raise SystemExit(f"need ≥2 diverse families for multi-list CRC; have {len(fams)} "
                         f"(--no-glm dropped GLM?)")
    out = {}
    for name, spec in fams.items():
        cfg = copy.copy(base_cfg)
        cfg.backend = spec["backend"]
        out[name] = LLMBackend(spec["model"], f"proposer_{name}", cfg)
    return out


def _slug(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return s[:60]


def _harvest_gepa(registry_root: str, slug: str) -> list:
    """Criteria mined from M_i's GEPA-optimized prompt versions (source (a)), IF a registry artifact
    exists for this metric. Empty for clusters without one (GEPA Phase A not yet run)."""
    import glob
    pool = []
    for vf in sorted(glob.glob(f"{registry_root}/metrics/{slug}/versions/v*__prompt.json")):
        try:
            obj = _parse_body(json.load(open(vf)).get("body"))
        except Exception:
            obj = None
        if obj:
            _criteria(obj, pool)
    return pool


def _groups_for_level(task: str, bucket: str, level: str):
    """Dispatch to the hierarchy accessor for `level`. R1 ignores `bucket` (single r1_families file)."""
    if level == "R1":
        return r1_groups(task)
    if level == "R3":
        return r3_groups(task, bucket)
    return r2_groups(task, bucket)


def _children_for_level(task: str, bucket: str, level: str, group_idx: int):
    if level == "R1":
        return r1_children(task, group_idx)
    if level == "R3":
        return r3_children(task, bucket, group_idx)
    return r2_children(task, bucket, group_idx)


def main(argv=None):
    ap_ = argparse.ArgumentParser(prog="run_alpha_probe", description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap_.add_argument("--task", default="creative-writing")
    ap_.add_argument("--r2-bucket", default="general", help="R2 bucket (creative-writing only has 'general')")
    ap_.add_argument("--target-model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap_.add_argument("--n-metrics", type=int, default=30, help="sample this many R2 clusters (0 = all)")
    ap_.add_argument("--metric-start", type=int, default=0, help="skip the first N clusters (resume)")
    ap_.add_argument("--largest-first", action="store_true",
                     help="sample clusters with the most leaves first (richer Ω_i)")
    ap_.add_argument("--target-name", default=None,
                     help="restrict to clusters whose merged_name contains this substring (case-insensitive)"
                          " — e.g. --target-name Catharsis for a single-metric Ω-scale-up")
    ap_.add_argument("--M-freegen", type=int, default=90, help="within-metric free-gen draws across families")
    ap_.add_argument("--n-probes", type=int, default=300)
    ap_.add_argument("--gepa-reserve", type=int, default=60)
    ap_.add_argument("--no-glm", action="store_true")
    ap_.add_argument("--families", default=None)
    ap_.add_argument("--tau-floor", type=float, default=0.02)
    ap_.add_argument("--delta", type=float, default=0.05)
    ap_.add_argument("--per-call", type=int, default=10)
    ap_.add_argument("--gepa-registry", default=None,
                     help="registry root with metrics/<slug>/versions/ (GEPA source; omitted ⇒ Ω_i=children+free-gen)")
    ap_.add_argument("--out-dir", default="/tmp/alpha_probe_metric")
    ap_.add_argument("--level", default="R2", choices=("R1", "R2", "R3"),
                     help="hierarchy level: R1 (families — over-fragmented for CW, SAMPLE), R2 (default), "
                          "R3 (coarsest merged)")
    ap_.add_argument("--skip-existing", action="store_true",
                     help="skip metrics whose checkpoint already exists (recompute α from cache for the "
                          "summary). Makes a re-run resume cleanly after a crash — the supervisor loop relies on this.")
    ap_.add_argument("--form-invariance-n", type=int, default=12,
                     help="STANDARD permutation test: re-score this many criteria under deterministic "
                          "form/boilerplate reformulations and report signature drift vs tau0 (0 = off). "
                          "Validates that sigma_c is robust to prompt phrasing/format/order.")
    ap_.add_argument("--text-first", action="store_true",
                     help="EFFICIENCY #1: score with the text-first YES/NO template so vLLM prefix-caches "
                          "the probe texts across all criteria (the same 300 texts prefilled once, not "
                          "649x). B_E/species invariance to this reorder is validated by "
                          "prompt_ordering_check before deploying at scale.")
    ap_.add_argument("--n-scan-probes", type=int, default=0,
                     help="EFFICIENCY #2/#5: if >0 and < n_probes, score ALL criteria on "
                          "probe_texts[:N] (the cheap scan). The binary CRC headline + soft-alpha run "
                          "at scan resolution (headline stats are binarized, so ~120 probes suffices). "
                          "0 = score on the full n_probes set.")
    ap_.add_argument("--cmi-thresh", type=float, default=0.15,
                     help="conditional-species CMI threshold (same default as crc_analyze).")
    ap_.add_argument("--gi-list", default=None,
                     help="comma-separated R-level group indices to run (e.g. '0,1,7,10'); restricts the "
                          "sweep to exactly those metrics (overrides largest-first/n-metrics selection).")
    ap_.add_argument("--orbit-target", type=int, default=4,
                     help="m̄_ω (theory §12.6.2): average the metric verdict over this many FORM variants "
                          "(canonical + deterministic reformulations) so the value-census target is "
                          "Φ-invariant by construction; Var_φ + flip-rate are saved as the instrument "
                          "error bar. 0/1 = legacy single-form metric_verdict. Cost: (n_forms−1) extra "
                          "rubric scorings on the probe set only (negligible vs the criteria sweep).")
    ap_.add_argument("--dry-run", action="store_true")
    a = ap_.parse_args(argv)
    os.makedirs(a.out_dir, exist_ok=True)

    cfg = cfgmod.apply_task_preset(cfgmod.ImplementerConfig(), a.task)
    cfg0 = cfgmod.ImplementerConfig()
    if a.dry_run:
        cfg0.vllm_fake = True

    # ---- executor loaded ONCE, reused across all clusters -----------------------------------------
    executor = make_judge_backend(a.target_model, cfg0, temperature=None)
    families = ({name: _MockProposer(name) for name in ("A", "B", "C")} if a.dry_run
                else _build_families(argparse.Namespace(families=a.families, no_glm=a.no_glm,
                                                        novelty_model=None, novelty_backend=None,
                                                        dry_run=False), cfg))
    print(f"executor X = {a.target_model} | families = {list(families)} | M_freegen={a.M_freegen}")

    probe_texts, _ = _load_texts(a.task, a.gepa_reserve + a.n_probes, cfg)
    if len(probe_texts) < a.gepa_reserve + min(a.n_probes, 50):
        print(f"only {len(probe_texts)} texts loadable (corpus on sk3?); aborting")
        return
    probe_texts = probe_texts[a.gepa_reserve: a.gepa_reserve + a.n_probes]
    print(f"probe set = {len(probe_texts)} task items (texts[{a.gepa_reserve}:{a.gepa_reserve + a.n_probes}])")

    # --- efficiency levers (#1 text-first prefix-cache, #2 scan-probe subset) ----------------------
    template = ap._YESNO_TEXTFIRST if a.text_first else None
    if a.n_scan_probes and 0 < a.n_scan_probes < len(probe_texts):
        score_texts = probe_texts[: a.n_scan_probes]
        print(f"SCAN: scoring all criteria on {len(score_texts)}/{len(probe_texts)} probes "
              f"(text_first={bool(a.text_first)}); #5 rep-refine runs on the full {len(probe_texts)}")
    else:
        score_texts = probe_texts

    groups = _groups_for_level(a.task, a.r2_bucket, a.level)
    if not groups:
        print(f"no R2 groups for {a.task}/{a.r2_bucket}; aborting")
        return
    order = list(range(len(groups)))
    if a.largest_first:
        order.sort(key=lambda i: -len(groups[i].get("all_leaves", [])))
    if a.target_name:                       # restrict to name-matching clusters (single-metric scale-up)
        tn = a.target_name.lower()
        order = [i for i in order if tn in (groups[i].get("merged_name", "") or "").lower()]
    if a.gi_list:                           # explicit gi allow-list (e.g. the pruned R3 set)
        gset = set(int(x) for x in str(a.gi_list).split(",") if x.strip())
        order = [i for i in order if i in gset]
    order = order[a.metric_start:]
    if a.n_metrics and a.n_metrics > 0:
        order = order[: a.n_metrics]
    print(f"{len(groups)} {a.level} clusters in '{a.r2_bucket}'; sweeping {len(order)} "
          f"({'largest-first' if a.largest_first else 'in-order from %d' % a.metric_start})")

    results = []
    for n_done, gi in enumerate(order):
        g = groups[gi]
        name = g.get("merged_name", f"cluster_{gi}")
        desc = g.get("merged_description", "")
        children = _children_for_level(a.task, a.r2_bucket, a.level, gi)
        ckpt_path = os.path.join(a.out_dir, f"{a.task}_{a.level}_metric{gi}_sigs.npz")
        if a.skip_existing and os.path.exists(ckpt_path):
            try:                                                   # resume: recompute α from cached sigs (fast, CPU)
                zz = np.load(ckpt_path, allow_pickle=True)
                _s = np.asarray(zz["sigs"], float)
                _t = list(zz["tags"])
                _tau = float(zz["tau"])
                _rep = ap.alpha_probe(_s, _t, tau=_tau, delta=a.delta, compute_cmi=False)
                _fg = [i for i, tt in enumerate(_t) if tt not in ("children", "gepa")]
                _repfg = (ap.alpha_probe(_s[_fg], [_t[i] for i in _fg], tau=_tau, delta=a.delta,
                                         compute_cmi=False) if len(_fg) >= 20 else None)
                results.append({"r2_idx": gi, "name": name, "cached": True,
                                "n_children": int(_t.count("children")), "n_gepa": int(_t.count("gepa")),
                                "n_freegen": int(len(_t) - _t.count("children") - _t.count("gepa")),
                                "D": _rep["D"], "N": _rep["N"], "tau": float(_tau),
                                "alpha_i": float(_rep["alpha_terminal"]), "decision": _rep["decision"],
                                "alpha_i_freegen": (float(_repfg["alpha_terminal"]) if _repfg else None)})
                print(f"  [{n_done+1}/{len(order)}] gi={gi} CACHED α_i={float(_rep['alpha_terminal']):.3f} "
                      f"{name[:40]}")
                continue
            except Exception as e:
                print(f"  gi={gi} cache load failed ({e}); recomputing")
        gepa = _harvest_gepa(a.gepa_registry, _slug(name)) if a.gepa_registry else []
        if a.orbit_target and a.orbit_target > 1:              # m̄_ω — Φ-orbit-averaged target (§12.6.2)
            orb = ap.orbit_metric_verdict(executor, desc, score_texts, cfg.max_text_chars,
                                          template=template, n_forms=a.orbit_target)
            M_i, m_var_phi, m_flip = orb["m_bar"], orb["var_phi_mean"], orb["flip_rate"]
        else:
            M_i = ap.metric_verdict(executor, desc, score_texts, cfg.max_text_chars,
                                    template=template)        # legacy single-form §12.3 target
            m_var_phi = m_flip = float("nan")
        try:
            sigs, tags, prompts = ap.breadth_sample_metric(
                families, executor, a.task, cfg.item_noun, name, desc, children, gepa,
                score_texts, cfg.max_text_chars, M_freegen=a.M_freegen, per_call=a.per_call,
                template=template)
        except Exception as e:
            print(f"  [{n_done+1}/{len(order)}] gi={gi} {name[:50]!r}: breadth FAILED {e}")
            results.append({"r2_idx": gi, "name": name, "error": str(e)})
            continue
        tau0 = ap.noise_floor_tau(executor, prompts[0], score_texts, cfg.max_text_chars,
                                  reps=3, template=template)
        tau = max(tau0, a.tau_floor)
        fi = (ap.form_invariance(executor, prompts, score_texts, cfg.max_text_chars,
                                 tau0=tau0, max_n=a.form_invariance_n, seed=gi, sigs=sigs,
                                 template=template)
              if a.form_invariance_n and a.form_invariance_n > 0 and prompts else None)
        rep = ap.alpha_probe(sigs, tags, tau=tau, delta=a.delta, compute_cmi=False)
        # EFFICIENCY #5: when scanning on a probe subset AND --text-first is caching the texts, re-score
        # ONE representative per species on the FULL probe set for a high-resolution inter-species
        # diagnostic. Gated on --text-first because without prefix-caching the ~43 reps x 300 probes
        # rescore is ~3 uncached minutes/metric (eats the #2 savings); the scan-res diagnostics already
        # cover the soft readout when caching is off.
        refine = None
        if a.text_first and score_texts is not probe_texts:
            try:
                lab = ap.conditional_species(sigs, cmi_thresh=a.cmi_thresh)
                reps = {}
                for i, label in enumerate(lab):          # first criterion per non-overlapping species
                    if int(label) >= 0:
                        reps.setdefault(int(label), i)
                rep_idx = sorted(reps.values())
                if len(rep_idx) >= 2:
                    rep_sigs = np.vstack([ap.signature(executor, prompts[i], probe_texts,
                                                        cfg.max_text_chars, template=template)
                                          for i in rep_idx])
                    refine = {"n_species_reps": len(rep_idx),
                              "diag_full": ap.signature_diagnostics(rep_sigs)}
                    np.savez(os.path.join(a.out_dir, f"{a.task}_{a.level}_metric{gi}_refine.npz"),
                             rep_sigs=rep_sigs, rep_idx=np.array(rep_idx), name=name)
            except Exception as e:
                print(f"    (refine skipped: {e})")
        # DUAL read: full Ω (children+gepa+freegen) is the FAVORED descriptor; the freegen-only subset is
        # the ~iid stream (children/gepa are curated/conditioned, not iid — they bias α down and the
        # Good–Turing/BK C_lo bound doesn't formally hold over them). Report BOTH; full is primary.
        fg_idx = [i for i, t in enumerate(tags) if t not in ("children", "gepa")]
        rep_fg = (ap.alpha_probe(sigs[fg_idx], [tags[i] for i in fg_idx], tau=tau,
                                 delta=a.delta, compute_cmi=False) if len(fg_idx) >= 20 else None)
        rec = {"r2_idx": gi, "name": name,
               "n_children": int(tags.count("children")), "n_gepa": int(tags.count("gepa")),
               "n_freegen": int(len(tags) - tags.count("children") - tags.count("gepa")),
               "D": rep["D"], "N": rep["N"], "tau": float(tau), "tau0": float(tau0),
               "alpha_i": float(rep["alpha_terminal"]), "decision": rep["decision"],
               "decision_census": rep.get("decision_census"),
               "f1_over_N": float(rep.get("f1_over_N", float("nan"))),
               "M_i_var_phi": float(m_var_phi), "M_i_flip_rate": float(m_flip),
               "C_lo": float(rep["coverage"]["C_lo"]), "chao1": float(rep["chao1"]),
               "disc_L1": float(rep["signature_diagnostics"]["mean_between_sig_l1"]),
               "alpha_i_freegen": (float(rep_fg["alpha_terminal"]) if rep_fg else None),
               "C_lo_freegen": (float(rep_fg["coverage"]["C_lo"]) if rep_fg else None),
               "D_freegen": (int(rep_fg["D"]) if rep_fg else None),
               "N_freegen": (int(rep_fg["N"]) if rep_fg else None),
               "form_drift_median": (fi.get("median_drift") if fi else None),
               "form_invariant": (fi.get("form_invariant") if fi else None),
               "refine_reps": (refine["n_species_reps"] if refine else None),
               "refine_inter_sig_l1_full": (refine["diag_full"]["mean_between_sig_l1"] if refine else None)}
        # per-metric checkpoint (stores M_i — the anchor-free value-census target; NEVER Y).
        # M_i is the Φ-orbit-averaged m̄_ω when --orbit-target > 1 (soft; binarize downstream), with
        # its Var_φ / flip-rate instrument error bars alongside (§12.6.2).
        probe_sha256 = np.asarray([hashlib.sha256(str(t).encode()).hexdigest()
                                   for t in score_texts])
        probe_set_sha256 = hashlib.sha256("\n".join(probe_sha256).encode()).hexdigest()
        np.savez(os.path.join(a.out_dir, f"{a.task}_{a.level}_metric{gi}_sigs.npz"),
                 sigs=sigs, tags=np.array(tags, dtype=object), prompts=np.array(prompts, dtype=object),
                 tau0=tau0, tau=tau, name=name, r2_idx=gi, M_i=M_i,
                 M_i_var_phi=float(m_var_phi), M_i_flip_rate=float(m_flip),
                 orbit_forms=int(a.orbit_target if a.orbit_target else 1),
                 scan_n=len(score_texts), text_first=bool(a.text_first),
                 probe_sha256=probe_sha256, probe_set_sha256=probe_set_sha256)
        if fi is not None:                                 # form-invariance sidecar (standard permutation test)
            json.dump(fi, open(os.path.join(a.out_dir, f"{a.task}_{a.level}_metric{gi}_forminv.json"), "w"))
        results.append(rec)
        fg_str = (f" α_fg={rec['alpha_i_freegen']:.3f}" if rec.get("alpha_i_freegen") is not None else "")
        print(f"  [{n_done+1}/{len(order)}] gi={gi} α_i={rec['alpha_i']:.3f}{fg_str} {rec['decision']} "
              f"D={rec['D']}/{rec['N']} (ch={rec['n_children']} gepa={rec['n_gepa']} fg={rec['n_freegen']}) "
              f"{name[:40]}")

    out_json = os.path.join(a.out_dir, f"{a.task}_{a.level}_metric_alpha_summary.json")
    with open(out_json, "w") as fout:
        json.dump({"task": a.task, "bucket": a.r2_bucket, "M_freegen": a.M_freegen,
                   "n_probes": a.n_probes, "n_clusters_swept": len(results), "results": results}, fout, indent=2)
    done = [r for r in results if "alpha_i" in r]
    if done:
        alphas = [r["alpha_i"] for r in done]
        n_low = sum(1 for x in alphas if x < 0.5)
        fg = [r["alpha_i_freegen"] for r in done if r.get("alpha_i_freegen") is not None]
        print("\n" + "=" * 72)
        print(f"METRIC-LEVEL α_i summary: {len(done)} metrics  |  FULL Ω: min={min(alphas):.3f} "
              f"med={float(np.median(alphas)):.3f} max={max(alphas):.3f} | α_i<0.5: {n_low}/{len(done)}")
        if fg:
            print(f"  FREEGEN-only (~iid): min={min(fg):.3f} med={float(np.median(fg)):.3f} "
                  f"max={max(fg):.3f} | α_fg<0.5: {sum(1 for x in fg if x < 0.5)}/{len(fg)}")
        for r in sorted(done, key=lambda x: x["alpha_i"])[:8]:
            fg_s = f" (α_fg={r['alpha_i_freegen']:.3f})" if r.get("alpha_i_freegen") is not None else ""
            print(f"   α_i={r['alpha_i']:.3f}{fg_s} {r['decision']:8s} D={r['D']}/{r['N']}  {r['name'][:46]}")
        print(f"summary → {out_json}")


if __name__ == "__main__":
    main()
