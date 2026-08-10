#!/usr/bin/env python3
"""Arm-comparison generation run: residual vs unconditional vs label_contrast proposers,
all through the SAME global-infill gate, on one task. Emits per-arm ledgers + the certificate
report (bits currency). Judge+proposer = glm-5.2 via z.ai anthropic endpoint (cached).

  ANTHROPIC_API_KEY=$(cat ~/.z-ai-api-key.txt) \
  ANTHROPIC_BASE_URL=https://api.z.ai/api/anthropic \
  python scripts/tools/run_arm_comparison.py --task creative-writing \
      --rubrics-dir datasets/creative-writing/medoid-bank --n 400 --max-rounds 4
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "methods")

import numpy as np
import pandas as pd

from metrics_tree_infilling.certificates import report_from_ledger, report_from_ledgers
from metrics_tree_infilling.config import InfillConfig
from metrics_tree_infilling.feature_gen import make_proposer
from metrics_tree_infilling.generators import (
    autometrics_iterative_generator, label_contrast_generator, metric_tree_generator,
    residual_generator, unconditional_generator)
from metrics_tree_infilling.global_infill import run_global_infill
from metrics_tree_infilling.io_metrics import (
    REPO_ROOT, load_rubric_metrics, load_rubric_metrics_from_dir,
    make_vllm_judge_scorer, materialize, three_way_split)
from metrics_tree_infilling.run import DATASET_CONFIGS

TASK_HINTS = {
    "creative-writing": "evaluating short creative fiction for literary quality",
    "creative-writing-wigleaf": "evaluating short literary fiction for editorial curation "
                                "(which flash-fiction pieces an expert editor promotes to a best-of list)",
    "creative-writing-royalroad": "evaluating the opening chapters of a web serial for commercial "
                                  "pickup (which serials get a Kindle Unlimited / publishing deal)",
    "press-release": "evaluating corporate press releases for newsworthiness/pickup",
    "peer-review": "evaluating scientific paper submissions for acceptance",
}
TASK_HINTS["math-pooled-12tags"] = ("evaluating answers to mathematics questions on Math "
                                    "StackExchange for community-judged quality (which answer "
                                    "the community scores higher)")
_CW_GENRE_DESC = {
    "abstract-premise": "abstract or conceptual premises (omniscient beings, embodiments of death)",
    "immortality": "immortality, reincarnation, and afterlife premises",
    "wakeup-mystery": "second-person wake-up mystery and thriller premises",
    "hell-deal": "hell, demons, and deal-with-the-devil premises",
    "pooled-4genres": "a mix of genres",
    "aliens": "alien first-contact and invasion premises",
    "villain": "supervillain and villain-protagonist premises",
    "soulmate": "soulmate, destined-love, and reincarnated-lover premises",
    "ai": "artificial intelligence and sentient-machine premises",
    "time-travel": "time travel and time-loop premises",
    "meta-experimental": "meta or experimental prompts (write a story that/about ...)",
}
TASK_HINTS.update({
    f"cw-genre-{g}": (f"evaluating short fiction written for WritingPrompts stories with {d}, "
                      "for community upvote-revealed preference")
    for g, d in _CW_GENRE_DESC.items()
})
_HUMOR_TOPIC_DESC = {
    "marriage": "marriage and relationship jokes", "bar-jokes": "walks-into-a-bar jokes",
    "family": "family and dad jokes", "doctor": "doctor and medical jokes",
    "pooled-4topics": "jokes across topics",
    "political-classroom": "political and classroom jokes",
    "police": "police and cop jokes",
    "chicken-crossing": "why-did-X-cross-the-road format jokes",
    "everyday-observational": "everyday observational jokes",
    "absurd-wordplay": "absurd and wordplay jokes",
    "topical-corona": "topical and pandemic-era jokes",
}
TASK_HINTS.update({
    f"humor-topic-{t}": (f"evaluating {d} from r/Jokes for community upvote-revealed funniness")
    for t, d in _HUMOR_TOPIC_DESC.items()
})
# Math sub-community legs: the hint names the SUBFIELD so proposals can carry tag-local
# criteria (the whole point of the within-subtask run — general craft is already in the bank).
TASK_HINTS.update({
    f"math-{t}": (f"evaluating answers to {t.replace('-', ' ')} questions on Math StackExchange "
                  "for community-judged quality (which answer the community scores higher)")
    for t in ["real-analysis", "calculus", "linear-algebra", "abstract-algebra", "probability",
              "algebra-precalculus", "general-topology", "combinatorics", "sequences-and-series",
              "complex-analysis", "geometry", "integration"]
})
# Peer-review ICLR subfields: name the subfield (from the by_subfield manifest's TF-IDF terms)
# so proposals can carry subfield-local acceptance criteria. General ICLR gets a plain hint.
try:
    import json as _json
    _man = _json.load(open("datasets/peer-review/by_subfield/_manifest.json"))
    for _sf in _man.get("subfields", []):
        _terms = ", ".join(_sf.get("terms", [])[:5])
        TASK_HINTS[f"peer-iclr-{_sf['slug']}"] = (
            f"evaluating ICLR submissions in the research area characterized by [{_terms}] "
            "for acceptance (which papers the reviewers accept)")
except Exception:
    pass
TASK_HINTS["peer-iclr-general"] = ("evaluating ICLR machine-learning submissions for acceptance "
                                   "(which papers the reviewers accept)")

# --- Scale-out wave (task #66): four more sibling axes (build_strata.py, class-balanced) ---
import json as _json2
# peer x venue (metadata): name the venue so proposals can carry venue-local acceptance norms.
_VENUE_DESC = {"iclr": "ICLR", "neurips": "NeurIPS", "icml": "ICML",
               "tmlr": "TMLR (Transactions on Machine Learning Research)"}
TASK_HINTS.update({
    f"peer-venue-{v}": (f"evaluating {d} machine-learning submissions for acceptance "
                        "(which papers the reviewers accept)")
    for v, d in _VENUE_DESC.items()})
TASK_HINTS["peer-venue-general"] = ("evaluating machine-learning submissions across venues for "
                                    "acceptance (which papers the reviewers accept)")
# code-review x language (metadata): name the language.
_LANG_DESC = {"python": "Python", "go": "Go", "java": "Java",
              "typescript": "TypeScript", "javascript": "JavaScript"}
TASK_HINTS.update({
    f"code-lang-{l}": (f"evaluating {d} pull-request changes in code review for approval/merge "
                       "(which changes the reviewers approve)")
    for l, d in _LANG_DESC.items()})
TASK_HINTS["code-lang-general"] = ("evaluating software pull-request changes in code review for "
                                   "approval/merge (which changes the reviewers approve)")
# notice-and-comment x topic & press-release x topic (topic-model): name the topic from manifest terms.
for _task, _pref, _templ in [
    ("notice-and-comment", "notice-topic",
     "evaluating public comments on proposed federal regulations about [{terms}] "
     "(which comments the dataset labels as the positive class)"),
    ("press-releases", "press-topic",
     "evaluating press releases about [{terms}] for newsworthiness / media pickup")]:
    try:
        _m = _json2.load(open(f"datasets/{_task}/by_topic/_manifest.json"))
        for _s in _m.get("siblings", []):
            _t = ", ".join(_s.get("terms", [])[:5])
            TASK_HINTS[f"{_pref}-{_s['slug']}"] = _templ.format(terms=_t)
    except Exception:
        pass
TASK_HINTS["notice-topic-general"] = ("evaluating public comments on proposed federal regulations "
                                      "(which comments the dataset labels as the positive class)")
TASK_HINTS["press-topic-general"] = "evaluating press releases for newsworthiness / media pickup"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=sorted(DATASET_CONFIGS))
    ap.add_argument("--rubrics-dir", default=None)
    ap.add_argument(
        "--arms",
        default="residual,unconditional,label_contrast,autometrics_iterative,metric_tree")
    ap.add_argument("--n", type=int, default=400, help="discover+guard row budget")
    ap.add_argument("--max-metrics", type=int, default=40)
    ap.add_argument("--max-rounds", type=int, default=4)
    ap.add_argument("--min-auc-gain", type=float, default=0.02)
    ap.add_argument("--min-bits-gain", type=float, default=0.01)
    ap.add_argument("--acceptance-eval", default="guard", choices=["guard", "cv"])
    ap.add_argument("--confirm-repeats", type=int, default=5,
                    help="fresh-seed CV repeats in the confirm stage (0 = off; cv mode only)")
    ap.add_argument("--gate-alpha", type=float, default=0.05)
    ap.add_argument("--bonferroni-m", type=int, default=0,
                    help="Bonferroni divisor; 0 = auto (n_arms * max_rounds, the planned study)")
    ap.add_argument("--dense-bits", type=float, default=None,
                    help="dense ceiling C in bits/item on THIS task's eval (wrap evidence)")
    ap.add_argument("--stack-bits", type=float, default=None,
                    help="dense-stack (dense features + bank) bits/item; supersedes --dense-bits")
    ap.add_argument("--dense-plateaued", action="store_true",
                    help="assert the dense data-curve has plateaued (dominance-gate evidence)")
    ap.add_argument("--patience", type=int, default=0,
                    help="consecutive-rejection stop; 0 = max_rounds (never early-stop — "
                         "every planned draw feeds the flux capture-recapture read)")
    ap.add_argument("--flux-c", type=float, default=1.0)
    ap.add_argument("--flux-tau", type=float, default=0.92)
    ap.add_argument("--no-flux", action="store_true")
    ap.add_argument("--content-only", action="store_true",
                    help="anti-surface prompt instruction + drop surface-only proposals")
    ap.add_argument("--measure-reliability", action="store_true",
                    help="per-metric judge test-retest (measurement-floor disambiguation)")
    ap.add_argument("--min-bank-auc-residual", type=float, default=0.0,
                    help="skip the residual contrast when bank AUC < this (0=off)")
    ap.add_argument("--judge-backend", default="anthropic",
                    choices=["anthropic", "vllm_offline", "openai_compatible"],
                    help="vllm_offline = in-process LLM.generate batches (the sk3 path; "
                         "large-scale scoring never goes through an HTTP server)")
    ap.add_argument("--judge-model", default="glm-5.2")
    ap.add_argument("--proposer-backend", default="anthropic",
                    choices=["anthropic", "vllm_offline"],
                    help="vllm_offline = proposals from the SAME resident engine as the judge "
                         "(executor-closed certificate, no external quota in the loop)")
    ap.add_argument("--proposer-model", default="glm-5.2")
    ap.add_argument("--operationalize", action="store_true",
                    help="GEPA-style rubric iteration (retest + MI-recovery, label-free) on "
                         "EVERY proposal before gate scoring (cfg.operationalize_proposals)")
    ap.add_argument("--executor-label", default=None,
                    help="certificate executor tag; default = judge model (E = who RUNS the metrics)")
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    executor = args.executor_label or args.judge_model

    dcfg = DATASET_CONFIGS[args.task]
    out = Path(args.out or f"outputs/ctree/arm_comparison/{args.task}")
    out.mkdir(parents=True, exist_ok=True)

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    # Bonferroni over the PLANNED study (all arms x rounds on this task), fixed before any
    # proposal is gated — m must not depend on how many proposals happened to be generated
    m_bonf = args.bonferroni_m or len(arms) * args.max_rounds

    cfg = InfillConfig(
        random_seed=0, n_permutations=999,
        proposer_backend=args.proposer_backend, proposer_model=args.proposer_model,
        materialize_backend=args.judge_backend, materialize_model=args.judge_model,
        llm_concurrency=args.concurrency, max_text_tokens=700, verbose=False,
        min_auc_gain=args.min_auc_gain, min_bits_gain=args.min_bits_gain,
        acceptance_eval=args.acceptance_eval,
        confirm_n_repeats=args.confirm_repeats, gate_alpha=args.gate_alpha,
        gate_bonferroni_m=m_bonf,
        content_only_guard=args.content_only,
        measure_reliability=args.measure_reliability,
        min_bank_auc_for_residual=args.min_bank_auc_residual,
        viability_min_applicability=0.10, viability_min_std=0.05,
        operationalize_proposals=args.operationalize,
        group_split_by_id=dcfg.get("group_split", False),
        id_column=dcfg["id"], text_column=dcfg["text"], label_column=dcfg["label"],
        output_dir=str(out), cache_dir="outputs/ctree/B_tree/judge_cache",
        curated_z_only=True, include_text_length_in_z=False)

    df = pd.read_csv(REPO_ROOT / dcfg["split"], low_memory=False).dropna(
        subset=[dcfg["text"], dcfg["label"]])
    df[dcfg["label"]] = pd.to_numeric(df[dcfg["label"]], errors="coerce")
    df = df.dropna(subset=[dcfg["label"]])
    df[dcfg["label"]] = df[dcfg["label"]].astype(int)
    # fixed budget: subsample BEFORE splitting so all arms share items exactly
    df = df.sample(min(args.n + args.n // 2, len(df)), random_state=7).reset_index(drop=True)
    df_d, df_g, df_t = three_way_split(df, cfg)
    print(f"[{args.task}] rows d/g/t = {len(df_d)}/{len(df_g)}/{len(df_t)}", flush=True)

    if args.rubrics_dir:
        bank = load_rubric_metrics_from_dir(args.rubrics_dir)[: args.max_metrics]
    else:
        bank = load_rubric_metrics(args.task, limit=args.max_metrics)
    judge = make_vllm_judge_scorer(cfg)
    if args.proposer_backend == "vllm_offline":
        from metrics_tree_infilling.io_metrics import make_offline_vllm_proposer
        proposer = make_offline_vllm_proposer(cfg)
    else:
        proposer = make_proposer(cfg)

    # viability probe on discover
    probe = df_d.sample(min(60, len(df_d)), random_state=1)[dcfg["text"]].astype(str).tolist()
    lv, apl = judge(bank, probe)
    viable = [bank[j] for j in range(len(bank))
              if apl[:, j].mean() > cfg.viability_min_applicability
              and np.std(lv[apl[:, j], j]) > cfg.viability_min_std]
    print(f"viable bank {len(viable)}/{len(bank)}", flush=True)

    sm_d = materialize(viable, df_d, cfg, judge)
    sm_g = materialize(viable, df_g, cfg, judge)
    y_d = df_d[dcfg["label"]].to_numpy()
    y_g = df_g[dcfg["label"]].to_numpy()
    texts_d = df_d[dcfg["text"]].astype(str).tolist()

    hint = TASK_HINTS.get(args.task, "this domain")
    arm_factories = {
        "residual": lambda: residual_generator(),
        "unconditional": lambda: unconditional_generator(hint, k=4),
        "label_contrast": lambda: label_contrast_generator(texts_d, y_d, seed=0),
        "autometrics_iterative": lambda: autometrics_iterative_generator(hint, k=4),
        "metric_tree": lambda: metric_tree_generator(
            hint, texts_d, y_d, sm_d.levels, sm_d.metric_names, k=2, seed=0),
    }

    summary = {}
    for arm in arms:
        print(f"\n=== ARM {arm} ===", flush=True)
        try:
            res = run_global_infill(
                sm_d, df_d, y_d, sm_g, df_g, y_g, list(viable), cfg,
                judge_scorer=judge, proposer=proposer,
                max_rounds=args.max_rounds, patience=args.patience or args.max_rounds,
                measure_reconstruction=True,
                proposal_fn=arm_factories[arm]())
        except Exception as e:
            print(f"ARM {arm} FAILED ({type(e).__name__}: {e}) — continuing with next arm",
                  flush=True)
            continue
        arm_out = out / arm
        res.save(arm_out)
        kept = [l for l in res.ledgers if l.status == "kept"]
        # attenuation diagnostic: of the proposals dropped for low gain, how many were actually
        # UNRELIABLE (judge can't apply)? high share => the plateau is a measurement floor, not
        # a genuine null. Only meaningful when --measure-reliability is on.
        low_gain = [l for l in res.ledgers if l.status.startswith("dropped:auc_gain")
                    or l.status.startswith("dropped:bits_gain") or l.status.startswith("dropped:confirm")]
        attenuated = [l for l in low_gain if getattr(l, "attenuation_flag", False)]
        summary[arm] = {
            "proposals": len(res.ledgers), "kept": len(kept),
            "auc_trajectory": res.guard_auc_trajectory,
            "bits_trajectory": res.guard_bits_trajectory,
            "kept_names": [l.name for l in kept],
            "kept_bits_gains": [l.bits_gain for l in kept],
            "kept_recon_agreement": [l.reconstruction_agreement for l in kept],
            "kept_retest_spearman": [getattr(l, "retest_spearman", float("nan")) for l in kept],
            "n_dropped_surface": sum(1 for l in res.ledgers if l.status == "dropped:surface"),
            "low_gain_attenuated": f"{len(attenuated)}/{len(low_gain)}",
            "statuses": [l.status for l in res.ledgers],
        }
        print(json.dumps(summary[arm], indent=2, default=float), flush=True)
        try:
            rep = report_from_ledger(arm_out / "global_infill_ledger.json",
                                     task=args.task, executor=executor,
                                     delta_bits=args.min_bits_gain,
                                     dense_bits=args.dense_bits, stack_bits=args.stack_bits,
                                     dense_plateaued=args.dense_plateaued)
            rep.save(arm_out / "certificate.json")
            print(rep.render(), flush=True)
        except Exception as e:
            print(f"certificate: {e}", flush=True)

    # union certificate over ALL arms (MCC §2a: the certified artifact is the union ledger;
    # per-arm certificates always carry the single-arm anti-conservatism note)
    union_paths = [out / arm / "global_infill_ledger.json" for arm in summary
                   if (out / arm / "global_infill_ledger.json").exists()]
    flux_kwargs = {}
    if union_paths and not args.no_flux:
        try:
            from metrics_tree_infilling.flux import flux_from_ledgers
            fx = flux_from_ledgers(union_paths,
                                   base_rate=float(np.concatenate([y_d, y_g]).mean()),
                                   c=args.flux_c, tau=args.flux_tau)
            with open(out / "flux.json", "w") as f:
                json.dump(fx, f, indent=2, default=float)
            if fx.get("flux_tail_bits") is not None:
                flux_kwargs = {"flux_tail_bits": fx["flux_tail_bits"], "flux_meta": fx}
            print(f"flux read -> {out/'flux.json'} "
                  f"(N={fx.get('n_draws')}, species={fx.get('n_species')}, "
                  f"tail={fx.get('flux_tail_bits')})", flush=True)
        except Exception as e:
            print(f"flux read failed: {e}", flush=True)
    if union_paths:
        try:
            urep = report_from_ledgers(union_paths, task=args.task, executor=executor,
                                       delta_bits=args.min_bits_gain,
                                       dense_bits=args.dense_bits, stack_bits=args.stack_bits,
                                       dense_plateaued=args.dense_plateaued, **flux_kwargs)
            urep.save(out / "certificate_union.json")
            print("\n=== UNION CERTIFICATE ===\n" + urep.render(), flush=True)
        except Exception as e:
            print(f"union certificate: {e}", flush=True)

    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\nsummary -> {out/'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
