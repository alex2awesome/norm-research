"""Phase 5 — three optimizer arms over the GEPA paper's OWN programs/metrics/splits.

Everything evaluation-side is imported verbatim from vendor/gepa-artifact (the paper's code):
programs (HotpotMultiHop / HoverMultiHop / AIME CoT), metrics (EM / discrete_retrieval_eval /
int-match), splits (150/300/300; AIME 45/45/150). Only the OPTIMIZERS differ by arm:

  official   dspy.GEPA (the authors' maintained implementation) over module instructions
  inhouse    our monolithic mutate-and-accept loop (joint per-module rewrite via reflection LM)
  unitrecomb M_omega-style: mine instruction-clause units per module from BOTH other arms'
             trajectories, screen standalone, greedily recombine by conditional value

Candidates are dicts {predictor_name: instruction}. Every candidate evaluation (any arm) is
logged to runs_paperexact/<bench>/<arm>/proposals.jsonl with per-item scores (raw-draw logging
rule). Final: test-split evaluation with the paper metric.

  python paperexact_arms.py aime --arm official --task-lm <litellm-model> ...
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "vendor" / "gepa-artifact"))

import dspy  # noqa: E402

MIN_GAIN = 0.01
MAX_UNITS = 40


# ---------------------------------------------------------------- benchmark registry (paper's)
def load_bench(name: str):
    if name == "aime":
        from gepa_artifact.benchmarks.AIME import benchmark as metas
    elif name == "hover":
        from gepa_artifact.benchmarks.hover import benchmark as metas
    elif name == "hotpot":
        from gepa_artifact.benchmarks.hotpotQA import benchmark as metas
    elif name == "ifbench":
        from gepa_artifact.benchmarks.IFBench import benchmark as metas
    elif name == "livebench":
        from gepa_artifact.benchmarks.livebench_math import benchmark as metas
        _patch_livebench_amps_timeout()
    elif name == "pupa":
        from gepa_artifact.benchmarks.papillon import benchmark as metas
    else:
        raise ValueError(name)
    meta = metas[0]
    bench = meta.benchmark(dataset_mode="full")
    program = meta.program[0]
    if not isinstance(program, dspy.Module):  # AIME registers the class-built instance lazily
        program = program
    # Some artifact programs (IFBenchCoT2StageProgram) define __call__ but not forward;
    # dspy.Evaluate works while dspy.GEPA's trace bootstrap calls .forward and dies. Alias on
    # the CLASS (survives deepcopy) — same body, no behavior change, vendor/ untouched.
    if not hasattr(type(program), "forward") and hasattr(type(program), "__call__"):
        type(program).forward = type(program).__call__
    if name == "pupa":
        # The artifact hardcodes openai/gpt-4.1-mini at IMPORT time in two places:
        # papillon/__init__.py (`untrusted_lm`, baked into the PAPILLON program instance) and
        # papillon_utils.py (`llm_judge.set_lm(openai_lm)`, the metric's quality/leakage judge).
        # Rewire BOTH to GLM via z.ai here — never touching vendor/ — by set_lm on the
        # module-level judge instance (compute_metrics closes over it) and swapping the program
        # attribute. The judge is part of the METRIC, so it is pinned to glm-5.2 regardless of
        # --reflection-model; rescore inherits the same wiring via this function.
        from gepa_artifact.benchmarks.papillon import papillon_utils
        glm = make_reflection_lm("glm-5.2", patient=True)
        papillon_utils.llm_judge.set_lm(glm)
        program.untrusted_model = glm
    return bench, program, meta.metric, meta.metric_with_feedback


def _patch_livebench_amps_timeout():
    """macOS fix for the vendored AMPS_Hard sympy-equivalence timeout (vendor/ untouched).

    The artifact's run_with_timeout targets a LOCAL closure via multiprocessing.Process. On
    macOS the default start method is spawn, which must pickle the target -> every call raises
    "Can't pickle local object", is swallowed by `except Exception: warn`, and the item scores 0
    (diagnosed 2026-07-20 on a textually-IDENTICAL answer, \\frac{8599}{56}; AMPS_Hard is
    150/368 of livebench/math). The artifact's own Linux runs used fork, where this never
    triggers — so a fork-context clone IS the paper behavior, not a deviation. Only extra
    safety: queue.get gets a timeout so a hard-crashed child can't hang an arm run forever."""
    from gepa_artifact.benchmarks.livebench_math.livebenchmath_utils.AMPS_Hard import (
        utils as amps_utils,
    )
    import multiprocessing
    ctx = multiprocessing.get_context("fork")

    def run_with_timeout(func, args=(), timeout=8):
        def wrapper(queue):
            try:
                queue.put(func(*args))
            except Exception as e:  # noqa: BLE001 — mirror of the vendored contract
                queue.put(e)

        queue = ctx.Queue()
        process = ctx.Process(target=wrapper, args=(queue,))
        process.start()
        process.join(timeout)
        if process.is_alive():
            process.terminate()
            process.join()
            raise TimeoutError("Operation timed out")
        try:
            result = queue.get(timeout=10)
        except Exception as exc:
            raise TimeoutError("child exited without result") from exc
        if isinstance(result, Exception):
            raise result
        return result

    amps_utils.run_with_timeout = run_with_timeout


def robustify_aime_metrics(metric, metric_fb):
    """Wrap the paper's AIME metrics with LAST-integer extraction on the answer field.

    The paper metric does bare int(prediction.answer) and zeroes LaTeX-formatted answers
    ('$504$', '\\frac{...}') — a measurement artifact for long-reasoning LMs (diagnosed
    2026-07-20: free-form GLM-5.2 solved 5/5 where this metric scored it 0). Extraction is a
    no-op for bare-integer outputs, so paper-exact columns are unaffected; the vendored
    artifact is never modified. Both paper metrics read ONLY prediction.answer, so a minimal
    shim carrying the extracted answer is faithful.
    """
    class _P:
        def __init__(self, answer):
            self.answer = answer

    def _extract(prediction):
        nums = re.findall(r"-?\d+", str(getattr(prediction, "answer", "")))
        return nums[-1] if nums else getattr(prediction, "answer", "")

    def r_metric(example, prediction, trace=None):
        return metric(example, _P(_extract(prediction)), trace)

    def r_metric_fb(example, prediction, trace=None):
        return metric_fb(example, _P(_extract(prediction)), trace)

    return r_metric, r_metric_fb


# ---------------------------------------------------------------- candidate plumbing
def get_instructions(program) -> dict[str, str]:
    return {name: pred.signature.instructions for name, pred in program.named_predictors()}


def set_instructions(program, cand: dict[str, str]):
    prog = program.deepcopy()
    for name, pred in prog.named_predictors():
        if name in cand:
            pred.signature = pred.signature.with_instructions(cand[name])
    return prog


def cand_hash(cand: dict[str, str]) -> str:
    js = json.dumps({k: " ".join(v.lower().split()) for k, v in sorted(cand.items())})
    return hashlib.sha256(js.encode()).hexdigest()[:16]


HEALTH_PROBE = None  # set in main(); evaluate_cand refuses to record all-zero outage artifacts
EVAL_THREADS = 8     # set in main() via --eval-threads; 8 is tunnel-safe, 32 for localhost vLLM
EVAL_PASSES = 1      # v8: k independent generation passes per select-phase eval, scores averaged
                     # (aime same-prompt sd ~.09-.13 at n=150/temp-.6 — single-pass panels rank
                     # noise; measured twice, HB79b/HB80)


def evaluate_cand(program, cand, dataset, metric, log_path, tag, budget=None, n_threads=None,
                  passes=None):
    """v8 multi-pass: run `passes` independent generation passes and average (per-item scores
    averaged elementwise so paired item-level tests stay valid). Budget is charged
    passes*len(dataset) — a k-pass eval really spends k times the calls."""
    if n_threads is None:
        n_threads = EVAL_THREADS
    if passes is None:
        passes = EVAL_PASSES
    if budget is not None:
        # All-or-nothing: a truncated panel would make this score incomparable with every
        # other score on the same panel (paired selection assumes identical items).
        if budget["remaining"] < passes * len(dataset):
            return None
        budget["remaining"] -= passes * len(dataset)
    prog = set_instructions(program, cand)
    pass_means, pass_items = [], []
    for _ in range(passes):
        ev = dspy.Evaluate(devset=dataset, metric=metric, num_threads=n_threads,
                           display_progress=False, max_errors=10000, provide_traceback=False)
        res = ev(prog)
        pass_means.append(float(res.score) / 100.0)  # dspy reports percent
        try:
            pass_items.append([float(s) for (_, _, s) in res.results])
        except Exception:
            pass_items.append(None)
    score = sum(pass_means) / len(pass_means)
    per_item = None
    if all(p is not None and len(p) == len(dataset) for p in pass_items):
        per_item = [sum(col) / len(col) for col in zip(*pass_items)]
    # Mid-run outage guard (2026-07-21): an outage >retry-budget that begins mid-eval makes
    # dspy max_errors zero-score the dead window. An all-zero batch on a ≥20-item panel with a
    # dead endpoint is an outage artifact — raise, never record (the process dies loudly and the
    # heartbeat relaunches it clean).
    if HEALTH_PROBE and per_item and len(per_item) >= 20 and sum(per_item) == 0:
        if not HEALTH_PROBE():
            raise RuntimeError(f"all-zero batch ({tag}) with DEAD endpoint — outage artifact, "
                               "refusing to record; restore the endpoint and rerun")
    with open(log_path, "a") as fh:
        fh.write(json.dumps({"ts": time.time(), "candidate": cand, "hash": cand_hash(cand),
                             "n_batch": len(dataset), "mean_score": score,
                             "item_scores": per_item, "phase": tag, "passes": passes,
                             "pass_means": pass_means if passes > 1 else None}) + "\n")
    return score


# ---------------------------------------------------------------- reflection LM (GLM, z.ai)
def _zai_key() -> str:
    import os
    env = os.environ.get("ZAI_KEY_FILE")
    if env and Path(env).expanduser().exists():
        return Path(env).expanduser().read_text().strip()
    for name in (".z-ai-api-key-alexander-spangher.txt", ".z-ai-api-key-spangher.txt",
                 ".z-ai-api-key.txt"):
        k = Path.home() / name
        if k.exists():
            return k.read_text().strip()
    raise RuntimeError("no z.ai key file found")


def make_reflection_lm(model: str, patient: bool = False, cache: bool = True):
    # z.ai is flappy two ways (2026-07-21 outage post-mortem): 1302 rate-limit bursts AND
    # multi-hour hangs on dead sockets (observed: 149-min wedged reads). timeout recycles hung
    # attempts in minutes; the deep retry stack with litellm's exponential backoff rides out
    # bursts. 10 retries x <=5-min attempts ~ 50 min of per-call resilience.
    # v8 (2026-07-24 audit): `patient` is for the PUPA METRIC JUDGE, whose 1302 rate-limit
    # failures do not merely retry-and-recover — a failed judge call errors the ITEM, which
    # dspy scores 0 (123 such zeros in pupa v4sk3 = deflated panels, noise-chased selection).
    # Retrying harder changes no judgment, only whether the judgment is obtained at all.
    # v9 (2026-07-27): z.ai account is DEAD (1113 insufficient balance, user-confirmed).
    # claude-* models route to the real Anthropic API (ANTHROPIC_KEY_FILE env, else
    # ~/.anthropic-usc-key.txt; user directive: "we'll be using Sonnet"). `cache` kw added:
    # stochastic replicates (mining, k-passes) MUST pass cache=False — F3c rule.
    if model.startswith("claude"):
        import os
        kf = os.environ.get("ANTHROPIC_KEY_FILE") or str(Path.home() / ".anthropic-usc-key.txt")
        return dspy.LM(f"anthropic/{model}", api_key=Path(kf).read_text().strip(),
                       temperature=1.0, max_tokens=16000, cache=cache,
                       num_retries=40 if patient else 10, timeout=300)
    return dspy.LM(f"anthropic/{model}", api_base="https://api.z.ai/api/anthropic",
                   api_key=_zai_key(), temperature=1.0, max_tokens=16000, cache=cache,
                   num_retries=40 if patient else 10, timeout=300)


# ---------------------------------------------------------------- arms
def arm_official(program, bench, metric_fb, log_path, budget_calls, reflection_model,
                 use_merge=False):
    def gepa_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
        out = metric_fb(gold, pred, trace)
        return out if isinstance(out, dspy.Prediction) else dspy.Prediction(score=out, feedback="")

    gepa = dspy.GEPA(metric=gepa_metric, max_metric_calls=budget_calls,
                     reflection_lm=make_reflection_lm(reflection_model),
                     use_merge=use_merge,
                     track_stats=True, log_dir=str(Path(log_path).parent / "gepa_logs"))
    optimized = gepa.compile(program.deepcopy(), trainset=list(bench.train_set),
                             valset=list(bench.val_set))
    return get_instructions(optimized)


def arm_mipro(program, bench, metric, log_path, budget_calls, reflection_model):
    # MIPROv2 = the GEPA paper's principal baseline. Paper-literal init per
    # vendor/gepa-artifact/scripts/experiment_configs.py "MIPROv2-Heavy":
    # auto="heavy", max_errors=10000, requires_permission_to_run=False.
    # DEVIATION (deliberate, flagged): max_*_demos=0 — instruction-only regime so it
    # optimizes the same object as every other arm; demos would be silently dropped by
    # get_instructions() and misrepresent MIPROv2. The paper's own runs allow demos;
    # a demo-carrying variant needs a direct-program final-test path (not yet built).
    # budget_calls is NOT enforced here: heavy mode spends what it spends (in the paper,
    # MIPROv2-Heavy's realized metric calls DEFINE the budget handed to GEPA variants).
    opt = dspy.MIPROv2(metric=metric, prompt_model=make_reflection_lm(reflection_model),
                       auto="heavy", max_errors=10_000,
                       max_bootstrapped_demos=0, max_labeled_demos=0,
                       num_threads=EVAL_THREADS, seed=0)
    optimized = opt.compile(program.deepcopy(), trainset=list(bench.train_set),
                            valset=list(bench.val_set),
                            max_bootstrapped_demos=0, max_labeled_demos=0,
                            requires_permission_to_run=False)
    return get_instructions(optimized)


def arm_inhouse(program, bench, metric, metric_fb, log_path, budget_calls, reflection_model,
                panel_n=25, mutations_per_round=6):
    rng = random.Random(0)
    refl = make_reflection_lm(reflection_model)
    panel = list(bench.train_set)[:panel_n]
    seed_cand = get_instructions(program)
    budget = {"remaining": budget_calls}
    cur, cur_score = seed_cand, None
    cur_score = evaluate_cand(program, cur, panel, metric, log_path, "incumbent", budget)
    t0, n_evals = time.time(), 1
    while budget["remaining"] >= panel_n:
        if time.time() - t0 > 6 * 3600 and n_evals < 5:
            # Starvation guard (hotpot-GLM post-mortem: 2 evals in 14h of reflection-spin
            # through provider flaps). Die loudly; a starved search result is void anyway.
            raise RuntimeError(f"inhouse arm STARVED: {n_evals} evals in "
                               f"{(time.time()-t0)/3600:.1f}h — provider too unstable, rerun later")
        fb_examples = []
        for ex in rng.sample(panel, min(4, len(panel))):
            try:
                pred = set_instructions(program, cur)(**ex.inputs())
                out = metric_fb(ex, pred)
                fb_examples.append(getattr(out, "feedback", str(out))[:1200])
            except Exception as exc:
                fb_examples.append(f"program error: {exc}")
        prompt = (
            "You are optimizing the per-module instructions of a compound LLM program. "
            "Current instructions (JSON):\n" + json.dumps(cur, indent=1)[:6000] +
            "\n\nFeedback from recent failures/successes:\n- " + "\n- ".join(fb_examples) +
            "\n\nPropose an improved FULL set of instructions. Reply with ONLY a JSON object "
            "mapping each module name to its new instruction string.")
        accepted = False
        for _ in range(mutations_per_round):
            if budget["remaining"] < panel_n:
                break
            try:
                raw = refl(prompt)[0]
                raw = raw[raw.index("{"): raw.rindex("}") + 1]
                cand = {k: str(v) for k, v in json.loads(raw).items() if k in cur}
                if not cand or cand_hash({**cur, **cand}) == cand_hash(cur):
                    continue
                cand = {**cur, **cand}
            except Exception:
                continue
            s = evaluate_cand(program, cand, panel, metric, log_path, "mutation", budget)
            n_evals += 1
            if s is not None and cur_score is not None and s >= cur_score + MIN_GAIN:
                cur, cur_score, accepted = cand, s, True
                break
        if not accepted:
            break
    return cur


def _suggest_units_paper(bench_name, init_cand, refl, n=12, train_examples=None):
    """D2 third unit source: novel per-module clauses proposed by a strong LLM.

    v3: two independent asks with different framings (failure-general vs expert-strategy) —
    source diversity beats more draws from one framing (capture-recapture doctrine).
    v5: +example-grounded framing (the suggester finally SEES the task, not just the
    instructions) and +unconventional framing (clauses different in KIND from the existing
    instruction content)."""
    framings = [
        f"Propose {n} candidate instruction clauses that could each independently improve "
        "end-task accuracy.",
        f"Think like a domain expert who has watched many failures on this task. Propose {n} "
        "SPECIFIC strategy clauses (decomposition rules, verification steps, output-format "
        "discipline, common-trap warnings) that a weaker model following instructions could "
        "actually execute.",
        f"You are auditing near-miss failures: answers that were almost right but failed on a "
        f"final check, a format detail, or an unverified assumption. Propose {n} clauses that "
        "force explicit SELF-VERIFICATION before finalizing (re-derive the key step, re-read "
        "the constraint list, confirm the output format matches exactly).",
        f"Propose {n} UNCONVENTIONAL clauses that are different in KIND from everything "
        "already in the instructions — new reasoning orders, explicit counterexample search, "
        "restating the goal in the model's own words before answering, intermediate "
        "representations (enumerate candidates, tabulate constraints, then decide). Do NOT "
        "rephrase or strengthen existing content; cover angles it does not touch at all.",
    ]
    if train_examples:
        framings.append(
            f"Here are REAL examples from this task's training set:\n{train_examples}\n\n"
            f"Ground yourself in what these examples actually require. Propose {n} clauses "
            "targeting the specific skills, traps, intermediate steps, and output formats "
            "these examples reveal — clauses a weaker model could follow mechanically.")
    out, seen = [], set()
    for framing in framings:
        prompt = (
            f"A multi-module LLM program for the task '{bench_name}' has these module "
            f"instructions:\n{json.dumps(init_cand, indent=1)[:4000]}\n\n{framing} Reply "
            'with ONLY a JSON array of objects {"module": <one of the module names above>, '
            '"clause": <one or two self-contained sentences>}.')
        try:
            raw = refl(prompt)[0]
            arr = json.loads(raw[raw.index("["): raw.rindex("]") + 1])
            for d in arr:
                mod, c = str(d.get("module")), str(d.get("clause", "")).strip()
                key = (mod, " ".join(c.lower().split()))
                if mod in init_cand and 25 <= len(c) <= 400 and key not in seen:
                    seen.add(key)
                    out.append((mod, c))
        except Exception as exc:
            print(f"  LLM unit suggestion failed ({exc})", flush=True)
    return out


def _suggest_units_failures(bench_name, init_cand, refl, program, panel, metric,
                            log_path, budget, n=12, n_failures=12):
    """v8 unit source: FAILURE-GROUNDED mining (2026-07-24, user-approved for the pupa/livebench
    losses). The generic framings in _suggest_units_paper never see what actually goes wrong;
    on high-baseline benches (pupa .86, livebench .70) the residual failures are specific and
    the guard keeps falling back because nothing in the pool addresses them. Here we run the
    init candidate on the select panel ONCE (budget-charged), take the worst-scoring items with
    their real predictions, and ask the reflection LM to diagnose-and-fix.
    Returns (units, diagnostics_dict)."""
    prog = set_instructions(program, init_cand)
    if budget is not None:
        if budget["remaining"] < len(panel):
            print("  failure-mine: no budget for diagnostic pass — skipping", flush=True)
            return [], {}
        budget["remaining"] -= len(panel)
    ev = dspy.Evaluate(devset=panel, metric=metric, num_threads=EVAL_THREADS,
                       display_progress=False, max_errors=10000, provide_traceback=False)
    res = ev(prog)
    try:
        triples = [(ex, pred, float(s)) for (ex, pred, s) in res.results]
    except Exception:
        print("  failure-mine: dspy results unavailable — skipping", flush=True)
        return [], {}
    with open(log_path, "a") as fh:
        fh.write(json.dumps({"ts": time.time(), "candidate": init_cand,
                             "hash": cand_hash(init_cand), "n_batch": len(panel),
                             "mean_score": float(res.score) / 100.0, "item_scores":
                             [t[2] for t in triples], "phase": "failure_mine_diag"}) + "\n")
    worst = sorted(triples, key=lambda t: t[2])[:n_failures]
    cases = []
    for ex, pred, s in worst:
        try:
            inp = {k: str(v)[:400] for k, v in
                   (ex.inputs().toDict() if hasattr(ex, "inputs") else dict(ex)).items()}
            gold = {k: str(v)[:200] for k, v in ex.labels().toDict().items()} \
                if hasattr(ex, "labels") else {}
            out = {k: str(v)[:400] for k, v in
                   (pred.toDict() if hasattr(pred, "toDict") else {"output": str(pred)}).items()}
            cases.append({"score": s, "input": inp, "gold": gold, "model_output": out})
        except Exception:
            continue
    if not cases:
        return [], {}
    framings = [
        f"Diagnose the RECURRING failure modes across these cases, then propose {n} instruction "
        "clauses that would each fix one diagnosed mode. Target the actual mechanism of each "
        "failure, not generic advice.",
        f"For each case, state what a correct response required that the current instructions "
        f"never induce. Propose {n} clauses supplying exactly those missing inductions — "
        "concrete steps a weaker model could follow mechanically.",
    ]
    out_units, seen = [], set()
    for framing in framings:
        prompt = (
            f"A multi-module LLM program for the task '{bench_name}' has these module "
            f"instructions:\n{json.dumps(init_cand, indent=1)[:3500]}\n\n"
            f"Here are its LOWEST-SCORING real cases (input, gold, model output, score):\n"
            f"{json.dumps(cases, indent=1)[:9000]}\n\n{framing} Reply with ONLY a JSON array "
            'of objects {"module": <one of the module names above>, "clause": <one or two '
            "self-contained sentences>}.")
        try:
            raw = refl(prompt)[0]
            arr = json.loads(raw[raw.index("["): raw.rindex("]") + 1])
            for d in arr:
                mod, c = str(d.get("module")), str(d.get("clause", "")).strip()
                key = (mod, " ".join(c.lower().split()))
                if mod in init_cand and 25 <= len(c) <= 400 and key not in seen:
                    seen.add(key)
                    out_units.append((mod, c))
        except Exception as exc:
            print(f"  failure-grounded suggestion failed ({exc})", flush=True)
    diag = {"n_failure_cases": len(cases),
            "failure_case_scores": [c["score"] for c in cases],
            "n_failure_units": len(out_units)}
    print(f"  failure-mine: {len(cases)} cases -> {len(out_units)} clauses", flush=True)
    return out_units, diag


def arm_unitrecomb(program, bench, metric, log_path, budget_calls, bench_dir,
                   reflection_model="glm-5.2", select_n=None, confirm_n=None, max_units=48,
                   confirm_add_val=False, prefix_cap=32, pool_file=None,
                   failure_mine=False, confirm_passes=1):
    """M_ω v3 — v2 geometry + the selection-power upgrade (2026-07-20 user sign-off).

    v2 fixed the geometry (init-from-GEPA, no screen, paired marginals, prefix+drop-one,
    no-regret guard). The remaining binding constraint was SELECTION POWER: on AIME the 27-item
    panel's ranking was uncorrelated with test value (Spearman +.19 n.s.) and the best unit in
    the pool went unshipped. v3 therefore: (a) panels sized to the data — confirm = min(50,
    train/3), select = the rest (150-item trains: 100/50); (b) cross-LM unit mining — clauses
    from EVERY task-LM column's trajectories for this bench (sources tagged, reported
    separately); (c) doubled + framing-diversified LLM suggestions.
    """
    seed_cand = get_instructions(program)
    init_cand = dict(seed_cand)
    official_res = bench_dir / "official" / "result.json"
    if official_res.exists():                    # D2: start where official GEPA finished
        init_cand = {**seed_cand, **json.loads(official_res.read_text())["best_candidate"]}
        print(f"  init = official-GEPA shipped candidate (init==seed: {init_cand == seed_cand})",
              flush=True)

    train = list(bench.train_set)
    if confirm_n is None:
        confirm_n = min(50, len(train) // 3)
    if select_n is None:
        select_n = len(train) - confirm_n
    if len(train) < select_n + confirm_n:        # explicit sizes that overflow small trains
        select_n = max(1, int(len(train) * 0.6))
        confirm_n = len(train) - select_n
        print(f"  small train ({len(train)}): select_n={select_n} confirm_n={confirm_n}",
              flush=True)

    if pool_file:
        # OSL staircase mode (design v2): consume a FROZEN pool verbatim — no mining, no LLM
        # suggestion, no z.ai dependency. Pool order preserved; sources carried through.
        pf = json.loads(Path(pool_file).read_text())
        llm_units = []
        units = [(d["module"], d["unit"]) for d in pf["units"] if d["module"] in seed_cand]
        unit_source = {(d["module"], d["unit"]): d.get("source", "frozen")
                       for d in pf["units"]}
        print(f"  FROZEN POOL {pool_file}: {len(units)} units (no mining/suggestion)",
              flush=True)

    # v3 unit mining: own-LM trajectories first, then EVERY OTHER task-LM column's trajectories
    # for this bench (cross-LM units — fair game per 2026-07-20 directive, tagged separately).
    units, seen, xlm_keys = (units, set(), set()) if pool_file else ([], set(), set())
    def _mine(pdir, tag):
        p = pdir / "proposals.jsonl"
        if not p.exists():
            return
        for line in open(p):
            for mod, text in json.loads(line)["candidate"].items():
                if mod not in seed_cand:
                    continue
                for clause in re.split(r"(?<=[.!?])\s+|\n+|^[-*•]\s*", str(text)):
                    c = clause.strip().strip("-*• ").rstrip()
                    if not 25 <= len(c) <= 400:
                        continue
                    h = hashlib.sha256(f"{mod}|{c.lower()}".encode()).hexdigest()[:16]
                    if h not in seen:
                        seen.add(h)
                        units.append((mod, c))
                        if tag == "xlm":
                            xlm_keys.add((mod, " ".join(c.lower().split())))
    def _arm_dirs(lm_dir):
        # v5.1: include prior unitrecomb variants so pools INHERIT past winners (hover-v5
        # lesson: v3.2's 5 winning units were absent from v5's pool). INVALID_* excluded.
        if pool_file:
            return []          # frozen-pool mode: no mining at all
        return [d for d in sorted(lm_dir.glob("*")) if d.is_dir()
                and (d.name in ("official", "inhouse") or d.name.startswith("unitrecomb"))]
    for d in _arm_dirs(bench_dir):
        _mine(d, "own")
    for other_lm in sorted(bench_dir.parent.glob("*")):
        if other_lm.is_dir() and other_lm != bench_dir:
            for d in _arm_dirs(other_lm):
                _mine(d, "xlm")
    ex_strs = []
    for ex in train[:3]:
        try:
            d = ex.toDict() if hasattr(ex, "toDict") else dict(ex)
            ex_strs.append(json.dumps({k: str(v)[:300] for k, v in d.items()})[:1000])
        except Exception:
            pass
    fail_units, fail_diag = [], {}
    if not pool_file:
        llm_units = _suggest_units_paper(bench.__class__.__name__, init_cand,
                                         make_reflection_lm(reflection_model),
                                         train_examples="\n".join(ex_strs) or None)
        llm_set = {(m, " ".join(c.lower().split())) for m, c in llm_units}
        units = (llm_units + units)[:max_units]
        def _src(u):
            key = (u[0], " ".join(u[1].lower().split()))
            return ("llm" if key in llm_set else
                    "trajectory_xlm" if key in xlm_keys else "trajectory")
        unit_source = {u: _src(u) for u in units}
        n_x = sum(1 for u in units if unit_source[u] == "trajectory_xlm")
        print(f"  units: {len(llm_units)} LLM + {n_x} cross-LM + own-trajectory -> "
              f"{len(units)} used", flush=True)

    select_panel = train[:select_n]
    confirm_panel = train[select_n:select_n + confirm_n]
    if confirm_add_val:
        # v4: fold the val split into the confirm slice (selection still never touches test).
        # Official GEPA selected its init ON val, so val favors init — a CONSERVATIVE bias for
        # the no-regret guard (harder to ship a compile, impossible to ship a val-overfit one).
        confirm_panel = confirm_panel + list(bench.val_set)
        print(f"  confirm_add_val: confirm slice -> {len(confirm_panel)} items", flush=True)
    budget = {"remaining": budget_calls}
    if failure_mine and not pool_file:
        fail_units, fail_diag = _suggest_units_failures(
            bench.__class__.__name__, init_cand, make_reflection_lm(reflection_model),
            program, select_panel, metric, log_path, budget)
        # prepend (never cap-evict the other sources): failure units get marginal slots first
        units = fail_units + [u for u in units if u not in fail_units]
        for u in fail_units:
            unit_source[u] = "failure_grounded"
    # Minimal plan: init + every unit marginal + prefix + drop-one + add-back + 2x confirm.
    need = (1 + len(units) + 3 * prefix_cap) * len(select_panel) + 2 * len(confirm_panel)
    if budget_calls < need:
        print(f"  WARNING: budget {budget_calls} < minimal selection plan {need}; "
              "later stages will be skipped and the no-regret guard will ship init "
              "(pass a larger --budget-calls)", flush=True)

    def compile_cand(chosen):
        cand = dict(init_cand)
        for mod, clause in chosen:
            cand[mod] = cand[mod] + "\n- " + clause
        return cand

    base = evaluate_cand(program, init_cand, select_panel, metric, log_path, "select_init",
                         budget)
    marginals = []
    for u in units:
        s = evaluate_cand(program, compile_cand([u]), select_panel, metric, log_path,
                          "select_marginal", budget)
        if s is None:
            print("  budget exhausted during marginal pass", flush=True)
            break
        marginals.append((s - (base or 0.0), u))
    marginals.sort(key=lambda x: -x[0])
    units_info = {
        "n_llm_suggested": len(llm_units), "n_pool_used": len(units),
        "pool_file": str(pool_file) if pool_file else None,
        "select_init": base,
        "marginals": [{"module": u[0], "unit": u[1], "delta": d,
                       "source": unit_source.get(u, "trajectory")} for d, u in marginals],
    }

    chosen, best_score, cum = [], base or 0.0, []
    # v3.2 (2026-07-21, user directive): prefix cap lifted to 32 = every positive-marginal unit
    # gets a prefix slot. v5: --prefix-cap flag (48 for the ifbench/livebench pushes). The
    # drop-one pass still prunes non-earners, and the confirm-slice guard still gates the
    # whole compile.
    for _, u in [m for m in marginals if m[0] > 0][:prefix_cap]:
        cum = cum + [u]
        s = evaluate_cand(program, compile_cand(cum), select_panel, metric, log_path,
                          f"prefix_k{len(cum)}", budget)
        if s is None:
            break
        if s > best_score:
            chosen, best_score = list(cum), s
    for u in list(chosen):
        s = evaluate_cand(program, compile_cand([x for x in chosen if x != u]), select_panel,
                          metric, log_path, "drop_one", budget)
        if s is not None and s >= best_score:
            chosen, best_score = [x for x in chosen if x != u], s

    # v5: greedy ADD-BACK — positive-marginal units the prefix pass skipped get one shot at
    # joining the pruned set (catches value that only appears in combination; the confirm
    # guard below still gates the whole compile).
    for _, u in [m for m in marginals if m[0] > 0][:prefix_cap]:
        if u in chosen:
            continue
        s = evaluate_cand(program, compile_cand(chosen + [u]), select_panel, metric, log_path,
                          "add_back", budget)
        if s is not None and s > best_score:
            chosen, best_score = chosen + [u], s

    c_init = evaluate_cand(program, init_cand, confirm_panel, metric, log_path, "confirm_init",
                           budget, passes=confirm_passes)
    c_comp = evaluate_cand(program, compile_cand(chosen), confirm_panel, metric, log_path,
                           "confirm_compiled", budget, passes=confirm_passes)
    units_info.update({
        "confirm_passes": confirm_passes, "failure_mine": fail_diag or None,
        "select_compiled": best_score, "confirm_init": c_init, "confirm_compiled": c_comp,
        "compiled_units": [{"module": u[0], "unit": u[1],
                            "source": unit_source.get(u, "trajectory")} for u in chosen],
        "n_compiled": len(chosen),
    })
    if c_comp is None or c_init is None or c_comp < c_init:
        print(f"  NO-REGRET GUARD: confirm {c_comp} < init {c_init} -> shipping GEPA init",
              flush=True)
        units_info.update({"fell_back_to_init": True, "n_compiled": 0, "compiled_units": []})
        return init_cand, units_info
    print(f"  compiled {len(chosen)} units; select {base:.3f}->{best_score:.3f}, "
          f"confirm {c_init:.3f}->{c_comp:.3f}", flush=True)
    units_info["fell_back_to_init"] = False
    return compile_cand(chosen), units_info


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bench", choices=["aime", "hover", "hotpot", "ifbench", "livebench", "pupa"])
    ap.add_argument("--arm", required=True,
                    choices=["official", "official_merge", "inhouse", "unitrecomb", "mipro"])
    ap.add_argument("--task-lm", required=True,
                    help="litellm model id, e.g. openai/Qwen/Qwen3-8B or anthropic/glm-4.7")
    ap.add_argument("--lm-cache-off", action="store_true")
    ap.add_argument("--api-base", default=None)
    ap.add_argument("--api-key-file", default=None)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=None,
                    help="paper Appendix E.2 uses top-k 20 for Qwen3-8B; reaches vLLM via "
                         "extra_body (arms before 2026-07-21 ran without it — recorded deviation)")
    ap.add_argument("--max-tokens", type=int, default=8000,
                    help="paper-exact is 8000 (Appendix E.2); sk2 server max_model_len is "
                         "16384, so leave prompt headroom")
    ap.add_argument("--budget-calls", type=int, default=None,
                    help="default: 600 (official/inhouse, the declared paper budget) or 2400 "
                         "(unitrecomb — its paired selection needs the larger declared budget; "
                         "recorded in result.json, mirrors run_momega_v2's 6000-vs-600)")
    ap.add_argument("--test-n", type=int, default=0, help="0 = full paper test split")
    ap.add_argument("--reflection-model", default="glm-5.2")
    ap.add_argument("--max-units", type=int, default=48,
                    help="[unitrecomb] unit-pool cap (v4: 96 for the aime/ifbench pushes)")
    ap.add_argument("--pool-file", default=None,
                    help="[unitrecomb] FROZEN pool JSON (build_frozen_pool.py) — skips all "
                         "mining + LLM suggestion (OSL staircase primary arm; no z.ai dep)")
    ap.add_argument("--eval-threads", type=int, default=8,
                    help="dspy.Evaluate concurrency per eval; 8 is tunnel-safe (dead-socket "
                         "exposure scales with sockets), 32 for localhost vLLM on sk2")
    ap.add_argument("--prefix-cap", type=int, default=32,
                    help="[unitrecomb] cap on the prefix/drop-one/add-back passes (v5: 48 "
                         "for the ifbench/livebench pushes)")
    ap.add_argument("--run-tag", default=None,
                    help="run-dir suffix (<arm>_<tag>) so parallel variant arms never "
                         "collide; unit mining + init still read the untagged "
                         "official/inhouse dirs")
    ap.add_argument("--eval-passes", type=int, default=1,
                    help="[v8] independent generation passes per SELECT-phase eval, averaged "
                         "(k=3 for aime-class noise; budget charged k*panel per eval)")
    ap.add_argument("--confirm-passes", type=int, default=1,
                    help="[v8] passes for the no-regret confirm evals (k=3 makes the guard "
                         "decision reliable on high-baseline benches where true gains are "
                         "smaller than single-pass noise)")
    ap.add_argument("--test-passes", type=int, default=1,
                    help="[v8] passes for the two final test evals (k=3-5; result.json "
                         "best_test becomes the k-pass mean)")
    ap.add_argument("--failure-mine", action="store_true",
                    help="[v8 unitrecomb] add FAILURE-GROUNDED units: run init on the select "
                         "panel once, show the worst cases (input/gold/output) to the "
                         "reflection LM, mine clauses that fix the diagnosed modes")
    ap.add_argument("--confirm-add-val", action="store_true",
                    help="[unitrecomb] fold the val split into the no-regret confirm slice "
                         "(conservative guard bias; selection never touches test)")
    ap.add_argument("--robust-answer-extract", action="store_true",
                    help="AIME only: score by the LAST integer in the answer field instead of "
                         "bare int() (the paper metric zeroes LaTeX-formatted answers like "
                         "'$504$' — measurement artifact for long-reasoning LMs; no-op for "
                         "bare-integer models, so the Qwen paper-exact column never needs it)")
    a = ap.parse_args()

    if a.budget_calls is None:
        a.budget_calls = 12000 if a.arm == "unitrecomb" else 600

    if a.api_base and "127.0.0.1" in a.api_base:
        # PRE-FLIGHT (2026-07-21 landmine): with max_errors=10000, a dead local endpoint makes
        # dspy zero-score every item and write a plausible-looking 0.0 result (840/840
        # connection errors produced seed_test=0.0 on livebench). Abort loudly instead.
        import urllib.request
        try:
            urllib.request.urlopen(a.api_base.rstrip("/") + "/models", timeout=15)
        except Exception as exc:
            raise SystemExit(f"task-LM endpoint {a.api_base} is DEAD ({exc}); refusing to run "
                             "an arm that would zero-score everything — restore the tunnel/"
                             "server first") from exc

    key = Path(a.api_key_file).read_text().strip() if a.api_key_file else "EMPTY"
    # deep retries + per-attempt timeout: see make_reflection_lm comment (z.ai hang post-mortem)
    lm_kwargs = dict(api_base=a.api_base, api_key=key, temperature=a.temperature,
                     top_p=a.top_p, max_tokens=a.max_tokens, num_retries=10, timeout=300,
                     cache=not a.lm_cache_off)   # v9.1: --lm-cache-off makes k-passes real (F3)
    if a.top_k is not None:
        lm_kwargs["extra_body"] = {"top_k": a.top_k}
    lm = dspy.LM(a.task_lm, **lm_kwargs)
    dspy.configure(lm=lm)

    def _health_probe():
        import urllib.request
        try:
            if a.api_base and "127.0.0.1" in a.api_base:
                urllib.request.urlopen(a.api_base.rstrip("/") + "/models", timeout=15)
            elif a.api_base and "api.z.ai" in a.api_base:
                req = urllib.request.Request(
                    a.api_base.rstrip("/") + "/v1/messages",
                    data=json.dumps({"model": "glm-4.7", "max_tokens": 8,
                                     "messages": [{"role": "user", "content": "ok"}]}).encode(),
                    headers={"x-api-key": key, "anthropic-version": "2023-06-01",
                             "content-type": "application/json"})
                urllib.request.urlopen(req, timeout=45)
            return True
        except Exception:
            return False
    global HEALTH_PROBE, EVAL_THREADS, EVAL_PASSES
    HEALTH_PROBE = _health_probe
    EVAL_THREADS = a.eval_threads
    EVAL_PASSES = a.eval_passes

    bench, program, metric, metric_fb = load_bench(a.bench)
    if a.robust_answer_extract:
        if a.bench != "aime":
            raise SystemExit("--robust-answer-extract is AIME-only")
        metric, metric_fb = robustify_aime_metrics(metric, metric_fb)
    # One directory level per task LM: the two columns (Qwen3-8B paper-exact, GLM-5.2) must
    # never share result.json/proposals.jsonl, and unitrecomb must init from ITS OWN LM's
    # official run.
    lm_tag = re.sub(r"[^A-Za-z0-9._-]+", "_", a.task_lm.split("/")[-1])
    bench_dir = HERE / "runs_paperexact" / a.bench / lm_tag
    rundir = bench_dir / (f"{a.arm}_{a.run_tag}" if a.run_tag else a.arm)
    rundir.mkdir(parents=True, exist_ok=True)
    log_path = rundir / "proposals.jsonl"
    print(f"[{a.bench}|{a.arm}] train={len(bench.train_set)} val={len(bench.val_set)} "
          f"test={len(bench.test_set)} budget={a.budget_calls}", flush=True)

    if a.arm == "official":
        best = arm_official(program, bench, metric_fb, log_path, a.budget_calls,
                            a.reflection_model)
    elif a.arm == "official_merge":
        best = arm_official(program, bench, metric_fb, log_path, a.budget_calls,
                            a.reflection_model, use_merge=True)
    elif a.arm == "mipro":
        best = arm_mipro(program, bench, metric, log_path, a.budget_calls,
                         a.reflection_model)
    elif a.arm == "inhouse":
        best = arm_inhouse(program, bench, metric, metric_fb, log_path, a.budget_calls,
                           a.reflection_model)
    else:
        best, units_info = arm_unitrecomb(program, bench, metric, log_path, a.budget_calls,
                                          bench_dir, reflection_model=a.reflection_model,
                                          max_units=a.max_units,
                                          confirm_add_val=a.confirm_add_val,
                                          prefix_cap=a.prefix_cap, pool_file=a.pool_file,
                                          failure_mine=a.failure_mine,
                                          confirm_passes=a.confirm_passes)

    test = list(bench.test_set)
    if a.test_n:
        test = test[:a.test_n]
    seed_score = evaluate_cand(program, get_instructions(program), test, metric,
                               log_path, "final_test_seed", passes=a.test_passes)
    best_score = evaluate_cand(program, best, test, metric, log_path, "final_test_best",
                               passes=a.test_passes)
    if (seed_score is not None and best_score is not None
            and best_score < seed_score - 0.05):
        # A shipped candidate grossly below seed on test = either a corrupted search (outage
        # zero-scoring poisoned selection) or genuine guardless overfit — either way REVIEW
        # before quoting. Never auto-quarantine (could be real); flag loudly instead.
        print(f"⚠ REGRESSION FLAG: best_test {best_score:.3f} < seed_test {seed_score:.3f} "
              "- 0.05 — review for outage corruption before quoting", flush=True)
    result = {"bench": a.bench, "arm": a.arm, "task_lm": a.task_lm,
              "regression_flag": bool(seed_score is not None and best_score is not None
                                      and best_score < seed_score - 0.05),
              "budget_calls": a.budget_calls, "n_test": len(test),
              "max_tokens": a.max_tokens, "top_k": a.top_k,
              "prefix_cap": a.prefix_cap, "run_tag": a.run_tag,
              "robust_answer_extract": bool(a.robust_answer_extract),
              "eval_passes": a.eval_passes, "confirm_passes": a.confirm_passes,
              "test_passes": a.test_passes, "failure_mine": bool(a.failure_mine),
              "seed_test": seed_score, "best_test": best_score,
              "best_candidate": best}
    if a.arm == "unitrecomb":
        result["units"] = units_info
    (rundir / "result.json").write_text(json.dumps(result, indent=2))
    print(f"[{a.bench}|{a.arm}] DONE seed_test={seed_score:.3f} best_test={best_score:.3f}",
          flush=True)


if __name__ == "__main__":
    main()
