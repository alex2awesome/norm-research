"""Driver for the Certified Unit Framework pilot (notes/2026-07-04__unit-certification-theory.md).

One invocation = one executor (Def 1/9: units are executor-indexed). The U5 ladder is three
invocations (3B/8B/70B) + `--aggregate` to type units across them. Hosts come from a host-JSON
built by prep_cuf_hosts.py: {"metric": ..., "m_rubric": ..., "hosts": {"description": ..., "gepa": ...,
"checklist": ...}}.

Trust gates (must pass before any real row is read): a PLANTED span (known-strong instruction) must
certify; a PLACEBO span (inert filler sentence) must land SUBTHRESHOLD/UNDERSAMPLED.

Usage (sk3):
  python -m methods.metric_implementer.experiments.run_unit_certificate \
      --host-json outputs/unit_cert/hosts_g24.json \
      --target-model meta-llama/Llama-3.1-8B-Instruct \
      --out-dir /lfs/.../outputs/unit_cert/g24/llama8b [--n-ctx 10 --n-sham 12]
  python -m ... --aggregate outputs/unit_cert/g24/llama3b outputs/unit_cert/g24/llama8b \
      outputs/unit_cert/g24/llama70b --ladder 3B,8B,70B
"""
from __future__ import annotations

import argparse, json, os, sys
import numpy as np

from . import alpha_probe as ap
from . import unit_certificate as uc
from .run_real_test import _load_texts
from .. import config as cfgmod

PLANTED = "Regardless of every other consideration, always answer YES about this text."
PLACEBO = "As has been noted in passing at this point in the present document, this continues."


# --------------------------------------------------------------------------------------------
def _fake_score_fn(probe_n=48):
    """Deterministic synthetic executor for --fake smoke runs (no vLLM). Keyword-block design
    mirroring the CPU tests so planted/placebo gates behave as in test_unit_certificate."""
    def fn(prompts):
        out = []
        base = np.linspace(-0.5, 0.5, probe_n)
        for p in prompts:
            z = base.copy()
            pl = p.lower()
            if "always answer yes" in pl:
                z = z + 3.0
            for j, kw in enumerate(["worldbuilding", "dialogue", "pacing", "research",
                                    "authentic", "setting"]):
                if kw in pl:
                    b = probe_n // 8
                    z[j * b:(j + 1) * b] += 2.5
            out.append(1.0 / (1.0 + np.exp(-z)))
        return np.asarray(out)
    return fn


def _real_score_fn(executor, probe_texts, max_chars):
    calls = {"n": 0}

    def fn(prompts):
        sigs = []
        for i, p in enumerate(prompts):
            sigs.append(ap.signature(executor, p, probe_texts, max_chars,
                                     template=ap._YESNO_TEXTFIRST))
            calls["n"] += 1
            if calls["n"] % 100 == 0:
                print(f"  scored {calls['n']} host variants ...", flush=True)
        return np.stack(sigs)
    return fn


def _glm_paraphrases(hosts: dict, cache_path: str, model: str = "glm-4.7") -> dict:
    """Best-effort GLM paraphrase generation for U2, cached to JSON (be sparing with GLM:
    one call per host, all nodes batched). Returns {host_key: {node_id: [paraphrases]}}."""
    if os.path.exists(cache_path):
        return json.load(open(cache_path))
    out = {}
    try:
        from ..backends import LLMBackend
        _cfg = cfgmod.ImplementerConfig()
        _cfg.backend = "zai_anthropic"
        be = LLMBackend(model, role="paraphrase", cfg=_cfg, temperature=0.4)
        for key, host in hosts.items():
            nodes = [n for n in uc.address_lattice(host) if n.level == 1]
            listing = "\n".join(f"{n.node_id}: {n.span}" for n in nodes)
            prompt = ("Rewrite each numbered sentence below preserving its EXACT meaning "
                      "(a strict paraphrase; same criterion content, different wording). "
                      "Return STRICT JSON: {\"<id>\": [\"variant1\", \"variant2\"], ...}.\n\n"
                      + listing)
            for attempt in range(3):
                try:
                    txt = be.generate(prompt, system="You paraphrase text precisely.",
                                      max_tokens=1500)
                    j = json.loads(txt[txt.index("{"): txt.rindex("}") + 1])
                    out[key] = {int(k): [str(v) for v in vs][:2] for k, vs in j.items()}
                    break
                except Exception as e:                                    # retry, then give up
                    if attempt == 2:
                        print(f"[paraphrase] {key}: giving up ({type(e).__name__}: {e})")
    except Exception as e:
        print(f"[paraphrase] backend unavailable ({type(e).__name__}: {e}); running without U2")
    json.dump(out, open(cache_path, "w"))
    return out


def run_one(a):
    spec = json.load(open(a.host_json))
    hosts = dict(spec["hosts"])
    os.makedirs(a.out_dir, exist_ok=True)

    if a.fake:
        probe_texts = [f"probe {i}" for i in range(48)]
        score_fn = _fake_score_fn(48)
        m_bar = None
    else:
        cfg = cfgmod.apply_task_preset(cfgmod.ImplementerConfig(), spec.get("task", "creative-writing"))
        cfg0 = cfgmod.ImplementerConfig()
        from ..vllm_backend import make_judge_backend
        executor = make_judge_backend(a.target_model, cfg0, temperature=None)
        texts, _ = _load_texts(spec.get("task", "creative-writing"), 60 + a.n_probes, cfg)
        probe_texts = texts[60: 60 + a.n_probes]
        print(f"probe set = {len(probe_texts)} items; executor = {a.target_model}")
        score_fn = _real_score_fn(executor, probe_texts, cfg.max_text_chars)
        # m̄_ω for the δ^M arm: the metric's own orbit-averaged verdict (Def 5)
        orb = ap.orbit_metric_verdict(executor, spec["m_rubric"], probe_texts,
                                      cfg.max_text_chars, n_forms=4)
        m_bar = np.asarray(orb["m_bar"], float)
        np.save(os.path.join(a.out_dir, "m_bar.npy"), m_bar)

    paras = {}
    if a.paraphrase_cache:
        paras = (_glm_paraphrases(hosts, a.paraphrase_cache) if a.gen_paraphrases
                 else (json.load(open(a.paraphrase_cache)) if os.path.exists(a.paraphrase_cache) else {}))

    # ---- trust gates on an augmented description host (plan §Verification 2) ----------------
    gate_host = hosts[next(iter(hosts))].rstrip() + " " + PLANTED + " " + PLACEBO
    gate = uc.certify_host(gate_host, score_fn, n_ctx=max(4, a.n_ctx // 2), n_sham=a.n_sham,
                           alpha=a.alpha, m_bar=m_bar, delta_min=a.delta_min, seed=a.seed)
    g_planted = next((r for r in gate["rows"] if "always answer yes" in r["span"].lower()), None)
    g_placebo = next((r for r in gate["rows"] if "noted in passing" in r["span"].lower()), None)
    gate_ok = bool(g_planted and g_planted.get("detect_free")) and \
              bool(g_placebo and not g_placebo.get("detect_free"))
    json.dump({"planted": g_planted, "placebo": g_placebo, "ok": gate_ok},
              open(os.path.join(a.out_dir, "trust_gate.json"), "w"), default=str)
    print(f"TRUST GATE: planted detect={bool(g_planted and g_planted.get('detect_free'))} "
          f"placebo inert={bool(g_placebo and not g_placebo.get('detect_free'))} -> "
          f"{'PASS' if gate_ok else 'FAIL'}")
    if not gate_ok and not a.ignore_gate:
        open(os.path.join(a.out_dir, "STATUS"), "w").write("GATE-FAILED\n")
        sys.exit(2)

    # ---- certify every host --------------------------------------------------------------
    for key, host in hosts.items():
        print(f"\n=== host: {key} ({len(host)} chars) ===")
        res = uc.certify_host(host, score_fn, n_ctx=a.n_ctx, n_sham=a.n_sham, alpha=a.alpha,
                              depth=a.depth, m_bar=m_bar, delta_min=a.delta_min,
                              company_profile=a.company_profile,
                              paraphrases={int(k): v for k, v in paras.get(key, {}).items()},
                              seed=a.seed)
        out = {"metric": spec.get("metric"), "host": key, "executor": a.target_model, **res}
        json.dump(out, open(os.path.join(a.out_dir, f"host_{key}.json"), "w"), default=str)
        fp = res["fingerprints"]
        np.savez(os.path.join(a.out_dir, f"host_{key}_fps.npz"),
                 node_ids=np.array(list(fp.keys())),
                 fps=np.array([fp[k] for k in fp.keys()], float),
                 spans=np.array([r["span"] for r in res["rows"]], dtype=object))
        vd = {}
        for r in res["rows"]:
            vd[r["verdict"]] = vd.get(r["verdict"], 0) + 1
        n_free = sum(bool(r.get("detect_free")) for r in res["rows"])
        n_M = sum(bool(r.get("detect_M")) for r in res["rows"])
        print(f"  verdicts: {vd} | detect_free {n_free}/{len(res['rows'])} | detect_M {n_M}")
    open(os.path.join(a.out_dir, "STATUS"), "w").write("DONE\n")
    print(f"\nDONE -> {a.out_dir}")


def run_bank(a):
    """Tier-1 metric-bank census: certify the DESCRIPTION host of every R2 metric in a task's
    hierarchy (merged_description). Compact hosts only (the company-profile upgrade gates the
    redundant GEPA/checklist hosts). One shared trust gate per (task, executor); per-metric rows
    appended to bank_units.jsonl; fingerprints per metric in fps/ subdir."""
    mg = json.load(open(a.bank_r2))["merged_groups"]
    metrics = [(g["merged_name"], (g.get("merged_description") or "").strip()) for g in mg]
    metrics = [(n, d) for n, d in metrics if len(d.split()) >= 10]
    if a.bank_limit:
        metrics = metrics[: a.bank_limit]
    os.makedirs(a.out_dir, exist_ok=True)
    os.makedirs(os.path.join(a.out_dir, "fps"), exist_ok=True)
    done = set()
    out_path = os.path.join(a.out_dir, "bank_units.jsonl")
    if os.path.exists(out_path):                                   # resumable
        done = {json.loads(l)["metric"] for l in open(out_path)}
        print(f"[bank] resuming: {len(done)} metrics already done")

    cfg = cfgmod.apply_task_preset(cfgmod.ImplementerConfig(), a.bank_task)
    cfg0 = cfgmod.ImplementerConfig()
    from ..vllm_backend import make_judge_backend
    executor = make_judge_backend(a.target_model, cfg0, temperature=None)
    texts, _ = _load_texts(a.bank_task, 60 + a.n_probes, cfg)
    probe_texts = texts[60: 60 + a.n_probes]
    print(f"[bank] task={a.bank_task} metrics={len(metrics)} probes={len(probe_texts)} "
          f"executor={a.target_model}")
    score_fn = _real_score_fn(executor, probe_texts, cfg.max_text_chars)

    # U2 (paraphrase-identity): GLM paraphrases per metric description, one call/host, cached.
    # Off unless --paraphrase-cache given (keeps the original bank census unchanged).
    paras = {}
    if a.paraphrase_cache:
        _hosts = {name: desc for name, desc in metrics}
        paras = (_glm_paraphrases(_hosts, a.paraphrase_cache) if a.gen_paraphrases
                 else (json.load(open(a.paraphrase_cache)) if os.path.exists(a.paraphrase_cache) else {}))
        print(f"[bank] U2 ON: paraphrases for {len(paras)}/{len(metrics)} metrics", flush=True)

    # one shared trust gate per (task, executor)
    g_host = metrics[0][1].rstrip() + " " + PLANTED + " " + PLACEBO
    gate = uc.certify_host(g_host, score_fn, n_ctx=4, n_sham=a.n_sham, alpha=a.alpha,
                           delta_min=a.delta_min, seed=a.seed)
    gp = next((r for r in gate["rows"] if "always answer yes" in r["span"].lower()), None)
    gb = next((r for r in gate["rows"] if "noted in passing" in r["span"].lower()), None)
    # Placebo inertness via CI-materiality (certified_lo), NOT the point-δ two-gate used for
    # detect_free: an inert filler that nudges δ≈0.05 but whose CI lower bound is 0 is within noise
    # of the floor and must count inert. detect_free's point-δ≥δ_min is anti-conservative here and
    # spuriously failed doc-like tasks (code-review, peer-review) whose fillers are mildly
    # on-topic while humor/math had truly-zero placebos (user decision 2026-07-07).
    placebo_inert = bool(gb) and (gb.get("certified_lo") or 0.0) < a.delta_min
    ok = bool(gp and gp.get("detect_free")) and placebo_inert
    json.dump({"planted": gp, "placebo": gb, "ok": ok, "placebo_inert_by": "certified_lo<delta_min"},
              open(os.path.join(a.out_dir, "trust_gate.json"), "w"), default=str)
    print(f"[bank] TRUST GATE: {'PASS' if ok else 'FAIL'}")
    if not ok and not a.ignore_gate:
        open(os.path.join(a.out_dir, "STATUS"), "w").write("GATE-FAILED\n"); sys.exit(2)

    fout = open(out_path, "a")
    for k, (name, desc) in enumerate(metrics):
        if name in done:
            continue
        try:
            orb = ap.orbit_metric_verdict(executor, f"{name}: {desc}", probe_texts,
                                          cfg.max_text_chars, n_forms=4)
            m_bar = np.asarray(orb["m_bar"], float)
            res = uc.certify_host(desc, score_fn, n_ctx=a.n_ctx, n_sham=a.n_sham,
                                  alpha=a.alpha, depth=a.depth, m_bar=m_bar,
                                  delta_min=a.delta_min, seed=a.seed,
                                  paraphrases={int(kk): v for kk, v in paras.get(name, {}).items()})
            slim = [{kk: r.get(kk) for kk in
                     ("node_id", "level", "span", "delta_free", "p_free", "delta_M", "p_M",
                      "sign_stability", "kappa", "eps_ctx", "eps_id", "r_self", "verdict", "atom",
                      "detect_free", "detect_M", "certified_lo")} for r in res["rows"]]
            fout.write(json.dumps({"metric": name, "k": k, "rows": slim,
                                   "meta": res["meta"]}) + "\n")
            fout.flush()
            fp = res["fingerprints"]
            np.savez(os.path.join(a.out_dir, "fps", f"m{k:04d}.npz"),
                     metric=name, node_ids=np.array(list(fp.keys())),
                     fps=np.array([fp[x] for x in fp], float),
                     spans=np.array([r["span"] for r in res["rows"]], dtype=object),
                     m_bar=m_bar)
            nf = sum(bool(r.get("detect_free")) for r in res["rows"])
            nM = sum(bool(r.get("detect_M")) for r in res["rows"])
            print(f"[bank] {k+1}/{len(metrics)} {name[:48]:48s} nodes={len(slim)} "
                  f"free={nf} M={nM}", flush=True)
        except Exception as e:
            print(f"[bank] {name[:60]}: ERROR {type(e).__name__}: {str(e)[:120]}", flush=True)
            if "EngineDead" in type(e).__name__ or "EngineDead" in str(e):
                # dead engine fails every remaining metric identically — abort (resumable), don't loop
                print("[bank] engine dead — aborting bank loop (resume with same command)", flush=True)
                fout.close()
                open(os.path.join(a.out_dir, "STATUS"), "w").write("ENGINE-DIED\n")
                sys.exit(3)
    fout.close()
    open(os.path.join(a.out_dir, "STATUS"), "w").write("DONE\n")
    print(f"[bank] DONE -> {out_path}")


def aggregate(a):
    """U5: type units across executor runs (Def 9). Matches nodes by (host, span)."""
    ladder = a.ladder.split(",")
    runs = []
    for d, e in zip(a.aggregate, ladder):
        rows = {}
        for f in os.listdir(d):
            if f.startswith("host_") and f.endswith(".json"):
                j = json.load(open(os.path.join(d, f)))
                z = np.load(os.path.join(d, f.replace(".json", "_fps.npz")), allow_pickle=True)
                fpm = {int(i): v for i, v in zip(z["node_ids"], z["fps"])}
                for r in j["rows"]:
                    rows[(j["host"], r["span"])] = (bool(r.get("detect_free")), fpm[r["node_id"]])
        runs.append(rows)
    keys = set().union(*[set(r) for r in runs])
    out = []
    for k in sorted(keys):
        det = {e: (runs[i][k][0] if k in runs[i] else False) for i, e in enumerate(ladder)}
        fps = {e: runs[i][k][1] for i, e in enumerate(ladder) if k in runs[i]}
        scope = uc.cross_executor_scope(fps, det, ladder, r_star=a.r_star)
        out.append({"host": k[0], "span": k[1], "scope": scope, **{f"det_{e}": det[e] for e in ladder}})
    path = a.out_dir or os.path.dirname(a.aggregate[0])
    json.dump(out, open(os.path.join(path, "u5_scope.json"), "w"))
    from collections import Counter
    print("U5 scope distribution:", dict(Counter(r["scope"] for r in out)))
    print(f"-> {os.path.join(path, 'u5_scope.json')}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host-json")
    p.add_argument("--target-model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--out-dir")
    p.add_argument("--n-probes", type=int, default=300)
    p.add_argument("--n-ctx", type=int, default=10)
    p.add_argument("--n-sham", type=int, default=12)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--paraphrase-cache", default=None)
    p.add_argument("--gen-paraphrases", action="store_true")
    p.add_argument("--ignore-gate", action="store_true")
    p.add_argument("--delta-min", type=float, default=0.01,
                   help="materiality floor (Def 6 two-gate; pilot lesson from the 70B placebo)")
    p.add_argument("--company-profile", action="store_true",
                   help="Tier-2: solo/LOO effect bracket + within-host species merge")
    p.add_argument("--fake", action="store_true", help="synthetic executor smoke run (no vLLM)")
    p.add_argument("--aggregate", nargs="+", help="U5 mode: run dirs, weak->strong")
    p.add_argument("--ladder", default="3B,8B,70B")
    p.add_argument("--r-star", type=float, default=0.5)
    p.add_argument("--bank-r2", help="Tier-1 bank mode: <task>_general_r2_expanded.json")
    p.add_argument("--bank-task", help="config/manifest task name for the bank corpus")
    p.add_argument("--bank-limit", type=int, default=0)
    a = p.parse_args(argv)
    if a.aggregate:
        aggregate(a)
    elif a.bank_r2:
        assert a.bank_task and a.out_dir, "--bank-task and --out-dir required"
        run_bank(a)
    else:
        assert a.host_json and a.out_dir, "--host-json and --out-dir required"
        run_one(a)


if __name__ == "__main__":
    main()
