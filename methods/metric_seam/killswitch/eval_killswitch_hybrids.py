"""Kill-switch hybrid arm: run the 7 blind h0 hybrids on both channels, ablate media,
certify op marginals, and emit the FINAL S6 placement verdicts vs designed truth.

Ablation lattice per plant: {fields on/off} x {ops real/null}; op marginal = paired
bootstrap P(rho(hybrid, real ops) > rho(hybrid, null ops)) on held-out; field marginal
analogous. Placement rules (pre-registered in DESIGN.md):
  CODE   best no-field channel (code rung or fieldless hybrid) >= 85% ceiling (P>=.5 boot)
  MIXED  fielded hybrid >= 70% ceiling AND beats best no-field channel by >= 0.10
  EVIDENCE-OP  op marginal certified (P>=.95) AND op-type readout = evidence
  A-LAYER nothing certifies as CODE and gate floor unreached by any code channel
  DEGENERATE rel1 ~ 0 -> nothing certifiable
"""
import importlib.util, json, pathlib, random, signal, sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "methods/metric_seam"))
sys.path.insert(0, str(ROOT / "methods/metric_seam/hybrids"))
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from certificates import spearman, attenuation_ceiling  # noqa: E402
from harness import split_ids                            # noqa: E402
from ops import Ops                                      # noqa: E402
from plants import TRUTH_TYPE                            # noqa: E402
from eval_killswitch import load_two_pass, PIDS          # noqa: E402

OUT = ROOT / "outputs/metric_seam_pilot/killswitch"
HYB = pathlib.Path(__file__).parent / "programs_ks"
B = 2000


class NullOps:
    @staticmethod
    def normalize(text):
        return text

    @staticmethod
    def extract_dates(text):
        return []

    @staticmethod
    def sent_stats(text):
        return (0, 0.0, 0.0)

    @staticmethod
    def retrieve_similar(text, k=5, exclude_id=None):
        return []


def _alarm(sig, frame):
    raise TimeoutError()


signal.signal(signal.SIGALRM, _alarm)


def load_fields():
    out = {}
    f = OUT / "field_results_ks.jsonl"
    if not f.exists():
        print("WARNING: no field results yet")
        return out
    for line in open(f):
        r = json.loads(line)
        pid, field = r["aspect_id"].split("__", 1)
        ans = (r.get("raw") or "").strip()
        if ans.upper() == "NONE":
            ans = ""
        out.setdefault(pid, {}).setdefault(r["datapoint_id"], {})[field] = ans
    return out


class _SelfExcludingOps:
    """Wraps an Ops instance so retrieve_similar always excludes the CURRENT document by
    its true datapoint_id, regardless of what the hybrid program passes as exclude_id.
    Fixes the p903 self-hit interface gap: the score(text, extracted, ops) contract has no
    dpid parameter, so h0 programs can only approximate self-exclusion via a similarity
    threshold, which misses truncated/edited docs whose self-similarity falls below it."""

    def __init__(self, base, dpid):
        self._base, self._dpid = base, dpid

    def __getattr__(self, name):
        return getattr(self._base, name)

    def retrieve_similar(self, text, k=5, exclude_id=None):
        return self._base.retrieve_similar(text, k=k, exclude_id=self._dpid)


def run_hybrid(mod, texts, fields, ops):
    col = {}
    for dpid, t in texts.items():
        try:
            signal.alarm(20)
            v = float(mod.score(t, fields.get(dpid, {}), _SelfExcludingOps(ops, dpid)))
            col[dpid] = max(0.0, min(1.0, v))
        except Exception:
            col[dpid] = None
        finally:
            signal.alarm(0)
    return col


def boot_pair(sel, a, b, ch, seed=23):
    """P(rho_a > rho_b), P(rho_a >= rho_b + 0.10), paired item bootstrap.
    If b is CONSTANT (rho undefined — e.g. op ablation destroys the channel entirely),
    treat rho_b as 0: the comparison degenerates to P(rho_a > 0), which is the honest
    reading of 'the op carries all the signal'."""
    rng = random.Random(seed)
    n = len(sel)
    gt = m10 = used = 0
    for _ in range(B):
        idx = [sel[rng.randrange(n)] for _ in range(n)]
        ra = spearman([a[d] for d in idx], [ch[d] for d in idx])
        rb = spearman([b[d] for d in idx], [ch[d] for d in idx])
        if ra != ra:
            continue
        if rb != rb:
            rb = 0.0
        used += 1
        gt += ra > rb
        m10 += ra >= rb + 0.10
    return (gt / used if used else None, m10 / used if used else None)


def boot_frac(sel, a, ch, frac_ceil, seed=29):
    rng = random.Random(seed)
    n = len(sel)
    hit = used = 0
    for _ in range(B):
        idx = [sel[rng.randrange(n)] for _ in range(n)]
        r = spearman([a[d] for d in idx], [ch[d] for d in idx])
        if r != r:
            continue
        used += 1
        hit += r >= frac_ceil
    return hit / used if used else None


def main():
    items = {x["datapoint_id"]: x["ctext"]
             for x in json.load(open(ROOT / "outputs/metric_seam_pilot/v1/items_v1.json"))}
    _, test = split_ids()
    cs = json.load(open(OUT / "code_scores_ks.json"))["scores"]
    code_report = json.load(open(OUT / "killswitch_report.json"))
    fields = load_fields()
    ops_real = Ops(corpus_path=str(ROOT /
                                   "runs/validity_full/v2/press_releases/datapoints.json"))
    ops_null = NullOps()
    arms = {"S": load_two_pass(OUT / "channels_synth.jsonl"),
            "J": load_two_pass(OUT / "results_judge.jsonl")}

    cols = {}
    for pid in PIDS:
        prog = HYB / f"{pid}_h0.py"
        spec = importlib.util.spec_from_file_location(prog.stem, prog)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        fmap = fields.get(pid, {})
        has_fields = bool(getattr(mod, "LLM_FIELDS", {}))
        cols[pid] = {
            "full": run_hybrid(mod, items, fmap, ops_real),
            "noops": run_hybrid(mod, items, fmap, ops_null),
            "nofields": run_hybrid(mod, items, {}, ops_real) if has_fields else None,
            "has_fields": has_fields}
        print(f"{pid}: hybrid run (fields={has_fields})")

    report = {}
    print("\narm plant type              hyb_te  code_te  op_marg(P>) field_marg(P>=.10) "
          "P(hyb>=.85c) verdict")
    for arm, (chan, rel) in arms.items():
        for pid in PIDS:
            ch = chan.get(pid)
            if not ch:
                continue
            r1 = rel[pid]
            ceil = (attenuation_ceiling(max(0.0, min(1.0, r1)), 2)
                    if r1 == r1 and r1 > 0.05 else None)
            c = cols[pid]
            sel = [d for d in test if d in ch and c["full"].get(d) is not None]

            def rho(col):
                s = [d for d in sel if col.get(d) is not None]
                return spearman([col[d] for d in s], [ch[d] for d in s])

            best_code_key = code_report[f"{arm}:{pid}"].get("best", {}).get("flavor")
            code_col = cs.get(f"{pid}_{best_code_key}") if best_code_key else None
            hyb_te, code_te = rho(c["full"]), (rho(code_col) if code_col else None)
            op_p, _ = boot_pair(sel, c["full"], c["noops"], ch, seed=31)
            fld = (boot_pair(sel, c["full"], c["nofields"], ch, seed=37)
                   if c["nofields"] else (None, None))
            p85 = boot_frac(sel, c["full"], ch, 0.85 * ceil) if ceil else None
            p70 = boot_frac(sel, c["full"], ch, 0.70 * ceil) if ceil else None

            # no-field channel = max(code rung, fieldless hybrid variant)
            nofield_cols = [code_col] if code_col else []
            nofield_cols.append(c["nofields"] if c["nofields"] else c["full"])
            nf_best = max(nofield_cols, key=lambda col: rho(col))
            nf_te = rho(nf_best)
            p85_nf = boot_frac(sel, nf_best, ch, 0.85 * ceil, seed=41) if ceil else None
            _, p_mix10 = boot_pair(sel, c["full"], nf_best, ch, seed=43)

            if ceil is None:
                verdict = "DEGENERATE"
            elif p85_nf is not None and p85_nf >= 0.5:
                verdict = "CODE"
                if op_p is not None and op_p >= 0.95:
                    ot = code_report.get("op_type_readout", {}).get(pid, {})
                    verdict = ("CODE+EVIDENCE_OP" if ot.get("verdict") == "evidence"
                               else "CODE+COMP_OP")
            elif (c["has_fields"] and p70 is not None and p70 >= 0.5
                  and p_mix10 is not None and p_mix10 >= 0.5):
                verdict = "MIXED"
            elif (op_p is not None and op_p >= 0.95
                  and code_report.get("op_type_readout", {}).get(pid, {})
                  .get("verdict") == "evidence" and p85 is not None and p85 >= 0.5):
                verdict = "CODE+EVIDENCE_OP"
            else:
                # nothing certified at the pre-registered bars; distinguish a knife-edge
                # miss from a clear non-codable readout
                best_te = max(v for v in [hyb_te, code_te, nf_te] if v == v)
                verdict = ("UNCERTIFIED(near-miss)"
                           if ceil and best_te >= 0.75 * ceil else "UNCERTIFIED")
            report[f"{arm}:{pid}"] = {
                "rho_hybrid_test": round(hyb_te, 3) if hyb_te == hyb_te else None,
                "rho_code_test": round(code_te, 3) if code_te and code_te == code_te
                else None,
                "rho_nofield_best": round(nf_te, 3) if nf_te == nf_te else None,
                "ceiling": round(ceil, 3) if ceil else None,
                "P_op_marginal_gt0": op_p, "P_field_marg_ge10": fld[1],
                "P_hyb_ge_85ceil": p85, "P_hyb_ge_70ceil": p70,
                "P_nofield_ge_85ceil": p85_nf, "P_mixed_margin10": p_mix10,
                "verdict": verdict, "designed": TRUTH_TYPE[pid], "n_test": len(sel)}
            print(f"{arm}   {pid} {TRUTH_TYPE[pid]:17s} "
                  f"{hyb_te:+.3f}  {code_te if code_te else float('nan'):+.3f}   "
                  f"{op_p if op_p is not None else -1:.2f}        "
                  f"{fld[1] if fld[1] is not None else -1:.2f}            "
                  f"{p85 if p85 is not None else -1:.2f}      {verdict}")

    json.dump(report, open(OUT / "killswitch_hybrid_report.json", "w"), indent=1)
    print(f"-> {OUT/'killswitch_hybrid_report.json'}")


if __name__ == "__main__":
    main()
