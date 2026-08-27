"""Run the 21 blind description-compiled kill-switch programs on the 250 canonical texts.

Output: outputs/metric_seam_pilot/killswitch/code_scores_ks.json
        {"<pid>_<flavor>": {dpid: float|None}}; broken programs recorded as failed rungs.
"""
import importlib.util, json, pathlib, signal, sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
OUT = ROOT / "outputs/metric_seam_pilot/killswitch"
CG = pathlib.Path(__file__).parent / (sys.argv[1] if len(sys.argv) > 1 else "codegen")
OUTNAME = sys.argv[2] if len(sys.argv) > 2 else "code_scores_ks.json"


def _alarm(sig, frame):
    raise TimeoutError()


signal.signal(signal.SIGALRM, _alarm)


def main():
    items = {x["datapoint_id"]: x["ctext"]
             for x in json.load(open(ROOT / "outputs/metric_seam_pilot/v1/items_v1.json"))}
    scores, failed = {}, []
    for prog in sorted(CG.glob("p9*_v*.py")):
        key = prog.stem
        try:
            spec = importlib.util.spec_from_file_location(key, prog)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
        except Exception as e:
            failed.append((key, f"import: {e}"))
            continue
        col = {}
        for dpid, t in items.items():
            try:
                signal.alarm(10)
                v = float(mod.score(t))
                col[dpid] = v if 0.0 <= v <= 1.0 else None
            except Exception:
                col[dpid] = None
            finally:
                signal.alarm(0)
        n_ok = sum(1 for v in col.values() if v is not None)
        distinct = len({round(v, 4) for v in col.values() if v is not None})
        scores[key] = col
        print(f"{key}: {n_ok}/250 scored, {distinct} distinct values"
              + ("  << DEGENERATE" if distinct < 3 else ""))
    for k, e in failed:
        print(f"FAILED RUNG {k}: {e}")
    json.dump({"scores": scores, "failed_rungs": failed},
              open(OUT / OUTNAME, "w"))
    print(f"-> {OUT/OUTNAME} ({len(scores)} programs)")


if __name__ == "__main__":
    main()
