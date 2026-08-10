"""Run the existing codegen flavors (v0/v1/v2) for the pilot aspects over the pilot items.

CPU-only; programs follow the score(text)->float [0,1] contract.
"""
import importlib.util, json, pathlib, signal

ROOT = pathlib.Path(__file__).resolve().parents[3]
CODEGEN = ROOT / "runs/validity_full/v2/press_releases/codegen_claude"
OUT = ROOT / "outputs/metric_seam_pilot"

ASPECTS = ["a79", "a80", "a110", "a100", "a101", "a86", "a105", "a118", "a117", "a73"]
FLAVORS = ["v0_keyword", "v1_structure", "v2_holistic"]

class Timeout(Exception):
    pass

def _alarm(sig, frame):
    raise Timeout()

def load_score(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.score

def main():
    items = json.load(open(OUT / "items.json"))
    signal.signal(signal.SIGALRM, _alarm)
    results = {}
    for aid in ASPECTS:
        for fl in FLAVORS:
            path = CODEGEN / f"{aid}_{fl}.py"
            try:
                score = load_score(path)
            except Exception as e:
                print(f"{aid}_{fl}: BROKEN PROGRAM ({type(e).__name__}) — skipped")
                results[f"{aid}_{fl}"] = None
                continue
            col = {}
            errs = 0
            for it in items:
                try:
                    signal.alarm(10)
                    col[it["datapoint_id"]] = float(score(it["text"]))
                except Exception:
                    errs += 1
                    col[it["datapoint_id"]] = None
                finally:
                    signal.alarm(0)
            results[f"{aid}_{fl}"] = col
            vals = [v for v in col.values() if v is not None]
            import statistics as st
            spread = st.pstdev(vals) if len(vals) > 1 else 0.0
            print(f"{aid}_{fl}: n={len(vals)} errs={errs} "
                  f"mean={st.mean(vals):.3f} sd={spread:.3f}")
    json.dump(results, open(OUT / "code_scores.json", "w"))
    print("wrote", OUT / "code_scores.json")

if __name__ == "__main__":
    main()
