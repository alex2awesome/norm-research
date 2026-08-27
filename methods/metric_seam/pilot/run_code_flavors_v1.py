"""v1 code channel: same canonical text as the LLM channel (fixes v0 apples-to-apples bug).
Also scores the a86 CF-injected items with every working a86 flavor.
"""
import importlib.util, json, pathlib, signal

ROOT = pathlib.Path(__file__).resolve().parents[3]
CODEGEN = ROOT / "runs/validity_full/v2/press_releases/codegen_claude"
OUT = ROOT / "outputs/metric_seam_pilot/v1"

ASPECTS = ["a79", "a80", "a110", "a100", "a101", "a86", "a105", "a118", "a117", "a73"]
FLAVORS = ["v0_keyword", "v1_structure", "v2_holistic"]

def _alarm(sig, frame):
    raise TimeoutError()

def load_score(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.score

def run(score, texts):
    col = {}
    for dpid, t in texts.items():
        try:
            signal.alarm(10)
            col[dpid] = float(score(t))
        except Exception:
            col[dpid] = None
        finally:
            signal.alarm(0)
    return col

def main():
    signal.signal(signal.SIGALRM, _alarm)
    items = {x["datapoint_id"]: x["ctext"] for x in json.load(open(OUT / "items_v1.json"))}
    cf = {x["datapoint_id"]: x["ctext"] for x in json.load(open(OUT / "cf_items_a86.json"))}
    results, cf_results = {}, {}
    for aid in ASPECTS:
        for fl in FLAVORS:
            path = CODEGEN / f"{aid}_{fl}.py"
            try:
                score = load_score(path)
            except Exception as e:
                print(f"{aid}_{fl}: BROKEN ({type(e).__name__})")
                results[f"{aid}_{fl}"] = None
                continue
            results[f"{aid}_{fl}"] = run(score, items)
            if aid == "a86":
                cf_results[f"{aid}_{fl}"] = run(score, cf)
    json.dump(results, open(OUT / "code_scores_v1.json", "w"))
    json.dump(cf_results, open(OUT / "code_scores_cf_a86.json", "w"))
    ok = sum(1 for v in results.values() if v is not None)
    print(f"scored {ok} programs on canonical text; CF sets: {list(cf_results)}")

if __name__ == "__main__":
    main()
