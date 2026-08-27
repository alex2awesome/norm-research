"""Wave-2 code channel: run codegen flavors for the 20 new aspects on canonical text."""
import importlib.util, json, pathlib, signal

ROOT = pathlib.Path(__file__).resolve().parents[3]
CODEGEN = ROOT / "runs/validity_full/v2/press_releases/codegen_claude"
V1 = ROOT / "outputs/metric_seam_pilot/v1"
OUT = ROOT / "outputs/metric_seam_pilot/v2"
FLAVORS = ["v0_keyword", "v1_structure", "v2_holistic"]

def _alarm(sig, frame):
    raise TimeoutError()

def main():
    signal.signal(signal.SIGALRM, _alarm)
    aspects = json.load(open(OUT / "wave2_aspects.json"))
    items = {x["datapoint_id"]: x["ctext"] for x in json.load(open(V1 / "items_v1.json"))}
    results, broken = {}, []
    for aid in aspects:
        for fl in FLAVORS:
            path = CODEGEN / f"{aid}_{fl}.py"
            try:
                spec = importlib.util.spec_from_file_location(path.stem, path)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                score = mod.score
            except Exception as e:
                broken.append(f"{aid}_{fl}:{type(e).__name__}")
                results[f"{aid}_{fl}"] = None
                continue
            col = {}
            for dpid, t in items.items():
                try:
                    signal.alarm(10)
                    col[dpid] = float(score(t))
                except Exception:
                    col[dpid] = None
                finally:
                    signal.alarm(0)
            results[f"{aid}_{fl}"] = col
    json.dump(results, open(OUT / "code_scores_v2.json", "w"))
    ok = sum(1 for v in results.values() if v is not None)
    print(f"{ok}/{len(aspects)*3} programs ran; broken: {broken or 'none'}")

if __name__ == "__main__":
    main()
