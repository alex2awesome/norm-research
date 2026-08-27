"""Code channel for a generalized task survey: run codegen flavors over the task's items.
Usage: python3 run_code_flavors_task.py <task>
"""
import importlib.util, json, pathlib, signal, sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
FLAVORS = ["v0_keyword", "v1_structure", "v2_holistic"]

def _alarm(sig, frame):
    raise TimeoutError()

def main():
    task = sys.argv[1]
    CODEGEN = ROOT / "runs/validity_full/v2" / task / "codegen_claude"
    OUT = ROOT / "outputs/metric_seam_pilot/tasks" / task
    signal.signal(signal.SIGALRM, _alarm)
    aspects = json.load(open(OUT / "aspects_used.json"))
    items = {x["datapoint_id"]: x["ctext"] for x in json.load(open(OUT / "items.json"))}
    results, broken = {}, []
    for aid in aspects:
        for fl in FLAVORS:
            path = CODEGEN / f"{aid}_{fl}.py"
            if not path.exists():
                results[f"{aid}_{fl}"] = None
                broken.append(f"{aid}_{fl}:missing")
                continue
            try:
                spec = importlib.util.spec_from_file_location(f"{task}_{path.stem}", path)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                score = mod.score
            except Exception as e:
                results[f"{aid}_{fl}"] = None
                broken.append(f"{aid}_{fl}:{type(e).__name__}")
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
    json.dump(results, open(OUT / "code_scores.json", "w"))
    ok = sum(1 for v in results.values() if v is not None)
    print(f"{task}: {ok}/{len(aspects)*3} programs ran; broken({len(broken)}): "
          f"{broken[:6]}{'...' if len(broken) > 6 else ''}")

if __name__ == "__main__":
    main()
