"""One-off TEST-split field extraction for the CW held-out promotion batch.

Two census cells (a207 cell 10, a189 cell 12) declare NEW LLM_FIELDS that were only
ever extracted for the 150 TRAIN items (by design -- the dpid contract keeps TEST
sealed during cell-building). To score those candidates on TEST we must extract the
SAME fields for the 100 TEST items, using the EXACT SAME prompt template + per-field
instruction text the cell authors used -- reused here by importing T/FIELDS directly
from each cell's own build_new_fields_prompts.py (not modified, not re-typed).

Writes TEST-only prompts/results OUTSIDE the census cell dirs (a fresh
_heldout_batch/ dir), per the held-out-batch rule that no test-derived artifact is
ever written into a census cell directory.

Usage: python3 build_heldout_test_fields_cw.py            # build prompts only
       python3 build_heldout_test_fields_cw.py --extract   # also run api_field_runner.py
-> methods/metric_seam/battery/../../../outputs/metric_seam_pilot/battery/effort_ladder/
   census/_heldout_batch/{test_field_prompts.jsonl, test_field_results.jsonl}
"""
import importlib.util, json, pathlib, subprocess, sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "methods/metric_seam/battery"))
from battery_common import load_ctx  # noqa: E402

CENSUS = ROOT / "outputs/metric_seam_pilot/battery/effort_ladder/census"
OUTDIR = CENSUS / "_heldout_batch"
OUTDIR.mkdir(exist_ok=True)

# (aspect_id, cell_dir_name) for the 2 confirmed cells with test-side field gaps
# (confirmed by a full LLM_FIELDS x field-store-coverage scan over all 22 queue
#  candidates -- these are the ONLY 2 with <90% test coverage on a declared field).
CELLS = [("a207", "creative_writing__a207"), ("a189", "creative_writing__a189")]


def load_mod(path):
    spec = importlib.util.spec_from_file_location(path.stem + "_x" + str(id(path)), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ctx = load_ctx("creative_writing")
    items = ctx["items"]
    test = sorted(ctx["test"])
    out = OUTDIR / "test_field_prompts.jsonl"
    n = 0
    with open(out, "w") as f:
        for aid, cell_dir in CELLS:
            mod = load_mod(CENSUS / cell_dir / "build_new_fields_prompts.py")
            T, FIELDS = mod.T, mod.FIELDS  # EXACT same template + instructions, reused not retyped
            for field, instruction in FIELDS.items():
                for d in test:
                    if d not in items:
                        continue
                    f.write(json.dumps({
                        "channel": "field",
                        "aspect_id": f"{aid}__{field}",
                        "datapoint_id": d,
                        "prompt": T.format(instruction=instruction, text=items[d]),
                    }) + "\n")
                    n += 1
    print(f"wrote {n} TEST-only prompts -> {out}  "
          f"(cells: {[c for _, c in CELLS]}, fields: "
          f"{[list(load_mod(CENSUS / c / 'build_new_fields_prompts.py').FIELDS) for _, c in CELLS]})")

    if "--extract" in sys.argv:
        res = OUTDIR / "test_field_results.jsonl"
        cmd = [sys.executable, str(ROOT / "methods/metric_seam/battery/api_field_runner.py"),
               "--backend", "zai_anthropic", "--model", "glm-4.7",
               "--prompts", str(out), "--out", str(res),
               "--key-file", str(pathlib.Path.home() / ".z-ai-api-key.txt"),
               "--concurrency", "6"]
        print("running:", " ".join(cmd))
        r = subprocess.run(cmd)
        sys.exit(r.returncode)


if __name__ == "__main__":
    main()
