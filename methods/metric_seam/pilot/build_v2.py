"""Wave-2 build: 20 NEW aspects x same 250 canonical items x 2 judge passes.
Reuses v1 canonical items and templates; scope channel reused from v1 (same items).
"""
import importlib.util, json, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[3]
V2DIR = ROOT / "runs/validity_full/v2/press_releases"
V1 = ROOT / "outputs/metric_seam_pilot/v1"
OUT = ROOT / "outputs/metric_seam_pilot/v2"
OUT.mkdir(parents=True, exist_ok=True)

spec = importlib.util.spec_from_file_location(
    "build_v1", pathlib.Path(__file__).parent / "build_v1.py")
b1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b1)
T1, T2 = b1.T1, b1.T2

# 20 new aspects: ~7 structural (expected V-ish), ~7 boundary, ~6 soft (expected A)
WAVE2 = ["a75", "a76", "a104", "a111", "a112", "a81", "a41",          # structural
         "a66", "a97", "a115", "a2", "a28", "a31", "a103", "a119",     # boundary
         "a64", "a65", "a25", "a87", "a42"]                            # soft


def main():
    aspects = {x["aspect_id"]: x for x in json.load(open(V2DIR / "aspects.json"))}
    items = json.load(open(V1 / "items_v1.json"))
    missing = [a for a in WAVE2 if a not in aspects]
    assert not missing, f"unknown aspects: {missing}"
    n = 0
    with open(OUT / "prompts_v2.jsonl", "w") as f:
        for aid in WAVE2:
            a = aspects[aid]
            for it in items:
                for ch, T in [("pass1", T1), ("pass2", T2)]:
                    f.write(json.dumps({
                        "channel": ch, "aspect_id": aid,
                        "datapoint_id": it["datapoint_id"],
                        "prompt": T.format(name=a["name"],
                                           description=a["description"],
                                           text=it["ctext"])}) + "\n")
                    n += 1
    json.dump(WAVE2, open(OUT / "wave2_aspects.json", "w"))
    print(f"wrote {n} prompts for {len(WAVE2)} aspects -> {OUT}")


if __name__ == "__main__":
    main()
