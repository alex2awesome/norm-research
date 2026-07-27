"""TASK 11: pool-to-test content-overlap audit (CPU-free).
Decisive comparison: if NATIVE and FOREIGN pools overlap the hover TEST text similarly, then
train->test content leakage cannot explain the native-foreign gap (+.097), because both pools
would leak equally. Also count proper-entity spans: vocabulary priming from entity leakage
requires entities to exist in the units at all."""
import json, re, sys
import paperexact_arms as px
bench, program, metric, _ = px.load_bench("hover")
test_text = " ".join(str(getattr(ex, "claim", "")) for ex in bench.test_set).lower()
STOP = set("the a an of to and or in on for with that this those these is are was were be been from by as at it its".split())
def cw(t):
    return [w for w in re.findall(r"[a-z]+", t.lower()) if len(w) > 3 and w not in STOP]
def audit(units, name):
    n_big = n_caps = 0
    fracs = []
    for u in units:
        c = cw(u)
        bigs = [a + " " + b for a, b in zip(c, c[1:])]
        if any(b in test_text for b in bigs):
            n_big += 1
        fracs.append(sum(1 for w in set(c) if w in test_text) / max(1, len(set(c))))
        if re.search(r"[A-Z][a-z]+ [A-Z][a-z]+", u):
            n_caps += 1
    print("  %-8s n=%3d  units with a test-matching BIGRAM: %3d (%.0f%%)   mean single-word overlap: %.2f   proper-entity spans: %d"
          % (name, len(units), n_big, 100.0 * n_big / len(units), sum(fracs) / len(fracs), n_caps))
nat = [x["unit"] for x in json.load(open("pools/hover_Qwen3-8B_frozen.json"))["units"]]
frn = [x["unit"] for x in json.load(open("pools/hotpot_Qwen3-8B_frozen.json"))["units"]]
audit(nat, "native"); audit(frn, "foreign")
