"""STAGE 2 (code_review): scope results_dedup.jsonl to the new (E2 ladder) run only,
join on items.json dpids (crb-prefixed) x the 18 aspects_candidates.json aspects (+ scope
channel). Writes results_newrun.jsonl (never touches results_dedup.jsonl / sk3 originals).
"""
import json, pathlib, collections

ROOT = pathlib.Path(__file__).resolve().parents[3]
TASK = ROOT / "outputs/metric_seam_pilot/tasks/code_review"

items = json.load(open(TASK / "items.json"))
item_dpids = set(x["datapoint_id"] for x in items)
aspects18 = json.load(open(TASK / "aspects_used.json"))
aspects18_set = set(aspects18)

assert len(item_dpids) == 250, len(item_dpids)
assert len(aspects18_set) == 18, len(aspects18_set)

n_in, n_out = 0, 0
kept_rows = []
skipped_reasons = collections.Counter()
with open(TASK / "results_dedup.jsonl") as f:
    for line in f:
        n_in += 1
        r = json.loads(line)
        dpid = r["datapoint_id"]
        aid = r["aspect_id"]
        if dpid not in item_dpids:
            skipped_reasons["dpid_not_in_new_items"] += 1
            continue
        if aid != "scope" and aid not in aspects18_set:
            skipped_reasons["aspect_not_in_18"] += 1
            continue
        kept_rows.append(r)
        n_out += 1

with open(TASK / "results_newrun.jsonl", "w") as f:
    for r in kept_rows:
        f.write(json.dumps(r) + "\n")

print(f"read {n_in} rows from results_dedup.jsonl")
print(f"kept {n_out} rows -> results_newrun.jsonl")
print("skip reasons:", dict(skipped_reasons))

# --- reconciliation: expected 18*250*2 = 9000 judge rows + 250 scope rows = 9250 ---
by_channel = collections.Counter(r["channel"] for r in kept_rows)
print("by channel:", dict(by_channel))

by_cell = collections.Counter((r["aspect_id"], r["channel"]) for r in kept_rows if r["aspect_id"] != "scope")
expected_cells = {(aid, ch) for aid in aspects18 for ch in ("pass1", "pass2")}
missing_cells = []
for aid in aspects18:
    for ch in ("pass1", "pass2"):
        n = by_cell.get((aid, ch), 0)
        if n != 250:
            missing_cells.append({"aspect_id": aid, "channel": ch, "n": n, "expected": 250})

print(f"n aspect x channel cells with != 250 rows: {len(missing_cells)}")
for m in missing_cells:
    print(" ", m)

# duplicate check within kept rows
dup_counter = collections.Counter((r["aspect_id"], r["channel"], r["datapoint_id"]) for r in kept_rows)
dups = {k: v for k, v in dup_counter.items() if v > 1}
print(f"n duplicate (aspect,channel,dpid) keys: {len(dups)}")
if dups:
    print(" sample dups:", list(dups.items())[:5])

# scope channel reconciliation
scope_rows = [r for r in kept_rows if r["channel"] == "scope"]
scope_dpids = set(r["datapoint_id"] for r in scope_rows)
print(f"scope rows: {len(scope_rows)}, distinct dpids: {len(scope_dpids)}, "
      f"== item_dpids: {scope_dpids == item_dpids}")
