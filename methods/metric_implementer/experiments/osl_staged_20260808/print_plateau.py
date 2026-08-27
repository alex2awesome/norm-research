import json
d = json.load(open("/lfs/skampere3/0/alexspan/outputs/osl_multi/family_verdict_join_v1.json"))
rows_by_key = {(r["task"], r["name"]): r for r in d["full_rows"]}
for r in d["plateau_set"]["top30"]:
    fr = rows_by_key[(r["task"], r["name"])]
    print("%-18s %+.4f  n_fr=%3d  %s" % (r["task"], r["combined_flatness"],
                                          fr["n_frontier_items"], r["name"][:70]))
print()
print("dialect owner_by_task:", d["dialect_set"]["owner_by_task"])
