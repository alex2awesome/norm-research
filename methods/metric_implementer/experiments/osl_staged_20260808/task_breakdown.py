import json
from collections import Counter, defaultdict
d = json.load(open("/lfs/skampere3/0/alexspan/outputs/osl_multi/family_verdict_join_v1.json"))
rows = d["full_rows"]
by_task = defaultdict(list)
for r in rows:
    by_task[r["task"]].append(r)
FAMS = ["llama", "qwen25", "qwen3"]
print("task | n | llama R/F/Fa | qwen25 R/F/Fa | qwen3 R/F/Fa | plateau | dialect")
for t, rs in by_task.items():
    n = len(rs)
    def rf(fam):
        c = Counter(r["verdict3"][fam] for r in rs)
        return f"{c.get('RISING',0)}/{c.get('FLAT',0)}/{c.get('FALLING',0)}"
    plateau = sum(1 for r in rs if all(r["verdict3"][f] in ("FLAT","FALLING") for f in FAMS))
    dialect = sum(1 for r in rs if sum(r["verdict3"][f]=="RISING" for f in FAMS)==1)
    print(f"{t:18s} {n:4d}  {rf('llama'):>10s}  {rf('qwen25'):>10s}  {rf('qwen3'):>10s}  {plateau:4d}  {dialect:4d}")
