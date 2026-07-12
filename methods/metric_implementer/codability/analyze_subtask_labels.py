import json, numpy as np
from collections import Counter, defaultdict

# 1. load workflow result (robust)
raw = open("/private/tmp/claude-502/-Users-spangher-Projects-stanford-research-norm-research/a5b7c93c-c623-4a16-b97b-2f7cf3d3de15/tasks/wnmoah253.output").read()
obj = json.loads(raw)['result']
labels = {L['pair_id']: L for L in obj['labels']}
print("labels loaded:", len(labels))

# 2. load pairs
pairs = {}
for line in open("/tmp/subtask_pairs/all_pairs.jsonl"):
    r = json.loads(line); pairs[r['pair_id']] = r
print("pairs loaded:", len(pairs))

# 3. join -> labeled dataset
rows = []
for pid, p in pairs.items():
    L = labels.get(pid)
    if not L: continue
    rows.append({**p, "label": L['label'], "confidence": L['confidence'], "reason": L['reason']})
with open("/tmp/subtask_pairs/labeled.jsonl","w") as f:
    for r in rows: f.write(json.dumps(r)+"\n")
print("joined:", len(rows), "-> /tmp/subtask_pairs/labeled.jsonl\n")

score = {'SAME':2,'RELATED':1,'DIFFERENT':0}
lab = np.array([score[r['label']] for r in rows])
cos = np.array([r['cos_label'] for r in rows])

# 4. overall + by stratum
print("=== OVERALL ===")
c = Counter(r['label'] for r in rows)
for k in ['SAME','RELATED','DIFFERENT']: print("  %-9s %4d (%.1f%%)"%(k,c[k],100*c[k]/len(rows)))
print("  corr(cos_S, graded_label) = %.3f"%np.corrcoef(cos,lab)[0,1])

print("\n=== BY SURFACE STRATUM (rows: stratum | n | %SAME %REL %DIFF | mean cos) ===")
for st in ['high','mid','low','random']:
    sr = [r for r in rows if r['stratum']==st]
    cc = Counter(r['label'] for r in sr); n=len(sr)
    print("  %-7s n=%3d | SAME %4.1f  REL %4.1f  DIFF %4.1f | cos %.3f"%(
        st,n,100*cc['SAME']/n,100*cc['RELATED']/n,100*cc['DIFFERENT']/n,
        np.mean([r['cos_label'] for r in sr])))

# 5. KEY DIAGNOSTIC: low-surface SAME pairs (truly-different encoding, same aim)
print("\n=== TRULY-DIFFERENT-ENCODING: SAME label with cos_S < 0.35 ===")
lowsame = sorted([r for r in rows if r['label']=='SAME' and r['cos_label']<0.35], key=lambda r:r['cos_label'])
print("count: %d  (of %d SAME total)"%(len(lowsame), c['SAME']))
for r in lowsame[:12]:
    print("  cos=%.2f [%s,%s] %s  <=>  %s"%(r['cos_label'],r['confidence'][:2],r['stratum'][:1].upper(),
        r['A']['subtask'][:42], r['B']['subtask'][:42]))
    print("       reason: %s"%r['reason'][:140])

# 6. P(SAME) and P(SAME or RELATED) vs surface bins -- codability curve at subtask level
print("\n=== CODABILITY CURVE (subtask level): P(label | cos_S bin) ===")
print("  bin        n   P(SAME)  P(SAME|REL)")
edges=[0,0.2,0.3,0.4,0.5,0.6,0.7,1.01]
for a,b in zip(edges,edges[1:]):
    m=(cos>=a)&(cos<b); n=int(m.sum())
    if n==0: continue
    ps=np.mean(lab[m]==2); psr=np.mean(lab[m]>=1)
    print("  [%.1f,%.1f) %4d   %.3f    %.3f"%(a,b,n,ps,psr))

# 7. confidence
print("\n=== CONFIDENCE ===")
for k in ['high','medium','low']:
    cc=Counter(r['label'] for r in rows if r['confidence']==k); n=sum(cc.values())
    print("  %-7s n=%3d | SAME %d REL %d DIFF %d"%(k,n,cc['SAME'],cc['RELATED'],cc['DIFFERENT']))
