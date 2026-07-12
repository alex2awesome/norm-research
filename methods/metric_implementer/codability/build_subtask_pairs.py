import pandas as pd, numpy as np, json, os, itertools
rng = np.random.RandomState(42)
df = pd.read_parquet("notebooks/_explore_cache/rubrics.parquet")
pr = df[df['task']=='peer-review'].copy()

# Build per-subtask context record
recs = {}
for st, sub in pr.groupby('subtask_short'):
    sub = sub.sort_values('rubric_idx')
    exrub = []
    for _, r in sub.head(3).iterrows():
        nm = str(r['rubric_name'])[:70]; ds = str(r['rubric_description'])[:110]
        exrub.append(f"{nm} — {ds}")
    recs[st] = {
        "subtask": str(st),
        "breadth": str(sub.iloc[0]['subtask_breadth']),
        "n_rubrics": int(len(sub)),
        "example_rubrics": exrub,
    }
subs = sorted(recs.keys())
print("n_subtasks:", len(subs))

# Embed labels for stratification (semantic spread). Fall back to TF-IDF if model unavailable.
texts = [recs[s]['subtask'] + " :: " + (recs[s]['example_rubrics'][0] if recs[s]['example_rubrics'] else "") for s in subs]
try:
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer('all-MiniLM-L6-v2')
    E = m.encode(texts, normalize_embeddings=True, batch_size=256, show_progress_bar=False)
    print("embed: MiniLM", E.shape)
except Exception as e:
    print("MiniLM failed (%s); TF-IDF fallback" % str(e)[:60])
    from sklearn.feature_extraction.text import TfidfVectorizer
    V = TfidfVectorizer(min_df=1, ngram_range=(1,2)).fit_transform(texts)
    from sklearn.preprocessing import normalize
    E = normalize(V).toarray().astype('float32')

idx = {s:i for i,s in enumerate(subs)}
def cos(a,b): return float(np.dot(E[idx[a]], E[idx[b]]))

# --- Sample pairs ---
pairs = {}  # frozenset -> stratum
def add(a,b,stratum):
    if a==b: return False
    k = frozenset((a,b))
    if k in pairs: return False
    pairs[k] = stratum; return True

# 300 uniform random
tries=0
while sum(1 for v in pairs.values() if v=='random')<300 and tries<20000:
    a,b = rng.choice(subs,2,replace=False); add(a,b,'random'); tries+=1

# Stratified by cosine: draw random candidate pairs, bucket, fill quotas
quota = {'high':280,'mid':210,'low':210}
got = {'high':0,'mid':0,'low':0}
tries=0
while sum(got.values())<sum(quota.values()) and tries<400000:
    a,b = rng.choice(subs,2,replace=False); tries+=1
    c = cos(a,b)
    s = 'high' if c>=0.55 else ('mid' if c>=0.35 else 'low')
    if got[s]>=quota[s]: continue
    if add(a,b,s): got[s]+=1
print("strata filled:", got, " random:", sum(1 for v in pairs.values() if v=='random'), " total:", len(pairs))

# Emit
os.makedirs("/tmp/subtask_pairs", exist_ok=True)
recs_out=[]
for i,(k,stratum) in enumerate(pairs.items()):
    a,b = sorted(k)
    recs_out.append({"pair_id":i,"stratum":stratum,"cos_label":round(cos(a,b),3),
                     "A":recs[a],"B":recs[b]})
rng.shuffle(recs_out)
for i,r in enumerate(recs_out): r['pair_id']=i  # renumber after shuffle
# batches of 50
B=50
nb=(len(recs_out)+B-1)//B
for bi in range(nb):
    chunk = recs_out[bi*B:(bi+1)*B]
    with open(f"/tmp/subtask_pairs/batch_{bi:02d}.jsonl","w") as f:
        for r in chunk: f.write(json.dumps(r)+"\n")
with open("/tmp/subtask_pairs/all_pairs.jsonl","w") as f:
    for r in recs_out: f.write(json.dumps(r)+"\n")
print("wrote", len(recs_out), "pairs in", nb, "batches to /tmp/subtask_pairs/")

# Spot-check: print 6 pairs across strata
print("\n=== SPOT CHECK ===")
for stratum in ['high','mid','low','random']:
    ex = [r for r in recs_out if r['stratum']==stratum][:2]
    for r in ex:
        print(f"\n[{stratum} cos={r['cos_label']}] pair {r['pair_id']}")
        print("  A: %s (%s, n=%d) | e.g. %s" % (r['A']['subtask'][:55], r['A']['breadth'], r['A']['n_rubrics'], r['A']['example_rubrics'][0][:80]))
        print("  B: %s (%s, n=%d) | e.g. %s" % (r['B']['subtask'][:55], r['B']['breadth'], r['B']['n_rubrics'], r['B']['example_rubrics'][0][:80]))
