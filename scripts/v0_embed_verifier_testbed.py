"""V0 embedding verifier on the true-cites testbed (task #52).

Question: can retrieval-embedding similarity verify anticipation at the CLAIM
level? score(claim) = max cosine over all paragraphs of the attached art docs.
Eval: separate fell_102 claims from standing claims.

Key contrast (Alex's hypothesis): within an app the topic is constant, so a
topic-confounded model (v3, claim1<->claim1 trained) should show weak WITHIN-
RECORD discrimination; the paragraph-trained v6a should do better.

Slices:
  - pooled AUC (all claims)
  - within-record AUC (mean over records that have both classes)
  - independent-claims-only (dependent claims reference 'claim N' and often
    stand for structural reasons -> a depth giveaway we must not credit)
"""
import gzip, glob, json, re
import numpy as np

BASE = "/lfs/skampere3/0/alexspan/norm-research"
PROC = f"{BASE}/datasets/patents/processed"
TESTBED = f"{PROC}/truecite_testbed_v1/testbed.jsonl.gz"
MODELS = {
    "v3_claim_trained": f"{BASE}/models/bge-m3-anticipation-v3",
    "v6a_para_trained": f"{BASE}/models/bge-m3-anticipation-v6a",
}
MAX_PARAS_PER_DOC = 200

norm = lambda x: "".join(c for c in str(x) if c.isdigit()).lstrip("0")
log = lambda m: print(m, flush=True)
DEP_RE = re.compile(r"\b(?:according to|of|in)\s+claim\s+\d|\bclaim\s+\d+\s*,?\s*wherein", re.I)

# ---------- load testbed ----------
log("Loading testbed ...")
records = []
need_docs = set()
with gzip.open(TESTBED, "rt") as f:
    for line in f:
        r = json.loads(line)
        docs = [d["doc_id"] for d in r["art"] if d["in_gp_corpus"]]
        if not docs:
            continue
        fell = [c for c in r["claims"] if c["fell_102"]]
        stand = [c for c in r["claims"] if not c["fell_102"]]
        if not fell or not stand:
            continue
        records.append({"app_id": r["app_id"], "ifw": r["ifw_number"],
                        "claims": r["claims"], "docs": docs})
        need_docs.update(docs)
log(f"  {len(records):,} records with GP art + both classes; {len(need_docs):,} docs")

# ---------- load GP paragraphs for needed docs ----------
log("Loading GP paragraphs ...")
doc_paras = {}
def eat_gp(path, op):
    with op(path) as f:
        for line in f:
            try: rr = json.loads(line)
            except Exception: continue
            dn = norm(rr.get("pgpub_id"))
            if dn in need_docs and rr.get("paragraphs") and dn not in doc_paras:
                ps = list(rr["paragraphs"].values())[:MAX_PARAS_PER_DOC]
                doc_paras[dn] = [p for p in ps if p and len(p) > 40]
eat_gp(f"{PROC}/paragraph_keyed_specs.jsonl.gz", lambda p: gzip.open(p, "rt"))
for fn in sorted(glob.glob(f"{PROC}/paragraph_keyed_specs_v2/*.jsonl")):
    eat_gp(fn, open)
n_paras = sum(len(v) for v in doc_paras.values())
log(f"  {len(doc_paras):,} docs, {n_paras:,} paragraphs")

# ---------- unique texts to encode ----------
claim_texts, claim_key = [], {}
for r in records:
    for c in r["claims"]:
        k = (r["app_id"], c["num"])
        if k not in claim_key:
            claim_key[k] = len(claim_texts)
            claim_texts.append(c["text"][:2000])
para_index = {}   # doc -> (start, end) in para_texts
para_texts = []
for d, ps in doc_paras.items():
    para_index[d] = (len(para_texts), len(para_texts) + len(ps))
    para_texts.extend(p[:2000] for p in ps)
log(f"encoding {len(claim_texts):,} claims + {len(para_texts):,} paragraphs per model")

from sentence_transformers import SentenceTransformer
from sklearn.metrics import roc_auc_score

results = {}
for name, path in MODELS.items():
    log(f"--- {name} ---")
    model = SentenceTransformer(path, device="cuda")
    model.max_seq_length = 512
    C = model.encode(claim_texts, batch_size=256, normalize_embeddings=True,
                     show_progress_bar=False).astype(np.float32)
    P = model.encode(para_texts, batch_size=256, normalize_embeddings=True,
                     show_progress_bar=False).astype(np.float32)
    ys, ss, dep, recs_auc = [], [], [], []
    for r in records:
        rows = [i for d in r["docs"] if d in para_index
                for i in range(*para_index[d])]
        if not rows:
            continue
        Pm = P[rows]
        y_r, s_r = [], []
        for c in r["claims"]:
            ci = claim_key[(r["app_id"], c["num"])]
            score = float((C[ci] @ Pm.T).max())
            ys.append(int(c["fell_102"])); ss.append(score)
            dep.append(bool(DEP_RE.search(c["text"][:300])))
            y_r.append(int(c["fell_102"])); s_r.append(score)
        if len(set(y_r)) == 2:
            recs_auc.append(roc_auc_score(y_r, s_r))
    ys, ss, dep = np.array(ys), np.array(ss), np.array(dep)
    pooled = roc_auc_score(ys, ss)
    within = float(np.mean(recs_auc))
    indep = roc_auc_score(ys[~dep], ss[~dep]) if len(set(ys[~dep])) == 2 else float("nan")
    log(f"  pooled AUC={pooled:.4f}  within-record AUC={within:.4f} "
        f"(n_rec={len(recs_auc)})  independent-only AUC={indep:.4f} "
        f"(n_indep={int((~dep).sum()):,}/{len(ys):,})")
    results[name] = {"pooled": pooled, "within": within, "indep": indep}
    del model, C, P

# depth giveaway baseline: predict standing if dependent
dep_auc = roc_auc_score(ys, ~dep)
log(f"depth-only baseline (independent => fell): AUC={dep_auc:.4f}")
json.dump(results, open(f"{PROC}/truecite_testbed_v1/v0_embed_results.json", "w"), indent=2)
log("DONE")
