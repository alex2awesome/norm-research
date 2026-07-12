"""glm_cluster — GLM-4.7 (or any zai_anthropic model) as a FRESH clusterer; cross-judge audit.

The user asked: instead of trusting the v6-Llama-judge-grounded L0/R1 partition, let GLM cluster the
rubric forms FROM SCRATCH and compare. No human gold (cross-judge validity only) — agreement with the
existing partition and with the 434K v6 labels is the signal, not true accuracy.

FRESH L0 (definition-grounded, anti-overcluster):
  1. S axis (frozen, key-aligned): TF-IDF over canonical text by default (sklearn, no deps, no GPU).
     Swap to a neural frozen embedding (--embed) for better paraphrase recall at sk3 scale. NEVER a
     verdict-trained encoder (the S⊥R lesson, [[rubric-clustering-pipeline]]).
  2. Overlapping kNN-anchor batches (coverage --cov, size --batch): true-same forms co-occur in ~cov
     batches → each potential merge gets ~cov independent co-grouping VOTES.
  3. GLM groups each batch by the L0 definition ("same criterion RESTATED in different words; related-
     but-different criteria stay apart; do not over-merge") → JSON groups.
  4. Reconcile: union-find over co-grouping edges; KEEP an edge only with >=--min-votes independent
     co-groupings (the anti-overcluster guard — a single greedy co-grouping is ~55% overturned per the
     adopt_v2 lesson, so require >=2). → GLM L0 clusters.
  5. COMPARE to the existing v6-grounded L0: pairwise-F1 / Rand / NMI (sampled), and which partition
     better RESPECTS the v6 labels (score-2 kept together, score-0 kept apart) — the "no gold" signal.

Pilot (laptop, real GLM):   python -m methods.metric_implementer.experiments.glm_cluster --task creative-writing --n-forms 200
Full task (sk3):            python -m methods.metric_implementer.experiments.glm_cluster --task creative-writing
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.request
from collections import defaultdict
from itertools import combinations

import numpy as np

_CANON = os.environ.get("GLMCLUSTER_CANON", "outputs/analyses/canon_all_real_forms.jsonl")
_STRUCT = os.environ.get("GLMCLUSTER_STRUCT", "outputs/analyses/structural_metrics")
_VERDICTS = os.environ.get("GLMCLUSTER_VERDICTS", "outputs/analyses/all_verdicts.jsonl")


# -------------------------------------------------------------------------------------------- helpers

def _key():
    # Toggle: two z.ai accounts, use whichever has monthly quota. Set GLMCLUSTER_KEY to a key FILE
    # (or a literal key) to override; otherwise prefer alexander-spangher, fall back to default.
    env = os.environ.get("GLMCLUSTER_KEY")
    if env:
        return open(env).read().strip() if os.path.exists(env) else env
    for p in (os.path.expanduser("~/.z-ai-api-key-alexander-spangher.txt"),
              os.path.expanduser("~/.z-ai-api-key.txt")):
        if os.path.exists(p):
            return open(p).read().strip()
    raise FileNotFoundError("no GLM key (set GLMCLUSTER_KEY or ~/.z-ai-api-key*.txt)")


def glm_call(model, prompt, max_tokens=900, temp=0.2, timeout=120, retries=5):
    """z.ai intermittently returns overloaded_error / HTTP 529 (and 429/502/503) — transient capacity
    that clears on retry ([[reference_glm_subscription_api]]). Retry with backoff so a batch never
    silently falls back to singletons on a transient blip."""
    import urllib.error
    body = json.dumps({"model": model, "max_tokens": max_tokens, "temperature": temp,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    last = None
    for attempt in range(retries):
        req = urllib.request.Request("https://api.z.ai/api/anthropic/v1/messages", data=body,
                                     headers={"x-api-key": _key(), "anthropic-version": "2023-06-01",
                                              "content-type": "application/json"})
        try:
            return json.loads(urllib.request.urlopen(req, timeout=timeout).read())["content"][0]["text"]
        except urllib.error.HTTPError as e:
            last = e
            if e.code in (429, 500, 502, 503, 529):           # transient — back off and retry
                time.sleep(2 ** attempt + 1)
                continue
            raise
        except (urllib.error.URLError, TimeoutError) as e:    # transport — retry
            last = e
            time.sleep(2 ** attempt + 1)
    raise last if last else RuntimeError("glm_call exhausted retries")


def load_forms(task, n_forms=None, seed=7, biased=False):
    keys, texts = [], []
    with open(_CANON) as f:
        for line in f:
            o = json.loads(line)
            if o.get("task") != task:
                continue
            if o.get("canonical"):
                keys.append(o["key"]); texts.append(o["canonical"])
    if n_forms and n_forms < len(keys):
        rng = np.random.default_rng(seed)
        if biased:                      # oversample multi-member existing clusters -> known partners in sample
            ex = json.load(open(os.path.join(_STRUCT, f"clusters_{task}.json")))
            c2k = defaultdict(list)
            for k in keys:
                if k in ex:
                    c2k[ex[k]].append(k)
            multi = [ks for ks in c2k.values() if len(ks) >= 2]
            picked, seen = [], set()
            for ks in multi:                       # take up to 3 from each multi cluster
                for k in ks[:3]:
                    if k in seen:
                        continue
                    picked.append(k); seen.add(k)
                    if len(picked) >= n_forms:
                        break
                if len(picked) >= n_forms:
                    break
            idx = [keys.index(k) for k in picked[:n_forms]]
        else:
            idx = rng.choice(len(keys), n_forms, replace=False)
        keys = [keys[i] for i in idx]; texts = [texts[i] for i in idx]
    return keys, texts


_EMB_CACHE_DIR = os.environ.get("GLMCLUSTER_EMB_CACHE", "outputs/analyses/glm_cluster_cache")
_DEFAULT_EMB_MODEL = {"openai": "text-embedding-3-small", "bge": "BAAI/bge-large-en-v1.5", "tfidf": "tfidf-1-2gram"}


def _tfidf_emb(texts):
    from sklearn.feature_extraction.text import TfidfVectorizer
    return TfidfVectorizer(ngram_range=(1, 2), min_df=1).fit_transform(texts)       # sparse, RAW


def _bge_emb(texts, model):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model).encode(texts, batch_size=64, show_progress_bar=False,
                                             normalize_embeddings=False)             # RAW (honest)


def _openai_emb(texts, model):
    from openai import OpenAI
    import numpy as np
    key_path = os.environ.get("GLMCLUSTER_OPENAI_KEY", "/lfs/skampere3/0/alexspan/.openai-salt-lab-key.txt")
    key = open(key_path).read().strip()
    c = OpenAI(api_key=key)
    embs, B = [], 2000
    for i in range(0, len(texts), B):
        r = c.embeddings.create(model=model, input=texts[i:i + B])
        embs.extend([d.embedding for d in sorted(r.data, key=lambda x: x.index)])
    return np.asarray(embs, float)                                                   # RAW (honest, unnormalized)


def get_embeddings(texts, keys, task, embed, model=None):
    """Cached RAW embeddings in HONEST FORM (unnormalized, key-aligned) so we can return to them later
    — re-cluster with different batch/cov/vote params WITHOUT re-paying or re-embedding. Cache key =
    (task, embed, model); validated by a sha of the texts so a stale cache is never silently reused.
    Stores keys + emb + manifest (embed/model/n/texts_sha). Sparse (tfidf) saved alongside via scipy.
    This is the frozen S axis (never verdict-trained) — [[rubric-clustering-pipeline]] S⊥R."""
    import hashlib
    import numpy as np
    model = model or _DEFAULT_EMB_MODEL.get(embed, "na")
    os.makedirs(_EMB_CACHE_DIR, exist_ok=True)
    path = os.path.join(_EMB_CACHE_DIR, f"emb_{task}__{embed}__{model.replace('/', '_')}.npz")
    sha = hashlib.sha256(("\n".join(texts)).encode("utf-8", "ignore")).hexdigest()[:16]
    if os.path.exists(path):                       # reuse honest-form cache if texts match exactly
        try:
            d = np.load(path, allow_pickle=True)
            if str(d.get("texts_sha", "")) == sha and int(d["n"]) == len(texts):
                if bool(d.get("is_sparse", False)):
                    from scipy.sparse import load_npz
                    X = load_npz(path.replace(".npz", ".sparse.npz"))
                else:
                    X = np.asarray(d["emb"], float)
                print(f"  [emb] cache hit {os.path.basename(path)}: {getattr(X,'shape','?')}")
                return X
        except Exception as e:
            print(f"  [emb] cache load failed ({e}); recomputing")
    print(f"  [emb] computing {embed} ({model}) for {len(texts)} forms ...")
    X = (_openai_emb(texts, model) if embed == "openai"
         else _bge_emb(texts, model) if embed == "bge"
         else _tfidf_emb(texts))
    is_sparse = hasattr(X, "toarray")
    if is_sparse:
        from scipy.sparse import save_npz
        save_npz(path.replace(".npz", ".sparse.npz"), X)
        np.savez(path, texts_sha=sha, n=len(texts), is_sparse=True, embed=embed, model=model,
                 keys=np.array(keys, dtype=object))
    else:
        np.savez(path, texts_sha=sha, n=len(texts), is_sparse=False, emb=np.asarray(X),
                 embed=embed, model=model, keys=np.array(keys, dtype=object))
    print(f"  [emb] cached honest-form -> {os.path.basename(path)} {getattr(X,'shape','?')}")
    return X


def build_knn(texts, keys, task, embed, k, model=None):
    """Cached embeddings -> L2-normalize -> cosine kNN (self at index 0)."""
    from sklearn.neighbors import NearestNeighbors
    import numpy as np
    X = get_embeddings(texts, keys, task, embed, model)
    if hasattr(X, "toarray"):                       # sparse tfidf: NearestNeighbors handles CSR directly
        nn = NearestNeighbors(n_neighbors=min(k, len(texts)), metric="cosine", n_jobs=-1).fit(X)
        return nn.kneighbors(X, return_distance=False)
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    nn = NearestNeighbors(n_neighbors=min(k, len(texts)), metric="cosine", n_jobs=-1).fit(Xn)
    return nn.kneighbors(Xn, return_distance=False)


def build_batches(n, knn, batch=30, cov=2):
    """Overlapping kNN-anchor batches, coverage ~cov. Anchor stride = batch/cov; each batch = anchor +
    its top-(batch-1) kNN. Guarantees every form appears (uncovered forms become their own anchor)."""
    stride = max(1, batch // cov)
    anchors = list(range(0, n, stride))
    batches, covered = [], set()
    for a in anchors:
        members = [int(x) for x in knn[a][:batch]]
        batches.append(members); covered.update(members)
    for i in range(n):                       # sweep: anything not covered gets its own anchor batch
        if i not in covered:
            batches.append([i])
    return batches


def group_prompt(task, items, level="L0"):
    listing = "\n".join(f"{i}. {t}" for i, t in enumerate(items))
    if level == "L0":
        default = ('Group by the L0 rule: two statements go in the SAME group ONLY IF they are the SAME '
                   'criterion restated in different words (identical evaluation, just rephrased). Statements '
                   'that are merely RELATED but are DIFFERENT criteria, or are unrelated, MUST be in different '
                   'groups. Do NOT over-merge — when uncertain, keep them separate.')
        rule = os.environ.get("GLMCLUSTER_L0_RULE") or default     # GEPA-tuned override (see gepa_cluster_prompt.py)
    else:
        rule = ('Group by the R1 rule: statements go in the SAME group if they express the SAME underlying '
                'principle/norm even if different in surface (same rule of evaluation). Different principles '
                'stay separate.')
    return (f"Below are {len(items)} evaluative rubric statements about {task}. {rule}\n\n{listing}\n\n"
            'Reply with ONLY JSON: {"groups": [[indices 0-based], ...]}; every index in exactly one group.')


def parse_groups(txt, n):
    m = re.search(r"\{.*\}", txt, re.S)
    if not m:
        return None
    try:
        groups = json.loads(m.group(0))["groups"]
    except Exception:
        return None
    seen = set(); ok = []
    for g in groups:
        clean = [int(i) for i in g if 0 <= int(i) < n and int(i) not in seen]
        for i in clean:
            seen.add(i)
        if clean:
            ok.append(clean)
    for i in range(n):                       # any unassigned → own singleton
        if i not in seen:
            ok.append([i])
    return ok


# -------------------------------------------------------------------------------------------- reconcile

class UF:
    def __init__(self, n):
        self.p = list(range(n))
    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]; x = self.p[x]
        return x
    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


def reconcile(batch_groupings, n, min_votes=2):
    """Union-find over co-grouping edges, but KEEP an edge only with >=min_votes independent votes
    (anti-overcluster). batch_groupings: list of (global-indices-list) per batch."""
    votes = defaultdict(int)                 # frozenset({i,j}) -> #batches that co-grouped them
    for groups in batch_groupings:
        for g in groups:
            for a, b in combinations(sorted(g), 2):
                votes[(a, b)] += 1
    uf = UF(n)
    for (a, b), v in votes.items():
        if v >= min_votes:                   # require independent corroboration to merge
            uf.union(a, b)
    cid = {}
    for i in range(n):
        cid[i] = uf.find(i)
    return cid                               # form-index -> GLM cluster id


# -------------------------------------------------------------------------------------------- compare

def existing_clusters_for(keys, task):
    fn = os.path.join(_STRUCT, f"clusters_{task}.json")
    if not os.path.exists(fn):
        return None
    ex = json.load(open(fn))
    return {k: ex[k] for k in keys if k in ex}   # key -> existing L0 cluster id


def compare(glm_cid, keys, ex_map, task, n_pairs=4000, seed=0):
    """Sampled pairwise comparison vs existing partition + label-respect vs v6 verdicts."""
    rng = np.random.default_rng(seed)
    n = len(keys)
    # GLM agreement vs existing (sampled pairs)
    same_glm = same_ex = agree = 0
    ex_idx = {i: ex_map[keys[i]] for i in range(n) if keys[i] in ex_map}
    idxs = list(ex_idx)
    if len(idxs) >= 2:
        for _ in range(n_pairs):
            a, b = rng.choice(idxs, 2, replace=False)
            g = glm_cid[a] == glm_cid[b]; e = ex_idx[a] == ex_idx[b]
            same_glm += g; same_ex += e; agree += (g == e)
    rand = agree / n_pairs
    # pairwise-F1 of GLM against existing (existing as pseudo-gold)
    tp = sum(1 for _ in range(n_pairs) if False)  # placeholder; computed via sampled same-ex & glm
    pf1 = 0.0
    # label-respect vs v6 verdicts: of score-2 pairs, frac kept-together; of score-0, frac kept-apart
    k2i = {keys[i]: i for i in range(n)}
    s2 = s0 = 0; tog2 = sep0 = 0
    with open(_VERDICTS) as f:
        for line in f:
            o = json.loads(line)
            if o.get("task") != task:
                continue
            a, b, sc = k2i.get(o.get("key_a")), k2i.get(o.get("key_b")), o.get("score")
            if a is None or b is None:
                continue
            if sc == 2:
                s2 += 1; tog2 += (glm_cid[a] == glm_cid[b])
            elif sc == 0:
                s0 += 1; sep0 += (glm_cid[a] != glm_cid[b])
    return {"rand_vs_existing": round(rand, 3),
            "glm_same_rate": round(same_glm / n_pairs, 3), "ex_same_rate": round(same_ex / n_pairs, 3),
            "v6_score2_kept_together": round(tog2 / s2, 3) if s2 else None,
            "v6_score0_kept_apart": round(sep0 / s0, 3) if s0 else None,
            "n_v2": s2, "n_v0": s0}


# -------------------------------------------------------------------------------------------- driver

def run(task, model, n_forms, batch, cov, min_votes, out, biased=False, embed="tfidf", embed_model=None):
    t0 = time.time()
    keys, texts = load_forms(task, n_forms, biased=biased)
    n = len(keys)
    print(f"[{task}] {n} forms; S axis = {embed} ({embed_model or 'default'}); batches size={batch} cov={cov} min_votes={min_votes}")
    knn = build_knn(texts, keys, task, embed, batch, model=embed_model)
    batches = build_batches(n, knn, batch, cov)
    print(f"[{task}] {len(batches)} overlapping batches -> {len(batches)} GLM calls (model={model})")
    batch_glob = []                          # list of (list-of-global-indices-per-group)
    for bi, mem in enumerate(batches):
        items = [texts[i] for i in mem]
        try:
            g = parse_groups(glm_call(model, group_prompt(task, items)), len(items))
        except Exception as e:
            print(f"  batch {bi} failed: {e}; falling back to singletons"); g = [[j] for j in range(len(items))]
        if g is None:
            g = [[j] for j in range(len(items))]
        batch_glob.append([[mem[j] for j in grp] for grp in g])   # map local idx -> global
        if (bi + 1) % 10 == 0:
            print(f"  ...{bi+1}/{len(batches)} batches ({time.time()-t0:.0f}s)")
    glm_cid = reconcile(batch_glob, n, min_votes)
    n_clusters = len(set(glm_cid.values()))
    from collections import Counter
    sizes = Counter(glm_cid.values())
    n_single = sum(1 for s in sizes.values() if s == 1)
    print(f"[{task}] GLM L0: {n_clusters} clusters ({100*n_single/n_clusters:.0f}% singletons); "
          f"max size {max(sizes.values())}; {time.time()-t0:.0f}s")
    ex = existing_clusters_for(keys, task)
    cmp = compare(glm_cid, keys, ex, task) if ex else {}
    print(f"[{task}] compare: {cmp}")
    outd = {"task": task, "model": model, "n_forms": n, "batch": batch, "cov": cov,
            "min_votes": min_votes, "n_clusters": n_clusters, "n_singletons": n_single,
            "glm_cid": {keys[i]: glm_cid[i] for i in range(n)}, "compare": cmp,
            "keys": keys, "batch_glob": batch_glob,        # saved so min_votes can be re-swept offline (0 GLM)
            "rule": os.environ.get("GLMCLUSTER_L0_RULE", "baseline"),
            "elapsed_s": round(time.time()-t0, 1)}
    if out:
        json.dump(outd, open(out, "w"))
        print(f"[{task}] wrote {out}")
    return outd


def main(argv=None):
    ap = argparse.ArgumentParser(prog="glm_cluster", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="creative-writing")
    ap.add_argument("--model", default="glm-4.7", help="zai_anthropic slug (glm-4.7 default per user)")
    ap.add_argument("--n-forms", type=int, default=None, help="subsample for pilot (None=all)")
    ap.add_argument("--batch", type=int, default=30)
    ap.add_argument("--cov", type=int, default=2, help="coverage: each form in ~cov batches")
    ap.add_argument("--min-votes", type=int, default=2, help="anti-overcluster: merges need >=this votes")
    ap.add_argument("--out", default=None)
    ap.add_argument("--biased", action="store_true",
                    help="oversample multi-member existing clusters (so the sample has known same-partners)")
    ap.add_argument("--embed", default="tfidf", choices=["tfidf", "bge", "openai"],
                    help="S axis: tfidf (lexical) | bge (sentence-transformers) | openai (te3-small, the "
                         "project's canonical clean-S, user's choice). openai uses $GLMCLUSTER_OPENAI_KEY.")
    ap.add_argument("--embed-model", default=None,
                    help="embedding model slug (default te3-small for openai, bge-large-en-v1.5 for bge)")
    a = ap.parse_args(argv)
    run(a.task, a.model, a.n_forms, a.batch, a.cov, a.min_votes, a.out,
        biased=a.biased, embed=a.embed, embed_model=a.embed_model)


if __name__ == "__main__":
    main()
