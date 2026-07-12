#!/usr/bin/env python3
"""
Sonnet-based clustering with GEPA iteration, supervised by v6 pairwise verdicts.

Replaces GLM batch-clustering when z.ai is overloaded. Uses Max plan subagents for
Sonnet calls (user requirement: "ALWAYS use Max plan subagents for Claude work").

Strategy:
1. Load embeddings (cached OpenAI te3-small from glm_cluster.py)
2. Build kNN batches (same coverage/batch-size as GLM version)
3. For each batch, iteratively cluster via GEPA:
   - Initial grouping (Sonnet)
   - Score against v6 pairwise labels (score-2 = same, score-0 = different)
   - GEPA revise prompt based on mistakes
   - Iterate until convergence or max rounds
4. Reconcile via union-find with min_votes threshold
5. Compare to existing partition + v6 labels

No GLM dependency. Pure Sonnet + cached embeddings + v6 supervision.
"""

import os, sys, json, time, hashlib, tempfile
import numpy as np
from collections import defaultdict, Counter
from itertools import combinations

# ---- GEPA registry (persistent traces) ----
class Registry:
    def __init__(self, path):
        self.path = path
        os.makedirs(path, exist_ok=True)
        self.traces = {}  # criterion_id -> {initial, revisions: [...], final}

    def record(self, cid, initial=None, revision=None, final=None):
        if cid not in self.traces:
            self.traces[cid] = {"initial": None, "revisions": [], "final": None}
        if initial:
            self.traces[cid]["initial"] = initial
        if revision:
            self.traces[cid]["revisions"].append(revision)
        if final:
            self.traces[cid]["final"] = final
        # persist
        with open(os.path.join(self.path, f"{cid}.json"), "w") as f:
            json.dump(self.traces[cid], f, indent=2)


# ---- Load data (reuse glm_cluster helpers) ----
_CANON = os.environ.get("GLMCLUSTER_CANON", "outputs/analyses/canon_all_real_forms.jsonl")
_STRUCT = os.environ.get("GLMCLUSTER_STRUCT", "/lfs/skampere3/0/alexspan/norm_embed/match_out")
_VERDICTS = os.environ.get("GLMCLUSTER_VERDICTS", "/lfs/skampere3/0/alexspan/norm_embed/all_verdicts.jsonl")
_EMB_CACHE = os.environ.get("GLMCLUSTER_EMB_CACHE", "outputs/analyses/glm_cluster_cache")


def load_forms(task, n_forms=None, seed=0):
    keys, texts = [], []
    with open(_CANON) as f:
        for line in f:
            o = json.loads(line)
            if o.get("task") != task:
                continue
            if o.get("canonical"):
                keys.append(o["key"])
                texts.append(o["canonical"])
    if n_forms and n_forms < len(keys):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(keys), n_forms, replace=False)
        keys = [keys[i] for i in idx]
        texts = [texts[i] for i in idx]
    return keys, texts


def load_embeddings(task, embed="openai", model="text-embedding-3-small"):
    """Load cached honest-form embeddings from glm_cluster cache."""
    cache_path = os.path.join(_EMB_CACHE, f"emb_{task}__{embed}__{model}.npz")
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"No cached embeddings at {cache_path}. Run glm_cluster.py first to build cache.")
    d = np.load(cache_path, allow_pickle=True)
    return dict(d)  # keys, emb, texts_sha, embed, model, n


def load_v6_pairs(task):
    """Load v6 pairwise verdicts (score 0/2) as supervision."""
    pairs = defaultdict(lambda: None)  # (key_a, key_b) -> score
    with open(_VERDICTS) as f:
        for line in f:
            v = json.loads(line)
            if v.get("task") != task:
                continue
            if v.get("judge") != "v6":
                continue
            score = v.get("score")
            if score not in [0, 2]:  # only use confident same/diff
                continue
            a, b = sorted([v["key_a"], v["key_b"]])
            pairs[(a, b)] = score
    return dict(pairs)


def existing_clusters(keys, task):
    """Load existing v6-grounded partition."""
    ex_path = os.path.join(_STRUCT, f"clusters_{task}.json")
    if not os.path.exists(ex_path):
        return {k: i for i, k in enumerate(keys)}  # all singletons
    ex = json.load(open(ex_path))
    return {k: ex.get(k, f"singleton_{k}") for k in keys}


# ---- kNN batching (same as glm_cluster) ----
def build_knn(emb, k=30):
    """L2-normalize + cosine kNN."""
    from sklearn.neighbors import NearestNeighbors
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normed = emb / norms
    nn = NearestNeighbors(n_neighbors=min(k, len(emb)), metric="cosine").fit(normed)
    dist, idx = nn.kneighbors(normed)
    return idx  # [n, k] — self at index 0


def knn_batches(knn, batch_size=30, coverage=2):
    """Overlapping batches via anchor greedy coverage."""
    n = len(knn)
    covered_count = np.zeros(n, int)
    anchors = []
    while True:
        least = np.argmin(covered_count)
        if covered_count[least] >= coverage:
            break
        anchors.append(least)
        for nb in knn[least][:batch_size]:
            covered_count[nb] += 1
    batches = []
    covered_global = set()
    for a in anchors:
        members = [int(x) for x in knn[a][:batch_size]]
        batches.append(members)
        covered_global.update(members)
    for i in range(n):  # singletons for uncovered
        if i not in covered_global:
            batches.append([i])
    return batches


# ---- Sonnet clustering with GEPA ----
def sonnet_call(prompt, max_tokens=2000, temp=0.2):
    """Call Sonnet via Max plan subagent (user requirement: ALWAYS use Max subagents for Claude work)."""
    # Write prompt to temp file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(prompt)
        pf = f.name

    # Invoke subagent via skill (Max plan default for claude-api calls)
    # Since we can't call Agent tool directly, use a Max-model direct call via anthropic SDK
    # BUT user wants Max subagents — so we'll use the Anthropic SDK with max-tokens in the script
    # Actually, let me re-read: "ALWAYS use Max plan subagents for Claude work"
    # This means: spawn a Max-effort agent, not use the SDK directly
    # Let me use a hack: write a small wrapper that calls Agent tool

    # WAIT — I can't call Agent from within this script (it's a runtime tool, not Python API)
    # The user said "you can use sonnet locally with subagents" — meaning I (Claude Code) should
    # orchestrate this script to call subagents, not the script itself calling subagents.
    # So this script should return structured data, and I'll orchestrate the Agent calls.

    # Let me reframe: this script will be a LIBRARY, not a standalone runner.
    # I'll write functions that I (Claude Code) call, and I'll handle the Agent orchestration.

    # For now, placeholder that raises NotImplementedError — I'll handle calls in the orchestration
    raise NotImplementedError("sonnet_call must be orchestrated by Claude Code via Agent tool, not called directly")


def cluster_batch_gepa(items, task, level="L0", v6_pairs=None, max_rounds=3, registry=None, batch_id=None):
    """
    Cluster a batch of items via GEPA iteration.

    Returns:
    - groups: list of index-lists (0-based within batch)
    - trace: GEPA iteration history

    This function is ORCHESTRATED by Claude Code — it doesn't call Sonnet directly.
    Instead, it returns prompts for Claude Code to send to Max subagents.
    """
    # Build initial prompt
    listing = "\n".join(f"{i}. {t}" for i, t in enumerate(items))
    if level == "L0":
        rule = ('Group by the L0 rule: two statements go in the SAME group ONLY IF they are the SAME '
                'criterion restated in different words (identical evaluation, just rephrased). Statements '
                'that are merely RELATED but are DIFFERENT criteria, or are unrelated, MUST be in different '
                'groups. Do NOT over-merge — when uncertain, keep them separate.')
    else:
        rule = ('Group by the R1 rule: statements go in the SAME group if they express the SAME underlying '
                'principle/norm even if different in surface (same rule of evaluation). Different principles '
                'stay separate.')

    initial_prompt = (f"Below are {len(items)} evaluative rubric statements about {task}. {rule}\n\n{listing}\n\n"
                      'Reply with ONLY JSON: {"groups": [[indices 0-based], ...]}; every index in exactly one group.')

    # Return the orchestration plan
    return {
        "strategy": "gepa_iteration",
        "initial_prompt": initial_prompt,
        "items": items,
        "task": task,
        "level": level,
        "v6_pairs": v6_pairs,
        "max_rounds": max_rounds,
        "batch_id": batch_id,
    }


# ---- Reconcile (same as glm_cluster) ----
class UF:
    def __init__(self, n):
        self.p = list(range(n))
    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x
    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


def reconcile(batch_groupings, n, min_votes=1):
    """Union-find over co-grouping edges with min_votes threshold."""
    votes = defaultdict(int)
    for groups in batch_groupings:
        for g in groups:
            for a, b in combinations(sorted(g), 2):
                votes[(a, b)] += 1
    uf = UF(n)
    for (a, b), v in votes.items():
        if v >= min_votes:
            uf.union(a, b)
    cid = {}
    for i in range(n):
        cid[i] = uf.find(i)
    return cid


# ---- Compare (same metrics as glm_cluster) ----
def compare(glm_cid, keys, ex_map, task, v6_pairs, n_pairs=4000, seed=0):
    """Compare GLM partition to existing + v6 labels."""
    from sklearn.metrics import adjusted_rand_score

    # Rand vs existing
    glm_labels = [glm_cid[i] for i in range(len(keys))]
    ex_labels = [ex_map[k] for k in keys]
    rand_vs_existing = adjusted_rand_score(ex_labels, glm_labels)

    # Same-rate
    glm_same = sum(1 for i in range(len(keys)) for j in range(i+1, len(keys)) if glm_cid[i] == glm_cid[j])
    glm_same_rate = glm_same / (len(keys) * (len(keys) - 1) / 2) if len(keys) > 1 else 0
    ex_same = sum(1 for i in range(len(keys)) for j in range(i+1, len(keys)) if ex_labels[i] == ex_labels[j])
    ex_same_rate = ex_same / (len(keys) * (len(keys) - 1) / 2) if len(keys) > 1 else 0

    # v6 label respect
    v2_pairs = [(i, j) for i in range(len(keys)) for j in range(i+1, len(keys))
                if (keys[i], keys[j]) in v6_pairs and v6_pairs[(keys[i], keys[j])] == 2]
    v0_pairs = [(i, j) for i in range(len(keys)) for j in range(i+1, len(keys))
                if (keys[i], keys[j]) in v6_pairs and v6_pairs[(keys[i], keys[j])] == 0]

    v6_score2_kept_together = (sum(1 for i, j in v2_pairs if glm_cid[i] == glm_cid[j]) / len(v2_pairs)
                               if v2_pairs else None)
    v6_score0_kept_apart = (sum(1 for i, j in v0_pairs if glm_cid[i] != glm_cid[j]) / len(v0_pairs)
                            if v0_pairs else None)

    return {
        "rand_vs_existing": rand_vs_existing,
        "glm_same_rate": glm_same_rate,
        "ex_same_rate": ex_same_rate,
        "v6_score2_kept_together": v6_score2_kept_together,
        "v6_score0_kept_apart": v6_score0_kept_apart,
        "n_v2": len(v2_pairs),
        "n_v0": len(v0_pairs),
    }


# ---- Main orchestration plan ----
def build_plan(task, batch_size=30, coverage=2, min_votes=1, max_rounds=3, level="L0", registry_dir=None):
    """
    Build clustering plan for Claude Code to orchestrate.

    Returns a plan dict that Claude Code will execute via Agent calls.
    """
    # Load data
    keys, texts = load_forms(task)
    emb_cache = load_embeddings(task, embed="openai", model="text-embedding-3-small")
    emb = emb_cache["emb"]
    v6_pairs = load_v6_pairs(task)
    ex_map = existing_clusters(keys, task)

    # Build kNN batches
    knn = build_knn(emb, k=batch_size)
    batches = knn_batches(knn, batch_size=batch_size, coverage=coverage)

    # Build batch plans
    batch_plans = []
    for bid, batch_idx in enumerate(batches):
        batch_keys = [keys[i] for i in batch_idx]
        batch_texts = [texts[i] for i in batch_idx]
        # Extract v6 pairs for this batch (keyed by batch-local indices, not keys)
        batch_v6_local = {}
        for i in range(len(batch_keys)):
            for j in range(i+1, len(batch_keys)):
                global_pair = tuple(sorted([batch_keys[i], batch_keys[j]]))
                if global_pair in v6_pairs and v6_pairs[global_pair] in [0, 2]:
                    batch_v6_local[(i, j)] = v6_pairs[global_pair]  # LOCAL indices

        plan = cluster_batch_gepa(batch_texts, task, level=level, v6_pairs=batch_v6_local,
                                  max_rounds=max_rounds, batch_id=bid)
        plan["batch_idx"] = batch_idx  # global indices
        plan["batch_keys"] = batch_keys
        batch_plans.append(plan)

    return {
        "task": task,
        "n_forms": len(keys),
        "keys": keys,
        "texts": texts,
        "batches": batch_plans,
        "reconcile": {"min_votes": min_votes},
        "compare": {"ex_map": ex_map, "v6_pairs": v6_pairs},
        "registry_dir": registry_dir or f"outputs/analyses/sonnet_cluster_{task}.registry",
    }


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--batch", type=int, default=30)
    p.add_argument("--cov", type=int, default=2)
    p.add_argument("--min-votes", type=int, default=1)
    p.add_argument("--max-rounds", type=int, default=3)
    p.add_argument("--level", default="L0")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    plan = build_plan(args.task, batch_size=args.batch, coverage=args.cov,
                      min_votes=args.min_votes, max_rounds=args.max_rounds, level=args.level)

    print(f"Built plan: {len(plan['batches'])} batches for {plan['n_forms']} forms")
    print(f"Orchestration required: Claude Code must execute {len(plan['batches'])} GEPA iterations via Max subagents")
    print(f"Plan saved to {args.out}")

    with open(args.out, "w") as f:
        json.dump(plan, f, indent=2)
