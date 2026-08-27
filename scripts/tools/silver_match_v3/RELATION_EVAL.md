# Equivalence-aware retrieval evaluation

This evaluator keeps three claims separate:

1. **Exact recall:** the retrieved metric ID equals the teacher metric ID.
2. **Strict-equivalence recall:** the retrieved ID belongs to an R3
   same-construct merge independently supported by the norm-embed 0/1/2 pair
   labels.
3. **Family recall:** the retrieved ID shares either an R3 merge candidate or
   an R3 grandparent with the teacher metric.

R3 grandparents are explicitly subsumption groupings of related but distinct
constructs. Family recall is therefore diagnostic and must never replace exact
recall or be called accuracy.

## Conservative equivalence rule

Every unordered metric pair in an R3 merged group must independently have:

- at least 3 unique raw-rubric pairs labeled `2` (same rule);
- at least 90% `2` among labels `0/1/2`;
- no label `0` (unrelated).

Repeated judgments of one raw-rubric pair do not increase support. If repeated
judgments disagree, the minimum score wins. A three-metric group with evidence
for only one edge is not closed transitively and receives no equivalence
credit.

## Build relations on sk3

```bash
cd /lfs/skampere3/0/alexspan/norm-research
/lfs/skampere3/0/alexspan/envs/ai_usage/bin/python -m \
  scripts.tools.silver_match_v3.build_relations \
  --hierarchy-root /lfs/skampere3/0/alexspan/norm-research/outputs/hierarchy \
  --pair-labels /lfs/skampere3/0/alexspan/norm_embed/all_verdicts.jsonl \
  --output /lfs/skampere3/0/alexspan/data/silver_match_v3_20260712/relations.json
```

The artifact records SHA-256 hashes for the R2 bank, R3 hierarchy, and pair
labels, plus accepted and rejected R3 merge audits.

## Evaluate retrieval

```bash
python -m scripts.tools.silver_match_v3.evaluate_retrieval \
  --teachers /path/to/teacher_sample.jsonl \
  --candidates /path/to/candidates/*.jsonl \
  --relations /path/to/relations.json \
  --ks 1 3 5 10 16 \
  --output /path/to/retrieval_eval.json \
  --errors-output /path/to/retrieval_errors.jsonl
```

Teacher and candidate bank hashes are required to match the frozen relation
artifact. Missing candidate rows remain misses in the recall denominator and
are exposed through `candidate_coverage`.

## Limitations

- The norm-embed labels were produced by the older Llama-70B v6 judge, not the
  present Sonnet teacher panel. They provide independent model-family support,
  but not human gold.
- The pair-label pool spans specificity buckets, while this matcher uses only
  the general R2 bank. Many pair rows therefore cannot map into this universe.
- Absence of pair evidence means “unverified,” not “different.” The strict rule
  intentionally leaves such R3 merges uncredited.
- Family sets overlap and can be broad. Their recall is a navigation diagnostic
  for retrieval, not proof that the retrieved leaf is interchangeable with the
  teacher leaf.
