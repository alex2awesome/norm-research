"""
V1 of the pairwise rubric de-duplication via gpt-5-mini.

Goal: given two creative-writing rubrics, decide if they express the SAME
underlying evaluation criterion (and could be merged) or are genuinely
distinct criteria.

V1 strategy:
- Load all creative-writing keep rubrics from outputs/classifier_chunks_FULL
- Pick a small fixed set of test pairs spanning the expected verdict space:
  obvious dup, near-dup, related-but-distinct, clearly different
- Call gpt-5-mini with the v1 prompt
- Print results for human inspection

Reads OPENAI_API_KEY from the SALT-lab shared location on sk3.
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path
import numpy as np
import pandas as pd
from openai import OpenAI

ROOT = Path("/lfs/skampere3/0/alexspan/norm-research")
CHUNKS = ROOT / "outputs/classifier_chunks_FULL"
KEY_PATH = Path("/lfs/skampere3/0/alexspan/.openai-salt-lab-key.txt")

MODEL = "gpt-5-mini"

# Rubric names used as few-shot examples (across all specificity buckets) —
# excluded from test pairs to ensure we're testing generalization, not
# memorization. Keys are lowercase, stripped.
FEW_SHOT_NAMES = {
    # general bucket
    "use of profanity, gore, and sexuality",
    "use of profanity / gore / sexuality",
    "content boundaries — sex, violence, profanity",
    "manner of imitation (dramatic vs narrative)",
    "manner of imitation: drama vs narrative vs mixed forms",
    "plot coherence",
    "tonal control",
}

# ---- Prompt v5 ----
# Per-specificity few-shots. Each (task, specificity) bucket gets its own
# few-shot examples drawn from real corpus pairs in that bucket. Construct-
# identity remains the decision criterion. Reasoning-first via strict
# json_schema (property order preserved).
SYSTEM_PROMPT_BASE = """You are evaluating two rubric items from a creative-writing evaluation system to determine their relationship for the purpose of de-duplicating the rubric set.

The dedup target is CONSTRUCT IDENTITY — whether the two rubrics are attempting to measure the same underlying property of the work. This is distinct from SCORE AGREEMENT (whether they'd produce similar scores on the same texts). Two rubrics can correlate strongly on real data (a story scoring high on one tends to score high on the other) and still measure DIFFERENT constructs — collapsing them on correlation alone destroys information. Only collapse rubrics when they share the same underlying construct.

Use this 4-way verdict scheme:

- "duplicate": The two rubrics measure the IDENTICAL construct with substantively the same operational definition. One could be removed and the other kept with no loss of evaluative information. The wording may differ but the property being measured is the same and is described in essentially the same way.

- "paraphrase": The two rubrics measure the SAME underlying construct but with meaningfully different wording, level of detail, or surface framing. A reviewer would describe both as targeting the same property of the work. One could replace the other without losing distinct evaluative information; they could safely be merged into one canonical rubric.

- "related": The two rubrics may correlate (a piece doing well on one tends to do well on the other) and may share a topic or dimension, but they measure DIFFERENT underlying constructs. Each captures information the other does not. Removing one would lose evaluative signal — they should be kept as distinct items even though they may cluster on a shared axis.

- "different": The two rubrics measure unrelated constructs with no meaningful conceptual connection.

Important: Do NOT use score-agreement as your test. The questions are:
  - What construct does each rubric purport to measure (what property of the work)?
  - Could you describe both as measuring a single, coherent property — or do they require separate descriptions?
  - If you removed one, would you lose evaluative information distinct from the other, or would you only lose redundant signal?

Your output must be a JSON object with two fields: reasoning (your step-by-step analysis) followed by verdict (your committed label). Reason before deciding — do NOT pick the verdict first. Work through: (1) name the construct of each rubric (what property of the work is being captured), (2) is there a single coherent description that captures both, or do they require separate descriptions, (3) what distinct evaluative information does each provide that the other does not, (4) is any correlation between them merely empirical (covariance on real data) or constitutive (they describe the same property).

Each example below is drawn from the actual creative-writing rubric corpus at the same specificity level as the pair you are evaluating. Surface wording in some examples has been lightly normalized for clarity, but the underlying constructs are real.
"""


# Per-(task, specificity) few-shots. Each entry is a list of 4 dicts
# (one per verdict). The pair content is lightly normalized from real
# corpus rubrics in the corresponding bucket.

FEW_SHOTS_BY_SPECIFICITY: dict[str, list[dict]] = {
    "general": [
        {
            "verdict": "duplicate",
            "a": {
                "name": "Use of profanity, gore, and sexuality",
                "description": "There are no absolute barriers to profanity, gore, or sexuality; however, high amounts are generally used poorly. If you include heavy adult content, ensure it is used skillfully and serves the story rather than relying on shock value.",
            },
            "b": {
                "name": "Use of Profanity / Gore / Sexuality",
                "description": "No absolute barriers exist on profanity, gore, or sexual content. High quantities are typically handled poorly. Such content should be deployed skillfully and in service of the narrative.",
            },
            "reasoning": "Construct of A: whether a magazine's submission tolerates adult content (profanity, gore, sexuality) given the writer's skill — no hard ceiling but heavy quantities are usually badly handled. Construct of B: identical — same tolerance threshold with the same skill-modifier on heavy amounts. A single coherent description captures both: 'this magazine accepts any quantity of adult content provided it is handled skillfully and serves the story.' Neither rubric carries evaluative information the other lacks. The two rubrics are the same magazine's content policy paraphrased across two of its own pages. Safely mergeable.",
        },
        {
            "verdict": "paraphrase",
            "a": {
                "name": "Manner of imitation (dramatic vs narrative)",
                "description": "Poetry may be dramatic (drama), narrative (epic or lyrical narrative), or a mixture; manner of imitation affects form and audience engagement. Choose dramatic presentation when action is to be enacted; choose narrative when length and breadth of action favor exposition.",
            },
            "b": {
                "name": "Manner of imitation: drama vs narrative vs mixed forms",
                "description": "Poetry may take the form of dramatic presentation, pure narrative (including lyric), or pure drama; the manner of imitation distinguishes subgenres and affects technique. Be conscious whether you are showing action (dramatic) or telling it (narrative) and use the conventions appropriate to each.",
            },
            "reasoning": "Construct of A: the property of a poetic work concerning whether its manner of imitation is dramatic, narrative, or mixed — a generic-form classification with implications for technique. Construct of B: identical — the same Aristotelian distinction among dramatic, narrative, and mixed manner with the same technique consequences. A single coherent description captures both: 'is the manner of imitation dramatic, narrative, or mixed, and is technique aligned with that manner.' The two rubrics are two different Gutenberg parses of the same Poetics chapter; the wording diverges substantively but neither carries information the other doesn't. Safely mergeable.",
        },
        {
            "verdict": "related",
            "a": {
                "name": "Use of profanity, gore, and sexuality",
                "description": "There are no absolute barriers to profanity, gore, or sexuality; however, high amounts are generally used poorly. Include heavy adult content only if handled skillfully.",
            },
            "b": {
                "name": "Content boundaries — sex, violence, profanity",
                "description": "Profanity is fine. Sex or violence in a story should be artistically justified; no excessive gore.",
            },
            "reasoning": "Construct of A: whether the quantity of adult content is calibrated to the writer's skill — a quantity-vs-skill judgment that admits heavy adult content if handled well. Construct of B: whether each instance of adult content is artistically justified by the story's purpose — a sufficiency-of-purpose judgment that admits adult content only when it serves a clear artistic end. These are not the same construct: A asks 'is the dosage right for the writer's skill?' while B asks 'is each instance purpose-justified?' A story can pass A while failing B (heavy gore well-handled but without explicit narrative purpose) or vice versa (sparing but unmotivated adult content). On real submissions the two scores will covary, but that correlation is empirical, not constitutive — each rubric captures distinct evaluative information. They cluster on the adult-content-policy axis but should remain separate items.",
        },
        {
            "verdict": "different",
            "a": {
                "name": "Plot coherence",
                "description": "The audience perceives that the main events follow logically from one another. Each event must be motivated by what came before; nothing should appear arbitrary. Evaluate whether the causal chain is intact.",
            },
            "b": {
                "name": "Tonal control",
                "description": "Humor should not tip into smugness or meanness; the comedic voice must remain warm enough to avoid alienating readers.",
            },
            "reasoning": "Construct of A: the logical-causal integrity of the plot — whether events follow from preceding events through motivated causation. Construct of B: the tonal calibration of humor — whether comedic voice avoids smugness or meanness. These describe entirely separate properties of a work (narrative logic versus tonal modulation in humor) with no shared dimension. No coherent single description captures both. They are unrelated constructs.",
        },
    ],
    "specific": [
        {
            "verdict": "duplicate",
            "a": {
                "name": "Peripeteia (Reversal)",
                "description": "Reversal of fortune within the plot. Peripeteia is a structural device where the action turns in an unexpected but necessary way; when combined with recognition, it yields the most powerful tragic effect.",
            },
            "b": {
                "name": "Peripeteia (reversal) as a structural device",
                "description": "Peripeteia — a sudden reversal of the action — is a key moment that changes the hero's situation toward good or ill. Place reversals so they arise from the plot's internal logic; the reversal should be both surprising and necessary.",
            },
            "reasoning": "Construct of A: the structural device whereby the plot's action turns to its opposite under probability or necessity — the Aristotelian peripeteia. Construct of B: identical — the same reversal mechanism with the same constraints (must arise from internal logic, must be surprising and necessary). A single coherent description captures both: 'does the work include an Aristotelian reversal that follows from the plot's causal chain.' The wording diverges (A frames it as a fortune-reversal structural device; B foregrounds the reversal moment and placement guidance), but neither carries evaluative information the other lacks. These are two different parses of the same Aristotelian concept. Safely mergeable.",
        },
        {
            "verdict": "paraphrase",
            "a": {
                "name": "East Asian 4-act (Kishōtenketsu)",
                "description": "Originally Chinese qǐ chéng zhuǎn hé; in Japanese kishōtenketsu: four-part structure (introduction, development, twist/turn, conclusion). Emphasizes development and turn without necessarily centering on conflict.",
            },
            "b": {
                "name": "Four-part structure (ki/shō/ten/ketsu)",
                "description": "Kishōtenketsu describes the four-part structure of many classic Chinese, Korean, Japanese and Vietnamese narratives: introduction, development, twist or reversal, and resolution. Stories built with these four acts deliberately place a turning point in the third part.",
            },
            "reasoning": "Construct of A: whether a work uses the kishōtenketsu four-act structure — introduction, development, twist, conclusion — as its organizing principle. Construct of B: identical — the same four-act structure with the twist explicitly identified as the third movement. A single coherent description covers both: 'is the work organized in the four-act ki-shō-ten-ketsu structure with a turning point in part three?' The two rubrics are wikipedia-style descriptions of the same structural form on two different pages. Wording differs in elaboration but the construct is identical. Safely mergeable.",
        },
        {
            "verdict": "related",
            "a": {
                "name": "Avoidance of Deus Ex Machina / Unmotivated Introductions",
                "description": "Aristotle criticizes resolutions by arbitrary external intervention (deus ex machina) and the sudden unprepared appearance of agents. Do not resolve the plot by introducing an agent or device that has no antecedent in the action's causal chain.",
            },
            "b": {
                "name": "Role of divine intervention or fate (interaction with hamartia)",
                "description": "Debates exist over whether gods/fate play a role in tragedy; hamartia may include elements of fate or divine will, but plot action must still beget plot action in Aristotelian terms. If incorporating fate/divine forces, ensure they interact with human error in a way that preserves causal chains.",
            },
            "reasoning": "Construct of A: whether the plot's resolution is causally motivated by antecedent action rather than by an external intervention (deus ex machina). The rubric is a constraint on the resolution mechanism. Construct of B: the more theoretical question of how divine/fate forces — when present — should integrate with the protagonist's hamartia to preserve causal logic. A is a NO-DEM prohibition; B is a positive theory of how fate-elements may legitimately function alongside human error. They are not the same construct: A judges the resolution endpoint, B judges the structural integration of supernatural elements across the action. A story without supernatural elements can pass A vacuously while B is inapplicable. A story whose divine forces interact tightly with hamartia satisfies B but would also satisfy A (no DEM needed). They cluster on the plot-causation axis but capture distinct evaluative dimensions.",
        },
        {
            "verdict": "different",
            "a": {
                "name": "Avoidance of Deus Ex Machina",
                "description": "Do not resolve the plot by introducing an agent or device that has no antecedent in the action's causal chain.",
            },
            "b": {
                "name": "Sattvika bhavas: physiological manifestations of intense emotion",
                "description": "Sattvika bhavas are eight involuntary physical manifestations resulting from intense inner emotion (trembling, tears, change in voice). They are central to authentic dramatic performance: inner identification produces visible involuntary signs of feeling.",
            },
            "reasoning": "Construct of A: the causal integrity of plot resolution — whether the ending follows from the established action without deus-ex-machina intervention. Construct of B: the authenticity of emotional performance — whether characters' inner emotional states manifest in involuntary physiological signs (the Sanskrit dramaturgy concept of sattvika bhava). These describe entirely separate properties: A is about plot architecture, B is about the embodiment of emotion through dramaturgical detail. No coherent single description captures both. Unrelated constructs.",
        },
    ],
    "hyper_specific": [
        {
            "verdict": "duplicate",
            "a": {
                "name": "Word count limits and preference",
                "description": "Up to 10,000 words (under 5,000 preferred). We prefer stories under 5,000 words, but consider stories up to 10,000 words. We have no minimum wordcount requirement.",
            },
            "b": {
                "name": "Word-count limits and preferences",
                "description": "Up to 10,000 words (under 5,000 preferred). The longer the story, the less likely we are to be interested.",
            },
            "reasoning": "Construct of A: the magazine's word-count submission constraint — a 10K-word ceiling, a 5K-word preference, and no minimum. Construct of B: identical — the same 10K ceiling, the same 5K preference, with an additional length-bias clarification that doesn't introduce a new constraint. A single coherent description captures both: 'is the submission within this magazine's word-count window (under 10K, ideally under 5K).' The two rubrics are the same magazine's word-count policy described on two of its own pages with slight wording differences. Safely mergeable.",
        },
        {
            "verdict": "paraphrase",
            "a": {
                "name": "Eligibility — publication year (post-2009 rule)",
                "description": "Beginning with the 2009 awards, nominated works must have been published during that calendar year. This replaced the earlier rolling eligibility window.",
            },
            "b": {
                "name": "Publication-year eligibility",
                "description": "Nominations must have been published (or scheduled to be published) in the current calendar year. Eligible works are those published during the current calendar year.",
            },
            "reasoning": "Construct of A: the eligibility constraint requiring a work to have been published in the award's named calendar year, with explicit reference to a 2009 rule change. Construct of B: the same eligibility constraint — current-calendar-year publication — without the historical-rule-change framing. A single coherent description covers both: 'is the work's publication date within the same calendar year as the award.' The two rubrics describe the same operational eligibility test from two different awards' pages. Wording diverges (one historical, one prescriptive); construct is identical. Safely mergeable.",
        },
        {
            "verdict": "related",
            "a": {
                "name": "Word count requirement (One Story)",
                "description": "3,000–8,000 words. Submissions outside this range are not acceptable; select work that fits the specified length.",
            },
            "b": {
                "name": "Word count eligibility (Commonwealth Short Story Prize)",
                "description": "2,000–5,000 words. Stories should conform to the stated length range; entries outside this range risk disqualification or lower consideration.",
            },
            "reasoning": "Construct of A: the One Story magazine's specific word-count window (3K–8K). Construct of B: the Commonwealth Prize's specific word-count window (2K–5K). Both apply the same TYPE of constraint (word-count band on the submission), but they prescribe DIFFERENT operational windows: a 4,500-word story passes both, a 7,000-word story passes A but fails B, a 2,500-word story passes B but fails A. The CONSTRUCT could be described in two ways: (a) 'magazine-specific submission length policy' (then they share the same construct and would be a paraphrase), or (b) 'compliance with THIS magazine's exact length range' (then they are different constructs because each operationalizes a different threshold). For dedup purposes we use interpretation (b): collapsing them loses information about which magazine's policy applies. They cluster on the word-count-policy axis but should remain separate items because their operational tests yield different verdicts on real submissions.",
        },
        {
            "verdict": "different",
            "a": {
                "name": "Form possessive singular of nouns by adding 's",
                "description": "Form the possessive singular of nouns by adding 's. Follow this rule whatever the final consonant (Charles's friend; Burns's poems).",
            },
            "b": {
                "name": "Sijo structural rules",
                "description": "Sijo are characterized by a structure of three stanzas of four feet each; each foot contains three to four syllables except on the third stanza where the first foot has 3 syllables.",
            },
            "reasoning": "Construct of A: an English prose grammar rule about possessive formation — a low-level orthographic constraint. Construct of B: a Korean verse-form structural rule — a syllable-and-stanza specification for the sijo poetic form. These describe entirely separate kinds of properties: an English punctuation rule versus a Korean poetic meter. No coherent single description captures both. Unrelated constructs.",
        },
    ],
    "vague":           [],
}


def format_few_shots(few_shots: list[dict]) -> str:
    if not few_shots:
        return "\n(No bucket-specific examples provided.)\n"
    chunks = ["\n"]
    for i, ex in enumerate(few_shots, 1):
        chunks.append(f"--- EXAMPLE {i} ({ex['verdict']}) ---\n")
        chunks.append("Rubric A:\n")
        chunks.append(f"  name: {ex['a']['name']}\n")
        chunks.append(f"  description: {ex['a']['description']}\n")
        chunks.append("Rubric B:\n")
        chunks.append(f"  name: {ex['b']['name']}\n")
        chunks.append(f"  description: {ex['b']['description']}\n")
        chunks.append("Output:\n")
        chunks.append("{\n")
        chunks.append(f"  \"reasoning\": {json.dumps(ex['reasoning'])},\n")
        chunks.append(f"  \"verdict\": \"{ex['verdict']}\"\n")
        chunks.append("}\n\n")
    return "".join(chunks)


def build_system_prompt(specificity: str) -> str:
    examples = FEW_SHOTS_BY_SPECIFICITY.get(specificity, [])
    if not examples:
        # Fallback: vague is too small for its own examples; use general's.
        examples = FEW_SHOTS_BY_SPECIFICITY["general"]
    return SYSTEM_PROMPT_BASE + format_few_shots(examples)


# Update FEW_SHOT_NAMES to include all bucket-specific example names (for
# exclusion from generalization tests).
for _bucket_examples in FEW_SHOTS_BY_SPECIFICITY.values():
    for _ex in _bucket_examples:
        FEW_SHOT_NAMES.add(_ex['a']['name'].lower().strip())
        FEW_SHOT_NAMES.add(_ex['b']['name'].lower().strip())

# Strict schema preserves property order: reasoning emitted before verdict.
JSON_SCHEMA = {
    "name": "rubric_relationship",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "reasoning": {"type": "string"},
            "verdict":   {"type": "string", "enum": ["duplicate", "paraphrase", "related", "different"]},
        },
        "required": ["reasoning", "verdict"],
    },
}

def build_user_msg(a: dict, b: dict) -> str:
    def fmt(r):
        return (
            f"  name: {r['rubric_name']}\n"
            f"  description: {r['rubric_description'] or ''}\n"
            f"  guidance: {r.get('rubric_guidance','') or ''}\n"
            f"  source page: {r['page_id']}"
        )
    return f"Rubric A:\n{fmt(a)}\n\nRubric B:\n{fmt(b)}"


def load_keep_rubrics(task: str) -> list[dict]:
    rows = []
    for cf in sorted(CHUNKS.glob("chunk_*.jsonl")):
        with cf.open() as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if r.get("task") == task and r.get("cls_ok") and r.get("cls_keep") == "keep":
                        rows.append(r)
                except Exception:
                    pass
    return rows


import random as _random

STOPWORDS = {'a','an','the','and','or','of','for','in','on','to','vs','from',
             'with','by','as','at','but','if','is','it','this','that','these',
             'those','be','do','your','you','our','their','its','its'}

def _content_words(name: str) -> set[str]:
    return {w for w in str(name).lower().split()
            if len(w) > 3 and w.isalpha() and w not in STOPWORDS}


def select_test_pairs(rubrics: list[dict]) -> list[tuple[dict, dict, str, str]]:
    """Return (a, b, category_label, silver_label) tuples.

    Sampling categories:
      1. WITHIN-DOC: same page_id, different rubrics. Silver expectation:
         should NOT be duplicate/paraphrase (authors don't repeat themselves).
      2. CROSS-DOC, WORD OVERLAP: different page_id, rubric_name shares >= 2
         content words. Candidates for paraphrase/related.
      3. CROSS-DOC, META OVERLAP: different page_id, same (cls_target,
         cls_action, cls_verifiability). Candidates for related.
      4. CROSS-DOC, RANDOM: different page_id, no overlap heuristic. Likely
         different.

    Identical-name pairs are excluded throughout (uninteresting for iteration).
    """
    df = pd.DataFrame(rubrics)
    df['name_lc'] = df['rubric_name'].astype(str).str.lower().str.strip()
    df['words']   = df['rubric_name'].apply(_content_words)

    def name_diff(a, b): return a['name_lc'] != b['name_lc']
    def diff_doc(a, b):  return a['page_id'] != b['page_id']

    pairs: list[tuple[dict, dict, str, str]] = []
    seen: set[tuple] = set()

    def add(a: dict, b: dict, cat: str, silver: str) -> bool:
        if not name_diff(a, b): return False
        if a.get('rubric_idx') is None or b.get('rubric_idx') is None: return False
        k = tuple(sorted([f"{a['page_id']}::{a['rubric_idx']}",
                          f"{b['page_id']}::{b['rubric_idx']}"]))
        if k in seen: return False
        seen.add(k)
        pairs.append((a, b, cat, silver))
        return True

    rng = _random.Random(2026)

    # 1) WITHIN-DOC pairs (silver: NOT duplicate/paraphrase)
    page_sizes = df['page_id'].value_counts()
    candidate_pages = page_sizes[page_sizes >= 5].index.tolist()
    rng.shuffle(candidate_pages)
    needed = 2
    for page in candidate_pages:
        if needed == 0: break
        sub = df[df['page_id'] == page]
        if len(sub) < 2: continue
        sample = sub.sample(min(4, len(sub)), random_state=rng.randint(0, 999_999))
        recs = sample.to_dict('records')
        for i in range(len(recs)):
            for j in range(i+1, len(recs)):
                if add(recs[i], recs[j], "WITHIN-DOC", "NOT duplicate/paraphrase"):
                    needed -= 1
                    break
            if needed == 0: break

    # 2) CROSS-DOC, WORD OVERLAP (target: paraphrase / related)
    sample = df.sample(min(4000, len(df)), random_state=2026).reset_index(drop=True)
    needed = 3
    for i in range(len(sample)):
        if needed == 0: break
        a = sample.iloc[i]
        if not a['words']: continue
        for j in range(i+1, min(i+200, len(sample))):
            b = sample.iloc[j]
            if not diff_doc(a, b): continue
            if not name_diff(a, b): continue
            overlap = a['words'] & b['words']
            if len(overlap) >= 2:
                a_rec = a.to_dict(); b_rec = b.to_dict()
                a_rec['_overlap'] = ','.join(sorted(overlap))
                if add(a_rec, b_rec, "CROSS-DOC word-overlap", "candidate paraphrase/related"):
                    needed -= 1
                    break

    # 3) CROSS-DOC, META OVERLAP (target: related)
    meta_cols = ['cls_target', 'cls_action', 'cls_verifiability']
    meta_sizes = df.groupby(meta_cols).size()
    big_metas = meta_sizes[meta_sizes >= 30].sample(min(5, len(meta_sizes[meta_sizes >= 30])), random_state=11)
    needed = 2
    for meta_tuple in big_metas.index.tolist():
        if needed == 0: break
        mask = (df['cls_target']==meta_tuple[0]) & (df['cls_action']==meta_tuple[1]) & (df['cls_verifiability']==meta_tuple[2])
        sub = df[mask]
        if len(sub) < 4: continue
        for _ in range(10):
            pair = sub.sample(2, random_state=rng.randint(0, 999_999)).to_dict('records')
            if not diff_doc(pair[0], pair[1]): continue
            if add(pair[0], pair[1], f"CROSS-DOC meta-overlap ({'/'.join(meta_tuple)})", "candidate related"):
                needed -= 1
                break

    # 4) CROSS-DOC, RANDOM (likely: different)
    needed = 2
    for _ in range(50):
        if needed == 0: break
        pair = df.sample(2, random_state=rng.randint(0, 999_999)).to_dict('records')
        if not diff_doc(pair[0], pair[1]): continue
        if add(pair[0], pair[1], "CROSS-DOC random", "likely different"):
            needed -= 1

    return pairs


def add_high_cosine_test_pairs(pairs: list, rubrics_all: list[dict], task: str = "creative-writing",
                                specificity: str = "general", n_per_bucket: int = 2) -> list:
    """Load embeddings cache and append high-cosine test pairs from a specific
    (task, specificity) bucket. Excludes few-shot rubrics for generalization."""
    cache = ROOT / f"outputs/embeddings/{task}_embeddings.npz"
    if not cache.exists():
        print(f"WARNING: no embedding cache at {cache} — skipping high-cosine tests")
        return pairs
    d = np.load(cache, allow_pickle=True)
    expected_keys = [f"{r['page_id']}::{r['rubric_idx']}" for r in rubrics_all]
    if list(d['keys']) != expected_keys:
        print("WARNING: embedding cache key mismatch — skipping high-cosine tests")
        return pairs
    embs_all = d['embs'].astype(np.float32)
    print(f"loaded embeddings: shape={embs_all.shape}")

    # Filter to the Phase-1 bucket: target=work + specified specificity
    keep_idx = [i for i, r in enumerate(rubrics_all)
                if r.get('cls_target') == 'work'
                and r.get('cls_specificity') == specificity]
    rubrics_filt = [rubrics_all[i] for i in keep_idx]
    embs_filt    = embs_all[keep_idx]
    print(f"within-bucket rubrics (target=work, spec={specificity}): {len(rubrics_filt):,} / {len(rubrics_all):,}")

    rng_np = np.random.RandomState(2027)
    n_queries = min(2500, len(rubrics_filt))
    q_idx = rng_np.choice(len(rubrics_filt), size=n_queries, replace=False)

    buckets = {
        "HIGH-COS dup-zone (cos>=0.90)":       [],
        "HIGH-COS paraphrase-zone (0.80-0.90)": [],
        "HIGH-COS related-zone (0.65-0.80)":    [],
    }
    seen: set[tuple] = set()

    for qi in q_idx:
        sims = embs_filt @ embs_filt[qi]
        sims[qi] = -1
        top = np.argpartition(-sims, 8)[:8]
        for ti in top:
            if ti == qi: continue
            a = rubrics_filt[qi]; b = rubrics_filt[ti]
            if a['page_id'] == b['page_id']: continue
            name_a = str(a.get('rubric_name') or '').lower().strip()
            name_b = str(b.get('rubric_name') or '').lower().strip()
            if name_a == name_b: continue
            if name_a in FEW_SHOT_NAMES or name_b in FEW_SHOT_NAMES: continue
            pair_key = tuple(sorted([(a['page_id'], a['rubric_idx']),
                                     (b['page_id'], b['rubric_idx'])]))
            if pair_key in seen: continue
            seen.add(pair_key)
            cos = float(sims[ti])
            if cos >= 0.90:
                buckets["HIGH-COS dup-zone (cos>=0.90)"].append((a, b, cos))
            elif cos >= 0.80:
                buckets["HIGH-COS paraphrase-zone (0.80-0.90)"].append((a, b, cos))
            elif cos >= 0.65:
                buckets["HIGH-COS related-zone (0.65-0.80)"].append((a, b, cos))

    for k in buckets:
        buckets[k].sort(key=lambda t: -t[2])

    for bucket_name, items in buckets.items():
        zone_silver = {
            "HIGH-COS dup-zone (cos>=0.90)":        "expect duplicate/paraphrase",
            "HIGH-COS paraphrase-zone (0.80-0.90)": "expect paraphrase/related",
            "HIGH-COS related-zone (0.65-0.80)":    "expect related",
        }[bucket_name]
        print(f"  {bucket_name:<42s}: {len(items)} candidates")
        for a, b, cos in items[:n_per_bucket]:
            label = f"{bucket_name} cos={cos:.3f}"
            pairs.append((a, b, label, zone_silver))
    return pairs


def call_model(client, system: str, user: str) -> dict:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        response_format={"type": "json_schema", "json_schema": JSON_SCHEMA},
    )
    raw = resp.choices[0].message.content or ""
    try:
        return json.loads(raw)
    except Exception:
        return {"_parse_error": True, "_raw": raw}


def silver_check(category: str, verdict: str) -> tuple[bool, str]:
    """Return (passes, message). For within-doc pairs, duplicate/paraphrase
    is a silver-label violation."""
    if category == "WITHIN-DOC":
        if verdict in ("duplicate", "paraphrase"):
            return False, f"SILVER VIOLATION — within-doc pair marked {verdict}"
        return True, "ok (within-doc not flagged as same)"
    return True, "(no silver check applied)"


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="creative-writing")
    ap.add_argument("--specificity", default="general",
                    help="Run on pairs from this specificity bucket only; uses the bucket's few-shots.")
    args = ap.parse_args()

    rubrics_all = load_keep_rubrics(args.task)
    print(f"loaded {len(rubrics_all):,} {args.task} keep rubrics (cls_keep=='keep', cls_ok=True)")

    # Apply Phase-1 filter: target=work + specified specificity bucket
    rubrics = [r for r in rubrics_all
               if r.get('cls_target') == 'work'
               and r.get('cls_specificity') == args.specificity]
    print(f"after Phase-1 filter (target=work, specificity={args.specificity}): {len(rubrics):,} rubrics")

    pairs = select_test_pairs(rubrics)
    print(f"heuristic test pairs (from bucket): {len(pairs)}")
    pairs = add_high_cosine_test_pairs(pairs, rubrics_all, task=args.task,
                                        specificity=args.specificity, n_per_bucket=2)
    print(f"total test pairs after embedding-augment: {len(pairs)}\n")

    api_key = KEY_PATH.read_text().strip() if KEY_PATH.exists() else os.environ.get("OPENAI_API_KEY","")
    if not api_key:
        sys.exit(f"no API key — checked {KEY_PATH} and OPENAI_API_KEY env")
    client = OpenAI(api_key=api_key)

    system_prompt = build_system_prompt(args.specificity)
    print(f"system prompt: {len(system_prompt):,} chars; few-shots from bucket={args.specificity}\n")

    cat_verdicts: dict[str, list[str]] = {}
    violations: list[int] = []

    for i, (a, b, category, silver_label) in enumerate(pairs, 1):
        print("="*80)
        print(f"PAIR {i} [{category}]  silver-label: {silver_label}")
        print("-"*80)
        print(f"RUBRIC A  name: {a['rubric_name']}")
        print(f"          desc: {(a['rubric_description'] or '')[:160]}")
        print(f"          page: {a['page_id']}")
        print(f"          cls : target={a.get('cls_target')} action={a.get('cls_action')} verif={a.get('cls_verifiability')}")
        if a.get('_overlap'):
            print(f"          shared words with B: {a['_overlap']}")
        print(f"RUBRIC B  name: {b['rubric_name']}")
        print(f"          desc: {(b['rubric_description'] or '')[:160]}")
        print(f"          page: {b['page_id']}")
        print(f"          cls : target={b.get('cls_target')} action={b.get('cls_action')} verif={b.get('cls_verifiability')}")
        print("-"*80)

        user_msg = build_user_msg(a, b)
        result = call_model(client, system_prompt, user_msg)
        verdict = result.get('verdict', '?')
        reasoning = result.get('reasoning', '')

        print(f"VERDICT:   {verdict}")
        print(f"REASONING: {reasoning}")

        ok, msg = silver_check(category.split()[0] if ' ' in category else category, verdict)
        flag = "PASS" if ok else "FAIL"
        print(f"SILVER:    [{flag}] {msg}")
        if not ok:
            violations.append(i)

        cat_verdicts.setdefault(category, []).append(verdict)
        print()

    # Summary
    print("="*80)
    print("SUMMARY")
    print("="*80)
    for cat, verdicts in cat_verdicts.items():
        cnts = {v: verdicts.count(v) for v in sorted(set(verdicts))}
        print(f"  {cat}")
        for v, c in cnts.items():
            print(f"    {v:<12s} {c}")
    if violations:
        print(f"\nsilver-label VIOLATIONS: pairs {violations} (within-doc marked duplicate/paraphrase)")
    else:
        print("\nno silver-label violations")

if __name__ == "__main__":
    main()
