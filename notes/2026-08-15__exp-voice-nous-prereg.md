# EXP-VOICE-NOUS-1 — preregistration: can showing install a voice that telling cannot?

Status: PREREGISTERED 2026-08-15, before any confirmatory run. User-approved design
(session paper-2-writing, 2026-08-15): authorship leg approved verbatim; synthetic-nous leg is
the user's proposal ("construct an entirely synthetic, tacit nous... isn't in the text and
can't be described but can be shown, then test it via examples"). User correction folded in:
exemplar selection in prior pipelines is via flips, not crowd labels; here primary selection is
RANDOM (seeded), flips-selection is a declared secondary arm.

## Motivation
§4.2g found examples do not outperform definitions for tacit constructs, but four selection-side
confounds were identified (tacit bases skipped by the exemplar freeze; polarity labels from a
labeler blind to the construct; short-support examples; κ≈0 reference). This experiment removes
all four: ground truth is generator/author identity (no labeler, no consensus reference), and
support is long (median 644 words).

## Leg A — natural voice (McSweeney's authorship)
Corpus: `datasets/humor/mcsweeneys_archive/authorship_corpus_v1.jsonl.gz`
(4,485 pieces ≥150 words, 118 authors ≥20 raw pieces, 107 authors ≥20 long pieces;
sha256 prefix **24b7082601744efc**; built from the bylined archive scrape, JSON-LD authors).

Slate: 12 target authors drawn by seeded random (seed 0) from the 63 authors with ≥30 long
pieces, stratified into 4 high-volume (≥60 pieces; fame proxy) / 8 mid-volume. Construct per
author: "written in the personal comic voice of this author."

Splits per author (disjoint, seeded): exemplar pool 12 pieces; held-out positives 20; matched
negatives 20 (other kept authors, matched on length decile and publication-year band; negative
AUTHORS disjoint from every exemplar set). Excerpts capped at 300 words (declared support cap).

Arms (receiver sees exactly one):
1. `examples` — 6 positive + 6 negative excerpts, labeled "by the same author" / "by other
   authors" (contrastive; polarity from ground truth, no model labels anywhere).
2. `definition` — the authoring model (gpt-oss-120b if available on sk3, else llama70b; NEVER a
   receiver in the same cell) reads the identical 6+6 set and writes ≤150 words describing the
   voice; receiver gets the description only.
3. `name` — author's real name only (tests pre-enculturation: is the voice already in the
   weights?).
4. `examples_donorswap` — placebo: another slate author's exemplar set presented as this
   author's. Must collapse.
5. `definition_padded` — arm 2 plus neutral padding to match arm 1's token length.
Secondary (declared, run only if budget remains): `examples_flip` — exemplars selected by flips
on a selection split disjoint from all evaluation items.

Task and readout: for each of the 40 held-out items (20 pos / 20 neg), receiver answers YES/NO
"is this by the same author/voice?". Score = balanced accuracy per (author × arm × receiver)
against real authorship. No consensus reference anywhere. Receivers: sk3 local ladder
(llama 1/3/8/70B; qwen2.5 3/7/14/32/72B), single pass, offline batch vLLM per standing rules;
within-family curves only.

## Leg B — synthetic tacit nous
12 generated personas at 3 describability grades × 4 personas, ~40 long texts each
(topic-balanced across a shared topic list so topic never predicts identity):
- D1 codable: persona = explicit mechanical style rules (planted analog).
- D2 soft bundle: persona = rich private style card (rhythm, stance, imagery habits) that
  resists compact description.
- D3 enculturated: persona defined ONLY by seed exemplars (generator told to continue the
  voice; no card exists). By construction there is nothing to "tell."
Same arms and readout as Leg A (ground truth = generating persona). Construct-validity gate for
"can't be told": a frontier describer writes its best description from exemplars; a disjoint
frontier receiver must identify D1 from description alone but FAIL on D3 (< .6 balanced acc),
while succeeding on D3 from examples-passthrough (else D3 is not indescribable and the grade is
re-authored, disclosed).

## Gates (all must pass before any headline is read)
- G1 instrument: D1/planted personas transmit via examples at top receivers (bal acc ≥ .75).
- G2 placebo: donorswap within .05 of chance at every receiver, both legs.
- G3 length: examples vs definition_padded is the quoted contrast (not vs bare definition).
- G4 fame leak (Leg A): if `name` ≈ `examples` for high-volume authors, voice is
  pre-enculturated; the showing-vs-telling read then uses mid-volume authors only (declared).

## Preregistered decision rule
"Showing installs what telling cannot" is SUPPORTED iff, at the top two receivers per family:
pooled (examples − definition_padded) balanced-acc delta > +.05 with bootstrap 95% CI > 0,
G1–G4 passing — and in Leg B the delta must GROW monotonically D1→D2→D3 (Page trend, α=.05).
UNSUPPORTED if delta CI spans 0 or the D-grade trend is flat/reversed. Anything else = mixed,
reported descriptively. No optional stopping; smoke stage (2 authors × 2 arms × llama8b,
≤500 calls) may only abort for infrastructure failure, never for effect direction.

## Not yet run
No confirmatory calls have been made as of freeze. Leg B generation and all scoring go to sk3
(articulability-trials rule); sk3 status must be checked (last known: parked, docker-full /).
