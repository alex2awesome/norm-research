#!/usr/bin/env python3
"""Style Invitational MATURE A bank (v2) -- authored AGAINST the documented failure.

THE FAILURE THIS BANK IS DESIGNED AGAINST. The v1 32-criterion bank was declared
TERMINAL as "TIE, bank = length model": 0 of 32 criteria retained a .05
deviation from chance once length was held fixed, and the bank (.613) scored
BELOW the plain programmatic length/format block (.632). Its best criteria
collapsed under stratification -- Self-contained clarity .567 -> .513,
Prompt-specific relevance .566 -> .510, Vivid comic image .543 -> .494, Prompt
task completion .542 -> .485.

Two separate things went wrong, and this bank fixes both.

(1) THE POPULATION WAS 16.3% PARSE ARTIFACTS (see build_si_clean_population.py).
    Removing them drops the length nuisance from AUC .6227 to .5520 pooled
    (.6181 -> .5589 within-week). Most of the "length model" was the model
    detecting orphan bylines and "And last:" headers, which are 11x
    concentrated in the honorable-mention class. This bank is scored on the
    CLEAN population only.

(2) THE CRITERIA THEMSELVES WERE ELABORATION PROXIES. "Vivid", "specific",
    "complete", "sets up and pays off" all reward having more room. On a corpus
    where top-tier entries are genuinely a bit longer, such criteria re-measure
    length no matter how carefully they are worded.

DESIGN RULES APPLIED TO EVERY ENTRY BELOW:
  * LENGTH-ORTHOGONAL: for each criterion, "could a writer raise this score by
    adding words?" must be answerable NO. Ratio framings make the denominator
    grow with length on purpose; positional framings are about arrangement, not
    amount; obviousness framings are about the idea, not its execution length.
  * NEGATIVELY-ORIENTED criteria are a deliberate third of the real bank
    (`orientation: "negative"`, 1.0 = the entry is WORSE). These are the
    length-CANCELLING family: padding, self-explanation, post-punch trailing
    material and hedging all GROW with length while making an entry worse, so
    they enter the fitted stack with the opposite sign to the length channel.
    A bank of only positive criteria cannot do this.
  * RANK WITHIN PUBLISHED QUALITY (the Wigleaf saturation lesson). Every row is
    a published entry; an honorable mention already beat thousands of
    submissions. "Is it funny / clever / well-made" saturates. The
    obviousness-and-contestability family exists because it is the property
    that still varies inside a pool of publishable jokes.
  * SPLIT THE POOL. v1's criteria had modal shares .60-.98 and four were
    outright near-constant. Thresholds here are pitched for a realistic split;
    anything that still lands above .98 is DROPPED by the enforced collapse gate
    at Layer-1 rather than argued about.
  * NO FORM-CONDITIONAL CRITERIA. v1's verse criteria were NA on 95-97% of
    entries (Meter and scansion control .966, Verse form as comic leverage
    .949). Nothing here is conditional on the entry being verse.
  * THE BYLINE IS NOT PART OF THE JOKE. Every entry carries a trailing
    "(Jane Smith, Bethesda)". Criteria that turn on final position or on
    per-clause ratios say explicitly that the byline is excluded.

DUAL TRACK: 4 criteria are DECLARED SURFACE probes (`track: "B"`), scored in the
same matrix so the surface channel is measurable inside the instrument and
A_real / A_surface can be split at readout without a re-score. Raw length is
deliberately NOT among them -- the V block already carries v_char_count, and
putting it in A would re-import the exact confound this bank exists to escape.

SMOKE REVISION (2026-08-10, validate-before-scaling). A 48-item smoke spread
across 48 DISTINCT weeks -- the first 24-item smoke drew from a single contest
and made every form-conditional criterion look falsely collapsed -- showed five
criteria dead on this pool and they were removed before the full run:

  | dropped                        | modal | judge's reading                    |
  |--------------------------------|-------|------------------------------------|
  | Premise had to be found        | 1.00  | every contest premise is "found"   |
  | Commits to exactly one comic idea | .98 | an 80-char entry has one idea      |
  | Hedges its own premise         | .98   | contest entries do not hedge       |
  | Explains its own joke          | .96   | published entries do not self-gloss|
  | Carries a second competing joke| .94   | no room for two jokes              |

This is the Wigleaf saturation lesson landing exactly where it was predicted:
FLAW-detection saturates on a published pool, because the flaws were edited out
before publication. Negative orientation is not the problem -- the negative
criteria that survive are the ones naming flaws that published work still has
(stock templates .50 modal, stock referents .25). The four criteria added in
their place sit in the families the smoke showed actually split: obviousness,
prompt-portability and inferential distance.

  python3 datasets/humor/style_invitational/va_v2/build_rubrics.py
"""
from __future__ import annotations

import json
from pathlib import Path

IGNORE_BYLINE = ("Ignore the trailing attribution in parentheses, such as "
                 "\"(Jane Smith, Bethesda)\"; it is archive metadata and is not "
                 "part of the entry.")

# (name, description, track, orientation, why_length_orthogonal)
BANK = [
    # ================= NEGATIVE / dilution: the length-cancelling family =====
    ("Continues past its own punch",
     "Whether material follows the entry's funniest word or beat, so the joke lands and "
     "then keeps going. " + IGNORE_BYLINE + " Score 1.0 when a clear tail of words follows "
     "the punch and drains it; 0.5 when a few words follow but do not blunt the landing; "
     "0.0 when the entry stops on its strongest beat; NA when the entry has no identifiable "
     "punch to place.",
     "A", "negative",
     "Scores WORSE as words are added after the punch, so adding words lowers quality here."),
    ("Recycles the prompt's own wording",
     "Whether the entry reuses the prompt's distinctive vocabulary verbatim rather than "
     "supplying its own, so the prompt is carrying words the entrant did not have to find. "
     "Score 1.0 when a distinctive phrase from the prompt is lifted whole and does no new "
     "work; 0.5 when prompt wording recurs but is bent to a new sense; 0.0 when the entry "
     "supplies its own vocabulary; NA when the prompt fixes an obligatory template that all "
     "entries must repeat.",
     "A", "negative",
     "Borrowed words add length while contributing nothing the entrant authored."),
    ("Signals its own cleverness",
     "Whether the entry flags that a joke is occurring -- an exclamation mark on the punch, "
     "scare quotes around the pun, a nudging aside, an ellipsis inviting the reader to "
     "admire the turn -- rather than delivering deadpan. Score 1.0 when such self-signalling "
     "is present and conspicuous; 0.5 when one mild marker appears; 0.0 when the entry plays "
     "it straight; NA when the entry's form requires the marker, as in quoted dialogue.",
     "A", "negative",
     "Punctuation and nudging asides are available at any length; if anything they are "
     "cheaper in short entries, so this cannot proxy for elaboration."),
    ("Leans on a stock joke template",
     "Whether the entry fills in a joke shape the reader has met many times -- the tired "
     "formula, the standard reversal, the off-the-shelf comparison -- rather than finding a "
     "shape this prompt provoked. Score 1.0 when the entry is a recognisable template with "
     "the blanks changed; 0.5 when a familiar shape is used but given a genuine twist; 0.0 "
     "when the construction is particular to this prompt; NA when the prompt itself dictates "
     "a fixed form.",
     "A", "negative",
     "Template-following is a property of the joke's shape, not of how many words realise it."),
    ("Reaches for an era's stock referent",
     "Whether the entry's target is one of the period's automatic go-to names or events that "
     "any entrant would reach for first, used as a punchline in itself rather than for "
     "anything specific about it. Score 1.0 when a stock referent carries the joke with no "
     "particular observation attached; 0.5 when a familiar referent is used but something "
     "specific is said about it; 0.0 when the referent is chosen for a precise reason or is "
     "not a stock name; NA when the entry names no external referent.",
     "A", "negative",
     "Referent choice is one word either way; picking a tired name is not a length effect."),

    # --- added after the week-spread smoke (see SMOKE REVISION note below) ---
    ("Could serve a different prompt unchanged",
     "Whether the entry is a general-purpose joke that happens to have been filed here, versus "
     "one that only works for this prompt. Score 1.0 when the entry would sit equally well "
     "under a different contest prompt with no change; 0.5 when it would need only a small "
     "adjustment; 0.0 when the joke is inseparable from this particular prompt; NA when the "
     "prompt is so open that no entry could be specific to it.",
     "A", "negative",
     "Portability is a property of the idea's attachment to the prompt; a long generic joke "
     "scores worse than a short specific one."),
    ("Joke is available from the prompt alone",
     "Whether the entry's gag is one a reader could have produced from the prompt without any "
     "further thought -- the first association the prompt's own wording throws up. Score 1.0 "
     "when the joke is the prompt's own obvious echo; 0.5 when it is a near neighbour of that "
     "echo; 0.0 when reaching it required a step the prompt does not supply; NA when the "
     "prompt supplies no obvious association.",
     "A", "negative",
     "Measures distance from the prompt's default association, which is fixed before any "
     "words are chosen."),

    # =============== OBVIOUSNESS / CONTESTABILITY: the within-quality axis ===
    ("Comic leap takes more than one step",
     "Whether reaching the joke requires the reader to make two or more connected inferences "
     "rather than one. Score 1.0 when at least two linked steps are needed and both land; 0.5 "
     "when a second step is available but optional; 0.0 when a single obvious association "
     "completes the joke; NA when the entry is a plain statement making no inferential demand.",
     "A", "positive",
     "Counts inferential steps, not words; a one-line entry can demand two steps and a long "
     "one can demand none."),
    ("Rests on a real-world coincidence",
     "Whether the entry depends on a genuine accident of fact or language that the entrant "
     "found rather than manufactured -- two things that really do share a name, a date, a "
     "sound or a property. Score 1.0 when a real coincidence is doing the work; 0.5 when the "
     "resemblance is real but loose enough to be arranged; 0.0 when the connection is invented "
     "by the entry itself; NA when the entry does not rely on any resemblance.",
     "A", "positive",
     "Whether a coincidence is real is a fact about the world, wholly independent of how many "
     "words report it."),
    ("Target is past the prompt's first suggestion",
     "Whether the entry goes beyond the referent the prompt most obviously points at, to one "
     "a competent entrant would reach only after discarding the first. Score 1.0 when the "
     "target is clearly not the prompt's default; 0.5 when the default is used but from an "
     "unexpected angle; 0.0 when the entry takes exactly the target the prompt hands over; "
     "NA when the prompt specifies the target and leaves no choice.",
     "A", "positive",
     "Which target is chosen is an idea-level decision; the entry is the same length either "
     "way."),
    ("Unlikely to be independently duplicated",
     "Whether many competent entrants working from this prompt would plausibly arrive at this "
     "same joke. Score 1.0 when the entry rests on a connection few would independently find; "
     "0.5 when it is a natural but not automatic move; 0.0 when it is the joke a room of "
     "entrants would converge on; NA when the prompt admits essentially one answer.",
     "A", "positive",
     "Duplicability is a property of the idea. A long obvious joke and a short surprising one "
     "score in opposite directions."),
    ("Wordplay is latent, not forced",
     "Whether the pun or substitution was already sitting in the source phrase and is merely "
     "revealed, versus manufactured by bending a word until it approximately fits. Score 1.0 "
     "when the wordplay falls out of the source with no distortion; 0.5 when a small stretch "
     "is required and the entry gets away with it; 0.0 when a word is visibly mangled to "
     "force the fit; NA when the entry contains no wordplay.",
     "A", "positive",
     "Latency is about the fit between two fixed words, which no amount of surrounding text "
     "improves."),
    ("Imports a referent from outside the prompt",
     "Whether the entry brings in a domain, name or frame the prompt did not mention, and "
     "makes it fit. Score 1.0 when an outside referent is imported and earns its place; 0.5 "
     "when an outside referent appears but does little work; 0.0 when the entry stays entirely "
     "inside the prompt's own furniture; NA when the prompt forbids outside material.",
     "A", "positive",
     "Importing a referent costs a word or two at most and is an idea-level move."),
    ("Depends on contest in-group knowledge",
     "Whether the joke requires knowing the contest itself -- its regular entrants, its "
     "running gags, its prize conventions -- rather than working for an ordinary reader. "
     "Score 1.0 when the joke cannot land without that inside knowledge; 0.5 when inside "
     "knowledge sweetens a joke that still works without it; 0.0 when no inside knowledge is "
     "needed; NA when the prompt is explicitly about the contest.",
     "A", "negative",
     "Inside-reference dependence is a property of the referent, not of entry length."),

    # ================== POSITIONAL / STRUCTURAL: arrangement, not amount ====
    ("Punch word occupies the final position",
     "Whether the word that carries the joke is the last word of the entry proper. "
     + IGNORE_BYLINE + " Score 1.0 when the punch word is final; 0.5 when it sits in the "
     "final clause but with a few words after it; 0.0 when it is buried earlier and ordinary "
     "material follows; NA when no single word carries the punch.",
     "A", "positive",
     "Position is pure arrangement: a long and a short entry can equally end on the punch."),
    ("Surprise is withheld to the end",
     "Whether the entry's turn arrives at the last possible moment, versus disclosing the joke "
     "early and coasting. Score 1.0 when the reader cannot see the turn coming until the final "
     "beat; 0.5 when the turn is partly telegraphed but still lands late; 0.0 when the joke is "
     "given away in the opening and the rest is elaboration; NA when the entry has no turn.",
     "A", "positive",
     "About WHERE the surprise sits in the entry, not how much text surrounds it."),
    ("Has a single load-bearing word",
     "Whether one specific word is doing the comic work, such that replacing it with a neutral "
     "synonym would collapse the joke. Score 1.0 when exactly such a word is identifiable; 0.5 "
     "when the work is shared between two words; 0.0 when the humour is diffuse and no single "
     "word is load-bearing; NA when the entry is not attempting a verbal joke.",
     "A", "positive",
     "Concentration of the comic load; a longer entry is if anything less likely to satisfy it."),
    ("Form mimics the thing it mocks",
     "Whether the entry's own grammar, register or shape imitates its target -- bureaucratic "
     "syntax for bureaucracy, ad copy for advertising, warning-label voice for a warning "
     "label. Score 1.0 when the form is a deliberate imitation that carries part of the joke; "
     "0.5 when the register is gestured at inconsistently; 0.0 when the form is generic prose; "
     "NA when the target has no characteristic form to imitate.",
     "A", "positive",
     "Formal imitation is a stylistic choice available at any length."),

    # =========== DENSITY / RATIO: length-normalised by construction =========
    ("Every clause carries comic weight",
     "Taking the entry's clauses excluding the byline, whether each one contributes to the "
     "joke rather than merely holding the sentence together. Score 1.0 when every clause "
     "contributes; 0.5 when one clause is purely structural; 0.0 when several clauses are "
     "scaffolding; NA when the entry is a single clause, so the ratio is undefined.",
     "A", "positive",
     "An explicit RATIO: a longer entry has more clauses that must all pay, so extra words "
     "make this HARDER, not easier."),
    ("Substitution changes meaning at one point",
     "Where the entry works by altering a known phrase, whether the change is localised to a "
     "single precise point rather than smeared across the sentence. Score 1.0 when exactly one "
     "swap does all the work; 0.5 when a second small change tags along; 0.0 when the source "
     "is rewritten broadly so no single pivot is visible; NA when the entry alters no known "
     "phrase.",
     "A", "positive",
     "Precision of a pivot, not the size of the text around it; broad rewriting is longer AND "
     "worse."),
    ("A word does double duty",
     "Whether a single word is held in two senses at once, both of which are live in the "
     "entry. Score 1.0 when a word genuinely carries two active senses; 0.5 when a second "
     "sense is available but not activated by the context; 0.0 when every word is used in one "
     "sense; NA when the entry is not attempting verbal play.",
     "A", "positive",
     "Double duty is economy itself -- it is the opposite of adding words."),
    ("Phonetic distance is tight",
     "Where the entry depends on sound resemblance, whether the match is close enough to snap "
     "into place rather than merely approximate. Score 1.0 when the sound match is near-exact; "
     "0.5 when it is recognisable but loose; 0.0 when the reader must be told the words are "
     "supposed to sound alike; NA when the entry does not rely on sound resemblance.",
     "A", "positive",
     "A property of two words' phonetics, wholly independent of entry length."),
    ("Sound pattern is load-bearing",
     "Where alliteration, rhyme or rhythm is present, whether it does comic work rather than "
     "decorating. Score 1.0 when the sound pattern is part of why the entry is funny; 0.5 when "
     "it is pleasant but incidental; 0.0 when it is present and actively distracting; NA when "
     "the entry has no marked sound pattern.",
     "A", "positive",
     "Asks whether an existing pattern works, not whether more of it is present."),

    # ============== RISK / COMMITMENT / VOICE: length-neutral ===============
    ("Commits fully to the bit",
     "Whether the entry inhabits its conceit without an escape hatch -- no apology, no "
     "distancing, no signalling that the writer knows it is silly. Score 1.0 when the entry "
     "commits completely; 0.5 when commitment slips once; 0.0 when the entry visibly holds "
     "itself at arm's length; NA when the entry has no conceit to inhabit.",
     "A", "positive",
     "Commitment is a stance; escape hatches are themselves extra words."),
    ("Trusts the reader to close the gap",
     "Whether the entry leaves the final inferential step to the reader rather than completing "
     "it. Score 1.0 when a specific step is left open and the reader can clearly take it; 0.5 "
     "when the gap is left but is vague enough to risk being missed; 0.0 when the entry closes "
     "every step itself; NA when the entry's form admits no such gap.",
     "A", "positive",
     "Closing a gap requires MORE words; leaving it requires fewer, so this is "
     "length-negative if anything."),
    ("Takes a position that could have failed",
     "Whether the entry risks something -- an unfashionable target, a tasteless angle, an "
     "obscure reference, a joke that only works if the reader grants a premise -- rather than "
     "playing safe. Score 1.0 when a real risk is taken and carried off; 0.5 when a mild risk "
     "is taken; 0.0 when the entry is entirely safe; NA when the prompt leaves no room for "
     "risk.",
     "A", "positive",
     "Risk is in the choice of material, not in its quantity."),
    ("Sustains one register throughout",
     "Whether the entry holds a single consistent voice from first word to last, rather than "
     "sliding between registers. " + IGNORE_BYLINE + " Score 1.0 when one register is held "
     "throughout; 0.5 when there is one slip; 0.0 when the voice changes without comic "
     "purpose; NA when the entry is too short to establish a register.",
     "A", "positive",
     "Consistency is a ratio-like property: longer entries have MORE opportunities to slip."),
    ("Voice belongs to a specified speaker",
     "Whether the entry is spoken by an identifiable persona whose way of talking is part of "
     "the joke, rather than by a neutral narrator. Score 1.0 when a specific speaker is "
     "audible and does comic work; 0.5 when a persona is implied but thin; 0.0 when the voice "
     "is neutral; NA when the prompt requires an impersonal form.",
     "A", "positive",
     "A persona can be established in three words or thirty."),
    ("Mockery is aimed at a deserving target",
     "Whether the entry's mockery lands on something powerful, pretentious or self-important "
     "rather than on an easy or defenceless one. Score 1.0 when the target can bear the hit; "
     "0.5 when the target is neutral; 0.0 when the entry punches at the defenceless for the "
     "laugh; NA when the entry mocks nothing.",
     "A", "positive",
     "Target selection is an idea-level choice with no length component."),
    ("Topical hook is load-bearing",
     "Where the entry uses a news reference, whether the joke depends on something specific "
     "about that story rather than on the name being current. Score 1.0 when the specific "
     "facts of the reference are doing the work; 0.5 when the reference is apt but "
     "interchangeable with a similar one; 0.0 when any current name would have served; NA "
     "when the entry makes no topical reference.",
     "A", "positive",
     "Depth of a reference, not the number of words spent on it."),
    ("Reading is unambiguous on first pass",
     "Whether the intended reading arrives on one pass, versus the reader having to re-parse "
     "to find the joke. Score 1.0 when the intended reading is the first one available; 0.5 "
     "when a brief second look is needed; 0.0 when the entry is genuinely ambiguous about "
     "which reading is the joke; NA when the entry deliberately sustains two readings as its "
     "point.",
     "A", "positive",
     "Two-sided: both over-compression and rambling cause misreads, so it does not track "
     "length in either direction."),
    ("Ending is the strongest beat",
     "Whether the entry's final beat is its funniest, versus its best moment occurring earlier. "
     + IGNORE_BYLINE + " Score 1.0 when the last beat is the peak; 0.5 when the ending is "
     "level with an earlier beat; 0.0 when the entry peaks early and declines; NA when the "
     "entry has only one beat.",
     "A", "positive",
     "A comparison between positions inside the entry, invariant to overall length."),

    # ============ Track B: DECLARED SURFACE PROBES (spurious pole) ==========
    ("Contains a parenthetical aside",
     "A surface check for a parenthetical remark inside the entry proper, independent of "
     "whether it is funny or useful. " + IGNORE_BYLINE + " Score 1.0 when such an aside is "
     "present; 0.5 when a dash-set aside serves the same function; 0.0 when no aside appears; "
     "NA never applies.",
     "B", "surface", "declared surface probe"),
    ("Uses an exclamation mark",
     "A surface check for an exclamation mark anywhere in the entry proper, independent of "
     "what it marks. " + IGNORE_BYLINE + " Score 1.0 when one or more appear; 0.5 when the "
     "entry instead ends on an ellipsis or a question mark; 0.0 when it ends on a period or "
     "no terminal punctuation; NA never applies.",
     "B", "surface", "declared surface probe"),
    ("Contains quoted material",
     "A surface check for quotation marks around any span in the entry proper, independent of "
     "what is quoted. " + IGNORE_BYLINE + " Score 1.0 when a quoted span is present; 0.5 when "
     "only scare quotes around a single word appear; 0.0 when no quotation marks appear; NA "
     "never applies.",
     "B", "surface", "declared surface probe"),
    ("Contains a fully capitalised word",
     "A surface check for a word in all capitals in the entry proper, independent of purpose. "
     + IGNORE_BYLINE + " Score 1.0 when such a word is present; 0.5 when only an acronym or "
     "initialism appears; 0.0 when no fully capitalised word appears; NA never applies.",
     "B", "surface", "declared surface probe"),
]


def main():
    out = Path(__file__).resolve().parent / "rubrics.jsonl"
    seen, lines = set(), []
    n_a = n_b = n_neg = 0
    for name, desc, track, orient, lo in BANK:
        assert name not in seen, f"duplicate criterion: {name}"
        seen.add(name)
        if track == "A":
            n_a += 1
            rid = f"a{n_a:02d}"
        else:
            n_b += 1
            rid = f"s{n_b:02d}"
        if orient == "negative":
            n_neg += 1
        lines.append(json.dumps({
            "rubric_id": rid, "name": name, "description": desc,
            "track": track, "orientation": orient,
            "length_orthogonality": lo,
            "gepa_revision": "authored against the v1 length-model failure: "
                             "length-orthogonal phrasing, an explicit NA branch that "
                             "is not a synonym for 0.0, and a threshold pitched to "
                             "split the pool rather than saturate it",
        }, ensure_ascii=False))
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}  n={len(lines)}  trackA={n_a} (negatively-oriented {n_neg})  "
          f"trackB={n_b}")
    ls = [len(json.loads(l)["description"]) for l in lines]
    print(f"description chars: min {min(ls)} median {sorted(ls)[len(ls)//2]} max {max(ls)}")


if __name__ == "__main__":
    main()
