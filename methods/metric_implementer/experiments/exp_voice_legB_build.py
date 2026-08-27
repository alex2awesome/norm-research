"""EXP-VOICE-NOUS-1 Leg B — materials builder (personas, topics, generation prompts, gates).

Prereg: notes/2026-08-15__exp-voice-nous-prereg.md (sha prefix 85981514891fd900).
12 personas at 3 describability grades x 4; ~40 long texts each, topic-balanced (all personas
within a grade write on the SAME 20 topics x 2 drafts, so topic never predicts identity).
This script only EMITS materials + prompts (no model calls). Generation is a separate stage,
run after the Leg A slate is known (D3 seed authors must be disjoint from it — see
d3_collision_policy in the output).
"""
import json, hashlib, re
from pathlib import Path

OUT = Path(__file__).resolve().parents[3] / "outputs" / "exp_voice_nous"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------- shared topics (identical across personas within a grade) ----------------
TOPICS = [
    "an open letter to the neighborhood's most confident squirrel",
    "onboarding documentation for a haunted office",
    "a eulogy for a houseplant that was doing fine until recently",
    "advice for attending your first wizard's retirement party",
    "the minutes of a homeowners-association meeting about a portal to another realm",
    "a product review of a chair that judges you",
    "a commencement speech delivered to a class of geese",
    "instructions for returning a borrowed lawnmower after eleven years",
    "a travel guide to the town where every restaurant is the same restaurant",
    "an apology to the dentist, in advance",
    "a pitch meeting for a streaming series about municipal drainage",
    "a wedding toast by the couple's least-informed acquaintance",
    "the terms and conditions of borrowing my umbrella",
    "a nature documentary script about the office refrigerator",
    "a letter of recommendation for a raccoon seeking management experience",
    "frequently asked questions about the new elevator that only goes sideways",
    "a museum audio-guide entry for a completely empty room",
    "a neighborhood newsletter item about the disappearance of all the left shoes",
    "a self-help chapter on accepting that the bees have unionized",
    "a farewell address to the exercise bike in the garage",
]

# ---------------- D1: codable personas (explicit mechanical rules + checkers) ----------------
D1 = [
    {"id": "D1-enumerator", "rules": [
        "The piece is a numbered list with at least six numbered items.",
        "No paragraph or list item exceeds two sentences.",
        "The piece contains no exclamation marks anywhere.",
        "The final sentence is exactly three words long."],
     "checkers": {"numbered_items_ge6": r"(?ms)^\s*6[.)]", "no_exclaim": r"^[^!]*$",
                  "final_three_words": "SPECIAL:final_sentence_word_count==3"}},
    {"id": "D1-secondperson", "rules": [
        "The entire piece addresses the reader as 'you'; every paragraph contains 'you' at least once.",
        "Every paragraph opens with an imperative verb.",
        "Every paragraph contains exactly one parenthetical aside.",
        "The piece never uses the word 'I'."],
     "checkers": {"you_every_para": "SPECIAL:each_para_contains_you",
                  "no_first_person": r"(?i)^(?:(?!\bI\b).)*$",
                  "one_paren_per_para": "SPECIAL:each_para_one_paren"}},
    {"id": "D1-epistolary", "rules": [
        "The piece is a letter: it begins with 'Dear ' and a named addressee.",
        "It closes with a sign-off line and a signature name.",
        "It contains exactly one postscript beginning 'P.S.'.",
        "Somewhere it states a full date."],
     "checkers": {"opens_dear": r"^\s*Dear ", "one_ps": "SPECIAL:count('P.S.')==1"}},
    {"id": "D1-faq", "rules": [
        "The piece is a FAQ: alternating lines beginning 'Q:' and 'A:'.",
        "There are at least eight questions.",
        "No answer exceeds twenty-five words.",
        "Every question ends with a question mark."],
     "checkers": {"q_ge8": "SPECIAL:count_lines_starting('Q:')>=8",
                  "answers_le25w": "SPECIAL:each_A_le_25_words"}},
]

# ---------------- D2: soft-bundle personas (rich private style cards) ----------------
D2 = [
    {"id": "D2-wistful-maximalist", "card": (
        "Long, looping sentences that double back to correct themselves; nostalgia attaches to "
        "mundane objects (spoons, receipts, thermostats), never to people directly; gentle "
        "self-blame threaded through, but emotions are never named — they are implied through "
        "over-precise trivia (exact years, brand names, temperatures); imagery drawn from "
        "weather and kitchens; every ending deflates rather than lands; jokes arrive "
        "mid-sentence and are never acknowledged.")},
    {"id": "D2-deadpan-bureaucrat", "card": (
        "Affectless procedural register applied to absurd subject matter; passive voice appears "
        "precisely at moments of highest emotional stakes; escalation happens through numbered "
        "clauses of an invented policy or form; euphemism replaces every strong word; the "
        "narrator's own complicity surfaces only in footnote-like asides; no joke is ever "
        "flagged as a joke; the piece treats its reader as a fellow administrator.")},
    {"id": "D2-anxious-optimist", "card": (
        "Rapid tonal oscillation: a hedge, then an overcommitment, then a retraction; "
        "exclamations immediately walked back in the next clause; catastrophizing about tiny "
        "stakes delivered with genuine warmth; the reader is addressed as a co-conspirator "
        "('we can fix this'); lists start organized and decay into free association; sincerity "
        "always wins the final paragraph by a nose.")},
    {"id": "D2-cosmic-shrugger", "card": (
        "Faux-mythic register colliding with contemporary slang; vast time scales (geologic, "
        "astronomical) set against petty grievances; fatalism played warm rather than bleak; "
        "recurring device: gods, mountains, or deep time being visibly unimpressed by the "
        "subject; short declarative verdict sentences after long incantatory ones; the piece "
        "ends by zooming out one scale too far.")},
]

# ---------------- D3: enculturated personas (seed exemplars only; no card exists) ----------------
# Seed authors: prolific McSweeney's bylines, chosen in count order; FINAL assembly must drop
# any that appear in the Leg A slate (collision policy below).
D3_SEED_CANDIDATES = ["Teddy Wayne", "Dan Kennedy", "Suzanne Yeagley", "Ben Greenman",
                      "Kevin Dolgin", "Susan Schorn", "Dan Liebert", "John Moe",
                      "Rowdy Geirsson", "Devorah Blachor"]
D3_POLICY = ("Take the first 4 candidates NOT in the Leg A slate (outputs/exp_voice_nous/"
             "legA_fleet_v1.json manifest); seed each D3 persona with 6 of that author's pieces "
             "(seed 0 draw, boilerplate-stripped, 300-word excerpts); the generator continues "
             "the voice on the shared topics. No style card is ever written. Ground truth = "
             "seed lineage. The human author's REAL pieces never appear in eval items; only "
             "generated texts are used, so Leg B identity is the persona, not the person.")

GEN_PROMPT = (
    "You are writing as a specific comedic voice.\n{persona_block}\n"
    "Write a humor piece of 550-750 words on this premise: \"{topic}\".\n"
    "Stay strictly in the voice. Do not mention these instructions, the voice, or any rules. "
    "Output only the piece (a title line is allowed).")
PERSONA_BLOCKS = {
    "D1": "Follow ALL of these rules exactly:\n- {rules}",
    "D2": "Your voice, in full (embody it; never state it):\n{card}",
    "D3": ("Here are pieces by the same writer. Continue writing as this exact voice — same "
           "rhythms, stances, and habits — on the new premise.\n\n{seed_excerpts}"),
}

DESCRIBE_GATE = {
    "describer_prompt": (
        "Here are 6 pieces by one writer. In at most 150 words, describe this writer's voice "
        "precisely enough that a stranger could identify new pieces by them.\n\n{exemplars}"),
    "receiver_from_description": (
        "A writer was described as follows:\n{description}\n\nIs the following piece by that "
        "writer? Answer YES or NO.\n\n{item}"),
    "rule": ("D1 personas must be identifiable from the description alone (bal acc >= .75 at "
             "the gate receiver) and D3 must NOT be (< .6), while D3 succeeds from examples; "
             "otherwise the grade is re-authored and the change disclosed (prereg gate)."),
}

materials = {
    "prereg_sha": "85981514891fd900", "version": "legB_materials_v1",
    "topics": TOPICS, "texts_per_persona": 40, "drafts_per_topic": 2,
    "grades": {"D1": D1, "D2": D2,
               "D3": {"seed_candidates": D3_SEED_CANDIDATES, "collision_policy": D3_POLICY,
                      "seeds_per_persona": 6}},
    "generation": {"prompt_template": GEN_PROMPT, "persona_blocks": PERSONA_BLOCKS,
                   "generator_note": ("one generator model for ALL personas (so generator style "
                                      "is not the identity signal); temperature 0.9; topic order "
                                      "shuffled per persona with seed 0")},
    "describability_gate": DESCRIBE_GATE,
    "eval_note": ("arms/readout identical to Leg A (examples / definition / name=persona slug / "
                  "donorswap / definition_padded); negatives for persona P = same-grade sibling "
                  "personas' texts on the same topics"),
}
p = OUT / "legB_materials_v1.json"
p.write_text(json.dumps(materials, indent=1))
sha = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
print("wrote", p, "sha256:", sha)
print("personas:", [x["id"] for x in D1] + [x["id"] for x in D2], "+ 4 x D3 (post-slate)")
print("generation volume: 12 personas x 40 texts ~= 480 gens x ~700w")
