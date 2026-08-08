"""Shared prompt bank for the ossfix_* diagnostic scripts. Not a repo file -- lives under
outputs/osl_multi/ alongside the standalone test scripts that import it."""

TEXTS = [
    "The committee reviewed the draft policy and unanimously approved it after minor edits.",
    "I love pizza but hate pineapple on it, it's just wrong.",
    "The stock market fell 2% today amid inflation fears.",
    "Please find attached the quarterly report for your review.",
    "lol that's the funniest thing I've seen all week.",
    "The mitochondria is the powerhouse of the cell, as every biology student learns.",
    "Why did the chicken cross the road? To get to the other side, obviously.",
    "Section 4.2 of the contract specifies a 30-day termination notice period.",
    "I can't believe they cancelled the show after just one season, so disappointing.",
    "The recipe calls for two cups of flour, one egg, and a pinch of salt.",
    "Breaking: local city council votes to raise property taxes by 3 percent.",
    "OMG did you see that meme, I'm crying laughing right now.",
    "The patient presented with a fever of 101F and mild abdominal pain.",
    "In accordance with Article 12, all shareholders must be notified in writing.",
    "My cat knocked my coffee off the table again this morning, typical Monday.",
    "The algorithm runs in O(n log n) time using a standard merge sort.",
    "Congress passed the bill 218-210 after a heated floor debate.",
    "Honestly this new phone update is the worst thing that's ever happened to me.",
    "The museum's new exhibit features Renaissance paintings from the 15th century.",
    "Two guys walk into a bar. The bartender says, 'what is this, a joke?'",
]

RUBRICS = [
    "describes a formal approval or decision process",
    "expresses a strong personal preference or opinion",
    "contains a joke or attempt at humor",
    "is a formal business or professional communication",
    "expresses amusement or laughter",
]


def build_prompts(n: int = 20):
    """Pair text[i] with rubric[i % len(rubrics)] -- deterministic, no repeats in text."""
    out = []
    for i, text in enumerate(TEXTS[:n]):
        rubric = RUBRICS[i % len(RUBRICS)]
        out.append(
            f"TEXT: {text}\nQUESTION: does this text satisfy '{rubric}'? Answer YES or NO."
        )
    return out


FIVE_TEST_PROMPTS = build_prompts(5)
TWENTY_PROMPTS = build_prompts(20)
