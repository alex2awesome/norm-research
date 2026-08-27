# Sealed agent prompts used by the CW-community closure campaign

These are the exact instruction wrappers used with the Agent tool. They are recorded
here for provenance; the *content* each agent sees is the sealed prompt file named in
the wrapper (built by `fleet_cw.py build`), plus nothing else.

---

## 1. Sealed PROPOSER (Claude legs, one fresh agent per proposer per track)

> You are a sealed proposer in a measurement experiment. Below is your entire task.
>
> HARD SEAL RULES — violating any of them invalidates the round:
> - Do NOT read any file, do NOT search the repository, do NOT run any command other
>   than the single Write that saves your answer. Everything you need is in this
>   message.
> - Do NOT look for a criterion bank, a label column, or another proposer's output.
> - Answer only from the text below.
>
> When you have your answer, use Write to save it verbatim (one JSON object, no
> surrounding prose, no code fence) to exactly this path:
> `<SCRATCH>/<tag><track>/out_<proposer_id>.txt`
> Then reply with only the number of criteria you wrote.
>
> ---8<--- SEALED PROMPT BEGINS ---8<---
> {contents of prompt_<proposer_id>.txt}
> ---8<--- SEALED PROMPT ENDS ---8<---

## 2. Blind SPECIES PARTITION (fresh judge per round per track)

> You are given a pool of {N} candidate scoring criteria, in random order, with all
> authorship stripped. Several different proposers wrote them independently and some
> of them are the SAME underlying concept expressed differently.
>
> Partition the pool into distinct CONCEPT SPECIES. Two criteria belong to the same
> species when an independent judge scoring a story against them would be measuring
> the same property — not merely when they share vocabulary or a theme. Different
> facets of one broad theme are DIFFERENT species if a story could clearly satisfy one
> and clearly fail the other.
>
> Use full recall: consider every pairing, do not shortlist by surface similarity.
>
> Do NOT read any file or search anything; work only from this message.
>
> Output exactly one JSON object and nothing else:
> {"species": {"<short species label>": ["<pid>", ...], ...}}
> Every pid must appear in exactly one species.

## 3. Blind ROUTING AUDIT (fresh Sonnet-class auditor per round)

> You are auditing a pool of {N} candidate scoring criteria for a study of short
> stories posted to an online creative-writing community. Authorship and provenance
> have been stripped; some were written to capture genuine story quality and some were
> written to capture predictive-but-spurious surface channels, and you are not told
> which is which.
>
> For each item decide:
>   quality_relevant — a careful editor would treat this property as evidence about
>     how good the story IS.
>   incidental — the property might well predict how a story is received, but a
>     careful editor would not treat it as evidence of the story's quality (surface
>     form, platform furniture, fashionable topic, a fingerprint of who wrote it or
>     when it was posted).
>
> Judge the criterion as written, not the concept it gestures at. Do not read any
> file; work only from this message.
>
> Output exactly one JSON object and nothing else:
> {"verdicts": [{"aid": "...", "verdict": "quality_relevant"|"incidental",
>                "confidence": 0.0-1.0, "reason": "<= 25 words"}, ...]}

## 4. ARBITER (frontier model, disputes only)

> Two independent processes disagree about how to route these scoring criteria. For
> each item you are given the criterion, the track it was PROPOSED under (A = intended
> as quality-relevant, B = intended as a suspected-spurious channel) and the blind
> auditor's verdict with its reason. You are the arbiter; your call is final.
>
> Rule: a criterion belongs on the A side only if a careful editor would treat the
> property as evidence about how good the story IS. If the property mainly tracks who
> wrote it, when it was posted, what platform conventions it follows, or how it looks
> rather than how it reads, it belongs on the B side. When a property genuinely does
> both, say so and pick the side its SCORING INSTRUCTION actually measures.
>
> Output exactly one JSON object and nothing else:
> {"decisions": [{"aid": "...", "final": "A"|"B", "reason": "<= 30 words"}, ...]}
