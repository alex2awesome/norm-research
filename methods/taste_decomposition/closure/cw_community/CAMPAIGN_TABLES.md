### Closure curve

| round | criteria added (post-gate) | bank cols | VA_nl MONITOR | Δ_r (MONITOR) | Δ_beyond MONITOR | Δ_beyond population |
|---|---|---|---|---|---|---|
| 0 | — (45 A + 15 V) | 60 | 0.6564 | — | +0.1386 | +0.1412 |
| 1 | +15 A / +10 B | 75 | 0.6493 | -0.0071 | +0.1457 | +0.1416 |
| 2 | +14 A / +10 B | 89 | 0.6578 | +0.0085 | +0.1372 | +0.1377 |
| 3 | +11 A / +13 B | 100 | 0.6672 | +0.0094 | +0.1279 | +0.1327 |
| 4 | +14 A / +11 B | 114 | 0.6737 | +0.0066 | +0.1213 | +0.1274 |
| 5 | +13 A / +10 B | 127 | 0.6716 | -0.0021 | +0.1234 | +0.1283 |
| 6 | +15 A / +10 B | 142 | 0.6731 | +0.0016 | +0.1219 | +0.1290 |
| 7 | +2 A / +2 B | 144 | 0.6786 | +0.0055 | +0.1164 | +0.1269 |

### Per-round instrument bookkeeping

| round | scored | collapsed | misrouting | probe gate | anchor pos/neg AUC | coherent-vs-scrambled | sign-contradicting A | ΔC₊ | ΔC₋ |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 25 | 0 | 0% | PASS | 0.618 | 0.999 | 0 | +0.0001 | -0.0024 |
| 2 | 25 | 1 | 0% | PASS | 0.559 | 1.000 | 1 | +0.0032 | +0.0067 |
| 3 | 25 | 0 | 4% | PASS | 0.598 | 0.999 | 2 | +0.0066 | -0.0013 |
| 4 | 25 | 0 | 0% | PASS | 0.654 | 0.987 | 1 | +0.0071 | -0.0011 |
| 5 | 25 | 1 | 0% | PASS | 0.631 | 1.000 | 2 | -0.0001 | -0.0039 |
| 6 | 25 | 0 | 0% | PASS | 0.555 | 1.000 | 1 | +0.0007 | -0.0061 |
| 7 | 4 | 0 | 0% | PASS | 0.725 | 0.888 | 0 | +0.0019 | +0.0027 |

### Track-B discount band (FREEZE ADDENDUM 2: MIXED channels in both)

| round | nuisance cols | mixed | upstream-traced | spurious-alone (lin/gb) | estimator | T_adj | VA_adj | Δ_adj (full set) | Δ_adj (strict, no MIXED) | Δ undiscounted |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 10 | 4 | 5 | 0.590/0.600 | deciles | 0.7816 | 0.6278 | +0.1538 | +0.1456 | +0.1416 |
| 2 | 20 | 7 | 11 | 0.630/0.623 | deciles | 0.7752 | 0.6070 | +0.1683 | +0.1566 | +0.1377 |
| 3 | 33 | 10 | 17 | 0.656/0.651 | matched_sampling | 0.7572 | 0.5796 | +0.1775 | +0.1572 | +0.1327 |
| 4 | 44 | 13 | 22 | 0.664/0.661 | matched_sampling | 0.7617 | 0.5945 | +0.1672 | +0.1432 | +0.1274 |
| 5 | 54 | 17 | 27 | 0.664/0.662 | matched_sampling | 0.7631 | 0.5967 | +0.1664 | +0.1435 | +0.1283 |
| 6 | 64 | 20 | 31 | 0.666/0.663 | matched_sampling | 0.7517 | 0.5870 | +0.1647 | +0.1411 | +0.1290 |
| 7 | 58 | 14 | 27 | 0.659/0.661 | matched_sampling | 0.7703 | 0.6032 | +0.1672 | +0.1580 | +0.1269 |

### Stacked increment (stratification-free control)

| round | AUC(joint B) | AUC(B + dense) | dense increment over all named channels | AUC(bank) | AUC(bank + dense) | dense increment over bank |
|---|---|---|---|---|---|---|
| 1 | 0.5995 | 0.7919 | +0.1924 | 0.6504 | 0.7926 | +0.1421 |
| 2 | 0.6304 | 0.7926 | +0.1623 | 0.6544 | 0.7931 | +0.1387 |
| 3 | 0.6559 | 0.7933 | +0.1374 | 0.6593 | 0.7933 | +0.1339 |
| 4 | 0.6636 | 0.7936 | +0.1300 | 0.6647 | 0.7934 | +0.1287 |
| 5 | 0.6643 | 0.7939 | +0.1295 | 0.6638 | 0.7932 | +0.1294 |
| 6 | 0.6655 | 0.7940 | +0.1285 | 0.6631 | 0.7929 | +0.1298 |
| 7 | 0.6614 | 0.7938 | +0.1324 | 0.6651 | 0.7931 | +0.1279 |

### Spurious map — per-channel alone-AUC (population), upstream parent, MIXED

| channel | alone-AUC | upstream parent | MIXED | first seen |
|---|---|---|---|---|
| Copy-editing regularity | 0.550 | editing help or professional writing experience | yes | r1 |
| Markdown vertical-whitespace density | 0.537 | surface-only | no | r1 |
| External audience footprint | 0.527 | established following or cross-posting through social networks | yes | r1 |
| First-person pronoun count | 0.524 | surface-only | no | r1 |
| Body-part and sensory word count | 0.511 | surface-only | no | r1 |
| Reader-facing boilerplate | 0.507 | surface-only | no | r1 |
| Serial-instalment furniture and pre-assumed world lore | 0.507 | series momentum - an ongoing serial that brings its own returning readers to each instalment | yes | r1 |
| Transcript or log format | 0.499 | surface-only | no | r1 |
| Contest-compliance furniture: word counts and organizer address | 0.496 | participation in a judged or curated challenge, which supplies its own committed entrant-audience and organizer attention | no | r1 |
| Novice/non-native authorial disclaimer | 0.492 | author's actual novice or non-native-speaker background, disclosed directly | yes | r1 |
| Dialogue punctuation mark count | 0.570 | surface-only | no | r2 |
| Editorial polish signature | 0.552 | editing help or professional status | yes | r2 |
| Final-paragraph word count | 0.546 | surface-only | no | r2 |
| Dialogue-first openings | 0.544 | surface-only | no | r2 |
| External-audience promotion | 0.520 | established following, cross-posting, or social-network reach | yes | r2 |
| Borrowed famous characters and pre-existing fictional worlds | 0.510 | cross-posting and fandom audience overlap: the piece arrives with a readership that is already invested in the source | no | r2 |
| Serial continuity cues | 0.497 | series momentum from earlier installments | yes | r2 |
| Explicit novice disclaimer or feedback solicitation | 0.490 | community norm of extending goodwill/leniency to self-identified newcomers -- the disclaimer itself invites sympathetic engagement independent of the prose | no | r2 |
| Text is community apparatus rather than narrative | 0.483 | moderator and curation dynamics (author is staff, or the artefact is machine-generated by the platform) | no | r2 |
| Form mirrors content | 0.481 | unspecified | no | r2 |
| Question-mark count | 0.559 | surface-only | no | r3 |
| Editorial provenance | 0.558 | professional authorship or editing assistance | yes | r3 |
| Stock genre furniture: aliens, HFY, isekai, dragons, capes | 0.557 | the writer's fluency with the sub's current genre fashions, which tracks time spent in the community rather than skill | no | r3 |
| Fragmented lineation | 0.550 | surface-only | no | r3 |
| Sign-off naming a personal subreddit or author handle | 0.540 | author has an established following and a running personal writing sub they are funnelling readers to | yes | r3 |
| Appended author note, thanks, or request for feedback | 0.523 | the writer's habits of audience cultivation and their sense of standing in the community | yes | r3 |
| A speaking presence, judged separately from clean mechanics | 0.518 | unspecified | no | r3 |
| Event-format compliance | 0.511 | moderator or curation dynamics | no | r3 |
| Direct name-check of recognizable franchise/pop-culture figures | 0.508 | surface-only | no | r3 |
| Explicit meta-reference to writing speed or chain handoff | 0.495 | posting timing/speed-to-post premium under the prompt thread and community chain-game participation | no | r3 |
| Concrete-noun token count | 0.495 | surface-only | no | r3 |
| Form enacts the condition it describes | 0.493 | unspecified | no | r3 |
| Interpretive Work Left to the Reader | 0.481 | unspecified | no | r3 |
| Fragmented staccato paragraphing / line-break repetition | 0.582 | surface-only | no | r4 |
| Self-promotional sign-off naming the author's own subreddit or handle | 0.562 | an established personal following / author brand inside the community | yes | r4 |
| Markup punctuation density | 0.547 | surface-only | no | r4 |
| Immediate voice opening | 0.528 | surface-only | no | r4 |
| Proper-noun token count | 0.516 | surface-only | no | r4 |
| Borrowed recognisable characters or franchises doing the characterisation | 0.510 | pre-existing audience attachment to a franchise or figure; fandom pull that operates before the prose is read | no | r4 |
| Event-participation markers | 0.510 | moderator or curation dynamics | no | r4 |
| Repeated-phrase exact-match count | 0.510 | surface-only | no | r4 |
| Series-continuity residue | 0.505 | series momentum from earlier installments | yes | r4 |
| Form matches and amplifies content | 0.491 | unspecified | no | r4 |
| Non-standard spelling/grammar slips (unedited-draft marker) | 0.478 | whether the writer had editing help, used a spellchecker, or gave the piece a revision pass before posting (a proxy for a more professional/experienced production process) | yes | r4 |
| Copyedit-level consistency | 0.553 | professional status or access to editing help | yes | r5 |
| Paragraph-count tally | 0.548 | surface-only | no | r5 |
| Sentence-final ellipsis count | 0.542 | surface-only | no | r5 |
| Author self-promotion footer plugging a personal subreddit | 0.541 | the author has an established personal following they are actively recruiting to | yes | r5 |
| Dialogue-first opening within first two lines | 0.532 | surface-only | no | r5 |
| Direct-address paratext / meta notes outside the fiction | 0.507 | author's degree of habituation to this community's posting conventions (how many times they have posted here before) | yes | r5 |
| Institutional community framing | 0.506 | moderator or curation dynamics | no | r5 |
| Distinctive narrating idiom, scored with mechanics disregarded | 0.496 | unspecified | no | r5 |
| Document or chat framing | 0.486 | surface-only | no | r5 |
| Self-positioning disclaimers | 0.486 | author reputation or seniority in the community | yes | r5 |
| Closing self-promotion block naming the author's own subreddit | 0.541 | author has an established following and a personal subreddit, i.e. seniority plus an audience that travels with them | yes | r6 |
| Immediate voice mode | 0.534 | surface-only | no | r6 |
| Mean sentence-length variance | 0.517 | surface-only | no | r6 |
| Fashionable subgenre trope vocabulary (LitRPG stats, dragon-rider tropes) | 0.508 | surface-only | no | r6 |
| Decorative markup and divider load | 0.507 | surface-only | no | r6 |
| Distinct-adjective type count | 0.500 | surface-only | no | r6 |
| Continuation and prior-context dependence | 0.494 | series momentum from earlier instalments | yes | r6 |
| Author's explicit self-deprecating hedge about experience or quality | 0.491 | author's tenure/experience level in creative writing generally or in this community specifically | no | r6 |
| Non-story furniture and editor markup debris inside the body | 0.491 | surface-only | no | r6 |
| Blank-line and short-paragraph density | 0.577 | decomposed component (surface half) | no | r7 |
| Mechanical slip count (spellchecker-catchable surface errors) | 0.543 | decomposed component (surface half) | no | r7 |

### Fleet missing mass (Good-Turing, blind full-recall species)

| track | round | P | families | N | S_obs | f1 | f2 | M̂ | jackknife M̂ [min,max] | cross-proposer recapture | remaining AUC (odds form) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A | round1 | 4 | 2 | 60 | 35 | 18 | 12 | 0.300 | [0.378, 0.511] | 0.49 | +0.0011 |
| A | round2 | 4 | 2 | 60 | 35 | 20 | 8 | 0.333 | [0.289, 0.578] | 0.43 | +0.0012 |
| A | round3 | 4 | 2 | 60 | 38 | 23 | 8 | 0.383 | [0.311, 0.556] | 0.39 | +0.0015 |
| A | round4 | 4 | 2 | 60 | 39 | 23 | 11 | 0.383 | [0.400, 0.644] | 0.41 | +0.0015 |
| A | round5 | 4 | 2 | 60 | 34 | 17 | 11 | 0.283 | [0.333, 0.444] | 0.50 | +0.0010 |
| A | round6 | 4 | 2 | 60 | 37 | 22 | 9 | 0.367 | [0.311, 0.578] | 0.41 | +0.0014 |
| B | round1 | 4 | 2 | 40 | 18 | 8 | 2 | 0.200 | [0.133, 0.300] | 0.56 | +0.0006 |
| B | round2 | 4 | 2 | 40 | 18 | 6 | 4 | 0.150 | [0.133, 0.300] | 0.67 | +0.0004 |
| B | round3 | 4 | 2 | 40 | 25 | 13 | 9 | 0.325 | [0.367, 0.600] | 0.48 | +0.0012 |
| B | round4 | 4 | 2 | 40 | 25 | 17 | 4 | 0.425 | [0.433, 0.567] | 0.32 | +0.0018 |
| B | round5 | 4 | 2 | 40 | 21 | 10 | 6 | 0.250 | [0.200, 0.467] | 0.52 | +0.0008 |
| B | round6 | 4 | 2 | 40 | 18 | 6 | 5 | 0.150 | [0.133, 0.300] | 0.67 | +0.0004 |
