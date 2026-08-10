"""
Shared task taxonomy — single source of truth for per-task work definitions
and noise patterns. Imported by:
  - extract_rubric_features_v5_prompt.py (Llama extractor)
  - classify_rubric_llama_prompt.py (Llama classifier)
  - classify_rubric_v2_prompt.py (gpt-5-mini classifier)

When you discover a new task-specific anti-pattern or service-policy noise
type, add it HERE — all three prompt files will pick it up automatically.

Each task entry has five fields, all lists of strings:
  - work: 1-element list with the work-of-this-task description
  - meta_artifacts: derived objects that should be DROPped (review of the work,
    certificate, score sheet, etc.) — NOT bibliographies/citations/sections
    inside the work itself, which are PART OF the work.
  - service_or_logistics: pricing, scheduling, transactional, internal-process
    items that should be DROPped.
  - actor_attribute: biographies, qualifications, hiring criteria of the
    people producing/evaluating the work — DROPped.
  - inputs_examples: 2-3 task-specific worked examples showing what good
    `inputs` look like for rubrics in this domain.
  - axis_examples: 2-3 task-specific worked examples showing how to label
    rubrics in this domain on the (verifiability_type, tractability, specificity)
    axes.
  - verifiability_inline: dict {verifiability_type: short_example_phrase} —
    fills the {EX_*} slots in SCHEMA_HINT so each task's prompt has its OWN
    inline example per verifiability_type. Provide a phrase per type that is
    domain-appropriate (or skip a type if it rarely occurs in this task —
    build_prompt_for_task supplies a generic fallback).

To keep the lists usable in both bullet and semicolon-list formats, each
string should be a self-contained noun phrase (no leading dash or bullet).
"""

TASK_INFO: dict[str, dict[str, list[str]]] = {

    "creative-writing": {
        "work": [
            "the creative work — a novel, short story, poem, screenplay, etc."
        ],
        "meta_artifacts": [
            "the *review of the book* (length, format, distribution policy of the review — e.g., 'review lengths by option' or 'reviews are 250-300 words' constrain the *review*, NOT the book; this is the canonical Kirkus over-extraction failure)",
            "an *award citation* honoring the work (e.g., Caldecott Medal citation)",
            "*editorial notes* sent back to the author by Kirkus/an editor as a paid service deliverable",
            "the *blurb* a publisher writes for the back cover (about the book, not the book itself)",
        ],
        "service_or_logistics": [
            "paid-review pricing tiers (e.g., Standard $425 / Express $575)",
            "review turnaround time (e.g., 7-9 weeks standard, 4-6 weeks express)",
            "publication-control of the review (\"you may choose not to publish a negative review\")",
            "how Kirkus matches a book to a reviewer (internal staffing/triage)",
            "submission-portal mechanics, login, account management for the review service",
            "Project Gutenberg license clauses (royalty/refund/warranty/AS-IS/license-preservation boilerplate) appearing on republished public-domain works",
        ],
        "actor_attribute": [
            "reviewer biographies and credentials (\"our reviewers are librarians, journalists, and creative executives with 10+ years experience\")",
            "reviewer pool size",
            "editorial board composition statements",
        ],
        "inputs_examples": [
            'Three-Act Structure → inputs = ["location of the first plot point (% of total length)", "location of the midpoint", "location of the third plot point", "presence and timing of pinch points", "strength of the opening hook"]',
            'Honest and impartial evaluation → inputs = ["the reviewer\'s potential biases toward the author / genre / marketplace pressures", "the book\'s actual quality independent of those biases"]',
            'Compelling characters → inputs = ["the protagonist\'s interiority", "the antagonist\'s coherent motivation", "secondary-character roundness"]',
        ],
        "axis_examples": [
            'Three-Act Structure (with explicit plot-point percentages) → verifiability=computational, tractability=programmatic_check, specificity=hyper_specific (concrete beat positions can be scored by inspecting the manuscript structure)',
            'Compelling characters → verifiability=normative, tractability=llm_judge, specificity=general (a careful educated reader can score character roundness; defaults to llm_judge, NOT expert_judgment)',
            'Submissions must be ≤5000 words → verifiability=computational, tractability=programmatic_check, specificity=hyper_specific (a word counter is a Python function)',
        ],
        "verifiability_inline": {
            "computational": "e.g., word count of the manuscript; PDF format check",
            "consistency":   "e.g., character backstory in chapter 1 matches chapter 8",
            "procedural":    "e.g., submission filed before the contest deadline",
            "completeness":  "e.g., the application packet includes all required components",
            "pragmatic":     "e.g., does the opening hook engage readers",
            "normative":     "e.g., compelling characters; elegant prose; clear voice",
        },
        "tractability_inline": {
            "programmatic_check": "e.g., word count check; format conformance",
            "llm_judge": "e.g., clear voice; compelling characters; coherent plot (any careful reader can score)",
            "expert_judgment": "e.g., poetic meter and prosody (need craft training); narrative voice consistency in literary fiction",
            "intractable": "e.g., is this a 'good' work of literature in absolute terms (cross-cultural and faultless-disagreement-prone)",
        },
        "specificity_inline": {
            "vague": "e.g., 'be clear'; 'write with style'; 'be original'",
            "general": "e.g., 'compelling characters'; 'show don't tell'",
            "specific": "e.g., 'three-act structure with first plot point ~25%'; 'avoid passive voice in action scenes'",
            "hyper_specific": "e.g., '≤5000 words'; '12pt double-spaced PDF'; 'opening hook within first 250 words'",
        },
    },

    "peer-review": {
        "work": [
            "the manuscript being reviewed (when the page audience is reviewers/editors) OR the peer review report itself (when the page is about review-writing standards). Use the page audience to disambiguate."
        ],
        "meta_artifacts": [
            "the *editorial decision letter* (accept/reject communication)",
            "the *published peer-review file* (when a journal publishes review history alongside the paper)",
            "*reviewer scores / numeric ratings* assigned to a paper",
            "the *editor's transparency report* about a paper's review timeline",
        ],
        "service_or_logistics": [
            "internal reviewer-allocation procedures (\"we assign two reviewers per submission\"; \"Phase 2 reviewers can't see Phase 1 reviews until they submit\")",
            "submission-portal mechanics (Easychair, OpenReview, ScholarOne navigation)",
            "internal editorial Slack/email consensus procedures for ambiguous cases",
            "conference vs journal review-management timelines (when stated as platform deadlines)",
            "service performance metrics (submission rate, time-to-publication, average reviewers per manuscript)",
            "Project Gutenberg license clauses (royalty/refund/warranty/AS-IS/license-preservation boilerplate) — these appear on republished historical scientific texts",
        ],
        "actor_attribute": [
            "reviewer hiring criteria (\"reviewers must hold a PhD\")",
            "editorial board composition and member biographies",
            "AE (associate editor) qualifications",
        ],
        "inputs_examples": [
            'First-page content and anonymity → inputs = ["the title placement on the first page", "abstract placement", "whether content areas/ID are listed", "whether author names/affiliations are absent"]',
            'Methodology rigor → inputs = ["the experimental design", "controls and confounders addressed", "statistical analysis appropriateness", "reproducibility of methods"]',
            'Honest and impartial review → inputs = ["the reviewer\'s potential biases (advisor relationships, competing labs, citation politics)", "the paper\'s actual merit independent of those social pressures"]',
        ],
        "axis_examples": [
            'First-page anonymity (no names/affiliations) → verifiability=computational, tractability=programmatic_check, specificity=hyper_specific (regex over the first page can verify)',
            'Novelty / contribution → verifiability=normative, tractability=expert_judgment, specificity=general (peer reviewers consistently score novelty in their field — NOT intractable)',
            'All cited works must exist → verifiability=factual, tractability=programmatic_check, specificity=specific (lookup against Semantic Scholar / DOI registry)',
        ],
        "verifiability_inline": {
            "computational": "e.g., first page has title + abstract + ID but not author names (regex)",
            "factual":       "e.g., every citation resolves in Semantic Scholar / DOI registry",
            "consistency":   "e.g., claims in the abstract match the results section",
            "procedural":    "e.g., IRB approval documented; pre-registration linked",
            "statistical":   "e.g., p-values recalculable from reported test statistics + df",
            "causal":        "e.g., the proposed mechanism explains the observed effect",
            "completeness":  "e.g., revision addresses every reviewer concern from round 1",
            "normative":     "e.g., novelty; significance; clarity of contribution",
        },
        "tractability_inline": {
            "programmatic_check": "e.g., first-page anonymization (regex); citation count threshold",
            "llm_judge": "e.g., abstract claims match results; argument is internally consistent",
            "expert_judgment": "e.g., novelty of contribution within the subfield; appropriateness of statistical test",
            "intractable": "e.g., 'is this paper worth publishing in any venue?' (faultless disagreement common)",
        },
        "specificity_inline": {
            "vague": "e.g., 'be rigorous'; 'be novel'",
            "general": "e.g., 'methodology is appropriately rigorous'; 'novelty is clearly motivated'",
            "specific": "e.g., 'IRB approval documented'; 'first page: title + abstract + ID; no author names'",
            "hyper_specific": "e.g., '≤9 pages including references'; '11pt single-column'; 'p < 0.05 with reported test statistics'",
        },
    },

    "math-stackexchange": {
        "work": [
            "a math question/answer post (or, for contest-criteria pages, a competition solution / proof)."
        ],
        "meta_artifacts": [
            "post *upvotes / downvotes / score*",
            "*comments* attached to a post",
            "*badges* awarded to authors",
            "the *accepted-answer marker* the asker assigns",
        ],
        "service_or_logistics": [
            "Stack Exchange site features (login, reputation system, rate limits, captcha, edit-window mechanics)",
            "Project Gutenberg license clauses appearing on republished historical math texts (royalty / refund / warranty disclaimers)",
            "competition logistics (registration, submission portal, fee, deadline)",
            "MathJax / KaTeX rendering configuration",
        ],
        "actor_attribute": [
            "moderator qualifications",
            "contestant eligibility statements (\"must be a US college student\")",
            "Putnam committee member biographies",
        ],
        "inputs_examples": [
            'Proof / rigorous justification → inputs = ["whether each step of the proof is justified vs. asserted", "whether the problem requires existence vs. explicit construction", "whether the stated counterexample actually refutes the claim"]',
            'Answer-format conformity → inputs = ["the specific output form requested (rational number, polynomial, closed-form, etc.)", "whether the answer matches the form exactly"]',
            'Difficulty labeling (A1/B1) → inputs = ["the problem\'s position in the A1/B1 difficulty bucket", "the typical solution length expected for that bucket"]',
        ],
        "axis_examples": [
            'Answer-format conformity ("express as rational number r") → verifiability=computational, tractability=programmatic_check, specificity=hyper_specific (regex / parser checks output form)',
            'Proof / rigorous justification → verifiability=normative, tractability=expert_judgment, specificity=general (a mathematician judges whether each step is justified)',
            'Elegant proof → verifiability=normative, tractability=expert_judgment, specificity=vague (mathematicians DO consistently identify elegance — NOT intractable)',
        ],
        "verifiability_inline": {
            "computational": "e.g., answer in the specified form (a/b, polynomial, closed-form) — parseable",
            "factual":       "e.g., cited theorem appears in the named reference",
            "consistency":   "e.g., later steps of the proof don't contradict earlier ones",
            "completeness":  "e.g., proof covers all cases the problem stipulates",
            "pragmatic":     "e.g., solution actually solves the stated problem (run the computation)",
            "normative":     "e.g., elegance of the proof; intuitive explanation",
        },
        "tractability_inline": {
            "programmatic_check": "e.g., answer in specified form (parseable); citation to a real theorem",
            "llm_judge": "e.g., proof steps are each justified vs asserted; argument is clear",
            "expert_judgment": "e.g., is the proof elegant; is the technique novel for this class of problem",
            "intractable": "e.g., 'is this the canonical proof of the theorem' (mathematicians genuinely disagree)",
        },
        "specificity_inline": {
            "vague": "e.g., 'be rigorous'; 'show your work'",
            "general": "e.g., 'each step of the proof must be justified'",
            "specific": "e.g., 'A1/B1 difficulty marker'; 'find with proof a set S for which the minimum k is achieved'",
            "hyper_specific": "e.g., 'express the answer in the form a! b! c! / n!'; 'find the largest real R such that...'",
        },
    },

    "news-homepages": {
        "work": [
            "a news article / story / homepage layout / journalistic piece."
        ],
        "meta_artifacts": [
            "the *editor's note* attached to a published story",
            "*reader comments* on the article",
            "*award citations* honoring the journalism (Pulitzer, Polk citation text)",
            "*reprint permissions* metadata",
        ],
        "service_or_logistics": [
            "subscription / paywall mechanics",
            "ad placement and homepage CMS configuration",
            "press-pass application procedures",
            "subscription-driving potential or other business KPIs",
            "Project Gutenberg license clauses (royalty/refund/warranty/AS-IS/license-preservation boilerplate) — these appear on republished historical journalism / journalism-history texts",
        ],
        "actor_attribute": [
            "journalist hiring criteria",
            "editor biographies",
            "fact-checker organizational chart",
        ],
        "inputs_examples": [
            'Commitment to Nonpartisanship and Fairness → inputs = ["the consistency of the fact-checking standard across political actors", "evidence of fact-check concentration on one side", "whether the same process was applied to each claim"]',
            'Accuracy of reporting → inputs = ["source verification for each factual claim", "internal consistency of dates/numbers/quotes", "presence of corrections for prior errors"]',
            'Transparency of Sources → inputs = ["whether each source is identified to the reader", "whether enough detail is given to replicate the verification", "redactions made for source safety with justification"]',
        ],
        "axis_examples": [
            'Source verification for each factual claim → verifiability=factual, tractability=llm_judge, specificity=general (claim-by-claim lookup against authoritative sources)',
            'Commitment to Nonpartisanship → verifiability=procedural, tractability=expert_judgment, specificity=specific (auditors check process consistency across political actors)',
            'Newsworthy / will-readers-care → verifiability=pragmatic, tractability=expert_judgment, specificity=vague (editors consistently judge newsworthiness; downstream engagement also measurable)',
        ],
        "verifiability_inline": {
            "computational": "e.g., headline word count; image-credit format",
            "factual":       "e.g., named source actually said the quoted statement",
            "consistency":   "e.g., headline matches the lede",
            "procedural":    "e.g., correction posted per house style after error",
            "statistical":   "e.g., poll methodology adequate for stated confidence interval",
            "causal":        "e.g., did the policy cause the outcome the article describes",
            "completeness":  "e.g., article addresses all WP-style ledes (who/what/where/when/why)",
            "pragmatic":     "e.g., did the article drive subscriber engagement",
            "normative":     "e.g., nonpartisanship; fairness; clarity",
        },
        "tractability_inline": {
            "programmatic_check": "e.g., headline word count; image-credit format check",
            "llm_judge": "e.g., headline matches the lede; the story's framing is non-sensationalist",
            "expert_judgment": "e.g., did the journalist apply Reuters-style sourcing standards; statistical methodology adequate",
            "intractable": "e.g., 'is this story newsworthy' (across audiences; varies by context)",
        },
        "specificity_inline": {
            "vague": "e.g., 'tell the truth'; 'be fair'",
            "general": "e.g., 'avoid sensationalism'; 'verify each source'",
            "specific": "e.g., 'IFCN nonpartisanship commitment'; 'corrections posted per house style'",
            "hyper_specific": "e.g., 'headline ≤80 chars'; 'two independent sources required for any anonymous-source claim'",
        },
    },

    "press-releases": {
        "work": [
            "a press release / science-communication piece / regulatory disclosure communication."
        ],
        "meta_artifacts": [
            "*media coverage* (the news story written in response to the press release)",
            "*social-media engagement metrics* on a release",
            "the *agency response document* to a press inquiry",
            "*pickup reports* from PR-distribution wires",
        ],
        "service_or_logistics": [
            "PR Newswire / Business Wire pricing and distribution timing",
            "embargo administration mechanics",
            "press-list / journalist-contact-database access fees",
            "agency website's news-distribution feature (timely public updates, point-of-contact info)",
            "Project Gutenberg license clauses (royalty/refund/warranty/AS-IS/license-preservation boilerplate) — these appear on republished historical PR or rhetoric textbooks",
        ],
        "actor_attribute": [
            "PR firm staff biographies",
            "communications-officer hiring criteria",
            "spokesperson media training credentials",
        ],
        "inputs_examples": [
            'Define your audience → inputs = ["audience demographic profile (age, profession, online habits)", "the producer\'s communication goal", "the affordances of each candidate platform"]',
            'Definition of "intentional" disclosure (Reg FD) → inputs = ["the discloser\'s knowledge of the information\'s materiality", "the discloser\'s recklessness about whether the information is nonpublic"]',
            'Definition of "promptly" → inputs = ["the time elapsed since a senior official learned of the disclosure", "the next NYSE trading-day start time", "whether 24 hours has passed"]',
        ],
        "axis_examples": [
            'Definition of "promptly" (24-hour rule) → verifiability=computational, tractability=programmatic_check, specificity=hyper_specific (timestamp comparison)',
            'Definition of "intentional" disclosure (knowledge or recklessness) → verifiability=normative, tractability=expert_judgment, specificity=specific (named legal scienter standard; SEC enforcement attorney judges)',
            'Will the press release engage readers / drive coverage → verifiability=pragmatic, tractability=llm_judge, specificity=general (downstream engagement metrics are objective; LLM can pre-rate)',
        ],
        "verifiability_inline": {
            "computational": "e.g., headline ≤ X chars; embargo timestamp comparison",
            "factual":       "e.g., quote attributed to a person who actually said it",
            "consistency":   "e.g., spokesperson name in headline matches the byline attribution",
            "procedural":    "e.g., Form 8-K filed within 24 hours of selective disclosure",
            "completeness":  "e.g., release includes all stakeholder perspectives the topic requires",
            "pragmatic":     "e.g., did the release drive media pickup / coverage",
            "normative":     "e.g., authenticity of voice; appropriate tone",
        },
        "tractability_inline": {
            "programmatic_check": "e.g., headline character limit; embargo timestamp comparison",
            "llm_judge": "e.g., release identifies main audiences; voice is authentic",
            "expert_judgment": "e.g., Reg FD intentional-disclosure scienter; meets SEC materiality standard",
            "intractable": "e.g., 'will this release succeed for the brand in the long term'",
        },
        "specificity_inline": {
            "vague": "e.g., 'be authentic'; 'engage readers'",
            "general": "e.g., 'discuss costs of the treatment'; 'use plain language'",
            "specific": "e.g., 'Form 8-K filed within 24 hours'; 'embargo respected for medical-journal coordination'",
            "hyper_specific": "e.g., 'headline ≤100 chars'; '24-hour or next-NYSE-trading-day promptness rule'",
        },
    },

    "code-review": {
        "work": [
            "the code change / pull request / commit being reviewed (or, for design-principle pages, the codebase itself)."
        ],
        "meta_artifacts": [
            "the *review comment thread* (length, tone of comments, who can resolve them)",
            "the *merge approval* / status badge",
            "*CI pipeline reports* attached to the PR",
            "*coverage reports* generated alongside the PR",
        ],
        "service_or_logistics": [
            "GitHub / GitLab UI features (\"how to enable required reviewers\")",
            "CI pipeline configuration and runner cost",
            "IDE plugin installation instructions",
            "code-review SaaS pricing tiers",
            "linter/formatter installation steps (the tool itself, not the rules it enforces)",
        ],
        "actor_attribute": [
            "reviewer hiring criteria (\"senior engineers must approve database migrations\")",
            "team composition policies",
            "on-call rotation membership rules",
        ],
        "inputs_examples": [
            'Resources should represent a single API object → inputs = ["the underlying API\'s object boundaries", "the resource\'s CRUD method set", "whether the resource abstracts multiple API objects"]',
            'DRY Principle → inputs = ["repeated code blocks across files", "extractable utility functions", "candidate abstractions to introduce"]',
            'Error Handling → inputs = ["enumeration of failure modes", "the user-facing error message clarity", "whether exceptions propagate vs. are caught locally"]',
        ],
        "axis_examples": [
            'PR must include unit tests for changed lines → verifiability=procedural, tractability=programmatic_check, specificity=specific (compliance check; coverage tool runs)',
            'PR must actually fix the bug → verifiability=pragmatic, tractability=programmatic_check, specificity=specific (regression test fails before, passes after)',
            'DRY Principle → verifiability=normative, tractability=llm_judge, specificity=general (an LLM or senior engineer can identify repetition with high agreement)',
        ],
        "verifiability_inline": {
            "computational": "e.g., type-check passes; lint rule satisfied",
            "factual":       "e.g., referenced API actually exists in the SDK docs",
            "consistency":   "e.g., PR description matches the diff",
            "procedural":    "e.g., PR template filled; tests included for changed lines",
            "completeness":  "e.g., all test paths covered for the new function",
            "pragmatic":     "e.g., the PR actually fixes the bug (regression test green)",
            "normative":     "e.g., DRY principle; readable variable names; SOLID design",
        },
        "tractability_inline": {
            "programmatic_check": "e.g., type-check passes; lint rule satisfied; test coverage ≥80%",
            "llm_judge": "e.g., DRY principle followed; variable names are readable; PR description matches diff",
            "expert_judgment": "e.g., architectural impact of the change; whether it introduces security risk",
            "intractable": "e.g., 'is this codebase well-designed in absolute terms'",
        },
        "specificity_inline": {
            "vague": "e.g., 'write good code'; 'be maintainable'",
            "general": "e.g., 'DRY principle'; 'avoid magic numbers'; 'handle errors gracefully'",
            "specific": "e.g., 'PR template filled including issue link'; 'unit tests for every changed line'",
            "hyper_specific": "e.g., 'test coverage ≥90% for new code'; '11pt monospace formatting'",
        },
    },

    "grant-funding": {
        "work": [
            "a grant application / research proposal (claims, budget, methodology, team rationale)."
        ],
        "meta_artifacts": [
            "the *reviewer score sheet* / overall impact score",
            "the *summary statement* returned to applicants",
            "the *award notice / decision letter*",
            "*pink-sheet* reviewer comments",
        ],
        "service_or_logistics": [
            "submission-portal mechanics (eRA Commons, FastLane, Grants.gov navigation)",
            "deadlines (LOI date, full-application date, committee meeting date)",
            "budget-form templates and indirect-cost calculators",
            "Copyright Disclosure Forms / IP-reporting forms",
            "privacy/data-handling policies for application materials",
        ],
        "actor_attribute": [
            "PI eligibility (\"must hold a PhD\", \"must be a fully independent researcher with own lab space\")",
            "co-investigator hiring criteria",
            "institutional eligibility (\"applicant must be a US-based 501(c)(3)\")",
            "PI accountability / mentorship-program participation",
        ],
        "inputs_examples": [
            'Independence of Thought → inputs = ["evidence of intellectual ability in application essays", "novelty of research direction", "cross-cultural perspective evident in past work"]',
            'Significance (NIH R01) → inputs = ["importance of the problem the project addresses", "the scientific premise\'s strength", "potential for the research to drive the field forward"]',
            'Approach (NIH R01) → inputs = ["rigor of experimental design", "appropriateness of methods for the aims", "feasibility of timeline", "consideration of alternatives"]',
        ],
        "axis_examples": [
            'Budget within indirect-cost cap → verifiability=computational, tractability=programmatic_check, specificity=hyper_specific (parse budget; compare to threshold)',
            'Significance (NIH R01) → verifiability=normative, tractability=expert_judgment, specificity=general (peer reviewers in the study section consistently score significance)',
            'Independence of Thought (Knight-Hennessy) → verifiability=normative, tractability=expert_judgment, specificity=specific (a named criterion with its own definition; admissions committee scores it)',
        ],
        "verifiability_inline": {
            "computational": "e.g., budget within indirect-cost cap; page-limit check",
            "factual":       "e.g., PI ORCID resolves; applicant institution is 501(c)(3)",
            "consistency":   "e.g., Specific Aims match what's described in Approach",
            "procedural":    "e.g., IRB approval documented before submission",
            "statistical":   "e.g., proposed sample size adequate for stated effect",
            "completeness":  "e.g., all required sections (Significance/Innovation/Approach/Environment) present",
            "normative":     "e.g., significance; innovation; investigator quality; team strength",
        },
        "tractability_inline": {
            "programmatic_check": "e.g., budget within indirect-cost cap; page-limit check; PI institution is 501(c)(3) (DB lookup)",
            "llm_judge": "e.g., Specific Aims are concrete and distinct; Broader Impacts are clearly articulated",
            "expert_judgment": "e.g., scientific significance within the field; statistical-power adequacy of proposed design",
            "intractable": "e.g., 'will this PI succeed if funded' (faultless disagreement between reviewers common)",
        },
        "specificity_inline": {
            "vague": "e.g., 'be significant'; 'be innovative'",
            "general": "e.g., 'methodology is rigorous'; 'Specific Aims are distinct'",
            "specific": "e.g., 'Knight-Hennessy Independence of Thought'; 'NIH R01 Significance criterion'",
            "hyper_specific": "e.g., 'budget caps at $250K direct/year'; '≤12 pages for Research Strategy'",
        },
    },

    "humor": {
        "work": [
            "a comedic piece — a stand-up routine, joke, sketch, satirical essay, comic strip, or comedy performance."
        ],
        "meta_artifacts": [
            "*audience reaction* (laughter measurements, applause meters)",
            "*reviews / critic ratings* of comedy specials",
            "*award citations* (Mark Twain Prize, Edinburgh Comedy Award, etc.)",
            "*Variety / Hollywood Reporter writeups* of a special",
        ],
        "service_or_logistics": [
            "comedy-club booking procedures",
            "festival submission deadlines (e.g., Edinburgh Fringe deadlines)",
            "talent-agency representation contracts",
            "open-mic sign-up procedures",
            "ticket pricing and box-office mechanics",
            "Project Gutenberg license clauses (royalty/refund/warranty/AS-IS/license-preservation boilerplate) — these appear on republished historical comedy texts (Holmes, Meredith, Artemus Ward)",
        ],
        "actor_attribute": [
            "performer biographies (\"the comedian has 15 years of experience\")",
            "festival-programming staff credentials",
            "writers-room hiring criteria for variety shows",
        ],
        "inputs_examples": [
            'Clear ironic persona → inputs = ["consistency of the persona\'s traits across the bit", "tone/language cues distinguishing persona from performer", "presence of irony markers (contradictions, exaggeration)"]',
            'Reveals something true about the targeted bigotry → inputs = ["the systemic or logical flaw the satire exposes", "whether the audience experiences recognition/catharsis", "whether the target is the bigoted system rather than the oppressed group"]',
            'Irony detectable to the intended audience → inputs = ["the sophistication of the intended audience", "the strength of irony cues (setup, contradiction)", "the risk of literal misreading"]',
        ],
        "axis_examples": [
            'Clear ironic persona (consistent character traits) → verifiability=normative, tractability=expert_judgment, specificity=general (comedy critics consistently identify persona-clarity)',
            'Is this funny / will the joke land → verifiability=pragmatic, tractability=intractable, specificity=vague (cross-audience humor genuinely varies; downstream laughter measurable but expert pre-judgment unreliable)',
            'Reveals something true about the targeted bigotry → verifiability=normative, tractability=expert_judgment, specificity=specific (named craft criterion in satire-analysis; cultural critics can consistently score)',
        ],
        "verifiability_inline": {
            "factual":       "e.g., historical reference in the joke is accurate",
            "consistency":   "e.g., persona traits stay consistent across the bit",
            "pragmatic":     "e.g., did the audience laugh (measured by recording or live response)",
            "normative":     "e.g., clear ironic persona; originality of the bit",
        },
        "tractability_inline": {
            "programmatic_check": "e.g., joke length / punctuation conformance (rare)",
            "llm_judge": "e.g., is the punchline clearly the subversion; is the setup-payoff structure present",
            "expert_judgment": "e.g., comedy craft analysis (timing, persona-clarity, satirical target precision)",
            "intractable": "e.g., 'is this joke funny' (varies across cultures and audiences; faultless disagreement)",
        },
        "specificity_inline": {
            "vague": "e.g., 'be funny'; 'have a clear voice'",
            "general": "e.g., 'clear ironic persona'; 'subvert expectation'",
            "specific": "e.g., 'reveals something true about the targeted bigotry'; 'persona-proximity insulation'",
            "hyper_specific": "e.g., 'punchline ≤7 words'; 'rule of three: establish-reinforce-surprise'",
        },
    },

    "legal-outcome-prediction": {
        "work": [
            "a legal brief, motion, court filing, or appellate argument."
        ],
        "meta_artifacts": [
            "the *court order / opinion* issued in response to the brief",
            "the *case caption / docket entry* generated by the clerk",
            "*reporter citations* (e.g., 547 F.3d 962) assigned to a decided case",
            "*headnotes* added by the reporter publisher",
        ],
        "service_or_logistics": [
            "PACER / e-filing portal mechanics",
            "court fee schedules (filing fees, motion fees)",
            "hearing scheduling rules (Rule 78 procedures for setting hearings)",
            "deadlines for submitting comments on regulatory information collections",
            "PTAB precedential-decision designation procedures (who approves a decision as precedential)",
            "Project Gutenberg license clauses (royalty/refund/warranty/AS-IS/license-preservation boilerplate) — these appear on republished historical legal treatises",
        ],
        "actor_attribute": [
            "judicial appointment / continuity rules (\"judges remain in office until successors are appointed\")",
            "attorney-bar-admission requirements",
            "judicial qualifications (must be licensed attorney, X years experience)",
        ],
        "inputs_examples": [
            'Statement of the Case (FRAP 28(a)(6)) → inputs = ["narrative completeness of the facts section", "relevance of each fact to the legal theory", "appendix-page citations attached to each factual assertion"]',
            'Citations to the appendix (formatting) → inputs = ["the format of each appendix citation", "consistency with the Fourth Circuit Appendix Pagination & Brief Citation Guide"]',
            'Argument quality → inputs = ["the legal theory\'s grounding in precedent", "the chain of authorities cited", "responsiveness to opposing arguments"]',
        ],
        "axis_examples": [
            'Citations to appendix (specific format guide) → verifiability=computational, tractability=programmatic_check, specificity=hyper_specific (regex / parser against the citation guide)',
            'Plaintiff must establish but-for causation → verifiability=causal, tractability=expert_judgment, specificity=specific (named legal element; trial judge applies)',
            'Statement of the Case (FRAP 28(a)(6)) — facts cite appendix pages → verifiability=consistency, tractability=llm_judge, specificity=specific (cross-check each factual claim against an appendix-page citation)',
        ],
        "verifiability_inline": {
            "computational": "e.g., brief within page limit; FRAP 32 typeface compliance",
            "factual":       "e.g., cited statute text actually says what the brief quotes",
            "consistency":   "e.g., Statement of Facts is consistent with the appendix pages",
            "procedural":    "e.g., brief filed before deadline; service certificate included",
            "causal":        "e.g., plaintiff established but-for causation",
            "completeness":  "e.g., brief addresses every element of the claim",
            "pragmatic":     "e.g., did the brief persuade the court (case outcome)",
            "normative":     "e.g., quality of legal argument; doctrinal coherence",
        },
        "tractability_inline": {
            "programmatic_check": "e.g., brief within page limit (FRAP 32 compliance); citation format compliance",
            "llm_judge": "e.g., Statement of Facts cites the appendix consistently; argument is internally non-contradictory",
            "expert_judgment": "e.g., doctrinal coherence (but-for causation, statutory interpretation); whether the argument is dispositive",
            "intractable": "e.g., 'is this brief persuasive in absolute terms' (depends on the judge's priors)",
        },
        "specificity_inline": {
            "vague": "e.g., 'be persuasive'; 'write clearly'",
            "general": "e.g., 'cite all relevant authorities'; 'address opposing arguments'",
            "specific": "e.g., 'Statement of the Case per FRAP 28(a)(6)'; 'plaintiff must establish but-for causation'",
            "hyper_specific": "e.g., '350-word limit for Rule 28(j) letters'; '12pt Century-style typeface, FRAP 32'",
        },
    },

    "notice-and-comment": {
        "work": [
            "a comment letter on a proposed rule, OR a petition for rulemaking, OR a substantive notice-and-comment guide describing what makes a good comment."
        ],
        "meta_artifacts": [
            "the *agency response document* (\"comments received and our responses\")",
            "the *rulemaking docket entry* (regulations.gov record)",
            "the *Federal Register publication* of the final rule",
            "*petition disposition* notices",
        ],
        "service_or_logistics": [
            "comment portal mechanics (regulations.gov navigation)",
            "comment-period deadlines (\"comments must be received by February 27, 2026\")",
            "Federal Register filing format requirements",
            "internal agency processes for handling fraudulent comment campaigns",
            "list/heading entries from the Code of Federal Regulations (e.g., \"§ 1910.179 Overhead and gantry cranes\")",
            "Project Gutenberg license clauses (royalty/refund/warranty/AS-IS/license-preservation boilerplate) — these appear on republished historical political/regulatory texts (Federalist Papers, Goodnow, Wilson)",
        ],
        "actor_attribute": [
            "petitioner organizational status (must be a registered association)",
            "agency-rulemaking-staff biographies",
        ],
        "inputs_examples": [
            'Full factual and legal basis → inputs = ["the facts supporting the requested action", "legal authorities cited", "data presented (research reports, statistics)", "information unfavorable to the petitioner that is disclosed"]',
            'Supporting data standards → inputs = ["whether each research report meets peer-review-journal presentation standards", "the distribution of source types (journals, industry, government statistics)"]',
            'Full statement of action requested → inputs = ["the precise text of the proposed rule or amendment", "the statutory authority cited", "the existing rule the petition would modify"]',
        ],
        "axis_examples": [
            'Agency must respond to all substantive comments → verifiability=completeness, tractability=expert_judgment, specificity=specific (cross-reference comment-set against agency response document; APRM lawyers judge "substantiveness")',
            'Comments deadline (Feb 27, 2026) → verifiability=procedural, tractability=programmatic_check, specificity=hyper_specific (timestamp check)',
            'Petition must include statutory authority cited → verifiability=factual, tractability=programmatic_check, specificity=specific (lookup against U.S. Code; the petition either cites a real statute or doesn\'t)',
        ],
        "verifiability_inline": {
            "computational": "e.g., comment submitted in a valid file format",
            "factual":       "e.g., comment cites a real provision of the proposed rule (lookup CFR)",
            "consistency":   "e.g., comment's substantive argument matches its requested action",
            "procedural":    "e.g., comment filed within the comment period",
            "statistical":   "e.g., supporting research report meets peer-review-journal presentation standards",
            "causal":        "e.g., the agency's stated reasons would actually cause its proposed effect",
            "completeness":  "e.g., the agency response document addresses every substantive comment received",
            "normative":     "e.g., substantiveness / persuasiveness of the petition's argument",
        },
        "tractability_inline": {
            "programmatic_check": "e.g., comment within filing period (timestamp); valid regulations.gov format",
            "llm_judge": "e.g., comment's substantive points are coherent; cites the right docket",
            "expert_judgment": "e.g., legal substantiveness of objections; APA-compliance of agency response",
            "intractable": "e.g., 'will the agency be persuaded by this comment in absolute terms'",
        },
        "specificity_inline": {
            "vague": "e.g., 'be substantive'; 'engage with the proposal'",
            "general": "e.g., 'provide a full factual and legal basis'; 'disclose unfavorable information'",
            "specific": "e.g., 'cite §1.31(b)(3) basis'; 'follow regulations.gov submission protocol'",
            "hyper_specific": "e.g., 'comments due by Feb 27, 2026'; '20% royalty obligation under Project Gutenberg license clause'",
        },
    },

    "patents": {
        "work": [
            "a patent application — its claims, specification, drawings, and supporting documents."
        ],
        "meta_artifacts": [
            "the *issued patent certificate*",
            "the *prior-art search report* attached to an examiner's office action",
            "the *Patent Trial and Appeal Board (PTAB) decision* on a patent",
            "the *gazette notice* publishing the granted patent",
        ],
        "service_or_logistics": [
            "USPTO fee schedules (filing fees, examination fees, maintenance fees, micro-entity fees)",
            "filing deadlines (RCE filing periods, revocation suit deadlines)",
            "USPTO Customer Number / EFS-Web account mechanics",
            "OED disciplinary referral procedures (referring a complaint to a hearing officer)",
            "form-eligibility-based-on-application-status rules (\"to use this form, current status must be 'A Non-final Action has been mailed'\")",
        ],
        "actor_attribute": [
            "practitioner-discipline-history rules (\"practitioner shall not have been publicly disciplined by any State authority\")",
            "patent-bar admission requirements",
            "registered-attorney status conditions",
        ],
        "inputs_examples": [
            'Statutory category requirement (Step 1) → inputs = ["the claim\'s category designation (process / machine / manufacture / composition of matter)", "whether the claim recites a method, apparatus, or substance"]',
            'Step 2B: significantly more than judicial exception → inputs = ["the additional elements beyond the judicial exception", "whether those elements are well-understood / routine / conventional", "the combination\'s specificity to the application domain"]',
            'Prohibition on using RCE to switch inventions → inputs = ["the claim set before the RCE", "the claim set after the RCE submission", "whether the two sets share unity-of-invention vs. are independent/distinct"]',
        ],
        "axis_examples": [
            'Statutory category requirement (§101 categories) → verifiability=normative, tractability=expert_judgment, specificity=specific (a patent examiner judges whether the claim fits a §101 category; not deterministic text-matching)',
            'Written description requirement (35 U.S.C. §112) → verifiability=normative, tractability=expert_judgment, specificity=specific (named legal standard; PHOSITA-judgment required by an examiner)',
            'Filing fee payment deadline → verifiability=procedural, tractability=programmatic_check, specificity=hyper_specific (timestamp check against the deadline)',
        ],
        "verifiability_inline": {
            "computational": "e.g., claim has the required 'comprising' transitional phrase",
            "factual":       "e.g., cited prior-art reference exists in the USPTO database",
            "consistency":   "e.g., specification supports each claim limitation",
            "procedural":    "e.g., RCE filed with required fee within 6 months of final office action",
            "causal":        "e.g., the cited prior-art combination teaches the claimed invention",
            "completeness":  "e.g., specification discloses every claim limitation",
            "normative":     "e.g., §112 written description; §103 obviousness over the prior art",
        },
        "tractability_inline": {
            "programmatic_check": "e.g., claim 'comprising' transitional phrase; filing fee paid (timestamp + DB lookup)",
            "llm_judge": "e.g., specification mentions every claim limitation; argument is internally consistent",
            "expert_judgment": "e.g., §112 written description (PHOSITA judgment); §103 obviousness over prior-art combination",
            "intractable": "e.g., 'is this a strong patent in absolute terms' (depends on litigation strategy)",
        },
        "specificity_inline": {
            "vague": "e.g., 'be novel'; 'be non-obvious'",
            "general": "e.g., 'satisfy §101 patentable subject matter'; 'avoid §112 indefiniteness'",
            "specific": "e.g., 'must fall within statutory categories (process, machine, manufacture, composition)'; 'Step 2B significantly-more inquiry'",
            "hyper_specific": "e.g., 'RCE within 6 months of final office action'; 'SHORTENED statutory period: 3 months from mailing date'",
        },
    },
}


# Convenience formatters used by prompt-assembly functions
def as_bullets(items: list[str], indent: str = "  ") -> str:
    return "\n".join(f"{indent}- {it}" for it in items)


def as_semicolon_list(items: list[str]) -> str:
    return "; ".join(items)


def get_task_info(task: str) -> dict[str, list[str]] | None:
    """Look up a task's info; returns None if the task isn't in TASK_INFO."""
    return TASK_INFO.get(task)
