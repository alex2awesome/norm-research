"""Shared news-homepages metric bank: 10 NEW (linguistic/structural) + 4 strongest existing (topical).
Each = (name, guidance) operational enough for a judge to score from a headline+context."""
NEW_METRICS=[
("concrete_numerical_specificity","Headline contains SPECIFIC numbers - exact dollar amounts, percentages, counts, ages, or ratios (e.g. '$800 million', '1-in-16', '530,000', '15-year-old') - rather than vague quantifiers ('many','major','large'). Score 1 if a precise concrete number appears, 0.5 if only rough, 0 if vague/none."),
("linguistic_action_intensity","Headline uses VIVID physical/action verbs conveying intensity or spectacle (torch, slam, explode, collapse, seize, deadlock, bury, blast, sweep, erupt) rather than neutral speech/reporting verbs (says, reports, announces, considers, states). Score 1 if a vivid action verb drives the headline, 0.5 if mixed, 0 if neutral/reportorial."),
("reader_personal_stakes","Headline conveys DIRECT actionable impact on the reader - their wallet (prices, jobs, taxes, mortgage), safety (what to do, evacuation, self-deport), or personal rights - as opposed to abstract institutional/policy impact. Score 1 if reader's personal stakes are concrete, 0.5 if indirect, 0 if abstract."),
("curiosity_gap_question","Headline poses an explicit question, promises an explanation, or creates an information void ('Why...','How...','What to know','Here's what', ends with '?'). Score 1 if it opens a curiosity gap, 0 if purely declarative."),
("moral_outrage_accountability","Headline names a perpetrator, institutional failure, or blame-worthy act that triggers moral outrage (cover-up, betrayal, scandal, exploiting, killing, fraud, abuses). Score 1 if it attributes blame/outrage, 0.5 if conflict without blame, 0 if neutral."),
("surprise_absurdity","Headline contains an unexpected, bizarre, or incongruous element (man-bites-dog, surreal juxtaposition, absurdity, record-breaking oddity). Score 1 if surprising/absurd, 0 if routine."),
("named_victim_personal_narrative","Headline centers on a NAMED or specifically-described individual's personal story (a named victim, a person's transformation/tragedy/decision) rather than abstract statistics or groups. Score 1 if an individual human narrative, 0 if abstract/group."),
("deadline_urgency_countdown","Headline references a ticking deadline, imminent cutoff, or countdown ('ticks down to zero','last chance','about to','hours away','today'). Score 1 if deadline/imminent-time pressure, 0 if not."),
("viral_currency_phraseology","Headline references a viral phrase, trending meme, or cultural-conversation marker (a show/moment/phrase in wide circulation, even without a celebrity name). Score 1 if viral/trending currency, 0 if not."),
("source_authority_exclusive","Headline cites authoritative source attribution ('Exclusive', a named official, 'officials say', 'scientists find', 'records show', 'court documents'). Score 1 if strong source authority, 0 if unsourced."),
]
EXISTING_TOP4=[
("hard_vs_soft","HARD NEWS (politics/war/economy/crime/disaster) rather than SOFT (lifestyle/entertainment/sports/service journalism). Score 1 hard, 0 soft."),
("elite_political_actor","Names a head of state, top government official, SCOTUS justice, legislature/party leader, or major US/foreign political figure as a central subject. Score 1 if yes, 0 if not."),
("ongoing_top_story","Part of a top-tier ongoing national/international story (the day's major storyline - war, election, major trial, crisis). Score 1 if clearly an ongoing major story, 0 if standalone."),
("breaking_developing","Timely/breaking/developing: a just-happened or actively-unfolding event (live updates, developing story, imminent). Score 1 if breaking/developing, 0 if evergreen/analysis."),
]
ALL = NEW_METRICS + EXISTING_TOP4
if __name__=="__main__":
    print(f"{len(NEW_METRICS)} new + {len(EXISTING_TOP4)} existing = {len(ALL)} metrics")
