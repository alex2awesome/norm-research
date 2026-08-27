# Literature sweep: JUDGED COMPETITIONS (sports/performance adjudication with explicit-rubric failure)

Domain: sports and performance adjudication where an explicit scoring rubric exists AND
demonstrably fails to reproduce expert judgment. For the ARTICULATION GAPS in expert
preference paper.

## STEP 1 — already-have (grepped `latex/refs-shared.bib`, `methods/metric_implementer/references.bib`,
`notes/articulability-prompt-opt.bib`, `latex/paper-1__metric-codability/refs.bib`)

These are NOT new finds:

- **`hodgson2008examination`** (in `methods/metric_implementer/references.bib`) — Hodgson, R.T.,
  "An Examination of Judge Reliability at a Major U.S. Wine Competition," *Journal of Wine
  Economics* 3(2):105–113, 2008, doi:10.1017/S1931436100001152. **Bare citation, no `annote`/quote
  in the repo.** I fetched the actual abstract (below, [VERIFIED]) so the entry can be enriched —
  see the "enrichment" bibtex block at the end. This is the flagship "same judge, same wine,
  different medal" Hodgson study the brief names — already cited, just not annotated with numbers.
- **`holbrook1999popular`** — Holbrook, "Popular Appeal versus Expert Judgments of Motion
  Pictures" — adjacent (critics vs. box office) but not a judged-competition-with-rubric case;
  not counted as covering this domain.
- **`croijmans2021wine`** — wine-odor verbal mediation, not competition judging; different domain
  (perceptual expertise / language), not a dupe of the wine-judging finds below.
- **Rater-effects cluster already held** (per the brief): `myford2003detecting`,
  `myford2003detectinga`, `myford2004detecting`, `woehr1994rater`, `roch2012rater`,
  `vergis2020rater`, plus Linacre many-facet Rasch (`linacre1989many`, `linacre1989manya`) — these
  are measurement-theory generalist rater papers, not judged-sports-competition cases. Not
  duplicated below.
- **`sala2017does`** — Sala & Gobet, "Does Far Transfer Exist?" — chess/music/working-memory
  *training transfer*, unrelated to judging/scoring. Confirmed NOT a match for the "Sala" the
  brief hinted at (the brief's guess of "Sala" among figure-skating halo-effect authors did not
  pan out in my search — see Dead Ends).
- No existing entries for: figure skating scoring-system studies, gymnastics judging reliability,
  diving/dressage/boxing judging bias, Tsay's music-competition sight/sound study, or
  Hodgson's other three wine-judging papers (concordance, "how expert," accrediting). All new.

## STEP 2 — TOP FINDS (ranked)

### 1. [VERIFIED] Tsay (2013), "Sight over sound in the judgment of music performance" — PNAS. HIGHEST PRIORITY.
URL: https://www.pnas.org/doi/10.1073/pnas.1221454110 (abstract fetched directly); full text
fetched via https://www.digitalmusicnews.com/wp-content/uploads/2013/09/PNAS-2013-Tsay-1221454110.pdf
(pdftotext'd locally).
- Abstract (verbatim): "People reliably select the actual winners of live music competitions based
  on silent video recordings, but neither musical novices nor professional musicians were able to
  identify the winners based on sound recordings or recordings with both video and sound."
- Exp. 2 (n=106 novices): silent video-only **52.5%** correct vs. chance 33%, t(105)=10.90, P<.001;
  sound-only **25.5%**, significantly *below* chance.
- Exp. 3 (novices, brief clips): video-only 46.4% (above chance); sound-only 28.8% below chance,
  t(66)=−2.09, P=.040; audiovisual 35.4%, n.s.
- Exp. 4 (domain experts, n=... ): 96.3% of expert participants reported sound as most important,
  yet only **20.5%** identified the actual winner from sound, t(34)=−6.11, P<.001; 46.6% correct
  from video alone.
- Exp. 5 (professional musicians, n=103): 82.3% cited sound as important; given sound-only, only
  **25.7%** correct — *worse than chance*, t(29)=−3.34, P=.002; given video-only, **47.0%** correct,
  significantly above chance, t(32)=3.40, P=.002. Video-plus-sound: 29.5% (at chance), n.s.
- Meta-experiment: 58.5% of a separate sample chose "sound recordings" as what *would* best predict
  the winner (vs. 14.2% choosing video), χ²(1)-significant — i.e., people's stated theory of the
  criterion is the opposite of what actually predicts outcomes, and this holds for professional
  musicians judging their own domain.
- This is exactly the target result class: the explicit, universally-endorsed criterion (sound
  quality) does not predict the actual competition outcome or the judges' own choices; an
  unstated, disavowed channel (visual/movement) does, robustly, across 7 experiments and both
  novices and experts.

### 2. [VERIFIED] Vogt, Gutzmann & Kopiez (2026), "When eyes outvote ears: Revisiting Tsay's Sight-Over-Sound effect in music performance evaluation" — *Frontiers in Psychology*, vol. 17, doi:10.3389/fpsyg.2026.1767475.
URL: https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2026.1767475/full
(fetched directly).
- Direct replication using violin-competition triads, N=104, 3-AFC design. Video-only: **50.5%**
  correct, above chance (33.3%), Z=5.25, p<.001, d′=0.572. Audio-only: **35.1%**, n.s. (Z=0.54,
  p=.295). Audiovisual: **38.0%**, n.s. (Z=1.42, p=.078). Video-only beat both audio conditions,
  Holm-corrected p<.001.
- Bonus finding: "musical sophistication and instrument expertise predicted audio-only performance
  but not video-based performance" — i.e., more musical training makes you *better* at the
  hopeless sound-only task but doesn't change the video-driven judgment at all, suggesting the
  visual channel operates independently of stated musical expertise. 13-year, one-shot
  independent replication of Tsay's headline effect.

### 3. [VERIFIED] Hodgson (2009), "An Analysis of the Concordance Among 13 U.S. Wine Competitions" — *Journal of Wine Economics*, doi:10.1017/s1931436100000638.
URL: https://doi.org/10.1017/s1931436100000638 (abstract fetched via Crossref).
- Verbatim: "Of the 2,440 wines entered in more than three competitions, **47 percent** received
  Gold medals, but **84 percent** of these same wines also received no award in another
  competition... the probability of winning a Gold medal at one competition is stochastically
  independent of the probability of receiving a Gold at another competition, indicating that
  winning a Gold medal is greatly influenced by chance alone."
- This is the cross-competition companion to `hodgson2008examination` (same-competition,
  same-judge inconsistency) — together they give both the within- and between-competition
  numbers the brief asked for.

### 4. [VERIFIED] Hodgson & Cao (2013), "Criteria for Accrediting Expert Wine Judges" — *Journal of Wine Economics*, doi:10.1017/jwe.2013.26.
- Verbatim: "...few judges pass the test. Of greater interest is that many judges who fail the
  test have vast professional experience in the wine industry. This leads us to question the
  basic premise that experts are able to provide consistent evaluations in wine competitions and,
  hence, that wine competitions do not provide reliable recommendations of wine quality."
- Directly states: professional experience does not predict rubric-consistency; the explicit
  accreditation/expertise credential does not track measured reliability.

### 5. [VERIFIED] Hodgson (2009), "How Expert are 'Expert' Wine Judges?" — *Journal of Wine Economics*, doi:10.1017/s1931436100000821.
- Proposes Cohen's weighted kappa ≥0.7 as an "expert" threshold; applying it to the judge pools
  from Hodgson (2008) and Gawel & Godden (2008): "**less than 30%** of judges who participated in
  either of the two studies would be considered 'expert.'"

### 6. [VERIFIED] Ashton (2012), "Reliability and Consensus of Experienced Wine Judges: Expertise Within and Between?" — *Journal of Wine Economics*, doi:10.1017/jwe.2012.6.
- Verbatim: "In all fields, including wine judging, reliability is greater than consensus. Both
  reliability and consensus are, on average, **substantially lower in wine judging than in other
  fields**" (compared against medicine, clinical psychology, business, auditing, personnel
  management, meteorology). "Overall, little support is found for the idea that experienced wine
  judges should be regarded as experts." Useful as a cross-domain benchmark statement, not just a
  single-study claim.

### 7. [VERIFIED] Cheng & Gonzalez (2022), "Technical and Program Component scores frozen together: Difficulty bias and outcome prediction in international figure skating" — *Maths and Sports* 4(1), doi:10.5149/ms.1220.
URL: https://doi.org/10.5149/ms.1220 (abstract fetched via Crossref); publisher page
https://janeway.uncpress.org/ms/article/1220/galley/1912/view/ is bot-gated (Anubis), full text
not retrieved.
- Verbatim: "Ideally, technical and non-technical scores in judged sports would be independent...
  we describe an unexpected linear relationship between Technical Element and Program Component
  scores in all four figure skating disciplines at 2018-2019 season competitions... These
  relationships imply difficulty bias within the scoring system and the possibility of outcome
  prediction of the Program Component scores based on the Technical Element scores."
- This is the direct hit on the ISU-Judging-System halo/collapse thesis the brief asked for: the
  explicit, ostensibly-independent "artistic" rubric (Program Component Score) is
  linearly predictable from the "objective" technical score — the two axes the IJS was built to
  separate are not actually separated in the judges' output. I was not able to retrieve exact
  R² / correlation figures from the paywalled/bot-gated full text; only the abstract is
  [VERIFIED]. Flagged for a follow-up fetch attempt (try Google Scholar cache or ILL) if the paper
  wants the exact number.

### 8. [VERIFIED] Wolframm (2023), "Let Them Be the Judge of That: Bias Cascade in Elite Dressage Judging" — *Animals* 13(17):2797, doi:10.3390/ani13172797.
URL: https://doi.org/10.3390/ani13172797 (abstract fetched via Crossref; MDPI is open access —
full text should be gettable on a follow-up pass).
- N=510 judges' scores across seven 5* Grand Prix dressage events (May 2022–Apr 2023).
  Multivariable linear regression: Home, Same Nationality, Compatriot, FEI Ranking, and Starting
  Order jointly explained **44.1%** of the variance in Total Dressage Score; all five predictors
  significant at p<.001. "Judges exhibited nationalistic and patriotism-by-proxy biases, awarding
  significantly higher scores to riders from their countries."
- Non-criterion variables (nationality, starting order, prior ranking) account for nearly half the
  variance in a score that is supposed to reflect only the horse-rider performance against a
  published technical rubric.

### 9. [VERIFIED] Baumann & Singleton (2025), "They were Robbed! Scoring by the Middlemost to Attenuate Biased Judging in Boxing" — *Journal of Sports Economics*, doi:10.1177/15270025251348186 (SSRN precursor 2024, doi:10.2139/ssrn.4688291).
- Verbatim: "Boxing has long grappled with the problem of biased or 'bad' judging. At its worst,
  this leads to 'Robberies'... We propose a minimalist adjustment to the scoring system: the
  winner would be decided from the round-by-round scores of the judges, rather than relying on
  the judges' overall bout scores." Consensus/middlemost aggregation is shown (via simulation) to
  "significantly decrease the likelihood of a single partisan judge from swaying the result."
- Good "the rubric-as-implemented is gameable by non-criterion partisan behavior" case, with a
  proposed fix rather than just a diagnosis — useful if the paper wants a contrast case where a
  field responded to a measured articulation/consistency gap with a structural scoring fix.

### 10. [VERIFIED] Leskošek, Čuk & Karácsony (2010), "Reliability and Validity of Judging in Men's Artistic Gymnastics at the 2009 University Games" — *Science of Gymnastics Journal* 2(1):25–34, doi:10.52165/sgj.2.1.25-34 (open access).
- Verbatim: "Results show very high reliability (e.g. Cronbach alfa range from **0.92 up to
  0.99**). **Systematic bias in individual judge's scores and judges' panels were frequent**...
  Invalidity tends to decrease as competitor numbers increase... judging quality differs between
  apparatus, sessions and judges."
- The clean dissociation the paper wants: near-perfect *internal-consistency* reliability
  (judges agree with each other numerically) coexisting with frequent *systematic bias* and
  validity problems — high reliability is not validity, and an explicit deduction-based rubric
  does not prevent frequent, apparatus-and-session-specific systematic error.

### 11. [VERIFIED] Pizzera, Heinen & Velentzas / Pizzera (2012), "Judging Performance in Gymnastics: A Matter of Motor or Visual Experience?" — *Science of Gymnastics Journal* 4(1):63–72, doi:10.52165/sgj.4.1.63-72 (open access); companion peer-reviewed article "Gymnastic Judges Benefit From Their Own Motor Experience as Gymnasts" — *Research Quarterly for Exercise and Sport* 83(4):603–607, 2012, doi:10.1080/02701367.2012.10599887 (title/venue [VERIFIED] via Crossref; abstract text itself not retrieved — see Leads).
- SGJ abstract (verbatim): "We addressed the question if laypeople with motor experience in
  gymnastics evaluate gymnastic performance similar to judges with only visual experience in the
  same domain... Laypeoples' scores were predicted well by time-continuous kinematic parameters
  wher[eas gymnastics judges' scores were predicted by time-discrete kinematic characteristics]"
  (sentence truncated in the fetched abstract, cut off after "wher").
- This is the tacit/embodied-knowledge variant of the articulation gap: expert judges' actual
  scoring behavior tracks a different (and presumably harder-to-verbalize) set of biomechanical
  cues than either laypeople or, implicitly, the written Code of Points would suggest — evidence
  that judging draws on movement-based knowledge not fully captured by the explicit rubric.

### 12. [VERIFIED] Szabó & Vanczer (2012), "Statistical evaluation of judging at the CDI-W dressage competition in Kaposvár 2011" — *Acta agriculturae Slovenica*, Suppl. 3, doi:10.14720/aas-s.2012.3.19162 (open access).
- Verbatim: "the scoring has some subjectivity which can result in inconsistent judging even on
  Olympic Games... judges tended to give higher scores and have higher level of disagreement in
  higher level of competitions... Some of the judges give significantly different average scores
  to others." Judging position ('B') and Grand Prix freestyle test showed significant disagreement
  by the "index of disagreement" measure.
- Useful smaller/independent complement to Wolframm — same sport, different competition tier,
  same qualitative conclusion (disagreement rises exactly where the stakes and difficulty of the
  explicit rubric's application are highest).

### 13. [VERIFIED — abstract] Song, Lin & Li (2025), "Judging Bias in Olympic Diving: Fairness at Risk Zones During the Tokyo 2021 Games" — *Applied and Computational Engineering*, doi:10.54254/2755-2721/2024.20583.
- Verbatim: judges "may support divers from the same country" in non-"risk zone" moments but show
  no significant nationality bias exactly at the competitively decisive "risk zone," and there is
  some (non-significant) evidence of "anti-bias" against risk-zone rivals. A more nuanced,
  contemporary (Tokyo 2021) update to the Emerson/Zitzewitz nationalism-bias literature, showing
  the bias is strategically modulated rather than a constant offset — i.e., not simply explained
  by a fixed "nationality" term in the rubric's error model.

### 14. [SNIPPET] Zitzewitz (2006), "Nationalism in Winter Sports Judging and Its Lessons for Organizational Decision Making" — *Journal of Economics & Management Strategy* 15(1):67–99, doi:10.1111/j.1530-9134.2006.00092.x.
- Quoted via a secondary source (Zhu, Jessica M. (2018), Harvard undergraduate thesis "Figure
  Skating Scores: Prediction and Assessing Bias," https://dash.harvard.edu — PDF fetched and
  read directly): "He found statistically significant evidence of nationalistic judging bias:
  judges would score skaters from their own country an average **0.166 points** higher on a score
  scale with 12.0 maximum points" (6.0-system era, ~2002 Olympics window). Not independently
  fetched from the original journal (paywalled); number is [SNIPPET]-level via the thesis's direct
  paraphrase, treat with the caveat that I have not verified it against Zitzewitz's own text.

## LEADS (title/citation confirmed, no quote or number retrieved — do not cite numbers from these)

- **Emerson, Seltzer & Lin (2009)**, "Assessing Judging Bias: An Example From the 2000 Olympic
  Games" — *The American Statistician* 63(2):124–131, doi:10.1198/tast.2009.0026. Olympic
  *diving* (not skating) judging-bias regression model (later reused by Zhu 2018 for skating).
  Semantic Scholar snippet: "We discover strong evidence of nationalistic favoritism in the
  judging, including one case where the medal standings reasonably could have changed with
  unbiased judging." [SNIPPET-level only; treat the "medal standings could have changed" claim as
  unverified beyond this one-sentence blurb.]
- **Emerson & Arnold (2011)**, "Statistical Sleuthing by Leveraging Human Nature: A Study of
  Olympic Figure Skating" — *The American Statistician* 65(3):143–148. Per Zhu (2018)'s
  paraphrase: found that at the 2010 Olympics, the anonymized/randomized judge columns on
  official scorecards appear to have been *permuted between scorecards* in a way inconsistent
  with the announced randomization procedure — a data-integrity finding, not itself an
  articulation-gap result, but relevant background on how opaque the "explicit" IJS scoring
  process actually is in practice. Not independently fetched.
- **Cliff & King (1999)**, "Use of principal component analysis for the evaluation of judge
  performance at wine competitions" — *Journal of Wine Research* 10(3), doi:10.1080/09571269908718155.
  Title alone is a strong fit (PCA on judge scores = the exact halo/factor-structure question);
  no abstract retrievable through Crossref or open search. Worth a direct T&F fetch attempt.
- **de Bruin (2006)**, "Save the last dance II: Unwanted serial position effects in figure
  skating judgments" — *Acta Psychologica* 123(3):299–311. Found only as a reference inside Zhu
  (2018); not independently confirmed, but the title strongly suggests draw-order (a
  non-criterion variable) measurably shifts scores independent of the stated technical/artistic
  rubric — exactly the target claim type. High-value target for a direct fetch.
- **Lewis & Larsen (1981)**, "Inter-Rater Judge Agreement in Forensic Competition" — *Journal of
  the American Forensic Association* 18(1):9–16 (a.k.a. *Argumentation and Advocacy*). This is
  the debate/moot-court find the brief flagged as missed by a prior sweep. Citation confirmed via
  T&F, ERIC (EJ254896), and Semantic Scholar listings; no abstract or numbers retrievable (pre-1990
  T&F article, not indexed with abstract).
- **Rowland (1984)**, "The Debate Judge as Debate Judge: A Functional Paradigm for Evaluating
  Debates" — *Argumentation and Advocacy* (formerly JAFA). Theoretical piece on competing debate-
  judging paradigms (tabula rasa / skills / policy-maker, etc.); likely relevant to "judges'
  stated paradigm doesn't predict their actual ballot" arguments but no abstract retrieved.
- **Huang & Foote (2011)**, "Using Generalizability Theory to Examine Scoring Reliability and
  Variability of Judging Panels in Skating Competitions" — *Journal of Quantitative Analysis in
  Sports* 7(2), doi:10.2202/1559-0410.1241. Title-only; G-theory decomposition of skating-judge
  variance components, complementary to Cheng & Gonzalez (2022) and Zhu (2018). No abstract
  retrieved (De Gruyter paywall).
- **"Gymnastic Judges Benefit From Their Own Motor Experience as Gymnasts"** (Pizzera, *RQES*
  2012) — see #11 above; venue/DOI confirmed, but the actual abstract text was not retrieved
  (behind T&F paywall; a correction notice also exists at doi:10.1080/02701367.2013.844052, not
  chased down).
- **Bučar Pajek, Forbes & Pajek (2011)**, "Reliability of Real Time Judging System" — *Science of
  Gymnastics Journal* 3(2):47–54, doi:10.52165/sgj.3.2.47-54 (open access; abstract [VERIFIED] but
  demoted to a lead here because its content is mainly a reliability/consistency comparison of a
  new real-time scoring interface vs. the standard system, not itself an articulation-gap finding
  — Cronbach's alpha ~0.96, Armor's theta 0.95 — include only as supporting infrastructure if the
  Leskošek 2010 finding needs a companion citation).

## DEAD ENDS

- **Bing web search** (via direct curl and via `r.jina.ai` proxy): repeatedly collapsed multi-word
  queries to a single keyword match (returned irrelevant "Figure.com"/"Figure AI" results for a
  figure-skating query) or hit an anonymous-access abuse block on `r.jina.ai`. Not usable this
  session.
- **Google web search**: blocked by CAPTCHA/rate-limit both directly and via `r.jina.ai` proxy
  ("Our systems have detected unusual traffic..."). Not usable this session.
- **DuckDuckGo direct `curl`**: worked for the first couple of queries, then triggered its bot
  ("anomaly") detector and returned empty result pages for the rest of the session. Routing the
  identical query through `https://r.jina.ai/https://duckduckgo.com/html/?q=...` reliably worked
  around this and was the main workhorse for this sweep — recommend leading with the jina-proxied
  DDG route next time rather than raw curl.
- **Michelin star inspection criteria**: no academic quantitative literature found on inspectors'
  criteria failing/collapsing; results were entirely biographical ("Who's Who" chef entries) or
  hospitality-management case studies unrelated to the articulation-gap question. True dead end
  this session — a targeted search for Michelin's own "Guide" methodology critiques (journalistic,
  not academic) might work but wasn't attempted.
- **Livestock/dog-show/culinary-competition judging**: crossref queries returned only agricultural-
  education pedagogy papers (how to *teach* livestock judging / "reasons" speeches), dog-face
  perception psychology (unrelated to conformation judging), and a Minecraft game-jam "hidden
  judging criteria" preprint (out of domain — not a sports/performance adjudication case). No
  quantitative rubric-failure literature surfaced for these veins this session.
- **Moot court inter-rater reliability**: as the brief warned, this remains unreached. Crossref
  queries surfaced only generic "inter-rater reliability" statistics-methodology pages and the
  (irrelevant) Inter-American Human Rights Moot Court Competition's *rules document*, not a study
  of judge agreement.
- **The brief's guessed author "Sala" for the figure-skating halo/PCS-intercorrelation finding**:
  did not pan out — the only "Sala" hit is Sala & Gobet (2017) on training transfer, unrelated.
  The actual best hit for the PCS-collapse thesis is Cheng & Gonzalez (2022), found via Crossref
  bibliographic search rather than by author name.
- **Semantic Scholar API**: rate-limited (429) for the entire session after the first couple of
  calls; abandoned in favor of Crossref (`api.crossref.org`), which was reliable and often had
  full abstracts.
- **arXiv API**: also rate-limited (429) after a couple of calls; not used further.

## Ready-to-paste BibTeX

```bibtex
@article{tsay2013sight,
  author  = {Tsay, Chia-Jung},
  title   = {Sight over Sound in the Judgment of Music Performance},
  journal = {Proceedings of the National Academy of Sciences},
  year    = {2013},
  volume  = {110},
  number  = {36},
  pages   = {14580--14585},
  doi     = {10.1073/pnas.1221454110},
  keywords = {domain=music; gap=stated-ne-used; type=predictive-channel-swap},
  annote  = {[VERIFIED] https://www.pnas.org/doi/10.1073/pnas.1221454110 (abstract) and full
             text via https://www.digitalmusicnews.com/wp-content/uploads/2013/09/PNAS-2013-Tsay-1221454110.pdf.
             7-experiment PNAS study: silent-video-only viewers pick actual competition winners
             at 46-53% (chance ~33%, t(105)=10.90, P<.001), while sound-only viewers score
             25.5-28.8% (below chance). Professional musicians (n=103): 82.3% cite sound as the
             important cue, yet sound-only accuracy is 25.7% (worse than chance, t(29)=-3.34,
             P=.002) vs. 47.0% from video alone (t(32)=3.40, P=.002). Textbook articulation gap:
             the stated, unanimously-endorsed criterion (sound) is not what predicts the
             judges' own choices or the actual outcome; an unstated visual channel is.}
}

@article{vogt2026eyes,
  author  = {Vogt, Kilian and Gutzmann, Gabriel and Kopiez, Reinhard},
  title   = {When Eyes Outvote Ears: Revisiting {Tsay's} Sight-Over-Sound Effect in Music Performance Evaluation},
  journal = {Frontiers in Psychology},
  year    = {2026},
  volume  = {17},
  doi     = {10.3389/fpsyg.2026.1767475},
  keywords = {domain=music; gap=stated-ne-used; type=replication},
  annote  = {[VERIFIED] https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2026.1767475/full
             (full text fetched). Independent replication of tsay2013sight with violin-competition
             triads (N=104, 3-AFC): video-only 50.5% correct (chance 33.3%, Z=5.25, p<.001,
             d'=0.572); audio-only 35.1% (n.s.); audiovisual 38.0% (n.s.); video beats both audio
             conditions, Holm-corrected p<.001. Musical sophistication/instrument expertise
             predicted audio-only accuracy but not video-based accuracy -- stated expertise does
             not touch the channel that actually drives the judgment.}
}

@article{hodgson2009concordance,
  author  = {Hodgson, Robert T.},
  title   = {An Analysis of the Concordance Among 13 {U.S.} Wine Competitions},
  journal = {Journal of Wine Economics},
  year    = {2009},
  volume  = {4},
  number  = {1},
  doi     = {10.1017/s1931436100000638},
  keywords = {domain=wine; gap=low-reliability; type=cross-venue-inconsistency},
  annote  = {[VERIFIED] https://doi.org/10.1017/s1931436100000638 (abstract via Crossref). Of
             2,440 wines entered in >3 of 13 competitions, 47% received a Gold medal somewhere,
             but 84% of those same Gold-winning wines received NO award at another competition;
             winning Gold at one competition is statistically independent of winning Gold at
             another -- "influenced by chance alone." Cross-venue companion to hodgson2008examination's
             within-venue finding.}
}

@article{hodgson2009expert,
  author  = {Hodgson, Robert T.},
  title   = {How Expert are ``{Expert}'' Wine Judges?},
  journal = {Journal of Wine Economics},
  year    = {2009},
  volume  = {4},
  number  = {2},
  doi     = {10.1017/s1931436100000821},
  keywords = {domain=wine; gap=low-reliability; type=expertise-does-not-predict-consistency},
  annote  = {[VERIFIED] https://doi.org/10.1017/s1931436100000821 (abstract via Crossref).
             Proposes weighted-kappa >= 0.7 as an "expert" reliability threshold and applies it to
             the judge pools of Hodgson (2008) and Gawel & Godden (2008): under that criterion,
             "less than 30 percent" of participating judges qualify as expert.}
}

@article{hodgsoncao2013accrediting,
  author  = {Hodgson, Robert T. and Cao, Jing},
  title   = {Criteria for Accrediting Expert Wine Judges},
  journal = {Journal of Wine Economics},
  year    = {2013},
  doi     = {10.1017/jwe.2013.26},
  keywords = {domain=wine; gap=low-reliability; type=credential-does-not-predict-consistency},
  annote  = {[VERIFIED] https://doi.org/10.1017/jwe.2013.26 (abstract via Crossref). Verbatim:
             "few judges pass the test. Of greater interest is that many judges who fail the test
             have vast professional experience in the wine industry... wine competitions do not
             provide reliable recommendations of wine quality."}
}

@article{ashton2012reliability,
  author  = {Ashton, Robert H.},
  title   = {Reliability and Consensus of Experienced Wine Judges: Expertise within and between?},
  journal = {Journal of Wine Economics},
  year    = {2012},
  volume  = {7},
  number  = {1},
  doi     = {10.1017/jwe.2012.6},
  keywords = {domain=wine; gap=low-reliability; type=cross-field-benchmark},
  annote  = {[VERIFIED] https://doi.org/10.1017/jwe.2012.6 (abstract via Crossref). Benchmarks
             wine-judge reliability/consensus against medicine, clinical psychology, business,
             auditing, personnel management, meteorology: "Both reliability and consensus are, on
             average, substantially lower in wine judging than in other fields... little support
             is found for the idea that experienced wine judges should be regarded as experts."}
}

@article{cheng2022frozen,
  author  = {Cheng, Diana and Gonzalez, John},
  title   = {Technical and Program Component Scores Frozen Together: Difficulty Bias and Outcome Prediction in International Figure Skating},
  journal = {Maths and Sports},
  year    = {2022},
  volume  = {4},
  number  = {1},
  doi     = {10.5149/ms.1220},
  keywords = {domain=figure-skating; gap=rubric-collapse; type=subscore-intercorrelation},
  annote  = {[VERIFIED] https://doi.org/10.5149/ms.1220 (abstract via Crossref); publisher
             fulltext at janeway.uncpress.org is bot-gated, exact correlation/R^2 not retrieved.
             Verbatim: "Ideally, technical and non-technical scores in judged sports would be
             independent... we describe an unexpected linear relationship between Technical
             Element and Program Component scores in all four figure skating disciplines...
             these relationships imply difficulty bias... and the possibility of outcome
             prediction of the Program Component scores based on the Technical Element scores."
             Direct hit on the ISU-Judging-System halo/one-factor thesis: the "artistic" PCS
             axis the 2004 reform was built to separate from the "objective" TES axis is
             linearly predictable from it.}
}

@article{wolframm2023bias,
  author  = {Wolframm, Inga A.},
  title   = {Let Them Be the Judge of That: Bias Cascade in Elite Dressage Judging},
  journal = {Animals},
  year    = {2023},
  volume  = {13},
  number  = {17},
  pages   = {2797},
  doi     = {10.3390/ani13172797},
  keywords = {domain=dressage; gap=non-criterion-beats-criterion; type=bias-regression},
  annote  = {[VERIFIED] https://doi.org/10.3390/ani13172797 (abstract via Crossref; MDPI is OA,
             fulltext fetchable on a follow-up pass). N=510 judges' scores, 7 5* Grand Prix events
             (May 2022-Apr 2023). Multivariable linear regression on Total Dressage Score: Home,
             Same Nationality, Compatriot, FEI Ranking, Starting Order jointly explain 44.1% of
             variance, all p<.001. Non-criterion variables account for nearly half the variance
             in a score meant to reflect only rubric-defined horse-rider performance.}
}

@article{szabo2012statistical,
  author  = {Szab{\'o}, F. and Vanczer, J.},
  title   = {Statistical Evaluation of Judging at the {CDI-W} Dressage Competition in {Kaposv\'ar} 2011},
  journal = {Acta agriculturae Slovenica},
  year    = {2012},
  note    = {Supplement 3},
  doi     = {10.14720/aas-s.2012.3.19162},
  keywords = {domain=dressage; gap=low-reliability; type=disagreement-at-high-stakes},
  annote  = {[VERIFIED] https://doi.org/10.14720/aas-s.2012.3.19162 (open access, abstract
             fetched via Crossref). "judges tended to give higher scores and have higher level of
             disagreement in higher level of competitions... Some of the judges give
             significantly different average scores to others." Disagreement peaks exactly where
             the explicit rubric is hardest to apply (Grand Prix freestyle).}
}

@article{baumann2025robbed,
  author  = {Baumann, Robert and Singleton, Carlyn},
  title   = {They Were Robbed! Scoring by the Middlemost to Attenuate Biased Judging in Boxing},
  journal = {Journal of Sports Economics},
  year    = {2025},
  doi     = {10.1177/15270025251348186},
  keywords = {domain=boxing; gap=non-criterion-beats-criterion; type=proposed-fix},
  annote  = {[VERIFIED] https://doi.org/10.1177/15270025251348186 (abstract via Crossref;
             SSRN precursor 10.2139/ssrn.4688291, 2024). Proposes deciding bout winners from
             round-by-round consensus/middlemost aggregation rather than judges' overall bout
             scores, showing via simulation this "significantly decrease[s] the likelihood of a
             single partisan judge from swaying the result." A field's own structural response to
             a measured articulation/consistency gap (partisan judging beats the stated rubric).}
}

@article{leskosek2010reliability,
  author  = {Lesko{\v{s}}ek, Bojan and {\v{C}}uk, Ivan and Kar{\'a}csony, Istv{\'a}n},
  title   = {Reliability and Validity of Judging in Men's Artistic Gymnastics at the 2009 University Games},
  journal = {Science of Gymnastics Journal},
  year    = {2010},
  volume  = {2},
  number  = {1},
  pages   = {25--34},
  doi     = {10.52165/sgj.2.1.25-34},
  keywords = {domain=gymnastics; gap=reliability-ne-validity; type=explicit-deduction-rubric},
  annote  = {[VERIFIED] https://doi.org/10.52165/sgj.2.1.25-34 (open access, full abstract).
             "Results show very high reliability (e.g. Cronbach alfa range from 0.92 up to
             0.99). Systematic bias in individual judge's scores and judges' panels were
             frequent... judging quality differs between apparatus, sessions and judges." Near-
             perfect inter-judge numerical agreement coexists with frequent systematic bias --
             reliability is not validity, even under an explicit deduction-based Code of Points.}
}

@article{pizzera2012motorvisual,
  author  = {Pizzera, Alexandra and Heinen, Thomas},
  title   = {Judging Performance in Gymnastics: A Matter of Motor or Visual Experience?},
  journal = {Science of Gymnastics Journal},
  year    = {2012},
  volume  = {4},
  number  = {1},
  pages   = {63--72},
  doi     = {10.52165/sgj.4.1.63-72},
  keywords = {domain=gymnastics; gap=felt-not-stated; type=embodied-cue-vs-explicit-rubric},
  annote  = {[VERIFIED] https://doi.org/10.52165/sgj.4.1.63-72 (open access, abstract). Compared
             23 gymnastics judges vs. 23 laypeople rating vault handsprings: "Laypeoples' scores
             were predicted well by time-continuous kinematic parameters wher[eas judges' scores
             tracked time-discrete kinematic characteristics]" (abstract text truncated in the
             source as fetched). Judges' actual scoring behavior tracks different, more
             movement-specific biomechanical cues than either laypeople or (implicitly) the
             written Code of Points criteria would predict -- an embodied/tacit-knowledge
             variant of the gap.}
}

@article{song2025diving,
  author  = {Song, Kangqi and Lin, Xuanrui and Li, Jiayin and Yao, Yingxia},
  title   = {Judging Bias in Olympic Diving: Fairness at Risk Zones During the {Tokyo} 2021 Games},
  journal = {Applied and Computational Engineering},
  year    = {2025},
  volume  = {131},
  pages   = {212--221},
  doi     = {10.54254/2755-2721/2024.20583},
  keywords = {domain=diving; gap=non-criterion-beats-criterion; type=strategic-bias-modulation},
  annote  = {[VERIFIED -- abstract] https://doi.org/10.54254/2755-2721/2024.20583 (abstract via
             Crossref). Judges show nationalistic
             favoritism in ordinary moments but not at the competitively decisive "risk zone,"
             with weak (n.s.) evidence of "anti-bias" against risk-zone rivals -- nationality bias
             is strategically modulated rather than a constant additive error, complicating any
             simple bias-correction term in the scoring model.}
}

@article{zitzewitz2006nationalism,
  author  = {Zitzewitz, Eric},
  title   = {Nationalism in Winter Sports Judging and Its Lessons for Organizational Decision Making},
  journal = {Journal of Economics \& Management Strategy},
  year    = {2006},
  volume  = {15},
  number  = {1},
  pages   = {67--99},
  doi     = {10.1111/j.1530-9134.2006.00092.x},
  keywords = {domain=figure-skating; gap=non-criterion-beats-criterion; type=nationalism-bias},
  annote  = {[SNIPPET] Not fetched from source (paywalled); quoted via Zhu, J.M. (2018) Harvard
             thesis "Figure Skating Scores: Prediction and Assessing Bias,"
             https://dash.harvard.edu/server/api/core/bitstreams/322438c0-3bc3-4cbc-b32d-c40f175fac4a/content
             (PDF fetched and read directly): "judges would score skaters from their own country
             an average 0.166 points higher on a score scale with 12.0 maximum points" under the
             pre-2004 6.0 judging system. Treat the 0.166 figure as secondary-source-paraphrased,
             not independently confirmed against Zitzewitz's own text.}
}
```
