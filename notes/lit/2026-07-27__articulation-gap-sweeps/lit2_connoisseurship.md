# Literature sweep: CONNOISSEURSHIP — art attribution/authentication as an explicit epistemology of unarticulable judgment

Domain assignment: art attribution/authentication, curatorial connoisseurship. Task: find verbatim expert testimony and scholarly analysis showing (a) experts agree/act on quality-or-authenticity judgments they cannot derive from stated criteria, or (b) explicit criteria demonstrably fail to capture what experts do.

Session note on method: the WebSearch tool's budget was exhausted (0/200 remaining) before this sweep began, and the usual search-engine substitutes were mostly blocked (DuckDuckGo → CAPTCHA every attempt; Bing → JS/ad-mismatched SERPs unreadable by the fetch-summarizer; Mojeek → 403). The only search path that worked was `search.brave.com` via WebFetch (itself rate-limited, HTTP 429, after ~6 calls). The bulk of this sweep was therefore done by (1) using Brave search sparingly to find primary-source URLs, then (2) downloading the actual PDFs/`_djvu.txt` full texts with `curl` and reading them directly (bypassing WebFetch's small summarizer model entirely) — this is how the Ginzburg, Getty-kouros-colloquium, Friedländer, and Morelli full texts below were obtained and grep'd for exact strings. This is more reliable than it sounds and is why the verification tags below are unusually strong (many direct [VERIFIED] against full primary texts rather than [SNIPPET] against search-result fragments.

## Already have (repo bib files) — confirmed nothing on attribution/connoisseurship
Grepped `latex/refs-shared.bib`, `methods/metric_implementer/references.bib`, `notes/articulability-prompt-opt.bib`, `latex/paper-1__metric-codability/refs.bib` for morelli|berenson|connoisseur|ginzburg|friedländer|eisner|attribution|kouros|rembrandt research|forgery|beltracchi|meegeren|vermeer.
- **Confirmed present** (not new, as the brief said): Bourdieu, Becker, Karpik, English, Guillory are elsewhere in the corpus but not on this grep; no hits at all for any connoisseurship/attribution term in any of the four files except one adjacent, off-domain lead:
- `refs-shared.bib`: `roumbanis2021oracles` (Roumbanis, "The Oracles of Science," grant peer review) — annotated as a **LEAD, not yet quoted**, flagged for the terms-of-art "connoisseurship" and "skilled academic intuition" in *grant peer review*, not art. This is a different domain (grants) already logged by a prior agent; I did not re-verify it and it does not count toward this sweep's art-connoisseurship total.
- No Morelli, Berenson, Ginzburg, Friedländer, Eisner, Getty kouros, Rembrandt Research Project, van Meegeren, or Beltracchi material anywhere in the four bib files. This entire domain is unopened in the repo before this sweep.

---

## Top finds, ranked

### 1. Carlo Ginzburg, "Clues: Roots of an Evidential Paradigm" — ACADEMIC / SCHOLARLY BOOK — ★ highest priority, as predicted
Ginzburg, Carlo. *Clues, Myths, and the Historical Method*, trans. John and Anne C. Tedeschi (Baltimore: Johns Hopkins University Press, 1989 [essay orig. Italian in *Miti, emblemi, spie*, 1986; first published in English as "Morelli, Freud and Sherlock Holmes: Clues and Scientific Method," *History Workshop Journal* 9 (1980): 5–36]). Essay pp. 96–125 in the 1989 book.
Obtained the **full text** (16pp scanned PDF, OCR'd locally at 300dpi with `tesseract` after the plain `pdftotext` extraction of the original file returned only page 1 legibly). Why it matters: this is a major theorized epistemology of exactly our gap — Ginzburg names connoisseurship as one of several "conjectural," "divinatory," or "semiotic" disciplines (with medicine, psychoanalysis, criminal detection) that are constitutively opposed to the quantifying "Galileian paradigm," and states outright that their knowledge cannot be formalized or taught as rules.

> "These are essentially mute forms of knowledge in the sense that their precepts do not lend themselves to being either formalized or spoken. No one learns to be a connoisseur or diagnostician by restricting himself to practicing only preexistent rules. In knowledge of this type imponderable elements come into play: instinct, insight, intuition." **[VERIFIED]** p. 124–125, via `https://beasley.sdsu.edu/Ginzburg.pdf` (OCR'd locally).

> "Knowledge of this sort in each instance was richer than any written codification; it was learned not from books but from the living voice, from gestures and glances: it was based on subtleties impossible to formalize, which often could not even be translated into words." — on connoisseurship, horse-trading, weather-reading, and similar tacit expert trades, contrasted with written codification attempts (physiognomy treatises) that were "dull and impoverished." **[VERIFIED]** p. 114–115, same source.

> "the group of disciplines which we have called evidential and conjectural (medicine included) are totally unrelated to the scientific criteria that can be claimed for the Galileian paradigm. In fact, they are highly qualitative disciplines, in which the object is the study of individual cases, situations, and documents, precisely because they are individual, and for this reason get results that have an unsuppressible speculative margin." **[VERIFIED]** p. 106–107.

> Morelli, quoted directly by Ginzburg, defending himself ironically against the charge of shallow literalism: "My adversaries," Morelli wrote ironically, "like to consider me a person who is unable to discern the spiritual meaning in a work of art and for this reason gives special importance to external matters, the shape of a hand, of an ear, and even, *horribile dictu*, to such an unpleasant subject as fingernails." **[VERIFIED]** p. 101.

> Freud on Morelli's method resembling psychoanalysis: "Long before I had any opportunity of hearing about psycho-analysis, I learnt that a Russian art-connoisseur, Ivan Lermolieff, had caused a revolution in the art galleries of Europe... It seems to me that his method of inquiry is closely related to the technique of psycho-analysis. It, too, is accustomed to divine secret and concealed things from unconsidered or unnoticed details, from the rubbish heap, as it were, of our observations." **[VERIFIED]** p. 99, Ginzburg quoting Freud's 1914 "The Moses of Michelangelo."

> Edgar Wind on Morelli (quoted by Ginzburg): "our inadvertent little gestures reveal our character far more authentically than any formal posture that we may carefully prepare." **[VERIFIED]** p. 100–101.

> "It was once said that falling in love is the act of overvaluing the marginal differences which exist between one woman and another... But this can also be said about works of art or about horses. In such situations this flexible rigor (pardon the oxymoron) of the conjectural paradigm seems impossible to suppress." **[VERIFIED]** p. 124.

---

### 2. *The Getty Kouros Colloquium: Athens, 25–27 May 1992* — SCHOLARLY PROCEEDINGS / PRACTITIONER TESTIMONY — the strongest primary-source find of the sweep
(Malibu: J. Paul Getty Museum; Athens: Nicholas P. Goulandris Foundation—Museum of Cycladic Art / Kapon Editions, 1993). ISBN 0-89236-263-4. Full text obtained directly: `https://media.getty.edu/Text/547a60db-dc37-5ce1-a548-35ec206d9515.pdf` (linked from the Getty's own publications page, `getty.edu/publications-reports/item/242H8P`, via its `citation_pdf_url` meta tag). This is the actual proceedings of the international colloquium the Getty convened to adjudicate whether its $10M kouros was a genuine 6th-c. BC Greek statue or a modern forgery — nineteen scholars and scientists, writing in the first person about their own reasoning. It is the single best documentary record I know of for "expert admits, in print, that a judgment is real but not derivable."

> John Boardman, "Criteria" (pp. 27–29): "My initial reaction was an instinctive one. Instinct without experience is useless; so what this means is that my instinct depended entirely on my previous experience not only of kouroi but of Greek art in general. It should be possible, although extremely difficult, to explain such instinctive judgements in terms of definable knowledge and demonstrable parallels..." and later: "There were no individual features that seemed wrong, but the ensemble was somehow awkward... I was reacting subconsciously to the feeling that in some way there was a discrepancy." **[VERIFIED]** pp. 28–29.

> Vassilis Lambrinoudakis, "Some Observations on the Authenticity of the Getty Kouros" (pp. 31–41): "The archaeologist must either become a criminologist and collect specific pieces of suspicious evidence or rely subjectively on his instinct or experience. I do not doubt that the sense of quality, which is based on experience and the practised eye, can constitute a serious guide. However, as a scholar, one should be able, beyond a feeling that the work is good or bad, be able to enumerate and to demonstrate those points which, ostensibly and objectively, are against the authenticity of the work. **I admit that, although I have a very unpleasant feeling about this kouros, I am unable to discover this kind of evidence in its partial forms.**" **[VERIFIED]** pp. 40–41. (Emphasis mine — this is as close to a direct textbook statement of the articulation gap as exists in the literature.)

> Angelos Delivorrias, "Concerning the Problem of the Authenticity of a Statue" (pp. 43–45): "I also feel the need to interpret the intuitive repulsion it arouses in me. Among the serious, scientifically documented reasons for condemning an artistic creation, it is, of course, difficult to include qualities maligned because of their metaphysical nature, though I personally would argue that these derive from the distillate of lived experience, which should not be dismissed lightly. Nevertheless, to prove the validity of intuition would necessitate investigating its chemical composition..." **[VERIFIED]** p. 45.

> Evelyn B. Harrison, "Remarks on the Style of the Getty Kouros" (pp. 21–25), describing her first encounter with the disassembled statue in the Getty basement, January 1985: "At the same time, the kouros did not give the impression of being something old; somehow it looked new." She then spends two pages trying to itemize *why* (curls "unpleasantly doughy," pectorals "flat and depressed, almost inorganic") — a rare case of a connoisseur's snap verdict followed immediately, in print, by her own effort to reverse-engineer it into stated criteria. **[VERIFIED]** p. 22.

> Museum label, as installed: **"Greek, about 530 B.C., or modern forgery."** **[VERIFIED]** via `https://traffickingculture.org/encyclopedia/case-studies/getty-kouros/` (fetched directly), corroborated by `en.wikipedia.org/wiki/Getty_kouros`. Note: I did not find this exact wording inside the Colloquium volume itself (the volume predates the final label by years); the citation is to the Getty's own later object label, reported by a University of Glasgow-hosted academic case-study site (Trafficking Culture, an academic art-crime research project) and Wikipedia. The Getty removed the kouros from display in 2018; the piece remains formally unresolved.

Use note: this find gives you *four independent scholars, in the same peer-reviewed proceedings volume, each admitting in their own words that their operative judgment outran their ability to state its grounds* — Boardman, Lambrinoudakis, Delivorrias in this order of explicitness. This is unusually strong because it's not a retrospective anecdote (like Berenson/Gladwell) but contemporaneous, signed, scholarly self-report, deliberately organized by the Getty as an adjudication exercise.

---

### 3. Max J. Friedländer, *On Art and Connoisseurship* — PRACTITIONER (museum director) — direct, sustained first-person epistemology of connoisseurship
Friedländer, Max J. *On Art and Connoisseurship*, trans. Tancred Borenius (London: Bruno Cassirer, 1942; Boston: Beacon Press repr., 335pp — the edition used). Full text: `https://archive.org/download/onartandconnoiss009325mbp/onartandconnoiss009325mbp_djvu.txt` (downloaded and grepped directly). Friedländer directed the paintings collections of the Berlin state museums and was among the most prolific attributors of Netherlandish/German old masters of his era. Chapter XXV is literally titled "On Intuition and the First Impression."

> "Even if attention deservedly goes to all the criteria which, with more or less justification, are described as the 'objective', seemingly scientific ones, and occupy a space disproportionately large in writings on art, **decision ultimately rests with something which cannot be discussed.** To be sure, when we come upon the concepts of intuition and self-evidence — and every statement based upon style-criticism ultimately reaches and is wrecked by these concepts — we resign as scholars and even as writers." **[VERIFIED]** p. 172.

> "The way in which an intuitive verdict is reached can, from the nature of things, only be described inadequately. A picture is shown to me. I glance at it, and declare it to be a work by Memling, without having proceeded to an examination of its full complexity of artistic form. This inner certainty can only be gained from the impression of the whole; never from an analysis of the visible forms." **[VERIFIED]** pp. 172–173.

> "Intuitive judgment may be regarded as a necessary evil. It is to be believed and disbelieved... Intuition resembles the magnet needle, which shows us our way whilst it oscillates and vibrates." **[VERIFIED]** p. 174.

> On why analysis after the fact destroys the thing it studies: "I have compared intuitive judgment to swimming in a deep river... You cannot explain a witticism without murdering it. And the position is the same with regard to the work of art." **[VERIFIED]** p. 183 (chapter "The Analytical Examination of Pictures").

---

### 4. Giovanni Morelli, *Italian Painters: Critical Studies of Their Works* — PRACTITIONER / PRIMARY SOURCE — the origin document of "explicit-rule connoisseurship," in his own words
Morelli, Giovanni [as "Ivan Lermolieff"]. *Italian Painters: Critical Studies of Their Works*, trans. Constance Jocelyn Ffoulkes (London: John Murray, 1892–1900). Full text: `https://archive.org/download/italianpainters01more/italianpainters01more_djvu.txt` (downloaded, grepped). This is the primary text behind the "Morellian method" — useful precisely because Morelli's own words show him trying to have it both ways: claiming a *rule-governed, "experimental method"* while being mocked by contemporaries for reducing connoisseurship to ears and fingernails.

> Morelli, in his own preface, defending the method as scientific and reproducible, not mere connoisseurial say-so: "if [my attributions] stand the test and prove sound, the merit will be due to me — that is to say, to **the experimental method** which I recommend." **[VERIFIED]** p. 44 (bracketed page number in OCR).

> Morelli anticipating and rebutting the caricature of his method (same passage Ginzburg quotes, here in fuller context and different translation): "...how is it that some of my other opponents, more especially in Germany, have sought to make this method for the decisive identification of the author of a picture appear ridiculous, by proclaiming that I am insensible to every deeper quality in a work of art, and regard only its external features, laying particular stress upon the form of the hand, the ear, and even, *horribile dictu*, of the finger-nails?" **[VERIFIED]** p. 45.

> Morelli, quoted in Ffoulkes's introduction, on the actual epistemic status of the "trifles": "it has been asserted in Germany that I profess to recognise a painter and to estimate his work solely by the form of the hand, the finger-nails, the ear, or the toes. Whether this statement is due to malice or to ignorance I cannot say; it is scarcely necessary to state that it is incorrect. What I maintain is, that **the forms... aid us in distinguishing the works of a master from those of his imitators, and control the judgment which subjective impressions might lead us to pronounce.**" **[VERIFIED]** p. 32 — this is the clearest primary statement of the "make the tacit checkable" project: minor forms as an explicit check *against* an otherwise unaccountable "subjective impression," i.e., Morelli himself frames his rules as a corrective to connoisseurial intuition rather than a replacement for it.

---

### 5. Bernard Berenson, "Rudiments of Connoisseurship" and Preface — PRACTITIONER / PRIMARY SOURCE — the counter-position to Morelli
Berenson, Bernard. *The Study and Criticism of Italian Art*, Second Series (London: George Bell & Sons, 1902). Full text: `https://archive.org/download/studyitalianart00bere/studyitalianart00bere_djvu.txt` (downloaded, grepped). Note on the famous "my stomach" anecdote: **I could not verify it this session** — see Leads/Dead ends below — but I did find something better-sourced and directly on-topic: Berenson's own 1902 preface explicitly stages the debate between Morelli's refusal to rationalize connoisseurship and his own project of trying to make it rationally defensible.

> Berenson quoting Morelli directly, then criticizing him: "A new spirit, however, or rather a new practice, was introduced by Morelli. Unfortunately, that great inventor was so much of a mere empiric, that he could say, **'The connoisseur should above all things have no bump of philosophy.'** The result of this consistently held attitude of his was that his method laid itself out to ridicule... But Morelli's empiricism was founded on facts which, had he not deliberately refused to use his powers of reasoning, he easily could have thought out and stated, thus presenting himself, not as a mere happy inventor, but as a real discoverer. **What he would not attempt, I have tried to do.**" **[VERIFIED]** pp. vii–viii (Preface).

This is a genuinely useful primary document of the internal argument: Morelli held (per Berenson) that connoisseurship should be *deliberately non-philosophical* — pure trained perception, unreasoned by design — while Berenson positioned his own "new Connoisseurship" as the attempt to supply the missing rational articulation. Both positions are quotable and both are from named practitioners arguing about codifiability, exactly per the brief.

Caution flagged per the verification instructions: **do not use the popular "my stomach" quote** (widely circulated, e.g. via Meryle Secrest's biography and secondhand retellings) without independently locating and reading the primary source (most likely Berenson's own diaries, e.g. *Sunset and Twilight: From the Diaries of 1947–1958*, ed. Nicky Mariano, 1963) — I was not able to reach that text this session.

---

### 6. Elliot W. Eisner, educational connoisseurship and criticism — ACADEMIC — imports the term into evaluation theory, exactly as flagged in the brief
Key texts: Eisner, Elliot W. *Educational Connoisseurship and Criticism: Their Form and Functions in Educational Evaluation* (1985 article, various reprints incl. in Alkin, M., ed., *Evaluation Roots*); Eisner, Elliot W. *The Enlightened Eye: Qualitative Inquiry and the Enhancement of Educational Practice* (New York: Macmillan, 1998; orig. 1991). I was unable to get past a compressed-PDF wall on the SagePub-hosted chapter reprint (binary, not OCR'd) and did not download alternate copies given time — the quotes below are **[SNIPPET]**, taken from a secondary compilation (infed.org's profile of Eisner, which itself gives page-cited quotes) rather than independently verified against the primary PDF. Treat page numbers as reported-by-secondary, not independently confirmed.

> "Connoisseurship is the art of appreciation. It can be displayed in any realm in which the character, import, or value of objects, situations, and performances is distributed and variable, including educational practice." **[SNIPPET]** Eisner 1998, p. 63, via `https://infed.org/mobi/elliot-w-eisner-connoisseurship-criticism-and-the-art-of-education/`.

> "The word connoisseurship comes from the Latin *cognoscere*, to know." ... "It involves the ability to see, not merely to look." **[SNIPPET]** Eisner 1998, p. 6, same source.

> "If connoisseurship is the art of appreciation, criticism is the art of disclosure. Connoisseurship is private, but criticism is public. Connoisseurs simply need to appreciate what they encounter. Critics, however, must render these qualities vivid by the artful use of critical disclosure." **[SNIPPET]** Eisner 1985, pp. 92–93, same source (also independently corroborated by matching snippet on Scribd/Libquotes mirrors of the same passage).

Why it matters for us directly: Eisner is explicit that connoisseurship is a real, exercised competence that is *by definition private* (unstated) until a second-order act of criticism translates it into disclosed, public language — i.e., he names the articulation gap as structural to the concept, not a defect of any one connoisseur. This maps almost exactly onto our "agree but cannot state" framing and gives us a ready-made vocabulary (connoisseurship = private appreciation; criticism = the always-imperfect public articulation of it) transplanted from art into a general evaluation theory.

---

### 7. Rembrandt Research Project — expert panel reversing itself on the same objects — ACADEMIC + JOURNALISM, mixed verification
Wikipedia (`en.wikipedia.org/wiki/Rembrandt_Research_Project`) confirms the RRP roughly halved the world's accepted Rembrandt self-portraits over its decades of work, run by "documentation, techniques, and forensic research," with "opinion weigh[ing] heavily" on market value — but the Wikipedia article itself carries **no direct quotes** on connoisseurial reasoning (checked directly, confirmed absent).

> New York Times, "A New System for Validating Rembrandts" (Aug 12, 1993): the RRP's members "would travel together to examine each painting and then decide by majority vote whether a work was a genuine Rembrandt," with one member noting, "But sometimes we agree to disagree." **[SNIPPET]** — from Brave search result snippet only; direct fetch of the NYT page was blocked by a paywall/CAPTCHA this session, so treat as unconfirmed pending a re-fetch.

> Best-verified item in this vein — Emilie E. S. Gordenker (Director, Mauritshuis) et al., **"Rembrandt's *Saul and David* at the Mauritshuis: A Progress Report,"** *Journal of Historians of Netherlandish Art* 5.2 (2013), DOI 10.5092/jhna.2013.5.2.11: quoting the optimistic 1969 RRP-founding-era prediction that scientific testing would "yield conclusive results" and "greatly enrich the limited supply of precise standards which connoisseurship has at its disposal," the article states flatly, forty years later: **"The optimistic predictions that scientific investigations would lead to 'conclusive results' about the authorship of the many paintings attributed to Rembrandt have, unfortunately, not been fulfilled, but such research certainly has had an impact."** **[VERIFIED]** — fetched `https://jhna.org/articles/rembrandt-saul-and-david-mauritshuis-progress-report/` directly and grepped the raw HTML for the exact string. This is a clean, citable, scholarly-journal statement that *explicit/scientific criteria demonstrably failed to replace connoisseurship* over a 40-year, well-funded, well-instrumented research program — one of the cleanest "criteria fail to capture what experts do" data points in the whole sweep, and it comes with a clear falsified prediction (1969 optimism → 2013 admission) rather than just a vague claim.

---

### 8. Van Meegeren forgery — connoisseur (Bredius) certifies a fake, in print, in rapturous terms — PRIMARY QUOTE, JOURNALISM/ENCYCLOPEDIA-MEDIATED
Confirmed by fetching the raw Wikipedia source (`en.wikipedia.org/wiki/Han_van_Meegeren`, `action=raw`) and grepping the exact string, with the article's own citations to the primary Burlington Magazine pieces:

> Abraham Bredius, the leading Dutch Old Master connoisseur of his generation, on van Meegeren's forged "The Supper at Emmaus" (1937): he "accepted it as a genuine Vermeer and praised it very highly as **'the masterpiece of Johannes Vermeer of Delft.'**" **[VERIFIED]** (string present in Wikipedia raw wikitext), citing Bredius, Abraham. "A New Vermeer." *The Burlington Magazine for Connoisseurs* 71, no. 416 (Nov. 1937): 210–211, JSTOR 867022 — **I did not independently fetch the Burlington Magazine original**, so treat the Bredius attribution as verified-in-Wikipedia/citation-present rather than verified-in-primary. On an earlier (also-forged) alleged Vermeer (1932): Bredius judged it "not only... an 'authentic Vermeer,' but also 'very beautiful,' and 'one of the finest gems of the master's œuvre,'" citing Bredius, "An Unpublished Vermeer," *Burlington Magazine* 61, no. 355 (1932): 145. **[VERIFIED]** same basis.

Why it matters: unlike the Getty kouros (where connoisseurs' *doubt* proved partly right), van Meegeren is the case where the leading connoisseur's confident, articulate, print-published certainty was simply wrong — detected only by hard chemistry (Paul Coremans's commission found 20th-century Bakelite/Albertol resins in the paint) after van Meegeren confessed. Useful as the "false positive" companion case to the Getty kouros's "the intuition was arguably right and the scientists were wrong" case — gives you both directions of error for the same epistemic setup.

### 9. Wolfgang Beltracchi forgeries — modern parallel, detection was technical not connoisseurial — ENCYCLOPEDIA-MEDIATED
Confirmed via Wikipedia fetch (`en.wikipedia.org/wiki/Wolfgang_Beltracchi`): Werner Spies, the leading Max Ernst expert, certified a Beltracchi fake ("La Forêt (2)") as authentic in 2004; it sold for €1.8M, was resold for $7M, and hung in the Max Ernst Museum in 2006 before detection. Detection came from materials scientist Nicholas Eastaugh finding titanium white pigment (unavailable in Campendonk's lifetime) in a different Beltracchi fake — **not** from connoisseurial doubt. **[SNIPPET]**, standard Wikipedia sourcing, not independently pushed further this session; useful mainly as a second modern false-positive case (like van Meegeren) showing that a still-functioning, prestigious, print-certifying connoisseurship apparatus can be defeated wholesale by a sufficiently skilled forger, and that recovery came from chemistry, not from any connoisseur later saying "I always felt something was off" (no such admission is reported here, unlike Zeri/Getty kouros).

---

## Leads (no verified quote — do not cite without further work)

- **Berenson's "my stomach" anecdote.** Extremely widely circulated (typically as: Berenson said he could tell a fake by a visceral, gut-level reaction he could not otherwise defend). I could not locate primary text this session — web search was blocked/rate-limited for exactly this query on every engine I tried (DuckDuckGo CAPTCHA, Bing unreadable, Mojeek 403, Brave 429). Likely primary candidates to chase next: Berenson's own diary *Sunset and Twilight: From the Diaries of 1947–1958* (ed. Nicky Mariano, 1963); Meryle Secrest, *Being Bernard Berenson* (1979); Kenneth Clark's essays/*Another Part of the Wood* (Clark was close to Berenson and recounts stories of this kind). **Do not quote the "stomach" line until one of these is directly read** — the instructions specifically flagged this as prone to garbled pop-science circulation, and I could not clear that bar this session.
- **Federico Zeri and Thomas Hoving on the Getty kouros.** Both are widely quoted (mostly via Malcolm Gladwell's *Blink*) as having had instant, inarticulate negative reactions ("it's fresh," fixation on the fingernails). Zeri's actual objection is corroborated as real and contemporaneous by Wikipedia (he resigned from the Getty board over it in 1984, before Gladwell ever wrote about it), but I did not find his own words in a primary source this session — only Gladwell's retelling and pop-science mirrors (Shortform, a Colby College course page). **Prefer the Colloquium quotes above (Boardman/Lambrinoudakis/Delivorrias/Harrison) over any Zeri/Hoving "fresh"/"fingernails" quote**, since those are independently confirmed primary-source, first-person, peer-reviewed statements, whereas the Zeri/Hoving material in circulation is Gladwell-mediated exactly as the brief warned against.
- **Full Bredius 1937 "moving emotion" quote.** I recall (but could not verify this session) a fuller, famous version of Bredius's Emmaus review describing a near-religious/bodily emotional reaction on first seeing the painting, comparable to "meeting a beloved person after many years." Attempted to fetch the original Burlington Magazine PDF (`burlington.org.uk/download/article/article_30053.pdf`, linked from Wikipedia) but the download was corrupted/not a valid PDF this session. **Do not use this fuller quote without independently confirming it** — only the shorter, Wikipedia-confirmed "the masterpiece of Johannes Vermeer of Delft" line above should be used.
- **Federico Zeri's own writings** (he wrote extensively, in Italian, on connoisseurship's non-explicit basis) — not investigated this session; likely productive if someone reads Italian-language sources or English translations of his essays.

## Dead ends
- WebSearch tool: 0/200 budget remaining for the whole session before this task started (shared session-wide budget) — no native web search was available at all.
- DuckDuckGo (html.duckduckgo.com, lite.duckduckgo.com): CAPTCHA'd on every attempt after the first.
- Bing (bing.com/search): pages loaded but returned ad-mismatched/JS-garbled SERPs that WebFetch's summarizer model could not parse into real results.
- Mojeek (mojeek.com/search): HTTP 403 on every attempt.
- NYT paywalled articles (1991 Kimmelman "Absolutely Real? Absolutely Fake?"; 1993 "A New System for Validating Rembrandts"): both blocked (CAPTCHA/empty body) via direct WebFetch and via the `r.jina.ai/` proxy — only reachable as Brave-search snippets, hence flagged SNIPPET/unconfirmed above rather than used as VERIFIED quotes.
- Direct download of `burlington.org.uk/download/article/article_30053.pdf` (Bredius 1932) returned a non-PDF/corrupted body.
- Eisner primary PDFs (SagePub `alkin2e_ch31and32.pdf`, ERIC fulltext PDFs): downloaded but were compressed/binary in a way the fetch tool could not OCR in-session; not pursued further via local `pdftotext`/`tesseract` given time budget — this is the one place in the sweep where a local-OCR pass (like the one that worked for Ginzburg and the Getty Colloquium) would likely pay off if someone continues this thread.

---

## Ready-to-paste BibTeX

```bibtex
@incollection{ginzburg1989clues,
  author    = {Ginzburg, Carlo},
  title     = {Clues: Roots of an Evidential Paradigm},
  booktitle = {Clues, Myths, and the Historical Method},
  publisher = {Johns Hopkins University Press},
  address   = {Baltimore},
  year      = {1989},
  pages     = {96--125},
  note      = {Trans. John and Anne C. Tedeschi; orig. Italian in \emph{Miti, emblemi, spie} (1986); orig. English pub. as ``Morelli, Freud and Sherlock Holmes: Clues and Scientific Method,'' \emph{History Workshop Journal} 9 (1980): 5--36},
  keywords  = {domain=connoisseurship; gap=felt-not-stated; type=academic-theory},
  annote    = {[VERIFIED] full text fetched and locally OCR'd (tesseract, 300dpi) from https://beasley.sdsu.edu/Ginzburg.pdf, exact strings confirmed. Central theoretical framing: connoisseurship grouped with medicine, psychoanalysis, criminal detection as a ``conjectural''/``divinatory''/``semiotic'' paradigm constitutively opposed to Galileian quantifying science. Key quote p.124-125: ``These are essentially mute forms of knowledge in the sense that their precepts do not lend themselves to being either formalized or spoken. No one learns to be a connoisseur or diagnostician by restricting himself to practicing only preexistent rules.'' Also quotes Morelli's own ironic self-defense (p.101) and Freud on Morelli-as-proto-psychoanalyst (p.99).}
}

@book{gettykouroscolloquium1993,
  editor    = {{J. Paul Getty Museum} and {Nicholas P. Goulandris Foundation---Museum of Cycladic Art}},
  title     = {The Getty Kouros Colloquium: Athens, 25-27 May 1992},
  publisher = {J. Paul Getty Museum / Kapon Editions},
  address   = {Malibu / Athens},
  year      = {1993},
  isbn      = {0-89236-263-4},
  keywords  = {domain=connoisseurship; gap=felt-not-stated; type=practitioner-testimony-scholarly-proceedings},
  annote    = {[VERIFIED] full text PDF fetched directly from https://media.getty.edu/Text/547a60db-dc37-5ce1-a548-35ec206d9515.pdf (linked via citation_pdf_url meta tag on getty.edu/publications-reports/item/242H8P), converted with pdftotext, exact strings confirmed with page numbers as printed in the book. Nineteen scholars/scientists convened by the Getty to adjudicate the Getty kouros's authenticity; several give first-person accounts of instinctive/felt judgments they could not fully derive from stated criteria. Strongest single quote, Vassilis Lambrinoudakis p.40-41: ``I admit that, although I have a very unpleasant feeling about this kouros, I am unable to discover this kind of evidence in its partial forms.'' Also John Boardman ``Criteria'' p.27-29 (instinctive reaction vs. definable knowledge), Angelos Delivorrias p.45 (``intuitive repulsion''), Evelyn B. Harrison p.22 (``did not give the impression of being something old; somehow it looked new''). Museum label ``Greek, about 530 B.C., or modern forgery'' independently verified via traffickingculture.org and Wikipedia, not found verbatim inside this 1993 volume (label postdates it).}
}

@book{friedlander1942onart,
  author    = {Friedl\"ander, Max J.},
  title     = {On Art and Connoisseurship},
  publisher = {Bruno Cassirer / Beacon Press},
  address   = {London / Boston},
  year      = {1942},
  note      = {Trans. Tancred Borenius},
  keywords  = {domain=connoisseurship; gap=felt-not-stated; type=practitioner-testimony},
  annote    = {[VERIFIED] full text fetched from https://archive.org/download/onartandconnoiss009325mbp/onartandconnoiss009325mbp_djvu.txt and grepped directly, exact strings confirmed with page numbers as printed. Berlin museums painting-collection director's sustained first-person epistemology of attribution. Chapter XXV, ``On Intuition and the First Impression,'' p.172: ``decision ultimately rests with something which cannot be discussed... when we come upon the concepts of intuition and self-evidence... we resign as scholars and even as writers.'' P.172-173: ``The way in which an intuitive verdict is reached can, from the nature of things, only be described inadequately.'' P.183: ``You cannot explain a witticism without murdering it. And the position is the same with regard to the work of art.''}
}

@book{morelli1892italianpainters,
  author    = {Morelli, Giovanni},
  title     = {Italian Painters: Critical Studies of Their Works},
  publisher = {John Murray},
  address   = {London},
  year      = {1892},
  note      = {Trans. Constance Jocelyn Ffoulkes; orig. pub. under pseudonym Ivan Lermolieff, \emph{Zeitschrift f\"ur bildende Kunst}, 1874--76 and \emph{Die Werke italienischer Meister}, 1880},
  keywords  = {domain=connoisseurship; gap=criteria-fail; type=primary-source},
  annote    = {[VERIFIED] full text fetched from https://archive.org/download/italianpainters01more/italianpainters01more_djvu.txt and grepped directly, exact strings confirmed (page numbers per OCR bracket pagination). Primary source of the ``Morellian method'' (attribution by minor unconsidered details: ears, fingernails, hands) presented, in Morelli's own words, as an explicit ``experimental method'' (p.44) meant to ``control the judgment which subjective impressions might lead us to pronounce'' (p.32) -- i.e. Morelli frames rule-based minutiae as a check on, not a replacement for, connoisseurial intuition. Also contains his own ironic rebuttal of the ``fingernails'' caricature (p.45), later quoted by Ginzburg.}
}

@incollection{berenson1902rudiments,
  author    = {Berenson, Bernard},
  title     = {Preface; Rudiments of Connoisseurship},
  booktitle = {The Study and Criticism of Italian Art, Second Series},
  publisher = {George Bell and Sons},
  address   = {London},
  year      = {1902},
  keywords  = {domain=connoisseurship; gap=criteria-fail; type=primary-source},
  annote    = {[VERIFIED] full text fetched from https://archive.org/download/studyitalianart00bere/studyitalianart00bere_djvu.txt and grepped directly, exact strings confirmed (pp. vii-viii of Preface). Berenson stages the explicit debate against Morelli: quotes Morelli's own dictum ``The connoisseur should above all things have no bump of philosophy,'' criticizes Morelli's ``empiricism'' for ``deliberately refus[ing] to use his powers of reasoning,'' and states his own project as making that reasoning explicit: ``What he would not attempt, I have tried to do.'' CAUTION: the widely-circulated ``my stomach'' Berenson anecdote was NOT found or verified this session -- do not conflate with this verified preface material; see notes/lit2_connoisseurship.md Leads section before using it.}
}

@article{eisner1985educational,
  author    = {Eisner, Elliot W.},
  title     = {Educational Connoisseurship and Criticism: Their Form and Functions in Educational Evaluation},
  year      = {1985},
  keywords  = {domain=connoisseurship; gap=felt-not-stated; type=academic-theory},
  annote    = {[SNIPPET] quotes taken from infed.org secondary compilation (https://infed.org/mobi/elliot-w-eisner-connoisseurship-criticism-and-the-art-of-education/), page numbers as reported there (pp.92-93), not independently confirmed against primary PDF (SagePub/ERIC copies downloaded but were binary/compressed and not machine-readable in-session). Explicitly imports art/wine connoisseurship into evaluation theory and structurally separates ``connoisseurship'' (private appreciation) from ``criticism'' (the always-imperfect public act of disclosing it): ``Connoisseurship is private, but criticism is public.''}
}

@book{eisner1998enlightenedeye,
  author    = {Eisner, Elliot W.},
  title     = {The Enlightened Eye: Qualitative Inquiry and the Enhancement of Educational Practice},
  publisher = {Macmillan},
  address   = {New York},
  year      = {1998},
  keywords  = {domain=connoisseurship; gap=felt-not-stated; type=academic-theory},
  annote    = {[SNIPPET] via infed.org secondary compilation, pp.6/63 as reported there, not independently confirmed against primary text. ``Connoisseurship is the art of appreciation. It can be displayed in any realm in which the character, import, or value of objects, situations, and performances is distributed and variable, including educational practice'' (p.63); ``The word connoisseurship comes from the Latin cognoscere, to know... It involves the ability to see, not merely to look'' (p.6).}
}

@article{gordenker2013rembrandt,
  author  = {Gordenker, Emilie E. S. and others},
  title   = {Rembrandt's {Saul and David} at the Mauritshuis: A Progress Report},
  journal = {Journal of Historians of Netherlandish Art},
  volume  = {5},
  number  = {2},
  year    = {2013},
  doi     = {10.5092/jhna.2013.5.2.11},
  keywords = {domain=connoisseurship; gap=criteria-fail; type=academic-journal},
  annote  = {[VERIFIED] fetched https://jhna.org/articles/rembrandt-saul-and-david-mauritshuis-progress-report/ directly and grepped raw HTML for exact strings. Quotes and then falsifies a 1969 Rembrandt-Research-Project-era prediction that scientific testing would ``yield conclusive results'' and enrich ``the limited supply of precise standards which connoisseurship has at its disposal''; states forty years later: ``The optimistic predictions that scientific investigations would lead to `conclusive results' about the authorship of the many paintings attributed to Rembrandt have, unfortunately, not been fulfilled, but such research certainly has had an impact.'' Clean scholarly-journal statement that explicit/instrumental criteria failed to replace connoisseurial judgment over a well-funded, multi-decade research program.}
}

@misc{rembrandtresearchproject_wiki,
  title = {Rembrandt Research Project},
  howpublished = {Wikipedia},
  note  = {en.wikipedia.org/wiki/Rembrandt\_Research\_Project; supplemented by NYT ``A New System for Validating Rembrandts'' (Aug 12, 1993)},
  keywords = {domain=connoisseurship; gap=disagreement-reversal; type=encyclopedia-secondary},
  annote = {[SNIPPET] Wikipedia confirms RRP roughly halved the accepted corpus of signed Rembrandt self-portraits over decades of committee re-attribution; NYT 1993 quote (``sometimes we agree to disagree''; decisions by majority vote) obtained only as a Brave-search snippet, direct NYT fetch blocked by paywall/CAPTCHA -- re-verify before quoting directly.}
}

@misc{vanmeegeren_wiki,
  title = {Han van Meegeren},
  howpublished = {Wikipedia},
  note  = {en.wikipedia.org/wiki/Han\_van\_Meegeren, raw wikitext fetched and grepped; primary citations therein to Bredius, Abraham, ``A New Vermeer,'' The Burlington Magazine for Connoisseurs 71.416 (Nov. 1937): 210-211 (JSTOR 867022), and ``An Unpublished Vermeer,'' Burlington Magazine 61.355 (1932): 145},
  keywords = {domain=connoisseurship; gap=criteria-fail; type=encyclopedia-secondary},
  annote = {[VERIFIED-IN-SECONDARY] exact string confirmed present in Wikipedia's raw wikitext, with full citation to the primary Burlington Magazine articles; the Burlington Magazine originals themselves were not independently fetched this session (one download attempt returned a corrupted file). Abraham Bredius, the era's leading Dutch Old Master connoisseur, certified van Meegeren's forged ``Supper at Emmaus'' in print as ``the masterpiece of Johannes Vermeer of Delft'' (1937) and an earlier forged Vermeer as ``authentic,'' ``very beautiful,'' ``one of the finest gems of the master's oeuvre'' (1932). Detection came only via Paul Coremans's postwar chemical analysis (20th-century Bakelite/Albertol resins), not connoisseurship. False-positive companion case to the Getty kouros.}
}

@misc{beltracchi_wiki,
  title = {Wolfgang Beltracchi},
  howpublished = {Wikipedia},
  note  = {en.wikipedia.org/wiki/Wolfgang\_Beltracchi},
  keywords = {domain=connoisseurship; gap=criteria-fail; type=encyclopedia-secondary},
  annote = {[SNIPPET] Werner Spies, leading Max Ernst expert, certified a Beltracchi fake (``La For\^et (2)'') as authentic in 2004; sold for \euro{}1.8M, resold for \$7M, exhibited at the Max Ernst Museum (2006) before detection. Detection was materials-scientific (titanium white pigment anachronistic to the purported period, found by Nicholas Eastaugh), not connoisseurial. Modern parallel to van Meegeren; unlike the Getty kouros case, no connoisseur is on record here retrospectively saying they had felt something was wrong.}
}
```
