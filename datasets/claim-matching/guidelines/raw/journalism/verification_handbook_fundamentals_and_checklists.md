# Verification Handbook: "Verification Fundamentals: Rules to Live By" (Ch. 2, Steve Buttry) and "Creating a Verification Process and Checklist(s)" (Ch. 9, Craig Silverman & Rina Tsubaki)
SOURCE_URL: https://datajournalism.com/read/handbook/verification-1
DOMAIN: journalism

The Verification Handbook (edited by Craig Silverman, published by the European Journalism Centre) is the standard training text for verifying breaking-news claims and user-generated content (UGC). Two chapters are the most directly actionable for claim-matching methodology: Chapter 2, "Verification Fundamentals: Rules to Live By" (Steve Buttry), which sets out the epistemics of why sources are unreliable and how to cross-check them, and Chapter 9, "Creating a Verification Process and Checklist(s)" (Craig Silverman and Rina Tsubaki), which operationalizes that epistemics into a concrete, repeatable checklist.

## Chapter 2 — Verification Fundamentals (Steve Buttry)

### Why sources cannot be trusted at face value

Buttry frames verification around a hard lesson from his own reporting: while covering a high-school basketball championship in 1996, he interviewed twelve people who had watched the same game, and their memories of a specific sequence of events (how many charging fouls a player named Tanya Bopp drew) did not match the video evidence. Eyewitnesses whom he had every reason to consider honest and credible were simply wrong. The lesson he draws: **"Don't trust even honest witnesses. Seek documentation."** Sources are wrong for a range of reasons that a verifier must actively account for, not assume away:
- They may be lying, maliciously or for self-interest.
- They may be innocently repeating misinformation they themselves were told.
- They may have faulty memories, even about events they witnessed directly and recently.
- They may lack context or understanding of what they saw.
- They may be physically or situationally unable to see the full picture of an unfolding event.

### The two core verification questions

Buttry reduces the discipline of verification to two recurring questions that should be applied at every step of the reporting/checking chain:
1. **"How do you know that?"** — asked by the reporter of the source, by the editor of the reporter, and, for anything that cannot be interrogated directly, investigated independently.
2. **"How else do you know that?"** — the push for a second, independent path to the same fact, rather than resting on a single line of confirmation.

### Three ingredients of successful verification

Buttry identifies three factors that jointly determine whether a verification effort succeeds:
1. The verifier's own resourcefulness, persistence, skepticism, and skill.
2. The reliability, honesty, and *quantity/variety* of the sources available — more independent sources, and more varied types of sources, make an error easier to catch.
3. Documentation — physical or digital records (video, transcripts, logs, official documents) that do not suffer from the memory and perspective failures of human witnesses.

### Illustrative failure case

The chapter cites West Virginia Governor Joe Manchin's 2006 (Sago Mine) claim that twelve trapped miners had been rescued alive, when in fact only one survived — a case where an ostensibly authoritative official source repeated bad information without independent verification, and news organizations that echoed the claim without their own corroboration were also wrong. The operative rule drawn from this: **authority of the source (e.g., a governor, an official spokesperson) does not substitute for independent corroboration** — official status raises credibility but does not eliminate the need to verify.

### Team verification

Buttry stresses that verification "is a team sport": reporters, editors, and other newsroom colleagues should communicate about sourcing and cross-check each other's work rather than each verifying in isolation, since a second set of eyes catches errors the original reporter's closeness to the story may hide.

---

## Chapter 9 — Creating a Verification Process and Checklist(s) (Craig Silverman & Rina Tsubaki)

This chapter converts the fundamentals above into an explicit, three-step operational checklist for verifying a piece of user-generated content (an image, video, or claim circulating online) before it is reported as fact. The chapter's opening instruction: **put a plan and procedures in place for verification before you need them** — verification should not be improvised under deadline pressure, and different classes of content call for different verification paths.

### Baseline stance

Start from the assumption that any given piece of UGC **may be inaccurate, repurposed from another event, or stripped of its original context.** Do not repeat or trust a witness, victim, or authority's account of content without independent scrutiny — the same "how do you know that?" test from Chapter 2 applies here to digital content, not just human testimony.

### Step 1 — Identify and verify the original source

**Provenance / originality checks:**
- Search for identical or visually similar versions of the content already circulating elsewhere online (the same photo or video may have been posted before, under a different, truer context).
- Establish the actual timeline: when was it filmed/shot, when was it first uploaded, when was it shared into the channel you found it in.
- Identify the location depicted, via geotags or visible landmarks.
- Find any website(s) associated with the account or the content.
- Contact the person who shared or posted the content directly.

**For images:** run a reverse image search (Google, TinEye) and compare file sizes/resolutions across matches to find the earliest/original version. Examine EXIF metadata (via Photoshop, Fotoforensics.com, Findexif.com) for camera model, timestamp, and dimensions — noting that most social platforms strip EXIF data on upload (Flickr is a notable exception that preserves it), so absence of metadata is not itself suspicious. GPS/location data tools such as Geofeedia or Ban.jo can supplement this.

**For videos:** run keyword searches (including place names, acronyms, and likely foreign-language terms via Google Translate) across YouTube, Vimeo, and regional platforms (e.g., Youku), applying date filters to isolate the earliest upload.

**Source (uploader) evaluation** — build a profile of the account before trusting its content:
- Can the poster's identity be confirmed, and can they be contacted directly?
- What is their track record — have they posted reliable content before, or do they have a history of hoaxes/conspiracy content?
- How active is the account, and does its activity pattern look organic?
- What biographical information and linked profiles (other social accounts, personal sites) are available?
- Does their stated or inferred location match the content's claimed location?
- What does their follower/connection network and interaction pattern look like?
- Is the account platform-verified?
- Cross-reference the same username across multiple platforms to build a fuller picture; people-search tools (Pipl, WebMii, Spokeo, White Pages) and LinkedIn can help confirm real-world identity.

When assessing whether someone plausibly created the image/video themselves (rather than lifting it from elsewhere), adopt the shooter's point of view and ask: who would be positioned to shoot this, from this location, at this time, with this framing, and why would they be there?

**Content evaluation — date verification:**
- Cross-check depicted weather conditions against historical forecasts and against other uploads from the same purported event (Wolfram Alpha is cited as a weather-lookup tool).
- Search contemporaneous news coverage for independent confirmation the depicted event happened when claimed.
- Search for earlier UGC from the same event that might better anchor the timeline.
- Look for visible temporal markers within the frame itself — clocks, screens, dated newspapers.

**Content evaluation — location verification:**
- Check for embedded/automated geolocation metadata.
- Read visible textual clues: shop signage, street signs, license plates, billboards, language on posters.
- Match distinctive natural landscape features (mountains, tree lines, rivers) and built landmarks (churches, stadiums, bridges) against known imagery.
- Cross-reference against Google Street View, Google Maps' photo layers, and Google Earth's historical satellite/terrain imagery; Wikimapia for landmark identification.
- Use sun position and shadow angles to estimate time of day and, combined with date, corroborate the season/location claim.
- For video specifically, also check: whether spoken language, accent, or dialect matches the claimed location; whether on-screen descriptions are internally consistent and specific; whether logos are consistent across multiple videos from the same purported source; whether the content is original or scraped from elsewhere; the technical file signature (e.g., ".MP4" vs ".AVI", or an "Uploaded via YouTube Capture" tag) which can indicate direct-from-device filming versus repurposed/edited footage.

### Step 2 — Triangulate and challenge the source

Before accepting a piece of content as verified, ask:
- Are other, independent media outlets or organizations distributing the same or corroborating material?
- Has this content (or claim) already been addressed by a fact-checking outlet such as Snopes?
- Does anything about the story feel implausible or "too convenient" given what is otherwise known?

When you do reach the original source, ask direct, specific questions and check their answers against what your independent research has already found — consistency (or its absence) between their account and your documented findings is itself diagnostic. For images, reflect your own EXIF/geolocation findings back to the source during questioning (without revealing exactly what you found, to test whether their account matches independently), and ask for additional images shot immediately before/after the one in question as a further authenticity check. For videos, when doubts remain about how a clip was constructed or edited, do a frame-by-frame review using tools such as VLC, Avidemux, or Vegas Pro to look for splice points, inconsistent lighting, or other manipulation artifacts.

### Step 3 — Obtain permission (post-verification)

Once content is verified, before publishing: confirm exactly which image/video will be used, explain how it will be used, clarify the credit the source wants (real name, pseudonym, or anonymity), and assess any safety or privacy risk to the source from being identified — including whether faces need to be blurred.

### Preparedness (building the infrastructure for fast verification)

The chapter also recommends pre-building, before a breaking story occurs: curated lists of trusted official and unofficial sources (first responders, academics, NGOs) organized by topic/geography (e.g., via Twitter Lists, Facebook Interest Lists); relationships with these sources cultivated as ongoing professional contacts rather than one-off tips; and internal newsroom protocols (toolsets, workflows, sign-off/approval steps, and communication chains) agreed upon in advance so that verification standards do not slip under deadline pressure.

## Relevance to claim-matching

Chapters 2 and 9 together supply a two-tier test that generalizes beyond breaking-news UGC to any claim-vs-evidence matching task: (1) a *source* tier — is the originator of the claim/content identifiable, independently corroborated by a second source, and free of a history of unreliability; and (2) a *content* tier — do the internal details of the claim (time, place, described specifics) survive cross-checking against independent, ideally primary, documentation. A "match" between claim and evidence is only as strong as the weakest of these two tiers, and a rich set of concrete, checkable sub-tests (reverse image search, EXIF check, weather/shadow cross-check, dialect/language check, cross-platform corroboration, "has this already been debunked") is provided for operationalizing both tiers.
