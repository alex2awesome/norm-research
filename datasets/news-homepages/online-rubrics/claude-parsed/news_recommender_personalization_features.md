---
source_url: https://arxiv.org/pdf/2106.08934
title: Personalized News Recommendation - Methods and Features
source_type: academic_paper
fetched: 2026-05-09
---

# Personalized News Recommendation — Algorithmic Newsworthiness Features

Survey of features used in personalized news recommendation systems (Wu et al., 2021, ACM TOIS). These features are the operational signals systems use to decide what news to surface.

## Feature Categories

### Content-Based Features
- **Title** — Text embedding
- **Body text** — Topic/topic model
- **Category / Topic** — Politics, sports, etc.
- **Entities** mentioned (NER on people, places, orgs)
- **Knowledge graph** entities
- **Sentiment / Emotion**
- **Length**
- **Reading difficulty**
- **Image presence and quality**

### Temporal / Recency Features
- **Publication time**
- **Decay / freshness score**
- **Time-of-day publishing**
- **Trending velocity**

### Engagement Features
- **Click-through rate (CTR)** — historical
- **Dwell time** — reading depth
- **Scroll depth**
- **Share count**
- **Comment count**
- **Like/reaction count**
- **Bounce rate** (negative signal)

### User-Based Features
- **Click history** — past articles
- **Topic affinity**
- **Source affinity**
- **Read time / dwell history**
- **Geographic location**
- **Device type**
- **Time of session**
- **Demographics** (where available)

### Social / Network Features
- **Friends' clicks**
- **Friends' shares**
- **Trending in social network**
- **Geographic neighbors' interests**

### Source / Publisher Features
- **Source reputation**
- **Source diversity**
- **Source recency** (when last good story)
- **Publisher size / brand**

### Diversity / Serendipity Features
- **Topic diversity** within feed
- **Source diversity** within feed
- **Anti-filter-bubble** mechanics
- **Novelty** — seen vs. unseen

## Implicit Newsworthiness Criteria from Algorithms

- **Predicted personal relevance** ≠ traditional news values
- **Predicted engagement** dominates
- **Recency** highly weighted
- **Topical match** over significance
- **Social proof** (peers liked it)
- **Geographic proximity** (when known)
- **Source familiarity / loyalty**
- **Diversity** is engineered, not organic

## Tensions

- Personalization vs. shared public information
- Engagement maximization vs. accuracy
- Filter bubbles vs. exposure to diverse views
- Click prediction vs. quality
- Algorithmic newsworthiness diverges from editorial newsworthiness

## Citation
Wu, C., Wu, F., Huang, Y., & Xie, X. (2022). Personalized News Recommendation: Methods and Challenges. ACM Transactions on Information Systems.
