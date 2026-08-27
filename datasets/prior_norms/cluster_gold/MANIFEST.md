# Cluster-gold corpora — external validation of the L0→R1→R2→R3 hierarchy

Purpose: answer Sanmi's question — is there human-labeled cluster/code structure out
there we can validate our clustering P/R against? Each entry records what levels it
supplies, whether SEGMENT→LABEL LINKS exist, licence, and local path.

Two distinct things a gold can supply, and almost no corpus supplies both:
- **LINKS** — which surface item was assigned to which code (validates SEATING).
- **TREE** — how codes nest into themes into categories (validates LEVEL STRUCTURE).

| corpus | L0 surface | R1 code | R2 theme | R3 category | links | licence | status |
|---|---|---|---|---|---|---|---|
| **O*NET 30.0** | 18,797 task statements | 2,087 DWA | 332 IWA | 37 GWA | **yes** (23,851) | public domain (US DOL) | ✅ local |
| **Code review ESEM'23** | 1,829 review comments | 19 categories | 5 comment groups | — | **yes** | open (repo) | ✅ local |
| Scrum (Alami & Krancher 2022) | 131 segments | 35 codes | 14 themes | — | yes | CC-BY-4.0 | ✅ local |
| ATLAS.ti sustainability | 397 quotations | 155 codes | 20 tag categories | — | yes (1,167) | vendor sample | ✅ local |
| DISAPERE | 734 review sentences | — | 8 aspects | — | yes | open | ✅ local |
| **UCSB Ithaka (Dryad)** | 10 PDFs, **uncoded** | 93 leaf codes | 31 mid nodes | 10 roots | **no** | CC0 | ✅ local |
| Dagstuhl ArgQuality | 320 arguments (ratings) | 11 sub-dimensions | 3 dimensions | 1 overall | ratings only | open | ⬜ to fetch |
| PDTB 3.0 | ~53k relation spans | subtypes | 14 types | 4 classes | yes | LDC (check Stanford) | ⬜ to fetch |
| Fora (NYC deliberation) | quotes | 20 sublabels | 7 themes | — | yes | RAIL, gated | ⛔ pending access |
| Fiesler Reddit coded corpus | — | — | — | — | — | — | ⛔ unavailable |

## Entries

### O*NET 30.0 — the structural match (`onet/db_30_0_text/`)
US Dept. of Labor occupational taxonomy. The only corpus found with a genuine
FOUR-level human-built hierarchy AND surface→label links at every level:
Task Statement → DWA → IWA → GWA, each node a short phrase like our construct names
(e.g. DWA "Review art or design materials." ⊂ IWA "Study details of artistic
productions." ⊂ GWA 4.A.1.a.1). Files: `Task Statements.txt`, `Tasks to DWAs.txt`,
`DWA Reference.txt`, `IWA Reference.txt`, `Work Activities.txt`.
CAVEAT: work activities, not evaluative criteria — validates the machinery's level
structure and seating at scale, not norms-domain coverage.
Source: https://www.onetcenter.org/database.html

### Code review ESEM'23 — the on-domain match (`codereview_esem23/labeled.xlsx`)
Turzo & Bosu, "Towards Automated Classification of Code Review Feedback to Support
Analytics" (ESEM 2023). 1,829 **manually labelled** OpenStack Nova review comments;
`message` = L0, `category` (19 values: Solution Approach, Naming Convention, Logical,
Visual Representation, Documentation, Praise, Validation, …) = R1, `comment_group`
(5 values: REFACTORING, FUNCTION, DOCUMENTATION, DISCUSS, FALSE POSITIVE) = R2.
DIRECTLY ON-DOMAIN — code review is one of our census tasks.
DO NOT USE `busraicoz/crc-py-dataset` (16,711 comments) as gold: its labels are
SVM-propagated from a manual seed, not human. Auto-labelled ≠ gold.
Source: https://github.com/WSU-SEAL/CR-classification-ESEM23

### UCSB Ithaka (`ucsb_dryad_ithaka/`)
Curty, Greer & White, DOI 10.25349/D9402J, CC0. `coodebook.csv` is an NVivo
backslash-path tree: 123 nodes / 93 leaves / depth 1–4 (10 roots → 31 → 58 → 24).
`deid-transcripts.zip` is 10 raw PDFs with NO coded segments — TREE ONLY, NO LINKS.
41 of 93 leaves sit under `Course\Demographics…` and are ATTRIBUTE/facet coding
(Division, Mode, Offering), not thematic; the registered filter excludes that branch.

### Scrum, ATLAS.ti, DISAPERE
See ledger 2026-07-25a for results already computed on these.
