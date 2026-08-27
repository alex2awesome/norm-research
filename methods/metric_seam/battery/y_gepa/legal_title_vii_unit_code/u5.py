import re

def score(text: str) -> float:
    try:
        if not text or not text.strip():
            return 0.0
        t = text.lower()
        words = re.findall(r"[a-z]+", t)

        if not words:
            return 0.0

        stats_terms = {
            "statistic", "statistical", "statistics", "underrepresentation",
            "underrepresented", "under-representation", "disparity",
            "disparities", "disproportionate", "disproportionately",
            "pattern", "practice", "systemic", "widespread", "pervasive",
            "numbers", "percentage", "percent", "proportion", "ratio",
            "workforce", "composition", "demographic", "concentration",
            "segregation", "segregated", "imbalance", "representation",
            "represented", "minority", "minorities", "women", "female",
            "african", "hispanic", "protected", "class"
        }
        proof_terms = {
            "evidence", "show", "shows", "showed", "shown", "demonstrate",
            "demonstrates", "demonstrated", "establish", "established",
            "establishes", "reveal", "reveals", "revealed", "indicate",
            "indicates", "indicated", "prove", "proves", "proven", "finding",
            "findings", "determined", "concluded", "support", "supports",
            "supported", "documented", "historical", "prior", "previous",
            "data", "analysis", "study", "studies", "report", "reports",
            "comparison", "compared", "measured", "calculation", "formula",
            "formulas", "algorithm", "methodology", "result", "results"
        }

        stats_hits = sum(1 for w in words if w in stats_terms)
        proof_hits = sum(1 for w in words if w in proof_terms)

        density = (stats_hits + proof_hits) / len(words)

        score_val = min(1.0, density * 25.0)

        phrases = [
            r"pattern (?:or|and) practice",
            r"prior (?:finding|findings) (?:of )?discrimination",
            r"previous (?:finding|findings) (?:of )?discrimination",
            r"under[- ]?representation",
            r"statistical (?:evidence|analysis|study|studies|data|proof|showing|significance)",
            r"(?:flawed|defective|biased|discriminatory) (?:formula|formulas|promotion|promotional|test|exam|procedure|criterion|criteria|algorithm|policy|policies)",
            r"disparate (?:impact|treatment)",
            r"(?:workforce|labor force) (?:composition|demographic|statistics|data)",
            r"comparator(?:s)? (?:data|statistics|evidence|group)",
            r"compelling (?:statistical|evidence|showing)",
            r"(?:failed|refused|lack(?:s|ed)?) (?:to hire|to promote)",
            r"protected (?:class|group|characteristic)",
            r"title vii",
            r"adverse impact",
            r"(?:showed|shows|demonstrated|revealed|established|indicated) (?:a |an |that )?(?:pattern|practice|disparity|disparities|underrepresentation|under-representation|statistical)",
            r"hiring (?:and|or) promotion",
            r"selection (?:rate|procedure|hiring)",
            r"force[d]? (?:comparison|rate)",
            r"labor market",
            r"standard deviation",
            r"binomial",
            r"regression analysis",
            r"flow (?:rate|statistics)",
            r"affected class"
        ]

        phrase_count = 0
        for p in phrases:
            try:
                c = len(re.findall(p, t))
                phrase_count += c
            except re.error:
                continue

        score_val += min(0.6, phrase_count * 0.06)

        bigrams = {
            ("statistical", "evidence"), ("prior", "findings"),
            ("statistical", "data"), ("under", "represented"),
            ("protected", "class"), ("protected", "group"),
            ("hiring", "promotion"), ("flawed", "formula"),
            ("disparate", "impact"), ("workforce", "composition"),
            ("pattern", "practice"), ("statistical", "analysis"),
            ("standard", "deviation"), ("selection", "rate"),
            ("compelling", "evidence"), ("regression", "analysis"),
            ("adverse", "impact"), ("labor", "market"),
            ("statistical", "disparity"), ("historical", "evidence")
        }
        bigrams_found = 0
        for i in range(len(words) - 1):
            if (words[i], words[i + 1]) in bigrams:
                bigrams_found += 1

        score_val += min(0.35, bigrams_found * 0.05)

        return round(min(1.0, score_val), 4)

    except Exception:
        return 0.5