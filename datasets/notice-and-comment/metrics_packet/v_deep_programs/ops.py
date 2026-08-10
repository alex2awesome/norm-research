"""Op library for hybrid (code/prompt/tool) metric channels — seam pilot v1.

Two op types (proposal §3.3):
  COMPUTATION ops (Z = f(X), widen the executor):  normalize, extract_dates, sent_stats
  EVIDENCE op (Z touches corpus state beyond x):   retrieve_similar  (TF-IDF over the 5k corpus)

Hybrid program contract:
  LLM_FIELDS = {"field_name": "one-line extraction instruction"}   # optional, <=2, thick extractor
  def score(text: str, extracted: dict, ops: "Ops") -> float       # [0,1]
`extracted` holds the LLM's short answer per field ("" if NONE/missing).
"""
import json, pathlib, re

def _b2c(x):
    try:
        return bytes([x]).decode("cp1252")
    except UnicodeDecodeError:
        return chr(x)

_MOJIBAKE = {}
for _ch, _rep in [("\u201c", '"'), ("\u201d", '"'), ("\u2018", "'"), ("\u2019", "'"),
                  ("\u2013", "-"), ("\u2014", "-"), ("\u2026", "...")]:
    _b = _ch.encode("utf-8")
    _MOJIBAKE["".join(_b2c(x) for x in _b)] = _rep   # cp1252-style mojibake (incl. mixed \x9d)
    _MOJIBAKE[_b.decode("latin-1")] = _rep            # latin-1-style mojibake
    _MOJIBAKE[_ch] = _rep                             # already-unicode form
_MOJIBAKE["\xa0"] = " "

_DATE_PAT = re.compile(
    r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|"
    r"Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\.?\s+"
    r"\d{1,2}(?:\s*,\s*\d{4})?|\b\d{1,2}/\d{1,2}/\d{2,4}\b|\b(19|20)\d{2}\b")


class Ops:
    def __init__(self, corpus_path=None):
        self._vec = self._mat = self._ids = None
        if corpus_path:
            self._load_corpus(corpus_path)

    # ---------- computation ops ----------
    @staticmethod
    def normalize(text):
        """Fix mojibake/smart quotes/nbsp; collapse repeated whitespace. Z = f(X)."""
        for k in sorted(_MOJIBAKE, key=len, reverse=True):
            text = text.replace(k, _MOJIBAKE[k])
        return re.sub(r"[ \t]+", " ", text)

    @staticmethod
    def extract_dates(text):
        """All date-like strings found in the text (surface forms)."""
        return [m.group(0) for m in _DATE_PAT.finditer(text)]

    @staticmethod
    def sent_stats(text):
        """(n_sentences, mean_words_per_sentence, frac_long_words>=3syll-approx)."""
        sents = [s for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        words = re.findall(r"[A-Za-z']+", text)
        longw = sum(1 for w in words if len(w) >= 9)
        mw = (len(words) / len(sents)) if sents else 0.0
        return len(sents), mw, (longw / len(words) if words else 0.0)

    # ---------- evidence op ----------
    def _load_corpus(self, corpus_path):
        from sklearn.feature_extraction.text import TfidfVectorizer
        data = json.load(open(corpus_path))
        self._ids = [d["datapoint_id"] for d in data]
        texts = [d["text"][:8000] for d in data]
        self._vec = TfidfVectorizer(max_features=50000, stop_words="english")
        self._mat = self._vec.fit_transform(texts)

    def retrieve_similar(self, text, k=5, exclude_id=None):
        """Top-k most similar corpus docs: [(similarity, datapoint_id)]. Evidence op —
        touches corpus state beyond this document."""
        if self._vec is None:
            return []
        import numpy as np
        q = self._vec.transform([text[:8000]])
        sims = (self._mat @ q.T).toarray().ravel()
        order = np.argsort(-sims)
        out = []
        for i in order:
            if self._ids[i] == exclude_id:
                continue
            out.append((float(sims[i]), self._ids[i]))
            if len(out) >= k:
                break
        return out
