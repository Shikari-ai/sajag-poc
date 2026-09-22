"""Message classifier sized for a phone: hashed character n-grams + a linear head.

Character n-grams work across scripts (Latin, Devanagari, Telugu) and across spelling chaos
in romanised Hinglish, need no tokenizer or vocabulary file, and make the whole model one
sparse weight vector. URLs are masked (the link checker owns them) and digits are masked
(so amounts and OTPs can never be memorised).
"""

from __future__ import annotations

import re

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline

from .links import URL_RE, check_url, extract_urls

N_FEATURES = 2**18
NGRAMS = (2, 5)
C_GRID = (0.25, 1.0, 4.0, 16.0)


def normalize(text: str) -> str:
    text = URL_RE.sub(" URL ", text)
    text = re.sub(r"\d", "0", text)
    return text.lower()


def make_model(C: float = 4.0) -> Pipeline:
    return make_pipeline(
        HashingVectorizer(analyzer="char_wb", ngram_range=NGRAMS, n_features=N_FEATURES,
                          alternate_sign=False, norm="l2", preprocessor=normalize),
        LogisticRegression(C=C, max_iter=3000, class_weight="balanced"),
    )


def assess(model: Pipeline, text: str) -> dict:
    """Full verdict for one message: text score, per-link verdicts, combined risk."""
    p_text = float(model.predict_proba([text])[0, 1])
    links = [check_url(u) for u in extract_urls(text)]
    p_link = max((lv.risk for lv in links), default=0.0)
    risk = 1 - (1 - p_text) * (1 - p_link)
    return {"risk": risk, "text_risk": p_text, "links": links}
