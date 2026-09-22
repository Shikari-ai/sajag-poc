"""Operating points, the shortcut check, and the demo-pair scores quoted in the README and deck.

Nested CV picked C=16 on every fold, so fixed C=16 reproduces its out-of-fold scores.
"""

import re

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline

from sajag.corpus import build_corpus
from sajag.evaluate import arrays
from sajag.model import N_FEATURES, NGRAMS, URL_RE, make_model


def strip_cues(t: str) -> str:
    """Remove links and every number entirely, leaving only the words."""
    t = URL_RE.sub(" ", t)
    t = re.sub(r"[+\d][\d ,.\-]*\d|\d", " ", t)
    return t.lower()


def words_only_model():
    return make_pipeline(
        HashingVectorizer(analyzer="char_wb", ngram_range=NGRAMS, n_features=N_FEATURES,
                          alternate_sign=False, norm="l2", preprocessor=strip_cues),
        LogisticRegression(C=16.0, max_iter=3000, class_weight="balanced"),
    )


rows = build_corpus()
X, y, g = arrays(rows)
langs = np.array([r["lang"] for r in rows])

oofs = []
for s in range(5):
    p = np.zeros(len(y))
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=s).split(X, y, g):
        p[te] = make_model(16.0).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    oofs.append(p)
for fa in (0.05, 0.10):
    rec = [roc_curve(y, p)[1][roc_curve(y, p)[0] <= fa].max() for p in oofs]
    print(f"scam recall at <= {fa:.0%} false alarms: {np.mean(rec):.3f} +- {np.std(rec):.3f}")

for train_langs, test_lang in [(("en", "hinglish"), "te"), (("en", "hinglish"), "hi")]:
    tr, te = np.isin(langs, train_langs), langs == test_lang
    a = roc_auc_score(y[te], make_model(16.0).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1])
    b = roc_auc_score(y[te], words_only_model().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1])
    print(f"zero-shot {'+'.join(train_langs)} -> {test_lang}: with cues {a:.3f}, cues removed {b:.3f}")

p = np.zeros(len(y))
for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(X, y, g):
    p[te] = words_only_model().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
print(f"grouped CV, cues removed (seed 0): {roc_auc_score(y, p):.3f}")

# out-of-fold scores for the "same topic, opposite verdict" pairs in the pitch deck
mean_oof = np.mean(oofs, axis=0)
for tid in ["kyc.scam.en.0", "kyc.legit.en.0", "parcel.scam.te.0", "parcel.legit.te.0",
            "family.scam.en.0", "family.legit.en.0"]:
    print(f"out-of-fold score {tid:20s} {100 * mean_oof[g == tid].mean():5.1f}%")
