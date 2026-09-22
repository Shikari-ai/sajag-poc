"""Guards that keep the headline numbers honest."""

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from sajag.corpus import LANGS, TEMPLATES, build_corpus
from sajag.evaluate import arrays
from sajag.model import make_model, normalize


def test_corpus_is_deterministic():
    assert build_corpus(seed=0) == build_corpus(seed=0)


def test_no_template_spans_train_and_test():
    X, y, g = arrays(build_corpus())
    for tr, te in StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0).split(X, y, g):
        assert not set(g[tr]) & set(g[te])


def test_imitated_topics_have_both_labels_in_every_language():
    paired = [t for t, labels in TEMPLATES.items() if {"scam", "legit"} <= set(labels)]
    assert len(paired) >= 10
    for t in paired:
        for lang in LANGS:
            assert TEMPLATES[t]["scam"].get(lang) and TEMPLATES[t]["legit"].get(lang), (t, lang)


def test_digits_and_urls_are_masked():
    s = normalize("Pay Rs.4500 now at http://sbi-kyc.xyz/v OTP 123456")
    assert not any(ch.isdigit() and ch != "0" for ch in s)
    assert "sbi-kyc" not in s and "url" in s


def test_shuffled_labels_collapse_to_chance():
    rows = build_corpus()
    X, _, g = arrays(rows)
    rng = np.random.default_rng(0)
    fake = {t: int(rng.integers(0, 2)) for t in np.unique(g)}  # label per template, not per row
    y = np.array([fake[t] for t in g])
    p = np.zeros(len(y))
    for tr, te in StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0).split(X, y, g):
        p[te] = make_model().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    assert 0.3 < roc_auc_score(y, p) < 0.7
