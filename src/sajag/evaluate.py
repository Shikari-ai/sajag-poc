"""Evaluation harness. Every number in the README is produced here.

Protocol:
* groups = template_id: no wording is ever in both train and test;
* nested CV: the regularisation strength C is chosen inside each training fold only;
* 5 split seeds, reported as mean +- sd;
* template-level bootstrap for confidence intervals;
* leave-one-topic-out: whole scam types the model has never seen;
* zero-shot cross-lingual: languages the model has never seen.
"""

from __future__ import annotations

import time
from collections import Counter

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from .corpus import LANGS, build_corpus
from .model import C_GRID, make_model

SEEDS = range(5)


def arrays(rows: list[dict]):
    X = np.array([r["text"] for r in rows], dtype=object)
    y = np.array([r["label"] for r in rows])
    g = np.array([r["template_id"] for r in rows])
    return X, y, g


def metrics(y, p, thr: float = 0.5) -> dict:
    pred = p >= thr
    out = {"n": int(len(y)), "accuracy": float(np.mean(pred == y))}
    if len(set(y)) == 2:
        out["auc"] = float(roc_auc_score(y, p))
    if (y == 1).any():
        out["scam_recall"] = float(pred[y == 1].mean())
    if (y == 0).any():
        out["false_alarm"] = float(pred[y == 0].mean())
    return out


def select_C(X, y, g, seed: int) -> float:
    inner = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed)
    splits = list(inner.split(X, y, g))
    best, best_auc = C_GRID[0], -1.0
    for C in C_GRID:
        aucs = [roc_auc_score(y[te], make_model(C).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1])
                for tr, te in splits]
        if np.mean(aucs) > best_auc:
            best, best_auc = C, float(np.mean(aucs))
    return best


def nested_oof(X, y, g, seed: int):
    """Out-of-fold scam probabilities; C picked on the training fold only."""
    oof = np.zeros(len(y))
    chosen = []
    for tr, te in StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed).split(X, y, g):
        C = select_C(X[tr], y[tr], g[tr], seed)
        chosen.append(C)
        oof[te] = make_model(C).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return oof, chosen


def bootstrap_auc(y, p, g, n: int = 2000, seed: int = 0):
    rng = np.random.default_rng(seed)
    ids = np.unique(g)
    idx_by = {t: np.flatnonzero(g == t) for t in ids}
    vals = []
    for _ in range(n):
        idx = np.concatenate([idx_by[t] for t in rng.choice(ids, size=len(ids))])
        if len(set(y[idx])) == 2:
            vals.append(roc_auc_score(y[idx], p[idx]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def leave_one_topic_out(X, y, g, topics):
    pooled = np.zeros(len(y))
    for t in np.unique(topics):
        tr = topics != t
        C = select_C(X[tr], y[tr], g[tr], seed=0)
        pooled[~tr] = make_model(C).fit(X[tr], y[tr]).predict_proba(X[~tr])[:, 1]
    per_topic = {t: metrics(y[topics == t], pooled[topics == t]) for t in np.unique(topics)}
    return metrics(y, pooled), per_topic


def cross_lingual(X, y, g, langs):
    out = {}
    for train_langs, test_lang in [(("en", "hinglish"), "hi"), (("en", "hinglish"), "te"),
                                   (("en", "hinglish", "hi"), "te")]:
        tr = np.isin(langs, train_langs)
        te = langs == test_lang
        C = select_C(X[tr], y[tr], g[tr], seed=0)
        p = make_model(C).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        out[f"{'+'.join(train_langs)} -> {test_lang}"] = metrics(y[te], p)
    return out


def efficiency(X, y, C: float):
    model = make_model(C).fit(X, y)
    coef = model[-1].coef_.ravel()
    nnz = int(np.count_nonzero(coef))
    # sparse on-device format: int32 index + float16 weight per non-zero
    size_kb = nnz * (4 + 2) / 1024
    msgs = list(X[:200])
    t0 = time.perf_counter()
    for _ in range(5):
        for m in msgs:
            model.predict_proba([m])
    ms = (time.perf_counter() - t0) / (5 * len(msgs)) * 1000
    return {"nonzero_weights": nnz, "size_kb_sparse_fp16": round(size_kb, 1),
            "ms_per_message_laptop_cpu": round(ms, 2), "C": C}


def run(fills: int = 6) -> dict:
    rows = build_corpus(fills=fills)
    X, y, g = arrays(rows)
    langs = np.array([r["lang"] for r in rows])
    topics = np.array([r["topic"] for r in rows])

    per_seed, oofs, chosen = [], [], []
    for s in SEEDS:
        oof, Cs = nested_oof(X, y, g, s)
        oofs.append(oof)
        chosen += Cs
        per_seed.append(metrics(y, oof))
    summary = {k: (float(np.mean([m[k] for m in per_seed])), float(np.std([m[k] for m in per_seed])))
               for k in ("auc", "accuracy", "scam_recall", "false_alarm")}
    mean_oof = np.mean(oofs, axis=0)
    ci = bootstrap_auc(y, mean_oof, g)
    by_lang = {lang: metrics(y[langs == lang], mean_oof[langs == lang]) for lang in LANGS}

    lofo, lofo_topics = leave_one_topic_out(X, y, g, topics)
    xling = cross_lingual(X, y, g, langs)
    c_mode = Counter(chosen).most_common(1)[0][0]
    eff = efficiency(X, y, c_mode)

    return {
        "corpus": {"messages": int(len(y)), "templates": int(len(np.unique(g))),
                   "scam": int(y.sum()), "legit": int((y == 0).sum()),
                   "topics": int(len(np.unique(topics))), "langs": list(LANGS)},
        "nested_cv": summary, "auc_ci95_template_bootstrap": ci,
        "by_language": by_lang, "leave_one_topic_out": lofo,
        "leave_one_topic_out_per_topic": lofo_topics, "cross_lingual_zero_shot": xling,
        "efficiency": eff, "C_chosen": dict(Counter(chosen)),
    }
