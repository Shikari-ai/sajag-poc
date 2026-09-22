"""sajag corpus | train | eval | check "message text" """

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib

DATA = Path("data")
MODEL = DATA / "model.joblib"


def _utf8() -> None:
    for s in (sys.stdout, sys.stderr):
        if hasattr(s, "reconfigure"):
            s.reconfigure(encoding="utf-8")


def cmd_corpus(_):
    from .corpus import build_corpus
    rows = build_corpus()
    DATA.mkdir(exist_ok=True)
    with open(DATA / "corpus.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} messages to data/corpus.jsonl")


def cmd_train(_):
    from .corpus import build_corpus
    from .evaluate import arrays, select_C
    from .model import make_model
    X, y, g = arrays(build_corpus())
    C = select_C(X, y, g, seed=0)
    DATA.mkdir(exist_ok=True)
    joblib.dump(make_model(C).fit(X, y), MODEL)
    print(f"trained on {len(y)} messages (C={C}) -> {MODEL}")


def cmd_eval(_):
    from .evaluate import run
    res = run()
    DATA.mkdir(exist_ok=True)
    (DATA / "results.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(res, indent=2, ensure_ascii=False))


def cmd_check(args):
    from .model import assess
    if not MODEL.exists():
        cmd_train(args)
    model = joblib.load(MODEL)
    v = assess(model, args.text)
    verdict = "SCAM" if v["risk"] >= 0.8 else "SUSPICIOUS" if v["risk"] >= 0.5 else "LOOKS SAFE"
    print(f"{verdict}  (risk {v['risk']:.2f})")
    print(f"  message wording: {v['text_risk']:.2f}")
    for lv in v["links"]:
        print(f"  link {lv.url}: {lv.risk:.2f}")
        for r in lv.reasons:
            print(f"    - {r}")


def main() -> None:
    _utf8()
    p = argparse.ArgumentParser(prog="sajag")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("corpus").set_defaults(fn=cmd_corpus)
    sub.add_parser("train").set_defaults(fn=cmd_train)
    sub.add_parser("eval").set_defaults(fn=cmd_eval)
    c = sub.add_parser("check")
    c.add_argument("text")
    c.set_defaults(fn=cmd_check)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
