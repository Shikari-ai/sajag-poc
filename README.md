# Sajag: proof of concept

**Can a phone catch scam messages and links on its own, the moment they arrive, without
sending them anywhere?**

**0.909 ROC AUC** (nested CV, 95% CI [0.869, 0.955]) from a **70 KB** linear model that
scores a message in **0.4 ms** on a laptop CPU. The test set is 798 synthetic messages in
English, Hinglish, Hindi and Telugu, and every test message uses wording the model never saw
in training.

The weakness, up front: at the default threshold **22.7% of genuine messages are
flagged**. Holding false alarms to 5% catches only **53% of scams**. Wording alone is
not enough (see [What this does not show](#what-this-does-not-show)).

> **On the hackathon rules.** This is a feasibility check run on a laptop, not the product.
> The Sajag Android app will be written from scratch during the on-site event window.
> Nothing here is app code, and none of it will be carried in.

---

## The design: zero-tap

Victims don't scan messages; they panic. So nothing here waits for a button press.

```
 SMS arrives ────────┐                          ┌─► link checker ── rules, explained ──┐
                     ├─► Android wakes Sajag ──►┤                                       ├─► risk ─► warning banner
 chat notification ──┘   (asleep otherwise)     └─► message model ── 70 KB, 0.4 ms ────┘
```

* **SMS:** a receiver for Android's SMS-received broadcast. Android wakes the app for each
  incoming SMS, even when it is closed.
* **WhatsApp, Telegram and other apps:** a notification listener gets each message
  notification as it is posted. It only sees what the notification shows.
* **Idle cost is zero.** Nothing runs between messages. On iQOO/OriginOS the user must allow
  autostart once, or the system blocks background wake-ups.
* **Nothing leaves the phone.** An app that reads every message is only acceptable if it
  never uploads one.
* **Play Store note:** Google restricts the SMS-received permission for Play Store apps. A
  sideloaded demo can use it; a store build would rely on the notification listener.

## The message model

```
 text ─► mask links (→ "URL") and digits (→ "0") ─► char 2–5-grams ─► hash into 2^18 buckets ─► logistic regression
```

* **Character n-grams** work the same on Latin, Devanagari and Telugu script. They survive
  the spelling chaos of romanised Hinglish, and need no tokenizer or vocabulary file.
* **Masking:** links are replaced so the link checker judges them. Digits are replaced so
  amounts and OTPs can never be memorised.
* **Size:** 11,978 non-zero weights, which is **70 KB** as sparse fp16. That's small enough
  to run on every message without touching the NPU.

## Results

Every number comes from `make eval`, which writes [`data/results.json`](data/results.json), or from
`make ablation` ([tools/ablation.py](tools/ablation.py)): the operating points and the cue-removal check.

**Corpus.** 798 messages from 133 templates across 16 topics:
* 408 scam, 390 genuine.
* Ten topics that scammers imitate (KYC, electricity, parcel, challan, reward points, loan,
  UPI, job, family, OTP) have **both a scam and a genuine version in every language**. The
  model cannot win by spotting the topic.

**Protocol.**
* **Grouped by template:** no wording is ever in both train and test.
* **Nested CV:** C is chosen inside each training fold only.
* **Repeated:** 5 split seeds.
* **Template-level bootstrap** for the confidence interval.

### Headline (message model alone)

| metric | value |
|---|---|
| ROC AUC | **0.909 ± 0.012** (95% CI [0.869, 0.955]) |
| accuracy @ 0.5 | 82.1% ± 1.7 |
| scam recall @ 0.5 | 86.6% ± 2.4 |
| false alarms @ 0.5 | **22.7% ± 1.4** |
| scam recall at ≤ 5% false alarms | 52.8% ± 4.2 |
| scam recall at ≤ 10% false alarms | 66.4% ± 6.7 |

### By language

| | English | Hinglish | Hindi | Telugu |
|---|---|---|---|---|
| AUC | 0.951 | 0.894 | 0.887 | 0.880 |
| false alarms @ 0.5 | 15.4% | 32.1% | 29.5% | 23.1% |

### Scam types it has never seen (leave-one-topic-out)

Pooled AUC is **0.873**, and **85.8%** of scams from unseen topics are caught. The failures
are informative:

| held-out topic | result |
|---|---|
| lottery / KBC | only **20%** caught |
| "Hi Mum, new number" impostor vs real family chat | AUC **0.569**, close to a coin toss |
| casual personal chat, never seen | **100%** flagged as scam |
| genuine KYC reminders | **80%** flagged |
| challan, parcel, digital arrest, investment | 100% caught, 0 false alarms |

### Languages it has never seen (zero-shot)

| trained on → tested on | AUC | recall @ 0.5 | AUC with link/number cues removed |
|---|---|---|---|
| English + Hinglish → Telugu | 0.932 | **7.7%** | 0.818 |
| English + Hinglish → Hindi | 0.930 | 33.3% | 0.730 |

The model still *ranks* unseen-language messages well, but its scores are badly off: at the
default threshold it misses 92% of Telugu scams. Some of that ranking comes from signals that
don't depend on language at all: the presence of a link or a phone number, plus Latin-script
words like "KYC", "OTP" and "Rs.". Removing links and numbers cuts even the in-language
grouped-CV AUC from 0.909 to **0.779** (seed 0). **The model leans on "has a link, has a
number" more than on reading the words.**

## The link checker

These are rules, not a model, and every flag comes with a plain-language reason:

```
$ make check MSG="Dear Customer, your SBI account will be blocked today due to pending KYC. Update PAN now: http://sbi-kyc-update.xyz/verify"
SCAM  (risk 1.00)
  message wording: 1.00
  link http://sbi-kyc-update.xyz/verify: 0.91
    - pretends to be SBI, but is not SBI's official site
    - .xyz domains are cheap and common in scams
    - connection is not encrypted (http)
    - unusually many hyphens in the name
```

It catches:
* brand impersonation in the domain name
* look-alike domains (`icicibnak.com`, `hdfcbank.co`)
* raw IP addresses
* disguised characters (punycode)
* link shorteners
* `.apk` app downloads
* cheap scam-favoured domain endings
* unencrypted links

Official bank and `.gov.in` sites score zero. It is unit-tested
([tests/test_links.py](tests/test_links.py)), but **it has no accuracy number**: we wrote
both the fake links in the corpus and the rules, so scoring one against the other would be
circular.

(Demo scores like the one above come from the shipped model trained on the whole corpus, and
this message is close to a training template. They are illustrations, not results. The tables
above are the results.)

## What this does not show

* **The corpus is synthetic and templated:** 133 machine-authored templates, each filled
  six times. The Hindi and Telugu templates need native-speaker review. Real scam SMS are
  messier, and real numbers will be lower.
* **No real messages yet.** The next step is a consented collection of real Indian scam and
  genuine SMS, used as a held-out test only.
* **False alarms are too high to show a warning on every message.** In the app, the message
  score has to be combined with the link checker and the sender (saved contact or unknown,
  DLT sender ID or a 10-digit number), and warn only above a stricter threshold. A
  multilingual text encoder on the NPU is the obvious upgrade for Hindi and Telugu.
* **The link checker is not measured** (see above).

## Reproduce

```bash
make install     # or: pip install -e ".[dev]"
make test        # 12 tests, including evaluation-integrity guards
make eval        # nested CV, leave-one-topic-out, zero-shot: ~4 minutes on a laptop CPU
make ablation    # operating points + shortcut check: ~1 minute
make check MSG="your message here"
```

No `make` (e.g. Windows)? Use `PYTHONPATH=src python -m sajag.cli eval`, `PYTHONPATH=src python
tools/ablation.py`, `PYTHONPATH=src python -m sajag.cli check "..."` and `python -m pytest`.

The tests guard the evaluation itself:
* no template ever spans train and test
* labels shuffled per template collapse to chance
* every imitated topic has both a scam and a genuine version in every language
* digits and links are masked before the model sees them

## License

MIT
