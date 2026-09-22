"""Offline link checker: explainable rules, no network, no blocklist download.

Every flag comes with a plain-language reason, because a warning a user cannot understand
is a warning they ignore. Rules, not a model — so there is no accuracy number for this part
(see README: we wrote both the fake links and the rules, so measuring one against the other
would be circular).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from urllib.parse import urlsplit

URL_RE = re.compile(r"(?:https?://|www\.)[^\s<>\"']+", re.IGNORECASE)

OFFICIAL = {
    "onlinesbi.sbi": "SBI", "sbi.co.in": "SBI", "hdfcbank.com": "HDFC Bank",
    "icicibank.com": "ICICI Bank", "axisbank.com": "Axis Bank", "kotak.com": "Kotak",
    "canarabank.com": "Canara Bank", "pnbindia.in": "PNB", "bankofbaroda.in": "Bank of Baroda",
    "paytm.com": "Paytm", "phonepe.com": "PhonePe", "npci.org.in": "NPCI",
    "amazon.in": "Amazon", "flipkart.com": "Flipkart", "irctc.co.in": "IRCTC",
    "delhivery.com": "Delhivery", "bluedart.com": "Blue Dart", "naukri.com": "Naukri",
}
OFFICIAL_SUFFIXES = ("gov.in", "nic.in")
# brand token in a hostname -> who it pretends to be
BRANDS = {
    "sbi": "SBI", "yono": "SBI YONO", "hdfc": "HDFC Bank", "icici": "ICICI Bank",
    "axis": "Axis Bank", "kotak": "Kotak", "canara": "Canara Bank", "pnb": "PNB",
    "paytm": "Paytm", "phonepe": "PhonePe", "gpay": "Google Pay", "amazon": "Amazon",
    "flipkart": "Flipkart", "parivahan": "Parivahan", "echallan": "e-Challan",
    "indiapost": "India Post", "aadhaar": "Aadhaar", "uidai": "UIDAI",
    "incometax": "Income Tax", "npci": "NPCI", "kbc": "KBC", "tgspdcl": "TGSPDCL",
}
SHORTENERS = {"bit.ly", "tinyurl.com", "cutt.ly", "is.gd", "t.co", "rb.gy", "shorturl.at",
              "goo.gl", "tiny.cc"}
CHAT_INVITES = {"wa.me", "t.me", "chat.whatsapp.com"}
RISKY_TLDS = {"xyz", "top", "click", "info", "live", "online", "site", "buzz", "icu", "vip",
              "shop", "cc", "rest", "monster", "cfd", "sbs"}
MULTI_SUFFIX = {"co.in", "org.in", "net.in", "gov.in", "ac.in", "nic.in", "co.uk", "com.au"}
IP_RE = re.compile(r"^\d{1,3}(?:\.\d{1,3}){3}$")


@dataclass
class LinkVerdict:
    url: str
    risk: float
    reasons: list[str] = field(default_factory=list)


def extract_urls(text: str) -> list[str]:
    return [u.rstrip(".,;:!?)") for u in URL_RE.findall(text)]


def registered_domain(host: str) -> str:
    parts = host.split(".")
    if len(parts) >= 3 and ".".join(parts[-2:]) in MULTI_SUFFIX:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:])


def _levenshtein(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def check_url(url: str) -> LinkVerdict:
    parts = urlsplit(url if "://" in url else "http://" + url)
    host = (parts.hostname or "").lower()
    if not host:
        return LinkVerdict(url, 0.5, ["link could not be read"])
    rd = registered_domain(host)
    if rd in OFFICIAL:
        return LinkVerdict(url, 0.0, [f"official {OFFICIAL[rd]} website"])
    if any(host == s or host.endswith("." + s) for s in OFFICIAL_SUFFIXES):
        return LinkVerdict(url, 0.0, ["official government (.gov.in) website"])

    flags: list[tuple[float, str]] = []
    if IP_RE.match(host):
        flags.append((0.7, "points to a raw IP address instead of a website name"))
    if "xn--" in host or not host.isascii():
        flags.append((0.7, "uses look-alike characters to imitate a real name"))
    if rd in SHORTENERS:
        flags.append((0.5, "shortened link hides where it really goes"))
    if rd in CHAT_INVITES:
        flags.append((0.15, "opens a chat with an unknown number or group"))
    for token, brand in BRANDS.items():
        if token in host:
            flags.append((0.8, f"pretends to be {brand}, but is not {brand}'s official site"))
            break
    label = rd.split(".")[0]
    for off in OFFICIAL:
        if rd != off and (label == off.split(".")[0] or _levenshtein(rd, off) <= 2):
            flags.append((0.8, f"look-alike of the real site {off}"))
            break
    tld = rd.rsplit(".", 1)[-1]
    if tld in RISKY_TLDS:
        flags.append((0.3, f".{tld} domains are cheap and common in scams"))
    if parts.path.lower().endswith(".apk"):
        flags.append((0.9, "downloads an Android app file (.apk) from outside the Play Store"))
    if parts.scheme == "http":
        flags.append((0.2, "connection is not encrypted (http)"))
    if host.count("-") >= 2:
        flags.append((0.15, "unusually many hyphens in the name"))

    safe = 1.0
    for w, _ in flags:
        safe *= 1 - w
    reasons = [r for _, r in flags] or ["no red flags found (not a guarantee)"]
    return LinkVerdict(url, round(1 - safe, 3), reasons)
