"""
Lexical URL feature extraction for phishing detection.

Design notes
------------
* Every feature here is computed from the URL *string alone*. No network
  access happens in this module, so extraction is fast, deterministic and safe
  to run on untrusted input (the URL is never fetched or resolved).
* Features that need the network (redirect chains, domain age) live in
  ``src/live_checks.py`` and are *not* model inputs, because public training
  sets contain URL strings only.
* ``FEATURE_SPECS`` is the single source of truth: it drives the column order,
  the generated docs (docs/FEATURES.md) and the consistency tests.
"""
from __future__ import annotations

import ipaddress
import math
import re
import sys
from collections import Counter
from dataclasses import dataclass
from urllib.parse import urlsplit


class InvalidURLError(ValueError):
    """Raised when a string cannot be treated as an http(s) URL."""


# --------------------------------------------------------------------------
# Heuristic vocabularies (deliberately small and easy to audit / extend)
# --------------------------------------------------------------------------
SUSPICIOUS_WORDS = (
    "login", "logon", "signin", "sign-in", "verify", "verification", "secure",
    "security", "account", "update", "confirm", "banking", "password", "passwd",
    "credential", "wallet", "billing", "invoice", "payment", "suspend",
    "unlock", "recover", "helpdesk", "webscr", "authenticate", "validate",
    "alert", "urgent", "limited", "support", "ebayisapi", "bonus", "prize",
)

BRANDS = (
    "paypal", "apple", "icloud", "microsoft", "office365", "outlook", "google",
    "gmail", "facebook", "instagram", "amazon", "netflix", "ebay", "whatsapp",
    "linkedin", "dropbox", "adobe", "docusign", "chase", "wellsfargo",
    "bankofamerica", "citibank", "hdfc", "hdfcbank", "icici", "sbi", "paytm",
    "phonepe", "irctc", "dhl", "fedex", "usps",
)

# TLDs that are cheap/free to register and therefore over-represented in abuse
# reports. This is a heuristic list: it drifts over time and a TLD on it is
# NOT evidence of phishing by itself.
SUSPICIOUS_TLDS = frozenset({
    "tk", "ml", "ga", "cf", "gq", "xyz", "top", "click", "link", "work", "zip",
    "mov", "country", "kim", "loan", "men", "review", "stream", "download",
    "racing", "win", "bid", "gdn", "cyou", "icu", "buzz", "rest", "monster",
    "sbs", "support", "cam", "fit",
})

URL_SHORTENERS = frozenset({
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd", "buff.ly",
    "rebrand.ly", "cutt.ly", "shorturl.at", "tiny.cc", "rb.gy",
})

# Legitimate platforms that anyone can publish on for free. Attackers abuse
# them, but so do millions of honest users -> weak signal only.
FREE_HOSTING_SUFFIXES = (
    "weebly.com", "wixsite.com", "blogspot.com", "web.app", "firebaseapp.com",
    "netlify.app", "github.io", "pages.dev", "glitch.me", "herokuapp.com",
    "000webhostapp.com", "godaddysites.com", "sites.google.com",
)

SUSPICIOUS_EXTENSIONS = (".exe", ".scr", ".apk", ".zip", ".rar", ".jar", ".msi", ".bat")

# Public suffixes that consist of two labels. A full Public Suffix List would be
# more complete, but this avoids a network download and covers common cases.
_TWO_LABEL_SUFFIXES = frozenset({
    "co.uk", "org.uk", "ac.uk", "gov.uk", "com.au", "net.au", "org.au",
    "co.in", "net.in", "org.in", "ac.in", "gov.in", "edu.in", "co.jp",
    "com.br", "com.cn", "com.mx", "co.za", "co.nz", "com.sg", "com.tr",
    "com.hk", "co.kr", "com.ar", "com.pk", "com.bd",
})

_SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://")
_OTHER_SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.\-]*:(?!\d)")
_CTRL_OR_SPACE_RE = re.compile(r"[\s\x00-\x1f\x7f]")
_PCT_RE = re.compile(r"%[0-9a-fA-F]{2}")
_TOKEN_SPLIT_RE = re.compile(r"[^a-z0-9]+")


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ParsedURL:
    url: str              # normalised: scheme added if it was missing
    scheme: str
    host: str             # lower-case, no port / userinfo / trailing dot
    port: int | None
    path: str
    query: str
    fragment: str
    subdomains: tuple
    sld: str              # second-level label, e.g. "paypal" in paypal.com
    suffix: str           # public suffix, e.g. "com" or "co.uk" ("" for IPs)
    is_ip: bool

    @property
    def registered_domain(self) -> str:
        return f"{self.sld}.{self.suffix}" if self.suffix else self.sld


def _is_ip_host(host: str) -> bool:
    """True for IPv4/IPv6 literals and for obfuscated forms (decimal, hex)."""
    h = host.strip("[]")
    try:
        ipaddress.ip_address(h)
        return True
    except ValueError:
        pass
    if re.fullmatch(r"0x[0-9a-f]+", h) or re.fullmatch(r"\d{8,10}", h):
        return True
    return bool(re.fullmatch(r"(?:0x[0-9a-f]+|\d+)(?:\.(?:0x[0-9a-f]+|\d+)){3}", h))


def parse_url(raw: str) -> ParsedURL:
    """Normalise and split a URL. Raises InvalidURLError for unusable input."""
    if not isinstance(raw, str):
        raise InvalidURLError("URL must be a string")
    url = raw.strip()
    if not url:
        raise InvalidURLError("URL is empty")
    if _CTRL_OR_SPACE_RE.search(url):
        raise InvalidURLError("URL contains whitespace or control characters")

    if not _SCHEME_RE.match(url):
        if url.startswith("//"):
            url = "http:" + url
        elif _OTHER_SCHEME_RE.match(url):
            raise InvalidURLError("Only http:// and https:// URLs are supported")
        else:
            url = "http://" + url

    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError as exc:
        raise InvalidURLError(f"Malformed URL: {exc}") from exc

    scheme = parts.scheme.lower()
    if scheme not in ("http", "https"):
        raise InvalidURLError("Only http:// and https:// URLs are supported")
    host = (parts.hostname or "").rstrip(".")
    if not host:
        raise InvalidURLError("URL has no hostname")

    is_ip = _is_ip_host(host)
    if is_ip:
        subdomains, sld, suffix = (), host, ""
    else:
        labels = host.split(".")
        if len(labels) >= 3 and ".".join(labels[-2:]) in _TWO_LABEL_SUFFIXES:
            suffix, sld, subdomains = ".".join(labels[-2:]), labels[-3], tuple(labels[:-3])
        elif len(labels) >= 2:
            suffix, sld, subdomains = labels[-1], labels[-2], tuple(labels[:-2])
        else:
            suffix, sld, subdomains = "", labels[0], ()

    return ParsedURL(url=url, scheme=scheme, host=host, port=port, path=parts.path,
                     query=parts.query, fragment=parts.fragment, subdomains=subdomains,
                     sld=sld, suffix=suffix, is_ip=is_ip)


def shannon_entropy(text: str) -> float:
    """Shannon entropy (bits per character) of a string."""
    if not text:
        return 0.0
    n = len(text)
    return -sum((c / n) * math.log2(c / n) for c in Counter(text).values())


# --------------------------------------------------------------------------
# Feature catalogue: (name, group, what it is, why it helps)
# --------------------------------------------------------------------------
FEATURE_SPECS = [
    # ---- Length ----
    ("url_length", "Length", "Total characters in the URL.",
     "Attackers pad URLs with extra paths, tokens and encoded redirect targets to bury the real domain, so phishing URLs are often longer than typical legitimate ones."),
    ("hostname_length", "Length", "Characters in the hostname (domain part).",
     "Long hostnames usually carry stuffed keywords or multiple fake subdomains (e.g. brand + 'secure' + 'login')."),
    ("path_length", "Length", "Characters in the path after the hostname.",
     "Deep, long paths (e.g. /wp-content/.../verify/login.php) are typical of pages planted on compromised sites."),
    ("query_length", "Length", "Characters in the query string (after '?').",
     "Long query strings can hide redirect targets or session/tracking tokens used by phishing kits."),
    ("sld_length", "Length", "Length of the registered domain label (the 'paypal' in paypal.com).",
     "Registered names that are unusually long (keyword stuffing) or very short random strings both deviate from typical brand domains."),
    # ---- Character counts ----
    ("num_dots", "Characters", "Number of '.' in the URL.",
     "Many dots mean many subdomain levels, a common trick for making evil.example look like brand.com.evil.example."),
    ("num_hyphens", "Characters", "Number of '-' in the whole URL.",
     "Hyphens are used to glue brand and urgency words together (paypal-secure-login) because the real brand domain has no hyphen."),
    ("num_underscores", "Characters", "Number of '_' in the URL.",
     "Underscores are rare in legitimate hostnames and more common in auto-generated paths and kit directories."),
    ("num_special_chars", "Characters", "Count of characters that are not letters, digits, '.', '-', '/' or ':'.",
     "Symbols such as ?, =, &, %, @, ~ and ! increase when URLs encode redirects, credentials or obfuscated payloads."),
    ("num_digits", "Characters", "Number of digits in the URL.",
     "Digits replace letters in typosquatting (paypa1) and appear in raw IPs and machine-generated names."),
    ("digit_ratio", "Characters", "Digits divided by URL length.",
     "A length-normalised version of num_digits, so long legitimate URLs are not penalised unfairly."),
    ("letter_ratio", "Characters", "Letters divided by URL length.",
     "Low letter ratios indicate heavy use of digits and symbols, typical of generated or obfuscated URLs."),
    ("host_digit_ratio", "Characters", "Digits divided by hostname length.",
     "High digit density inside the hostname is characteristic of IP-like hosts and algorithmically generated domains."),
    ("num_percent_encoded", "Characters", "Number of %XX percent-encoded sequences.",
     "Percent-encoding is used to hide characters or smuggle a full second URL inside the query string."),
    # ---- Structure ----
    ("path_depth", "Structure", "Number of non-empty path segments.",
     "Deeply nested paths are common when a phishing kit is dropped into subfolders of a hacked website."),
    ("num_subdomains", "Structure", "Number of labels left of the registered domain.",
     "Phishers add subdomains so the brand name sits where users look first (paypal.com.evil.tk has three)."),
    ("num_query_params", "Structure", "Number of key=value pairs in the query string.",
     "Many parameters can mean redirectors, tracking IDs or victim-specific tokens in targeted campaigns."),
    ("has_double_slash_in_path", "Structure", "1 if '//' appears inside the path.",
     "A '//' after the host can indicate an embedded redirect to another URL."),
    ("has_nonstandard_port", "Structure", "1 if the URL names a port other than 80/443.",
     "Legitimate public sites almost always use default ports; unusual ports show up on ad-hoc attacker servers."),
    # ---- Host / domain flags ----
    ("has_ip_host", "Host", "1 if the host is an IP address (including decimal/hex-obfuscated forms).",
     "Real organisations use domain names. An IP in place of a name is a classic phishing and malware-hosting indicator."),
    ("is_https", "Host", "1 if the scheme is https.",
     "Mostly informative in combination with other features: many phishing sites now use free TLS certificates, so HTTPS is NOT a safety guarantee."),
    ("has_at_symbol", "Host", "1 if the URL contains '@'.",
     "Browsers ignore everything before '@' in the authority part, so http://paypal.com@evil.tk really goes to evil.tk."),
    ("has_idn_host", "Host", "1 if the hostname uses Punycode ('xn--') or non-ASCII characters.",
     "Internationalised names enable homograph attacks, where look-alike Unicode letters imitate a trusted brand."),
    ("suspicious_tld", "Host", "1 if the top-level domain is on the abuse-heavy TLD list.",
     "Cheap or free TLDs (.tk, .xyz, .top ...) are over-represented in abuse reports. This is a prior, not proof."),
    ("is_shortener", "Host", "1 if the domain is a known URL shortener.",
     "Shorteners hide the true destination, which is why they are widely used in phishing messages."),
    ("is_free_hosting", "Host", "1 if hosted on a free website/hosting platform.",
     "Free platforms let attackers publish a page in minutes without registering a domain; weak signal because many honest sites use them."),
    ("num_hyphens_in_domain", "Host", "Number of '-' in the registered domain label.",
     "Hyphenated registered domains (secure-paypal-login.com) are a stronger imitation signal than hyphens deep in a path."),
    ("longest_host_token_len", "Host", "Length of the longest dot/hyphen-separated token in the hostname.",
     "Very long tokens suggest keyword stuffing or random strings."),
    # ---- Semantics ----
    ("suspicious_word_count", "Semantics", "Number of distinct phishing-style words (login, verify, secure, update, ...) anywhere in the URL.",
     "Phishing lures rely on urgency and account-security vocabulary to push victims to act."),
    ("suspicious_word_in_host", "Semantics", "Number of distinct such words inside the hostname.",
     "A security word in the hostname is far more suspicious than one in a path, because real brands rarely register such domains."),
    ("brand_in_host_mismatch", "Semantics", "1 if a well-known brand appears in the hostname but the registered domain is not that brand.",
     "Core impersonation pattern: paypal.com.evil.tk or paypal-secure.com borrow trust from a brand they do not own."),
    ("brand_in_path_mismatch", "Semantics", "1 if a well-known brand appears in the path/query but the registered domain is not that brand.",
     "Kits hosted on unrelated domains often use /paypal/ or /apple-id/ folders. Weak signal: news sites also mention brands."),
    ("suspicious_file_ext", "Semantics", "1 if the path ends in an executable/archive extension (.exe, .apk, .zip ...).",
     "Direct links to payload files indicate malware delivery rather than a normal web page."),
    ("https_token_in_host_or_path", "Semantics", "1 if the word 'https' appears in the hostname or path.",
     "Fake 'https' tokens in the wrong place (https-paypal.com) try to create a false sense of security."),
    # ---- Randomness ----
    ("url_entropy", "Randomness", "Shannon entropy of the URL, bits per character.",
     "Randomly generated or heavily encoded URLs have higher character entropy than readable ones."),
    ("host_entropy", "Randomness", "Shannon entropy of the hostname.",
     "Domain-generation algorithms and throw-away domains produce high-entropy, unpronounceable names."),
]

FEATURE_NAMES = [spec[0] for spec in FEATURE_SPECS]


# --------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------
def _words_in(text: str) -> int:
    return sum(1 for w in SUSPICIOUS_WORDS if w in text)


def _brand_hit(tokens: set, haystack: str, sld: str) -> int:
    for brand in BRANDS:
        if sld == brand:
            continue  # the page is on the brand's own domain
        if brand in tokens or (len(brand) >= 6 and brand in haystack):
            return 1
    return 0


def extract_features(url: str) -> dict:
    """Return an ordered {feature_name: number} dict for one URL."""
    p = parse_url(url)
    u = p.url
    ul = u.lower()
    host = p.host
    n = len(u)

    digits = sum(ch.isdigit() for ch in u)
    letters = sum(ch.isalpha() for ch in u)
    host_digits = sum(ch.isdigit() for ch in host)
    host_tokens = {t for t in _TOKEN_SPLIT_RE.split(host) if t}
    tail = (p.path + " " + p.query).lower()
    tail_tokens = {t for t in _TOKEN_SPLIT_RE.split(tail) if t}
    tld_last = p.suffix.rsplit(".", 1)[-1] if p.suffix else ""
    path_lower = p.path.lower()

    f = {
        "url_length": n,
        "hostname_length": len(host),
        "path_length": len(p.path),
        "query_length": len(p.query),
        "sld_length": len(p.sld),
        "num_dots": u.count("."),
        "num_hyphens": u.count("-"),
        "num_underscores": u.count("_"),
        "num_special_chars": sum(1 for ch in u if not ch.isalnum() and ch not in ".-/:"),
        "num_digits": digits,
        "digit_ratio": digits / n,
        "letter_ratio": letters / n,
        "host_digit_ratio": host_digits / len(host),
        "num_percent_encoded": len(_PCT_RE.findall(u)),
        "path_depth": len([s for s in p.path.split("/") if s]),
        "num_subdomains": len(p.subdomains),
        "num_query_params": len([q for q in p.query.split("&") if q]),
        "has_double_slash_in_path": int("//" in p.path),
        "has_nonstandard_port": int(p.port is not None and p.port not in (80, 443)),
        "has_ip_host": int(p.is_ip),
        "is_https": int(p.scheme == "https"),
        "has_at_symbol": int("@" in u),
        "has_idn_host": int("xn--" in host or not host.isascii()),
        "suspicious_tld": int(tld_last in SUSPICIOUS_TLDS),
        "is_shortener": int(p.registered_domain in URL_SHORTENERS),
        "is_free_hosting": int(any(host == s or host.endswith("." + s) for s in FREE_HOSTING_SUFFIXES)),
        "num_hyphens_in_domain": 0 if p.is_ip else p.sld.count("-"),
        "longest_host_token_len": max((len(t) for t in host_tokens), default=0),
        "suspicious_word_count": _words_in(ul),
        "suspicious_word_in_host": _words_in(host),
        "brand_in_host_mismatch": _brand_hit(host_tokens, host, p.sld),
        "brand_in_path_mismatch": _brand_hit(tail_tokens, "", p.sld),
        "suspicious_file_ext": int(path_lower.endswith(SUSPICIOUS_EXTENSIONS)),
        "https_token_in_host_or_path": int("https" in host or "https" in path_lower),
        "url_entropy": shannon_entropy(u),
        "host_entropy": shannon_entropy(host),
    }
    return {name: f[name] for name in FEATURE_NAMES}  # enforce catalogue order


# --------------------------------------------------------------------------
# Human-readable indicators (rule based, independent of the ML model)
# --------------------------------------------------------------------------
def describe_risk_indicators(f: dict) -> list:
    """Translate raw features into plain-language findings for the UI.

    These are *not* the model's reasoning; they are transparent heuristics that
    help a user understand what looks unusual about a URL.
    """
    out = []

    def add(severity: str, message: str) -> None:
        out.append({"severity": severity, "message": message})

    if f["has_ip_host"]:
        add("high", "The address uses an IP address instead of a domain name.")
    if f["has_at_symbol"]:
        add("high", "The URL contains '@'; browsers ignore the text before it, so the real destination can differ from what it looks like.")
    if f["brand_in_host_mismatch"]:
        add("high", "A well-known brand name appears in the hostname, but the site is not on that brand's own domain.")
    if f["has_idn_host"]:
        add("medium", "The hostname uses internationalised characters (Punycode), which can imitate trusted names.")
    if f["suspicious_tld"]:
        add("medium", "The top-level domain is one that is frequently abused (cheap or free to register).")
    if f["is_shortener"]:
        add("medium", "A link shortener hides the real destination.")
    if f["suspicious_word_in_host"]:
        add("medium", "The hostname contains account-security words such as 'login', 'verify' or 'secure'.")
    elif f["suspicious_word_count"] >= 2:
        add("medium", "The URL contains several account-security words (login, verify, update, ...).")
    if f["num_subdomains"] >= 3:
        add("medium", f"The hostname has {int(f['num_subdomains'])} subdomain levels, which can bury the real domain.")
    if f["num_hyphens_in_domain"] >= 2:
        add("medium", "The registered domain contains several hyphens, a common imitation pattern.")
    if f["suspicious_file_ext"]:
        add("medium", "The link points straight at an executable or archive file.")
    if f["has_double_slash_in_path"]:
        add("low", "A '//' inside the path can indicate an embedded redirect.")
    if f["has_nonstandard_port"]:
        add("low", "The URL uses an unusual port number.")
    if f["host_entropy"] >= 3.8 and f["hostname_length"] >= 12:
        add("low", "The hostname looks random (high entropy), typical of generated domains.")
    if f["url_length"] >= 100:
        add("low", f"The URL is long ({int(f['url_length'])} characters).")
    if f["brand_in_path_mismatch"]:
        add("low", "A brand name appears in the path of an unrelated domain.")
    if f["is_free_hosting"]:
        add("low", "The page is hosted on a free hosting platform (common for both honest and malicious pages).")
    if not f["is_https"]:
        add("low", "The connection is not HTTPS. (Note: many phishing sites do use HTTPS, so HTTPS alone does not mean a site is safe.)")
    return out


# --------------------------------------------------------------------------
# Documentation generator
# --------------------------------------------------------------------------
def render_feature_docs_markdown() -> str:
    lines = [
        "# Feature Catalogue",
        "",
        f"The model uses **{len(FEATURE_SPECS)} lexical features**, all computed from the URL string only "
        "(no page is fetched). This file is generated from `FEATURE_SPECS` in `src/features.py` "
        "by `python -m src.features --docs`.",
        "",
    ]
    group = None
    for name, grp, what, why in FEATURE_SPECS:
        if grp != group:
            group = grp
            lines += ["", f"## {grp}", "", "| Feature | What it measures | Why it helps |", "|---|---|---|"]
        lines.append(f"| `{name}` | {what} | {why} |")
    lines += [
        "",
        "## Network-based signals (not model inputs)",
        "",
        "`src/live_checks.py` can optionally report the **redirect count** and **domain age** of a URL. "
        "They are shown next to the prediction but are not fed to the model, because the public training "
        "datasets contain URL strings only. Both are disabled by default; see the README for the safety "
        "measures around them.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    if "--docs" in sys.argv:
        print(render_feature_docs_markdown())
    elif len(sys.argv) > 1:
        import json
        print(json.dumps(extract_features(sys.argv[1]), indent=2))
    else:
        print("usage: python -m src.features <url>   |   python -m src.features --docs")
