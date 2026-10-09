"""
Dataset loading, cleaning and (fallback) synthetic data generation.

IMPORTANT: ``generate_synthetic_dataset`` produces *artificial* URLs drawn from
distributions written by the project author. Metrics measured on it describe how
well the model learned that generator, NOT how it would perform on real-world
traffic. For the final report, train on a real labelled dataset (see README).
"""
from __future__ import annotations

import random
import string
from pathlib import Path

import numpy as np
import pandas as pd

from .features import FEATURE_NAMES, InvalidURLError, extract_features, parse_url

URL_COLUMN_CANDIDATES = ("url", "URL", "Url", "urls", "link")
LABEL_COLUMN_CANDIDATES = ("label", "Label", "type", "Type", "status", "class", "Class", "target")

_PHISH_LABELS = {"1", "phishing", "phish", "bad", "malicious", "malware", "defacement", "true"}
_LEGIT_LABELS = {"0", "legitimate", "legit", "good", "benign", "safe", "false"}


# --------------------------------------------------------------------------
# Loading / cleaning
# --------------------------------------------------------------------------
def load_dataset(path: str | Path, url_col: str | None = None, label_col: str | None = None,
                 phishing_values: set | None = None, legit_values: set | None = None) -> pd.DataFrame:
    """Load a CSV with a URL column and a label column; return columns url,label (1=phishing)."""
    df = pd.read_csv(path)
    url_col = url_col or next((c for c in URL_COLUMN_CANDIDATES if c in df.columns), None)
    label_col = label_col or next((c for c in LABEL_COLUMN_CANDIDATES if c in df.columns), None)
    if url_col is None or label_col is None:
        raise ValueError(
            f"Could not find URL/label columns in {path}. Columns present: {list(df.columns)}. "
            "Pass --url-col and --label-col.")

    phish = {str(v).strip().lower() for v in (phishing_values or _PHISH_LABELS)}
    legit = {str(v).strip().lower() for v in (legit_values or _LEGIT_LABELS)}

    def to_label(v):
        s = str(v).strip().lower()
        if s in phish:
            return 1
        if s in legit:
            return 0
        return np.nan

    out = pd.DataFrame({"url": df[url_col].astype("string"), "label": df[label_col].map(to_label)})
    unknown = out["label"].isna().sum()
    if unknown == len(out):
        raise ValueError(
            f"No label values in column '{label_col}' could be mapped. Values seen: "
            f"{sorted(map(str, df[label_col].dropna().unique()))[:10]}. "
            "Use --phishing-values / --legit-values to say which is which.")
    return out


def clean_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Drop missing/invalid/duplicate/conflicting rows. Returns (clean_df, report)."""
    report = {"rows_in": int(len(df))}
    df = df.copy()

    report["missing_url_or_label"] = int(df["url"].isna().sum() + df["label"].isna().sum())
    df = df.dropna(subset=["url", "label"])
    df["url"] = df["url"].astype(str).str.strip()
    df = df[df["url"] != ""]
    df["label"] = df["label"].astype(int)

    # Rows whose URL cannot be parsed as http(s) are unusable for this model.
    def parseable(u: str) -> bool:
        try:
            parse_url(u)
            return True
        except InvalidURLError:
            return False

    ok = df["url"].map(parseable)
    report["unparseable_urls"] = int((~ok).sum())
    df = df[ok]

    # Same URL carrying both labels is label noise: drop every copy.
    key = df["url"].str.lower()
    conflict = df.groupby(key)["label"].transform("nunique") > 1
    report["conflicting_labels"] = int(conflict.sum())
    df = df[~conflict]

    before = len(df)
    df = df.loc[~key[df.index].duplicated()]
    report["duplicates_removed"] = int(before - len(df))

    df = df.reset_index(drop=True)
    report["rows_out"] = int(len(df))
    report["phishing_share"] = round(float(df["label"].mean()), 4) if len(df) else 0.0
    return df, report


def build_feature_matrix(urls) -> pd.DataFrame:
    """Extract features for many URLs. Every URL must already be parseable."""
    rows = [extract_features(u) for u in urls]
    return pd.DataFrame(rows, columns=FEATURE_NAMES)


# --------------------------------------------------------------------------
# Synthetic data (fallback when no real dataset is supplied)
# --------------------------------------------------------------------------
_WORDS = (
    "green valley bright coffee garden river stone maple cloud pixel studio harbor "
    "summit lotus orbit canvas cedar amber breeze delta ember forest glacier hazel "
    "island jasmine kernel lantern meadow nimbus ocean pebble quartz ridge silver "
    "timber urban velvet willow zephyr atlas beacon cobalt dune echo fable grove "
    "haven iris juniper koala lumen mosaic nectar opal prairie quill rustic sage "
    "tundra union vista wander yonder zenith news daily health travel books learn "
    "fitness recipes market homes tech auto sports music photo design finance"
).split()

_LEGIT_KNOWN = [
    ("wikipedia", "org"), ("github", "com"), ("stackoverflow", "com"), ("bbc", "co.uk"),
    ("nytimes", "com"), ("python", "org"), ("mozilla", "org"), ("khanacademy", "org"),
    ("nasa", "gov"), ("harvard", "edu"), ("iitb", "ac.in"), ("thehindu", "com"),
    ("flipkart", "com"), ("amazon", "in"), ("microsoft", "com"), ("apple", "com"),
    ("paypal", "com"), ("google", "com"), ("linkedin", "com"), ("youtube", "com"),
    ("reddit", "com"), ("medium", "com"), ("coursera", "org"), ("nptel", "ac.in"),
    ("irctc", "co.in"), ("hdfcbank", "com"), ("facebook", "com"), ("netflix", "com"),
]
_BRANDS_FOR_PHISH = ["paypal", "apple", "microsoft", "google", "amazon", "netflix", "facebook",
                     "instagram", "linkedin", "dropbox", "hdfcbank", "icici", "paytm", "dhl", "fedex", "ebay"]
_LURE_WORDS = ["login", "secure", "verify", "update", "account", "confirm", "signin", "billing",
               "support", "wallet", "recover", "unlock", "validate", "alert"]
_COMMON_TLDS = ["com", "com", "com", "org", "net", "in", "co.uk", "io", "co.in", "edu"]
_RISKY_TLDS = ["tk", "xyz", "top", "click", "link", "ml", "ga", "cf", "gq", "icu", "buzz", "work"]
_FREE_HOSTS = ["weebly.com", "wixsite.com", "blogspot.com", "web.app", "firebaseapp.com",
               "netlify.app", "github.io", "pages.dev", "glitch.me", "herokuapp.com"]
_SHORTENERS = ["bit.ly", "tinyurl.com", "t.co", "is.gd", "cutt.ly", "rb.gy", "ow.ly"]
_LEGIT_SUBS = ["www", "www", "www", "blog", "docs", "support", "mail", "accounts", "shop", "m", "news", "login"]
_PATH_WORDS = ("article story product item category archive guide tutorial review profile "
               "news post topic search results page help faq pricing about contact download").split()
_KIT_DIRS = ["wp-content/plugins", "wp-includes", "includes", "cgi-bin", "images", "assets", "~user", "tmp", "modules"]
_KIT_FILES = ["login.php", "verify.html", "signin.php", "secure-update.php", "index.php", "account.html", "confirm.php"]


def _rand_token(rng: random.Random, n: int, digits: bool = True) -> str:
    alphabet = string.ascii_lowercase + (string.digits if digits else "")
    return "".join(rng.choice(alphabet) for _ in range(n))


def _word_domain(rng: random.Random, hyphen_p: float = 0.08) -> str:
    k = rng.choice([1, 2, 2, 2, 3])
    words = [rng.choice(_WORDS) for _ in range(k)]
    sep = "-" if rng.random() < hyphen_p else ""
    return sep.join(words)


def _legit_path(rng: random.Random) -> str:
    style = rng.random()
    if style < 0.15:
        return ""
    if style < 0.45:
        return "/" + "/".join(rng.choice(_PATH_WORDS) for _ in range(rng.randint(1, 2)))
    if style < 0.75:
        y, m = rng.randint(2015, 2025), rng.randint(1, 12)
        slug = "-".join(rng.choice(_WORDS) for _ in range(rng.randint(2, 6)))
        return f"/{y}/{m:02d}/{slug}"
    if style < 0.9:
        return f"/wiki/{rng.choice(_WORDS).title()}_{rng.choice(_WORDS).title()}"
    brand = rng.choice(_BRANDS_FOR_PHISH)  # honest news article that mentions a brand
    return f"/news/{brand}-" + "-".join(rng.choice(_WORDS) for _ in range(rng.randint(2, 4)))


def _legit_query(rng: random.Random) -> str:
    r = rng.random()
    if r < 0.65:
        return ""
    if r < 0.9:
        return f"?id={rng.randint(1, 99999)}&ref={rng.choice(_WORDS)}"
    return (f"?utm_source={rng.choice(_WORDS)}&utm_medium=email&utm_campaign={rng.choice(_WORDS)}"
            f"-{rng.randint(1, 99)}&session={_rand_token(rng, rng.randint(8, 24))}")


def _gen_legit(rng: random.Random) -> str:
    r = rng.random()
    scheme = "https" if rng.random() < 0.86 else "http"
    if r < 0.30:  # well-known sites incl. legit "login" style pages
        name, tld = rng.choice(_LEGIT_KNOWN)
        sub = rng.choice(_LEGIT_SUBS)
        path = _legit_path(rng) if rng.random() < 0.85 else rng.choice(["/signin", "/account/settings", "/login?next=/home", "/support/billing"])
        return f"{scheme}://{sub}.{name}.{tld}{path}{_legit_query(rng)}"
    if r < 0.93:  # long tail of ordinary sites
        name, tld = _word_domain(rng, 0.1), rng.choice(_COMMON_TLDS)
        sub = rng.choice(_LEGIT_SUBS + [""] * 3)
        host = f"{sub}.{name}.{tld}" if sub else f"{name}.{tld}"
        return f"{scheme}://{host}{_legit_path(rng)}{_legit_query(rng)}"
    if r < 0.96:  # honest personal sites on free platforms
        return f"{scheme}://{_word_domain(rng)}.{rng.choice(_FREE_HOSTS)}{_legit_path(rng)}"
    if r < 0.98:  # shortened links are used by honest senders too
        return f"{scheme}://{rng.choice(_SHORTENERS)}/{_rand_token(rng, rng.randint(5, 8))}"
    return f"http://192.168.{rng.randint(0, 255)}.{rng.randint(1, 254)}:{rng.choice([80, 8080, 3000])}{_legit_path(rng)}"


def _gen_phish(rng: random.Random) -> str:
    r = rng.random()
    scheme = "https" if rng.random() < 0.70 else "http"
    brand = rng.choice(_BRANDS_FOR_PHISH)
    lure = rng.choice(_LURE_WORDS)
    risky = rng.random() < 0.45
    tld = rng.choice(_RISKY_TLDS) if risky else rng.choice(_COMMON_TLDS)
    if r < 0.17:  # brand as subdomain of an unrelated registered domain
        host = f"{brand}.com.{lure}-{_word_domain(rng, 0)}.{tld}"
        return f"{scheme}://{host}/{lure}{rng.choice(['', '/index.php', '?id=' + _rand_token(rng, 10)])}"
    if r < 0.35:  # brand + lure glued with hyphens
        parts = [brand, lure] + ([rng.choice(_LURE_WORDS)] if rng.random() < 0.4 else [])
        rng.shuffle(parts)
        sub = rng.choice(["", "", "www.", "secure."])
        return f"{scheme}://{sub}{'-'.join(parts)}.{tld}/{rng.choice(_KIT_FILES)}"
    if r < 0.42:  # raw IP
        ip = ".".join(str(rng.randint(1, 254)) for _ in range(4))
        port = rng.choice(["", "", ":8080", ":8443"])
        return f"http://{ip}{port}/{rng.choice(_KIT_DIRS)}/{brand}/{rng.choice(_KIT_FILES)}"
    if r < 0.57:  # kit planted on a compromised, normal-looking site
        name, tld2 = _word_domain(rng, 0.05), rng.choice(_COMMON_TLDS)
        path = f"/{rng.choice(_KIT_DIRS)}/{rng.choice(_WORDS)}/{rng.choice(_KIT_FILES)}"
        q = rng.choice(["", f"?id={rng.randint(1, 99999)}", f"?email={_rand_token(rng, 8, False)}%40mail.com"])
        return f"{scheme}://{name}.{tld2}{path}{q}"
    if r < 0.67:  # free hosting
        return f"{scheme}://{brand}-{lure}-{_rand_token(rng, 4)}.{rng.choice(_FREE_HOSTS)}/{rng.choice(['', 'signin', 'verify'])}"
    if r < 0.70:  # shortener (scheme drawn like any other URL: no artificial shortcut)
        return f"{scheme}://{rng.choice(_SHORTENERS)}/{_rand_token(rng, rng.randint(5, 8))}"
    if r < 0.78:  # algorithmically generated hostname
        return f"{scheme}://{_rand_token(rng, rng.randint(9, 18))}.{tld}/{rng.choice(['', 'login', 'a/' + _rand_token(rng, 6)])}"
    if r < 0.81:  # '@' trick
        return f"http://www.{brand}.com@{_word_domain(rng, 0)}-{lure}.{tld}/"
    if r < 0.84:  # homograph / punycode
        return f"{scheme}://xn--{_rand_token(rng, 5, False)}-{_rand_token(rng, 3, False)}.{rng.choice(['com', 'net', 'xyz'])}/{lure}"
    if r < 0.90:  # heavy encoding / redirect parameter
        target = f"http%3A%2F%2F{_rand_token(rng, 8, False)}.{rng.choice(_RISKY_TLDS)}%2F{lure}"
        return f"{scheme}://{_word_domain(rng)}.{rng.choice(_COMMON_TLDS)}/redirect.php?url={target}&session={_rand_token(rng, 20)}"
    # hard cases: short, tidy, HTTPS look-alikes (deliberately hard to separate)
    return f"https://{_word_domain(rng, 0.03)}.{rng.choice(_COMMON_TLDS + _RISKY_TLDS[:3])}{_legit_path(rng)}"


def generate_synthetic_dataset(n: int = 24000, seed: int = 42) -> pd.DataFrame:
    """Create a balanced, artificial URL dataset with overlapping class patterns."""
    rng = random.Random(seed)
    seen: set = set()
    rows = []
    while len(rows) < n:
        label = len(rows) % 2
        url = _gen_phish(rng) if label else _gen_legit(rng)
        key = url.lower()
        if key in seen:
            continue
        seen.add(key)
        rows.append((url, label))
    df = pd.DataFrame(rows, columns=["url", "label"])
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def registered_domain_of(url: str) -> str:
    """Grouping key used to keep the same domain out of both train and test."""
    return parse_url(url).registered_domain
