"""
Optional network-based signals: redirect chain length and domain age.

These are SUPPLEMENTARY signals shown next to the model's verdict. They are not
model inputs (public training sets only contain URL strings).

Safety measures (this module fetches URLs supplied by untrusted users, which is a
classic Server-Side Request Forgery (SSRF) risk):
  * Disabled by default in the web app (needs ENABLE_LIVE_CHECKS=1).
  * Only http/https; every hop's hostname must resolve exclusively to *globally
    routable* addresses (blocks localhost, private LAN, link-local/cloud-metadata
    ranges such as 169.254.169.254).
  * Redirects are followed manually, so each hop is re-validated; hop count,
    time and response size are capped; only HEAD is used and no body is read.
  * Nothing is rendered or executed; no cookies are sent.
  * Domain-age lookups send ONLY the registered domain name (never the path or
    query) to the public RDAP bootstrap service.

Known limitation: DNS is resolved once for validation and again by the HTTP
client, so a DNS-rebinding attacker could in theory swap the answer between the
two. For a hardened production deployment, run these checks in an isolated
network sandbox or pin the validated IP.
"""
from __future__ import annotations

import ipaddress
import socket
from datetime import datetime, timezone
from urllib.parse import urljoin

import requests

from .features import InvalidURLError, parse_url

USER_AGENT = "PhishingDetectorStudentProject/1.0 (+redirect-check; HEAD only)"
MAX_HOPS = 5
TIMEOUT = 4  # seconds per request


def is_public_host(hostname: str) -> bool:
    """True only if *every* address the name resolves to is globally routable."""
    try:
        infos = socket.getaddrinfo(hostname, None, proto=socket.IPPROTO_TCP)
    except (socket.gaierror, UnicodeError):
        return False
    if not infos:
        return False
    for info in infos:
        addr = info[4][0].split("%")[0]
        try:
            if not ipaddress.ip_address(addr).is_global:
                return False
        except ValueError:
            return False
    return True


def follow_redirects(url: str) -> dict:
    """Count redirects without downloading page content."""
    result = {"redirect_count": None, "final_host": None, "note": None}
    try:
        current = parse_url(url).url
    except InvalidURLError as exc:
        result["note"] = str(exc)
        return result

    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    hops = 0
    try:
        while True:
            host = parse_url(current).host
            if not is_public_host(host):
                result["note"] = "Stopped: host is not a public internet address."
                break
            resp = session.head(current, allow_redirects=False, timeout=TIMEOUT)
            location = resp.headers.get("Location")
            if resp.is_redirect and location:
                hops += 1
                if hops > MAX_HOPS:
                    result["note"] = f"Stopped after {MAX_HOPS} redirects."
                    break
                current = urljoin(current, location)
                continue
            break
        result["redirect_count"] = hops
        result["final_host"] = parse_url(current).host
    except (requests.RequestException, InvalidURLError) as exc:
        result["note"] = f"Lookup failed: {type(exc).__name__}"
    return result


def domain_age_days(url: str) -> dict:
    """Domain age via the public RDAP service (registered domain name only)."""
    result = {"domain_age_days": None, "note": None}
    try:
        p = parse_url(url)
    except InvalidURLError as exc:
        result["note"] = str(exc)
        return result
    if p.is_ip:
        result["note"] = "IP address; no domain to look up."
        return result
    try:
        resp = requests.get(f"https://rdap.org/domain/{p.registered_domain}",
                            headers={"User-Agent": USER_AGENT, "Accept": "application/rdap+json"},
                            timeout=TIMEOUT)
        if resp.status_code != 200:
            result["note"] = f"RDAP returned HTTP {resp.status_code}."
            return result
        for event in resp.json().get("events", []):
            if event.get("eventAction") == "registration":
                registered = datetime.fromisoformat(event["eventDate"].replace("Z", "+00:00"))
                result["domain_age_days"] = max(0, (datetime.now(timezone.utc) - registered).days)
                return result
        result["note"] = "No registration date published."
    except (requests.RequestException, ValueError, KeyError) as exc:
        result["note"] = f"Lookup failed: {type(exc).__name__}"
    return result


def run_live_checks(url: str) -> dict:
    return {**follow_redirects(url), **domain_age_days(url)}
