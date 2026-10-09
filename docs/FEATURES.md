# Feature Catalogue

The model uses **36 lexical features**, all computed from the URL string only (no page is fetched). This file is generated from `FEATURE_SPECS` in `src/features.py` by `python -m src.features --docs`.


## Length

| Feature | What it measures | Why it helps |
|---|---|---|
| `url_length` | Total characters in the URL. | Attackers pad URLs with extra paths, tokens and encoded redirect targets to bury the real domain, so phishing URLs are often longer than typical legitimate ones. |
| `hostname_length` | Characters in the hostname (domain part). | Long hostnames usually carry stuffed keywords or multiple fake subdomains (e.g. brand + 'secure' + 'login'). |
| `path_length` | Characters in the path after the hostname. | Deep, long paths (e.g. /wp-content/.../verify/login.php) are typical of pages planted on compromised sites. |
| `query_length` | Characters in the query string (after '?'). | Long query strings can hide redirect targets or session/tracking tokens used by phishing kits. |
| `sld_length` | Length of the registered domain label (the 'paypal' in paypal.com). | Registered names that are unusually long (keyword stuffing) or very short random strings both deviate from typical brand domains. |

## Characters

| Feature | What it measures | Why it helps |
|---|---|---|
| `num_dots` | Number of '.' in the URL. | Many dots mean many subdomain levels, a common trick for making evil.example look like brand.com.evil.example. |
| `num_hyphens` | Number of '-' in the whole URL. | Hyphens are used to glue brand and urgency words together (paypal-secure-login) because the real brand domain has no hyphen. |
| `num_underscores` | Number of '_' in the URL. | Underscores are rare in legitimate hostnames and more common in auto-generated paths and kit directories. |
| `num_special_chars` | Count of characters that are not letters, digits, '.', '-', '/' or ':'. | Symbols such as ?, =, &, %, @, ~ and ! increase when URLs encode redirects, credentials or obfuscated payloads. |
| `num_digits` | Number of digits in the URL. | Digits replace letters in typosquatting (paypa1) and appear in raw IPs and machine-generated names. |
| `digit_ratio` | Digits divided by URL length. | A length-normalised version of num_digits, so long legitimate URLs are not penalised unfairly. |
| `letter_ratio` | Letters divided by URL length. | Low letter ratios indicate heavy use of digits and symbols, typical of generated or obfuscated URLs. |
| `host_digit_ratio` | Digits divided by hostname length. | High digit density inside the hostname is characteristic of IP-like hosts and algorithmically generated domains. |
| `num_percent_encoded` | Number of %XX percent-encoded sequences. | Percent-encoding is used to hide characters or smuggle a full second URL inside the query string. |

## Structure

| Feature | What it measures | Why it helps |
|---|---|---|
| `path_depth` | Number of non-empty path segments. | Deeply nested paths are common when a phishing kit is dropped into subfolders of a hacked website. |
| `num_subdomains` | Number of labels left of the registered domain. | Phishers add subdomains so the brand name sits where users look first (paypal.com.evil.tk has three). |
| `num_query_params` | Number of key=value pairs in the query string. | Many parameters can mean redirectors, tracking IDs or victim-specific tokens in targeted campaigns. |
| `has_double_slash_in_path` | 1 if '//' appears inside the path. | A '//' after the host can indicate an embedded redirect to another URL. |
| `has_nonstandard_port` | 1 if the URL names a port other than 80/443. | Legitimate public sites almost always use default ports; unusual ports show up on ad-hoc attacker servers. |

## Host

| Feature | What it measures | Why it helps |
|---|---|---|
| `has_ip_host` | 1 if the host is an IP address (including decimal/hex-obfuscated forms). | Real organisations use domain names. An IP in place of a name is a classic phishing and malware-hosting indicator. |
| `is_https` | 1 if the scheme is https. | Mostly informative in combination with other features: many phishing sites now use free TLS certificates, so HTTPS is NOT a safety guarantee. |
| `has_at_symbol` | 1 if the URL contains '@'. | Browsers ignore everything before '@' in the authority part, so http://paypal.com@evil.tk really goes to evil.tk. |
| `has_idn_host` | 1 if the hostname uses Punycode ('xn--') or non-ASCII characters. | Internationalised names enable homograph attacks, where look-alike Unicode letters imitate a trusted brand. |
| `suspicious_tld` | 1 if the top-level domain is on the abuse-heavy TLD list. | Cheap or free TLDs (.tk, .xyz, .top ...) are over-represented in abuse reports. This is a prior, not proof. |
| `is_shortener` | 1 if the domain is a known URL shortener. | Shorteners hide the true destination, which is why they are widely used in phishing messages. |
| `is_free_hosting` | 1 if hosted on a free website/hosting platform. | Free platforms let attackers publish a page in minutes without registering a domain; weak signal because many honest sites use them. |
| `num_hyphens_in_domain` | Number of '-' in the registered domain label. | Hyphenated registered domains (secure-paypal-login.com) are a stronger imitation signal than hyphens deep in a path. |
| `longest_host_token_len` | Length of the longest dot/hyphen-separated token in the hostname. | Very long tokens suggest keyword stuffing or random strings. |

## Semantics

| Feature | What it measures | Why it helps |
|---|---|---|
| `suspicious_word_count` | Number of distinct phishing-style words (login, verify, secure, update, ...) anywhere in the URL. | Phishing lures rely on urgency and account-security vocabulary to push victims to act. |
| `suspicious_word_in_host` | Number of distinct such words inside the hostname. | A security word in the hostname is far more suspicious than one in a path, because real brands rarely register such domains. |
| `brand_in_host_mismatch` | 1 if a well-known brand appears in the hostname but the registered domain is not that brand. | Core impersonation pattern: paypal.com.evil.tk or paypal-secure.com borrow trust from a brand they do not own. |
| `brand_in_path_mismatch` | 1 if a well-known brand appears in the path/query but the registered domain is not that brand. | Kits hosted on unrelated domains often use /paypal/ or /apple-id/ folders. Weak signal: news sites also mention brands. |
| `suspicious_file_ext` | 1 if the path ends in an executable/archive extension (.exe, .apk, .zip ...). | Direct links to payload files indicate malware delivery rather than a normal web page. |
| `https_token_in_host_or_path` | 1 if the word 'https' appears in the hostname or path. | Fake 'https' tokens in the wrong place (https-paypal.com) try to create a false sense of security. |

## Randomness

| Feature | What it measures | Why it helps |
|---|---|---|
| `url_entropy` | Shannon entropy of the URL, bits per character. | Randomly generated or heavily encoded URLs have higher character entropy than readable ones. |
| `host_entropy` | Shannon entropy of the hostname. | Domain-generation algorithms and throw-away domains produce high-entropy, unpronounceable names. |

## Network-based signals (not model inputs)

`src/live_checks.py` can optionally report the **redirect count** and **domain age** of a URL. They are shown next to the prediction but are not fed to the model, because the public training datasets contain URL strings only. Both are disabled by default; see the README for the safety measures around them.

