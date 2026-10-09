import unittest

import pandas as pd

from src.dataset import clean_dataset, generate_synthetic_dataset
from src.features import (FEATURE_NAMES, FEATURE_SPECS, InvalidURLError, describe_risk_indicators,
                          extract_features, parse_url, shannon_entropy)


class ParseTests(unittest.TestCase):
    def test_adds_missing_scheme(self):
        self.assertEqual(parse_url("example.com/a").url, "http://example.com/a")

    def test_two_label_suffix(self):
        p = parse_url("https://news.bbc.co.uk/x")
        self.assertEqual((p.sld, p.suffix, p.subdomains), ("bbc", "co.uk", ("news",)))

    def test_rejects_bad_input(self):
        for bad in ["", "   ", "javascript:alert(1)", "ftp://x.com", "http://", "http://exa mple.com",
                    "http://[::1", None, 42]:
            with self.subTest(bad=bad), self.assertRaises(InvalidURLError):
                parse_url(bad)

    def test_host_port_without_scheme_is_not_a_scheme(self):
        self.assertEqual(parse_url("example.com:8080/x").port, 8080)


class FeatureTests(unittest.TestCase):
    def test_catalogue_is_consistent(self):
        self.assertEqual(len(FEATURE_NAMES), len(set(FEATURE_NAMES)))
        self.assertEqual(list(extract_features("https://example.com").keys()), FEATURE_NAMES)
        for name, group, what, why in FEATURE_SPECS:  # every feature must be documented
            self.assertTrue(what and why, name)

    def test_ip_variants(self):
        for u in ["http://192.168.1.5/x", "http://3232235777/x", "http://0xC0A80105/x", "http://[2001:db8::1]/x"]:
            with self.subTest(u=u):
                self.assertEqual(extract_features(u)["has_ip_host"], 1)
        self.assertEqual(extract_features("http://example.com")["has_ip_host"], 0)

    def test_at_symbol_and_brand_mismatch(self):
        f = extract_features("http://www.paypal.com@evil-update.tk/")
        self.assertEqual(f["has_at_symbol"], 1)
        self.assertEqual(f["suspicious_tld"], 1)
        self.assertEqual(extract_features("http://paypal.com.secure-login.xyz/")["brand_in_host_mismatch"], 1)

    def test_brand_on_own_domain_is_not_a_mismatch(self):
        self.assertEqual(extract_features("https://www.paypal.com/signin")["brand_in_host_mismatch"], 0)
        self.assertEqual(extract_features("https://support.apple.com/account")["brand_in_host_mismatch"], 0)

    def test_https_flag_and_ports(self):
        self.assertEqual(extract_features("https://a.com")["is_https"], 1)
        self.assertEqual(extract_features("http://a.com")["is_https"], 0)
        self.assertEqual(extract_features("http://a.com:8080")["has_nonstandard_port"], 1)
        self.assertEqual(extract_features("https://a.com:443")["has_nonstandard_port"], 0)

    def test_counts(self):
        f = extract_features("https://a-b.example.com/x_y?p=1&q=2")
        self.assertEqual((f["num_hyphens"], f["num_underscores"], f["num_query_params"]), (1, 1, 2))
        self.assertEqual(f["num_subdomains"], 1)

    def test_entropy(self):
        self.assertEqual(shannon_entropy(""), 0.0)
        self.assertAlmostEqual(shannon_entropy("ab"), 1.0)
        self.assertGreater(shannon_entropy("x9k2jq0plz"), shannon_entropy("aaaaaaaaaa"))

    def test_features_are_finite_numbers(self):
        for u in ["https://example.com", "http://1.2.3.4", "http://xn--pypal-4ve.com/"]:
            for k, v in extract_features(u).items():
                self.assertIsInstance(v, (int, float), k)
                self.assertEqual(v, v, k)  # not NaN

    def test_indicators_flag_obvious_attack(self):
        msgs = " ".join(i["message"] for i in
                        describe_risk_indicators(extract_features("http://paypal.com.secure-login.xyz/a")))
        self.assertIn("brand", msgs)
        self.assertEqual([i for i in describe_risk_indicators(extract_features("https://www.wikipedia.org/wiki/Cat"))
                          if i["severity"] in ("high", "medium")], [])


class DatasetTests(unittest.TestCase):
    def test_cleaning_removes_bad_rows(self):
        raw = pd.DataFrame({
            "url": ["https://a.com", "https://A.com", "javascript:x", None, "https://b.com", "https://b.com", "https://c.com"],
            "label": [0, 0, 1, 1, 1, 0, 1],
        })
        clean, report = clean_dataset(raw)
        self.assertEqual(sorted(clean["url"]), ["https://a.com", "https://c.com"])
        self.assertEqual(report["conflicting_labels"], 2)   # b.com had both labels -> dropped entirely
        self.assertEqual(report["unparseable_urls"], 1)
        self.assertEqual(report["duplicates_removed"], 1)

    def test_synthetic_is_balanced_and_unique(self):
        df = generate_synthetic_dataset(400, seed=1)
        self.assertEqual(df["label"].value_counts().to_dict(), {0: 200, 1: 200})
        self.assertEqual(df["url"].str.lower().nunique(), 400)


if __name__ == "__main__":
    unittest.main()
