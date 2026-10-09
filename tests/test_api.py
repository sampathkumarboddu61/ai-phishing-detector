import os
import tempfile
import unittest
from unittest import mock

from app.app import create_app
from src.live_checks import follow_redirects, is_public_host
from src.predictor import DEFAULT_MODEL_PATH


@unittest.skipUnless(DEFAULT_MODEL_PATH.exists(), "train a model first: python -m src.train")
class ApiTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.app = create_app(db_path=os.path.join(self.tmp.name, "t.db"), enable_live_checks=False,
                              rate_limit_per_min=0)
        self.c = self.app.test_client()

    def tearDown(self):
        self.tmp.cleanup()

    def post(self, url, **extra):
        return self.c.post("/api/predict", json={"url": url, **extra})

    def test_health_and_index(self):
        self.assertTrue(self.c.get("/api/health").get_json()["model_loaded"])
        page = self.c.get("/")
        self.assertEqual(page.status_code, 200)
        self.assertIn("default-src 'self'", page.headers["Content-Security-Policy"])
        self.assertEqual(page.headers["X-Content-Type-Options"], "nosniff")

    def test_obvious_phish_flagged_and_clean_site_passes(self):
        bad = self.post("http://paypal.com.secure-login-verify.xyz/webscr?cmd=_login").get_json()
        good = self.post("https://www.wikipedia.org/wiki/Machine_learning").get_json()
        self.assertTrue(bad["is_phishing"])
        self.assertFalse(good["is_phishing"])
        self.assertGreater(bad["risk_score"], good["risk_score"])

    def test_shortener_is_marked_low_confidence(self):
        r = self.post("https://bit.ly/abc123").get_json()
        self.assertTrue(r["low_confidence"])
        self.assertTrue(r["warnings"])

    def test_validation_errors(self):
        self.assertEqual(self.post("").status_code, 400)
        self.assertEqual(self.post("javascript:alert(1)").status_code, 400)
        self.assertEqual(self.post("a" * 3000).status_code, 400)
        self.assertEqual(self.c.post("/api/predict", json={"url": 5}).status_code, 400)
        self.assertEqual(self.c.post("/api/predict", data="x", content_type="text/plain").status_code, 400)
        self.assertEqual(self.c.get("/api/predict").status_code, 405)
        self.assertEqual(self.c.get("/api/history?limit=abc").status_code, 400)
        self.assertEqual(self.c.post("/api/predict", data=b"x" * 20000, content_type="application/json").status_code, 413)

    def test_history_and_stats_roundtrip(self):
        self.post("https://example.com")
        self.post("http://192.0.2.1/login.php")
        self.assertEqual(self.c.get("/api/stats").get_json()["total"], 2)
        rows = self.c.get("/api/history?limit=1").get_json()
        self.assertEqual(len(rows), 1)
        self.assertNotIn("features", rows[0])  # privacy: only minimal data is stored

    def test_hostile_input_is_stored_and_returned_as_data_not_executed(self):
        # SQL injection / markup in a URL must be treated as plain text.
        evil = "http://x.com/?q=';DROP/**/TABLE/**/scans;--<script>alert(1)</script>"
        self.assertEqual(self.post(evil).status_code, 200)
        self.assertEqual(self.post("http://x.com/?q=1 OR 1=1").status_code, 400)  # raw whitespace is rejected
        self.assertEqual(self.c.get("/api/stats").get_json()["total"], 1)
        self.assertIn("<script>", self.c.get("/api/history").get_json()[0]["url"])
        self.assertTrue(self.c.get("/api/history").is_json)

    def test_live_checks_disabled_by_default(self):
        with mock.patch("src.live_checks.requests.Session") as session:
            r = self.post("http://example.com", live_checks=True).get_json()
            session.assert_not_called()
        self.assertIn("disabled", r["live_checks"]["note"])

    def test_rate_limit(self):
        app = create_app(db_path=os.path.join(self.tmp.name, "r.db"), rate_limit_per_min=3)
        c = app.test_client()
        codes = [c.post("/api/predict", json={"url": "https://example.com"}).status_code for _ in range(5)]
        self.assertEqual(codes, [200, 200, 200, 429, 429])


class SsrfGuardTests(unittest.TestCase):
    def test_private_and_loopback_hosts_are_not_public(self):
        for host in ["localhost", "127.0.0.1", "10.0.0.5", "192.168.1.1", "169.254.169.254", "[::1]".strip("[]")]:
            with self.subTest(host=host):
                self.assertFalse(is_public_host(host))

    def test_follow_redirects_refuses_internal_targets_without_connecting(self):
        with mock.patch("src.live_checks.requests.Session") as session:
            result = follow_redirects("http://169.254.169.254/latest/meta-data/")
            session.return_value.head.assert_not_called()
        self.assertIn("not a public", result["note"])


if __name__ == "__main__":
    unittest.main()
