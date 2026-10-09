"""Flask REST API + web UI.

    python -m app.app          # http://127.0.0.1:5000

Endpoints
    GET  /                  web interface
    GET  /api/health        liveness + whether the model is loaded
    POST /api/predict       {"url": "...", "live_checks": false} -> verdict
    GET  /api/history       recent scans (?limit=1..100)
    GET  /api/stats         totals of stored scans
    GET  /api/model-info    model name, threshold, held-out test metrics
"""
from __future__ import annotations

import logging
import os
import time
from collections import defaultdict, deque

from flask import Flask, jsonify, render_template, request

from src.features import InvalidURLError
from src.predictor import ModelNotFoundError, PhishingPredictor

from . import db

MAX_URL_LENGTH = 2048
log = logging.getLogger("phishing-app")


def create_app(model_path=None, db_path=None, enable_live_checks=None, rate_limit_per_min=30) -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 16 * 1024  # request bodies are tiny; reject anything bigger
    app.config["DB_PATH"] = db_path
    app.config["ENABLE_LIVE_CHECKS"] = (os.environ.get("ENABLE_LIVE_CHECKS") == "1"
                                        if enable_live_checks is None else enable_live_checks)
    db.init_db(db_path)

    try:
        predictor = PhishingPredictor(model_path)
    except ModelNotFoundError as exc:
        log.warning("%s", exc)
        predictor = None

    # ---- tiny in-memory rate limiter (per client address) ------------------
    hits: dict = defaultdict(deque)

    def rate_limited() -> bool:
        if not rate_limit_per_min:
            return False
        now, window = time.monotonic(), hits[request.remote_addr]
        while window and now - window[0] > 60:
            window.popleft()
        if len(window) >= rate_limit_per_min:
            return True
        window.append(now)
        return False

    # ---- security headers ----------------------------------------------------
    @app.after_request
    def add_headers(resp):
        resp.headers["Content-Security-Policy"] = (
            "default-src 'self'; img-src 'self' data:; base-uri 'none'; "
            "form-action 'self'; frame-ancestors 'none'")
        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers["Referrer-Policy"] = "no-referrer"
        resp.headers["X-Frame-Options"] = "DENY"
        if request.path.startswith("/api/"):
            resp.headers["Cache-Control"] = "no-store"
        return resp

    # ---- error handling (JSON for the API, no internals leaked) --------------
    def error(status: int, message: str):
        return jsonify({"error": message}), status

    @app.errorhandler(404)
    def _404(_):
        return error(404, "Not found")

    @app.errorhandler(405)
    def _405(_):
        return error(405, "Method not allowed")

    @app.errorhandler(413)
    def _413(_):
        return error(413, "Request body too large")

    @app.errorhandler(500)
    def _500(exc):
        log.exception("Unhandled error: %s", exc)
        return error(500, "Internal server error")

    # ---- routes ----------------------------------------------------------------
    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/api/health")
    def health():
        return jsonify({"status": "ok", "model_loaded": predictor is not None,
                        "live_checks_enabled": app.config["ENABLE_LIVE_CHECKS"]})

    @app.post("/api/predict")
    def predict():
        if predictor is None:
            return error(503, "Model not trained yet. Run: python -m src.train")
        if rate_limited():
            return error(429, "Too many requests. Please wait a minute.")

        payload = request.get_json(silent=True)
        if not isinstance(payload, dict) or not isinstance(payload.get("url"), str):
            return error(400, 'Send JSON like {"url": "https://example.com"}')
        url = payload["url"].strip()
        if not url:
            return error(400, "URL is empty")
        if len(url) > MAX_URL_LENGTH:
            return error(400, f"URL is longer than {MAX_URL_LENGTH} characters")

        wants_live = payload.get("live_checks") is True
        try:
            result = predictor.predict(url, live=wants_live and app.config["ENABLE_LIVE_CHECKS"])
        except InvalidURLError as exc:
            return error(400, str(exc))
        if wants_live and not app.config["ENABLE_LIVE_CHECKS"]:
            result["live_checks"] = {"note": "Live checks are disabled on this server."}

        db.save_scan(result, app.config["DB_PATH"])
        return jsonify(result)

    @app.get("/api/history")
    def history():
        try:
            limit = max(1, min(int(request.args.get("limit", 20)), 100))
        except ValueError:
            return error(400, "limit must be an integer")
        return jsonify(db.recent_scans(limit, app.config["DB_PATH"]))

    @app.get("/api/stats")
    def stats():
        return jsonify(db.stats(app.config["DB_PATH"]))

    @app.get("/api/model-info")
    def model_info():
        if predictor is None:
            return error(503, "Model not trained yet")
        return jsonify({"model": predictor.model_name,
                        "decision_threshold": round(predictor.threshold, 4),
                        "metadata": predictor.metadata})

    return app


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Never enable Flask's debug mode on a reachable server: the Werkzeug debugger allows code execution.
    create_app().run(host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", 5000)),
                     debug=os.environ.get("FLASK_DEBUG") == "1")
