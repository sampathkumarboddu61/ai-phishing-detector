"""
Inference pipeline: URL string -> verdict.

    from src.predictor import PhishingPredictor
    result = PhishingPredictor().predict("http://paypal.com.secure-login.xyz/webscr")

Security note: the model file is a joblib/pickle bundle. Unpickling can execute
arbitrary code, so only load model files that you trained yourself or fully trust.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from .features import (FEATURE_NAMES, InvalidURLError, describe_risk_indicators, extract_features,
                       parse_url)

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "phishing_model.joblib"
HIGH_RISK_SCORE = 0.80


class ModelNotFoundError(FileNotFoundError):
    pass


class PhishingPredictor:
    def __init__(self, model_path: str | Path | None = None):
        path = Path(model_path or DEFAULT_MODEL_PATH)
        if not path.exists():
            raise ModelNotFoundError(f"Model file not found: {path}. Train one with: python -m src.train")
        bundle = joblib.load(path)
        if bundle["feature_names"] != FEATURE_NAMES:
            raise RuntimeError("Model was trained with a different feature set; re-run python -m src.train")
        self.model = bundle["model"]
        self.model_name: str = bundle["model_name"]
        self.threshold: float = float(bundle["threshold"])
        self.metadata: dict = bundle["metadata"]

    def predict(self, url: str, live: bool = False) -> dict:
        """Classify one URL. Raises InvalidURLError for unusable input."""
        parsed = parse_url(url)
        feats = extract_features(parsed.url)
        score = float(self.model.predict_proba(pd.DataFrame([feats], columns=FEATURE_NAMES))[0, 1])
        is_phishing = score >= self.threshold

        if not is_phishing:
            risk = "low"
        elif score >= max(HIGH_RISK_SCORE, self.threshold):
            risk = "high"
        else:
            risk = "medium"

        warnings = []
        if feats["is_shortener"]:
            warnings.append("This is a shortened link: its real destination is hidden, so a URL-only model "
                            "cannot judge it reliably. Treat the verdict as low-confidence and expand the "
                            "link (or run the live redirect check) before trusting it.")

        result = {
            "url": parsed.url,
            "label": "phishing" if is_phishing else "legitimate",
            "is_phishing": bool(is_phishing),
            "risk_level": risk,
            # Model output scaled to 0-100. NOT a calibrated probability.
            "risk_score": round(score * 100, 1),
            "decision_threshold": round(self.threshold * 100, 1),
            "low_confidence": bool(feats["is_shortener"]),
            "warnings": warnings,
            "indicators": describe_risk_indicators(feats),
            "features": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in feats.items()},
            "model": self.model_name,
        }
        if live:
            from .live_checks import run_live_checks  # imported lazily: needs network
            result["live_checks"] = run_live_checks(parsed.url)
        return result


__all__ = ["PhishingPredictor", "ModelNotFoundError", "InvalidURLError"]
