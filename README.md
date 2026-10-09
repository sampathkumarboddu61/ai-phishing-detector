# AI-Powered Phishing Detection System

A machine-learning system that scores a URL as **legitimate** or **phishing** from the text of the address alone, with a REST API, a web interface and a SQLite scan history.

> **Read this first — about the numbers.** With no dataset supplied, the project trains on *synthetic* URLs produced by its own generator (`src/dataset.py`). Those results show that the pipeline works; they do **not** show how the model performs on real traffic. For your report, train on a real labelled dataset (see [Using a real dataset](#using-a-real-dataset)) and quote *those* numbers.

## Quick start

```bash
pip install -r requirements.txt
python -m src.train          # trains 4 models, writes models/ and outputs/ (about 1 minute)
python -m app.app            # open http://127.0.0.1:5000
python -m unittest discover -s tests -t .      # 25 tests
```

Single URL from the command line:
```bash
python -c "from src.predictor import PhishingPredictor as P; print(P().predict('http://paypal.com.secure-login.xyz/webscr')['label'])"
```

## Project layout

```
src/features.py      36 URL features + plain-language explanations (single source of truth)
src/dataset.py       CSV loader, cleaning, synthetic data generator
src/train.py         EDA -> split -> 4 models -> threshold tuning -> evaluation -> save
src/predictor.py     inference pipeline (URL -> verdict)
src/live_checks.py   OPTIONAL redirect count + domain age (SSRF-guarded, off by default)
app/app.py, db.py    Flask REST API + SQLite
app/templates, static  web UI (plain HTML/CSS/JS, no external CDNs)
tests/               unit + API + security tests
docs/FEATURES.md     every feature, what it measures, why it helps (generated)
outputs/             EDA charts, ROC/PR curves, confusion matrices, importance, reports
models/              phishing_model.joblib
```

## How it works

1. **Load & clean** — normalise labels, drop missing/unparseable/duplicate URLs and URLs that appear with *both* labels.
2. **EDA** — class balance, length distributions, per-class boxplots, correlation heatmap (`outputs/eda/`).
3. **Features** — 36 lexical features (length, character counts, structure, host flags, brand impersonation, suspicious words, entropy). Full catalogue: [`docs/FEATURES.md`](docs/FEATURES.md).
4. **Split** — 80/20, stratified **and grouped by registered domain**, so the same domain never appears in both train and test (avoids optimistic, leaky results).
5. **Models** — Logistic Regression, Decision Tree, Random Forest, and gradient boosting (XGBoost if installed, otherwise scikit-learn's `HistGradientBoostingClassifier`; the report states which was used).
6. **Recall-first tuning** — a missed phishing URL (false negative) is the costly error. For each model, the decision threshold is chosen on *out-of-fold predictions of the training set only* as the highest threshold that still reaches the target recall (default 98%). The model with the best precision at that recall is selected. The test set is never used for tuning or selection.
7. **Evaluate** — accuracy, precision, recall, F1, ROC-AUC, PR-AUC, confusion matrices, false-negative rate, permutation feature importance, and an automatic error analysis listing the missed URLs.

## Results on the bundled synthetic data (held-out test set)

Selected model: **Random Forest**, threshold ≈0.40.

| Setting | Recall | Precision | F1 | ROC-AUC | Missed phishing | False alarms |
|---|---|---|---|---|---|---|
| Default threshold 0.50 | 94.7% | 93.8% | 94.2% | 0.993 | 127 | 151 |
| **Recall-tuned threshold** | **97.7%** | 91.8% | 94.7% | 0.993 | **56** | 208 |

Tuning cut missed phishing URLs by more than half, at the price of ~57 extra false alarms. Target recall was 98%; the achieved 97.7% shows a normal small transfer gap between the tuning sample and the test set. All four models are compared in `outputs/model_comparison.csv` and `outputs/training_report.md`.

**Where the remaining misses come from:** mostly link-shortener URLs (`is.gd`, `tinyurl.com`). A short link hides its destination, so *no* URL-text model can judge it. The app therefore marks shortened links **low confidence** and tells the user to expand the link. Resolving the redirect (the optional live check) is the proper fix.

## REST API

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/predict` | `{"url": "...", "live_checks": false}` → verdict, risk score, indicators, features |
| GET | `/api/history?limit=20` | recent scans (max 100) |
| GET | `/api/stats` | totals |
| GET | `/api/model-info` | model, threshold, held-out metrics |
| GET | `/api/health` | liveness |

```bash
curl -s -X POST localhost:5000/api/predict -H 'Content-Type: application/json' \
     -d '{"url":"http://paypal.com.secure-login-verify.xyz/webscr"}'
```
`risk_score` is the model output scaled to 0–100. It is **not** a calibrated probability.

## Using a real dataset

Any CSV with a URL column and a label column works. Common free sources: the *PhiUSIIL Phishing URL* dataset (UCI), Kaggle's *Phishing Site URLs*, or PhishTank / OpenPhish feeds paired with a benign list such as Tranco.

```bash
python -m src.train --data data/real.csv                                  # auto-detects columns url / label
python -m src.train --data data/x.csv --url-col URL --label-col Label \
       --phishing-values bad --legit-values good                          # custom label words
```
Delete `models/phishing_model.joblib` before retraining only if you want a clean slate; training overwrites it. Expect lower scores than on the synthetic data — that is normal and is the honest result. Feature importance will also change: on synthetic data it is dominated by `num_subdomains`, `brand_in_host_mismatch` and `suspicious_word_count` because the generator uses those patterns.

## Security design

- URLs are **never fetched** during normal prediction; features come from the string only.
- Input is validated (http/https only, ≤2048 chars, no whitespace/control characters, 16 KB body cap) and rate-limited (30/min per client).
- All SQL is parameterised. The UI inserts text with `textContent` only and never renders URLs as links; a Content-Security-Policy blocks inline scripts.
- **Live checks** (redirect count, domain age via RDAP) are *off by default*; enable with `ENABLE_LIVE_CHECKS=1`. They resolve each hop and refuse non-public addresses (localhost, private ranges, cloud-metadata IP) to prevent SSRF, use HEAD only, cap hops/time, and send only the registered domain name to RDAP. Known limitation: DNS-rebinding between validation and connection; for production run them in an isolated network.
- Only the URL, verdict and score are stored. `python -m app.db --clear` wipes history.
- Never set `FLASK_DEBUG=1` on a reachable server. Only load `.joblib` models you trained yourself (pickle can execute code).

## Limitations (include these in your report)

- Lexical features only: a page that looks benign by URL but is malicious by content will be missed, and legitimate unusual URLs can be flagged. HTTPS gives no safety guarantee.
- Short links, freshly compromised legitimate sites, and look-alike domains with clean-looking text are intrinsically hard.
- Domain age and redirects are shown but are not model inputs, since public datasets contain URL strings only.
- Brand and TLD lists are small heuristics that age; extend them for your region.
- A single train/test split has variance; for a thesis, also report 5-fold grouped cross-validation on your real dataset.

## Ideas to extend

HTML/content features, WHOIS/DNS features in the training set, character-level CNN/LSTM on the raw URL, probability calibration, browser extension front end, retraining on user feedback.
