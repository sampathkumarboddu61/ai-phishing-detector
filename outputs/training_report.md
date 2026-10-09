# Training Report

- Data source: **SYNTHETIC demo data (not real-world traffic)**
- Rows after cleaning: 24000 (train 19202, test 4798; split grouped by registered domain)
- Target recall for threshold tuning: 98% (threshold chosen on out-of-fold training predictions only)
- Selected model: **Random Forest**, threshold **0.403**
- Gradient boosting implementation: scikit-learn HistGradientBoosting (XGBoost not installed)

## Test-set results

| Model | Setting | Accuracy | Precision | Recall | F1 | ROC-AUC | False neg. | False pos. |
|---|---|---|---|---|---|---|---|---|
| Logistic Regression | threshold 0.50 | 0.9398 | 0.9282 | 0.9533 | 0.9406 | 0.9927 | 112 | 177 |
| Logistic Regression | tuned 0.425 | 0.9450 | 0.9175 | 0.9779 | 0.9467 | 0.9927 | 53 | 211 |
| Decision Tree | threshold 0.50 | 0.9398 | 0.9407 | 0.9387 | 0.9397 | 0.9890 | 147 | 142 |
| Decision Tree | tuned 0.493 | 0.9398 | 0.9407 | 0.9387 | 0.9397 | 0.9890 | 147 | 142 |
| Random Forest (selected) | threshold 0.50 | 0.9421 | 0.9377 | 0.9471 | 0.9423 | 0.9931 | 127 | 151 |
| Random Forest (selected) | tuned 0.403 | 0.9450 | 0.9185 | 0.9767 | 0.9467 | 0.9931 | 56 | 208 |
| Gradient Boosting (sklearn HistGB) | threshold 0.50 | 0.9360 | 0.9380 | 0.9337 | 0.9359 | 0.9923 | 159 | 148 |
| Gradient Boosting (sklearn HistGB) | tuned 0.410 | 0.9396 | 0.9200 | 0.9629 | 0.9409 | 0.9923 | 89 | 201 |

## Top features (permutation importance)

| Feature | Importance |
|---|---|
| num_subdomains | 0.0705 |
| brand_in_host_mismatch | 0.0552 |
| suspicious_word_count | 0.0173 |
| suspicious_tld | 0.0019 |
| num_hyphens_in_domain | 0.0012 |
| host_digit_ratio | 0.0009 |
| num_query_params | 0.0008 |
| num_dots | 0.0007 |
| num_hyphens | 0.0006 |
| suspicious_word_in_host | 0.0005 |

## Error analysis (selected model)

- Target recall 98.0% (tuned out-of-fold on training data) vs **achieved test recall 97.7%**. A gap is normal: the threshold is tuned on one sample and applied to another, and a grouped split can concentrate a whole hard domain in the test set.
- Missed phishing URLs (false negatives): **56** of 2399 -> `false_negatives.csv`
- Wrongly flagged legitimate URLs (false positives): **208** -> `false_positives.csv`
- Registered domains most common among the misses (a handful of domains explaining many misses usually means those URLs are inherently ambiguous, e.g. link shorteners):

  - `is.gd`: 13
  - `tinyurl.com`: 5
  - `echolumen.io`: 1
  - `zenithautokernel.co.uk`: 1
  - `glacierjasminefitness.net`: 1

> **Caution:** these numbers come from synthetic data produced by this project's own generator. They show the pipeline works, not how it performs on real traffic. Re-run with `--data` for a real dataset.
