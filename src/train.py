"""
End-to-end training pipeline.

    python -m src.train                          # synthetic demo data
    python -m src.train --data data/real.csv     # real labelled dataset

Pipeline: load -> clean -> EDA -> features -> domain-grouped split -> train 4
models -> out-of-fold threshold tuning (recall-first) -> test evaluation ->
select best -> permutation importance / error analysis -> save model + reports.

Why recall-first: a phishing URL classified as safe (false negative) lets the
victim proceed to a credential-stealing page, whereas a false positive costs one
unnecessary warning. We therefore (1) pick each model's decision threshold on
out-of-fold TRAINING predictions so that it reaches a target recall, and (2)
select the model with the highest precision *at that recall*. The test set is
never used for tuning or selection.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             fbeta_score, precision_recall_curve, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .dataset import (build_feature_matrix, clean_dataset, generate_synthetic_dataset, load_dataset,
                      registered_domain_of)
from .features import FEATURE_NAMES

try:  # XGBoost is optional; fall back to scikit-learn's gradient boosting
    from xgboost import XGBClassifier
    HAVE_XGB = True
except Exception:  # pragma: no cover - depends on environment
    HAVE_XGB = False

ROOT = Path(__file__).resolve().parent.parent
PALETTE = {0: "#1f77b4", 1: "#d95f02"}  # legitimate = blue, phishing = orange (colour-blind safe)
LABEL_NAMES = {0: "Legitimate", 1: "Phishing"}


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
def build_models(seed: int, pos_weight: float) -> dict:
    models = {
        "Logistic Regression": Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed))]),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=12, min_samples_leaf=5, class_weight="balanced", random_state=seed),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, class_weight="balanced_subsample",
            n_jobs=-1, random_state=seed),
    }
    if HAVE_XGB:
        models["XGBoost"] = XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.08, subsample=0.9, colsample_bytree=0.9,
            eval_metric="logloss", scale_pos_weight=pos_weight, n_jobs=-1, random_state=seed)
    else:
        models["Gradient Boosting (sklearn HistGB)"] = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.08, class_weight="balanced", random_state=seed)
    return models


# --------------------------------------------------------------------------
# Metrics helpers
# --------------------------------------------------------------------------
def threshold_for_recall(y, p, target: float):
    """Largest threshold whose recall is still >= target (maximises precision for that recall)."""
    prec, rec, thr = precision_recall_curve(y, p)
    ok = np.where(rec[:-1] >= target)[0]
    idx = ok[-1] if len(ok) else 0
    return float(thr[idx])


def evaluate(y, p, thr: float) -> dict:
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(thr),
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "f2": fbeta_score(y, pred, beta=2, zero_division=0),
        "roc_auc": roc_auc_score(y, p),
        "pr_auc": average_precision_score(y, p),
        "false_negative_rate": fn / (fn + tp) if (fn + tp) else 0.0,
        "false_positive_rate": fp / (fp + tn) if (fp + tn) else 0.0,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


# --------------------------------------------------------------------------
# EDA
# --------------------------------------------------------------------------
def run_eda(df: pd.DataFrame, X: pd.DataFrame, y: pd.Series, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    names = y.map(LABEL_NAMES)
    pal = {LABEL_NAMES[k]: v for k, v in PALETTE.items()}

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.countplot(x=names, order=list(pal), palette=pal, hue=names, legend=False, ax=ax)
    for c in ax.containers:
        ax.bar_label(c)
    ax.set(title="Class balance", xlabel="", ylabel="URLs")
    fig.tight_layout(); fig.savefig(out / "class_balance.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    cap = X["url_length"].quantile(0.99)
    sns.histplot(data=X.assign(cls=names)[X["url_length"] <= cap], x="url_length", hue="cls",
                 palette=pal, bins=50, element="step", stat="density", common_norm=False, ax=ax)
    ax.set(title="URL length by class (clipped at 99th percentile)", xlabel="characters")
    fig.tight_layout(); fig.savefig(out / "url_length_distribution.png", dpi=150); plt.close(fig)

    show = ["url_length", "num_dots", "num_hyphens", "num_subdomains", "suspicious_word_count",
            "url_entropy", "host_entropy", "digit_ratio"]
    fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    for ax, col in zip(axes.ravel(), show):
        sns.boxplot(x=names, y=X[col], order=list(pal), palette=pal, hue=names, legend=False,
                    showfliers=False, ax=ax)
        ax.set(title=col, xlabel="", ylabel="")
    fig.suptitle("Key features by class")
    fig.tight_layout(); fig.savefig(out / "feature_boxplots.png", dpi=150); plt.close(fig)

    corr = X.assign(label=y).corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(13, 11))
    sns.heatmap(corr, cmap="vlag", center=0, square=True, cbar_kws={"shrink": 0.6}, ax=ax)
    ax.set_title("Feature correlation (including label)")
    fig.tight_layout(); fig.savefig(out / "correlation_heatmap.png", dpi=150); plt.close(fig)

    lab_corr = corr["label"].drop("label").sort_values(key=np.abs, ascending=False)
    means = X.groupby(y.values).mean().T.rename(columns={0: "mean_legit", 1: "mean_phishing"})
    summary = means.assign(corr_with_label=lab_corr).sort_values("corr_with_label", key=np.abs, ascending=False)
    summary.round(4).to_csv(out / "feature_summary_by_class.csv")
    (out / "eda_summary.txt").write_text(
        f"Rows: {len(df)}\nClass counts: {y.value_counts().to_dict()}\n"
        f"Missing values in feature matrix: {int(X.isna().sum().sum())}\n\n"
        f"Top 10 features by |correlation with label|:\n{lab_corr.head(10).round(3).to_string()}\n")


# --------------------------------------------------------------------------
# Plots for evaluation
# --------------------------------------------------------------------------
def plot_confusion_matrices(results: dict, out: Path) -> None:
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4))
    axes = np.atleast_1d(axes)
    for ax, (name, r) in zip(axes, results.items()):
        m = r["test_tuned"]
        cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["Pred safe", "Pred phish"], yticklabels=["Actual safe", "Actual phish"])
        ax.set_title(f"{name}\nthr={m['threshold']:.2f}  FN={m['fn']}", fontsize=10)
    fig.suptitle("Confusion matrices on held-out test set (recall-tuned thresholds)")
    fig.tight_layout(); fig.savefig(out / "confusion_matrices.png", dpi=150); plt.close(fig)


def plot_curves(results: dict, y_test, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, r in results.items():
        fpr, tpr, _ = roc_curve(y_test, r["test_proba"])
        ax.plot(fpr, tpr, label=f"{name} (AUC {r['test_default']['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set(xlabel="False positive rate", ylabel="True positive rate (recall)", title="ROC curves (test set)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout(); fig.savefig(out / "roc_curves.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    for name, r in results.items():
        pr, rc, _ = precision_recall_curve(y_test, r["test_proba"])
        ax.plot(rc, pr, label=name)
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision-recall curves (test set)", ylim=(0, 1.02))
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout(); fig.savefig(out / "pr_curves.png", dpi=150); plt.close(fig)


def plot_threshold_tradeoff(y_oof, p_oof, thr, target, name, out: Path) -> None:
    prec, rec, th = precision_recall_curve(y_oof, p_oof)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(th, prec[:-1], label="Precision", color=PALETTE[0])
    ax.plot(th, rec[:-1], label="Recall", color=PALETTE[1])
    ax.axvline(thr, color="k", ls="--", lw=1, label=f"chosen threshold {thr:.2f}")
    ax.axhline(target, color="grey", ls=":", lw=1, label=f"target recall {target:.0%}")
    ax.set(xlabel="Decision threshold", ylabel="Score", ylim=(0, 1.02),
           title=f"{name}: precision/recall vs threshold (out-of-fold, training data)")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "threshold_tradeoff.png", dpi=150); plt.close(fig)


def plot_importance(imp: pd.DataFrame, name: str, out: Path) -> None:
    top = imp.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"], color=PALETTE[0])
    ax.set(xlabel="Drop in ROC-AUC when feature is shuffled", title=f"Permutation importance - {name} (test set)")
    fig.tight_layout(); fig.savefig(out / "feature_importance.png", dpi=150); plt.close(fig)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", help="CSV with a URL column and a label column. Omit to use synthetic demo data.")
    ap.add_argument("--url-col"); ap.add_argument("--label-col")
    ap.add_argument("--phishing-values", nargs="*", help="label values meaning phishing (default: 1, phishing, bad, malicious ...)")
    ap.add_argument("--legit-values", nargs="*", help="label values meaning legitimate (default: 0, legitimate, good, benign ...)")
    ap.add_argument("--synthetic-n", type=int, default=24000)
    ap.add_argument("--target-recall", type=float, default=0.98)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default=str(ROOT / "outputs"))
    ap.add_argument("--model-path", default=str(ROOT / "models" / "phishing_model.joblib"))
    args = ap.parse_args(argv)

    warnings.filterwarnings("ignore", category=UserWarning)
    out = Path(args.out_dir)
    (out / "eda").mkdir(parents=True, exist_ok=True)
    (out / "evaluation").mkdir(parents=True, exist_ok=True)
    Path(args.model_path).parent.mkdir(parents=True, exist_ok=True)

    # 1. Load + clean --------------------------------------------------------
    if args.data:
        raw = load_dataset(args.data, args.url_col, args.label_col,
                           set(args.phishing_values) if args.phishing_values else None,
                           set(args.legit_values) if args.legit_values else None)
        data_source = f"real dataset: {Path(args.data).name}"
    else:
        raw = generate_synthetic_dataset(args.synthetic_n, args.seed)
        (ROOT / "data").mkdir(exist_ok=True)
        raw.to_csv(ROOT / "data" / "synthetic_urls.csv", index=False)
        data_source = "SYNTHETIC demo data (not real-world traffic)"
        print("!! No --data given: using synthetic data. Metrics are NOT representative of real-world performance.")
    df, clean_report = clean_dataset(raw)
    print("Cleaning report:", clean_report)

    # 2. Features + EDA ------------------------------------------------------
    X = build_feature_matrix(df["url"])
    y = df["label"].astype(int)
    groups = df["url"].map(registered_domain_of)
    run_eda(df, X, y, out / "eda")
    print(f"Features: {X.shape[1]} | rows: {len(X)} | EDA saved to {out / 'eda'}")

    # 3. Domain-grouped, stratified 80/20 split (same domain never in both sets)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.seed)
    tr, te = next(splitter.split(X, y, groups))
    Xtr, Xte, ytr, yte, gtr = X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te], groups.iloc[tr]
    print(f"Train {len(Xtr)} / Test {len(Xte)} (phishing share {ytr.mean():.2%} / {yte.mean():.2%})")

    # 4. Train + tune thresholds out-of-fold -----------------------------------
    cv = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.seed).split(Xtr, ytr, gtr))
    pos_weight = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))
    results = {}
    for name, model in build_models(args.seed, pos_weight).items():
        print(f"Training {name} ...")
        oof = cross_val_predict(model, Xtr, ytr, cv=cv, method="predict_proba")[:, 1]
        thr = threshold_for_recall(ytr, oof, args.target_recall)
        oof_tuned = evaluate(ytr, oof, thr)
        model.fit(Xtr, ytr)
        proba = model.predict_proba(Xte)[:, 1]
        results[name] = {
            "model": model, "oof_proba": oof, "test_proba": proba, "threshold": thr,
            "cv_tuned": oof_tuned,
            "test_default": evaluate(yte, proba, 0.5),
            "test_tuned": evaluate(yte, proba, thr),
        }

    # 5. Select best: highest CV precision at the target recall ----------------
    best = max(results, key=lambda n: (results[n]["cv_tuned"]["precision"], results[n]["cv_tuned"]["roc_auc"]))
    print(f"Selected model: {best}")

    # 6. Reports ---------------------------------------------------------------
    rows = []
    for name, r in results.items():
        for kind, key in (("test @ 0.50", "test_default"), ("test @ tuned thr", "test_tuned")):
            m = r[key]
            rows.append({"model": name, "setting": kind, **{k: round(v, 4) if isinstance(v, float) else v
                                                            for k, v in m.items()},
                         "selected": name == best})
    comp = pd.DataFrame(rows)
    comp.to_csv(out / "model_comparison.csv", index=False)

    plot_confusion_matrices(results, out / "evaluation")
    plot_curves(results, yte, out / "evaluation")
    plot_threshold_tradeoff(ytr, results[best]["oof_proba"], results[best]["threshold"],
                            args.target_recall, best, out / "evaluation")

    best_model = results[best]["model"]
    perm = permutation_importance(best_model, Xte, yte, scoring="roc_auc", n_repeats=5,
                                  random_state=args.seed, n_jobs=-1)
    imp = pd.DataFrame({"feature": FEATURE_NAMES, "importance_mean": perm.importances_mean,
                        "importance_std": perm.importances_std}).sort_values("importance_mean", ascending=False)
    imp.to_csv(out / "permutation_importance.csv", index=False)
    plot_importance(imp, best, out / "evaluation")

    # Error analysis: which phishing URLs slipped through?
    thr = results[best]["threshold"]
    scored = pd.DataFrame({"url": df.loc[Xte.index, "url"], "label": yte, "score": results[best]["test_proba"]})
    fn_df = scored[(scored.label == 1) & (scored.score < thr)].sort_values("score")
    fp_df = scored[(scored.label == 0) & (scored.score >= thr)].sort_values("score", ascending=False)
    fn_df.to_csv(out / "false_negatives.csv", index=False)
    fp_df.to_csv(out / "false_positives.csv", index=False)

    # 7. Save model bundle -----------------------------------------------------
    t = results[best]["test_tuned"]
    bundle = {
        "model": best_model,
        "model_name": best,
        "threshold": thr,
        "feature_names": FEATURE_NAMES,
        "metadata": {
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "data_source": data_source,
            "synthetic_data": not bool(args.data),
            "n_rows": int(len(df)), "n_train": int(len(Xtr)), "n_test": int(len(Xte)),
            "target_recall": args.target_recall,
            "test_metrics": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in t.items()},
            "xgboost_available": HAVE_XGB,
            "versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                         "numpy": np.__version__, "pandas": pd.__version__},
        },
    }
    joblib.dump(bundle, args.model_path)
    (out / "metrics.json").write_text(json.dumps(bundle["metadata"] | {"selected_model": best, "cleaning": clean_report}, indent=2))

    # Human-readable report ------------------------------------------------------
    def fmt(m): return (f"{m['accuracy']:.4f} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} | "
                        f"{m['roc_auc']:.4f} | {m['fn']} | {m['fp']}")
    lines = [
        "# Training Report", "",
        f"- Data source: **{data_source}**",
        f"- Rows after cleaning: {len(df)} (train {len(Xtr)}, test {len(Xte)}; split grouped by registered domain)",
        f"- Target recall for threshold tuning: {args.target_recall:.0%} (threshold chosen on out-of-fold training predictions only)",
        f"- Selected model: **{best}**, threshold **{thr:.3f}**",
        f"- Gradient boosting implementation: {'XGBoost' if HAVE_XGB else 'scikit-learn HistGradientBoosting (XGBoost not installed)'}",
        "", "## Test-set results", "",
        "| Model | Setting | Accuracy | Precision | Recall | F1 | ROC-AUC | False neg. | False pos. |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, r in results.items():
        star = " (selected)" if name == best else ""
        lines.append(f"| {name}{star} | threshold 0.50 | {fmt(r['test_default'])} |")
        lines.append(f"| {name}{star} | tuned {r['threshold']:.3f} | {fmt(r['test_tuned'])} |")
    lines += ["", "## Top features (permutation importance)", "", "| Feature | Importance |", "|---|---|"]
    lines += [f"| {row.feature} | {row.importance_mean:.4f} |" for row in imp.head(10).itertuples()]
    n_pos = int((yte == 1).sum())
    lines += ["", "## Error analysis (selected model)", "",
              f"- Target recall {args.target_recall:.1%} (tuned out-of-fold on training data) vs **achieved test recall {t['recall']:.1%}**. "
              "A gap is normal: the threshold is tuned on one sample and applied to another, and a grouped split can "
              "concentrate a whole hard domain in the test set.",
              f"- Missed phishing URLs (false negatives): **{len(fn_df)}** of {n_pos} -> `false_negatives.csv`",
              f"- Wrongly flagged legitimate URLs (false positives): **{len(fp_df)}** -> `false_positives.csv`"]
    if len(fn_df):
        top = fn_df["url"].map(registered_domain_of).value_counts().head(5)
        lines += ["- Registered domains most common among the misses (a handful of domains explaining many misses "
                  "usually means those URLs are inherently ambiguous, e.g. link shorteners):", ""]
        lines += [f"  - `{dom}`: {cnt}" for dom, cnt in top.items()]
    lines.append("")
    if not args.data:
        lines += ["> **Caution:** these numbers come from synthetic data produced by this project's own generator. "
                  "They show the pipeline works, not how it performs on real traffic. Re-run with `--data` for a real dataset.", ""]
    (out / "training_report.md").write_text("\n".join(lines))

    print("\n" + "\n".join(lines[:9]))
    print(comp[comp.setting == "test @ tuned thr"][["model", "accuracy", "precision", "recall", "f1", "roc_auc", "fn", "fp"]].to_string(index=False))
    print(f"\nModel saved to {args.model_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
