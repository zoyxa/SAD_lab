#!/usr/bin/env python3
"""
End-to-end analysis and modeling for `risk_factors_cervical_cancer.csv`,
following the same patterns as PWR-SAD-LABS Lab2–Lab6:

  Lab2 — pandas I/O, info/describe, missing values, duplicates, dtypes
  Lab3 — EDA plots, KMeans on preprocessed features, PCA 2D visualization
  Lab4 — train/test split, ColumnTransformer, SimpleImputer(median),
          StandardScaler, LogisticRegression pipeline, accuracy / ROC-AUC /
          confusion matrix / classification_report, coefficient ranking
  Lab5 — Shapiro–Wilk, Levene, Welch t-test, Mann–Whitney U, bootstrap CI
         for a key continuous variable (Age) across risk groups
  Lab6 — persist fitted sklearn Pipeline with joblib

Targets (risk):
  * biopsy      — binary: Biopsy (1 = histologic abnormality; strong risk proxy)
  * severity    — ordinal 0/1/2 from screening + biopsy (multinomial LR)

Features are restricted to columns *before* outcomes (no Dx/Hinselmann/…
in X) to avoid using downstream diagnostics as predictors of biopsy.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

OUTCOME_COLS = [
    "Dx:Cancer",
    "Dx:CIN",
    "Dx:HPV",
    "Dx",
    "Hinselmann",
    "Schiller",
    "Citology",
    "Biopsy",
]


def _project_root() -> Path:
    return Path(__file__).resolve().parent


def load_raw(csv_path: Path) -> pd.DataFrame:
    """Lab2-style load; `?` encoded as missing (common in this dataset)."""
    return pd.read_csv(csv_path, na_values=["?"])


def lab2_data_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Lab2-style cleaning summary and light transformations."""
    print("\n=== Lab 2: data quality ===")
    print("shape:", df.shape)
    print("\nhead():\n", df.head(3))
    print("\ninfo():")
    df.info(verbose=False)
    print("\ndescribe() (numeric sample):\n", df.describe().T.head(12))

    dup = df.duplicated().sum()
    print(f"\nduplicated rows: {dup}")
    df = df.drop_duplicates()

    na = df.isna().sum().sort_values(ascending=False)
    print("\ntop missing counts:\n", na[na > 0].head(15))

    for c in OUTCOME_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    feat_cols = [c for c in df.columns if c not in OUTCOME_COLS]
    for c in feat_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def build_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Add `y_biopsy` and `y_severity` label columns."""
    df = df.copy()
    df["y_biopsy"] = df["Biopsy"]

    h, s, cit, bio = df["Hinselmann"], df["Schiller"], df["Citology"], df["Biopsy"]
    sev = []
    for i in range(len(df)):
        b = bio.iloc[i]
        if pd.isna(b):
            sev.append(np.nan)
            continue
        if b == 1:
            sev.append(2)
            continue
        hi, si, ci = h.iloc[i], s.iloc[i], cit.iloc[i]
        if pd.isna(hi) or pd.isna(si) or pd.isna(ci):
            sev.append(np.nan)
        elif hi == 1 or si == 1 or ci == 1:
            sev.append(1)
        else:
            sev.append(0)
    df["y_severity"] = sev
    return df


def lab3_eda_clustering_pca(
    X: pd.DataFrame,
    y_display: np.ndarray,
    out_dir: Path,
    artifact_prefix: str,
    random_state: int = 42,
) -> None:
    """Lab3-style EDA, KMeans, PCA plot (y used only for coloring points)."""
    print("\n=== Lab 3: EDA, clustering, PCA ===")
    sns.set_theme(style="whitegrid")
    out_dir.mkdir(parents=True, exist_ok=True)

    n = min(8, X.shape[1])
    sample_cols = list(X.columns[:n])
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    axes = axes.ravel()
    for ax, col in zip(axes, sample_cols):
        sns.histplot(X[col].dropna(), kde=True, ax=ax)
        ax.set_title(col[:40])
    plt.tight_layout()
    fig.savefig(out_dir / f"lab3_histograms_sample_{artifact_prefix}.png", dpi=120)
    plt.close(fig)

    corr = X[sample_cols].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr,
        ax=ax,
        cmap="vlag",
        center=0,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 8},
        linewidths=0.5,
        linecolor="white",
    )
    ax.set_title("Correlation (sample of risk-factor columns)")
    fig.tight_layout()
    fig.savefig(out_dir / f"lab3_correlation_heatmap_{artifact_prefix}.png", dpi=120)
    plt.close(fig)

    pre = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    X_imp = pre.fit_transform(X)
    km = KMeans(n_clusters=3, random_state=random_state, n_init=20)
    clusters = km.fit_predict(X_imp)
    sizes = pd.Series(clusters).value_counts().sort_index()
    print("KMeans cluster sizes:\n", sizes)

    pca = PCA(n_components=2, random_state=random_state)
    xy = pca.fit_transform(X_imp)
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=y_display, cmap="viridis", alpha=0.75, s=22)
    ax.set_title("PCA (2D) of risk factors, colored by label (0/1 or severity)")
    plt.colorbar(sc, ax=ax)
    fig.tight_layout()
    fig.savefig(out_dir / f"lab3_pca_scatter_{artifact_prefix}.png", dpi=120)
    plt.close(fig)


def lab5_hypothesis_age(df: pd.DataFrame, group_col: str, out_dir: Path) -> None:
    """Lab5-style two-group comparisons on Age (continuous); group_col has values 0 vs 1."""
    print("\n=== Lab 5: hypothesis checks (Age vs groups) ===")
    out_dir.mkdir(parents=True, exist_ok=True)
    d = df[["Age", group_col]].dropna()
    g0 = d.loc[d[group_col] == 0, "Age"].values
    g1 = d.loc[d[group_col] == 1, "Age"].values
    if len(g0) < 3 or len(g1) < 3:
        print("Not enough data for two-group tests.")
        return

    sh_x = stats.shapiro(g0)
    sh_y = stats.shapiro(g1)
    lev = stats.levene(g0, g1)
    tt = stats.ttest_ind(g0, g1, equal_var=False)
    mw = stats.mannwhitneyu(g0, g1, alternative="two-sided")

    print(
        pd.DataFrame(
            {
                "test": ["shapiro_g0", "shapiro_g1", "levene", "welch_t", "mannwhitney"],
                "statistic": [sh_x.statistic, sh_y.statistic, lev.statistic, tt.statistic, mw.statistic],
                "p_value": [sh_x.pvalue, sh_y.pvalue, lev.pvalue, tt.pvalue, mw.pvalue],
            }
        )
    )

    rng = np.random.default_rng(42)
    n_boot = 2000
    diff = np.empty(n_boot)
    for b in range(n_boot):
        xb = rng.choice(g0, size=len(g0), replace=True)
        yb = rng.choice(g1, size=len(g1), replace=True)
        diff[b] = xb.mean() - yb.mean()
    ci_low, ci_high = np.quantile(diff, [0.025, 0.975])
    print(f"Bootstrap mean(Age|0) - mean(Age|1): mean={diff.mean():.3f}, 95% CI=({ci_low:.3f}, {ci_high:.3f})")


def lab4_train_logistic(
    X: pd.DataFrame,
    y: pd.Series,
    out_dir: Path,
    multiclass: bool,
    artifact_prefix: str,
    random_state: int = 42,
) -> Pipeline:
    """Lab4-style preprocessing + LogisticRegression (+ Lab6 save)."""
    print("\n=== Lab 4: supervised model ===")
    out_dir.mkdir(parents=True, exist_ok=True)

    strat = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=random_state,
        stratify=strat,
    )

    numeric_features = X.columns.tolist()
    numeric_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    preprocessor = ColumnTransformer([("num", numeric_transformer, numeric_features)])

    # sklearn >= 1.7: multinomial is automatic for >2 classes (multi_class arg removed).
    lr_kwargs: dict = dict(max_iter=4000, class_weight="balanced", random_state=random_state)
    model = Pipeline(
        [
            ("preprocess", preprocessor),
            ("model", LogisticRegression(**lr_kwargs)),
        ]
    )
    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    if multiclass:
        proba = model.predict_proba(X_test)
        roc = roc_auc_score(y_test, proba, multi_class="ovr", average="weighted")
        roc_tr = roc_auc_score(y_train, model.predict_proba(X_train), multi_class="ovr", average="weighted")
    else:
        proba = model.predict_proba(X_test)[:, 1]
        roc = roc_auc_score(y_test, proba)
        roc_tr = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])

    metrics = pd.DataFrame(
        {
            "split": ["train", "test"],
            "accuracy": [
                accuracy_score(y_train, train_pred),
                accuracy_score(y_test, test_pred),
            ],
            "roc_auc": [roc_tr, roc],
        }
    )
    print("\nMetrics:\n", metrics)

    labels = sorted(y.unique().tolist())
    print("\nclassification_report (test):\n")
    print(classification_report(y_test, test_pred, labels=labels, zero_division=0))

    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(
        confusion_matrix(y_test, test_pred, labels=labels),
        display_labels=[str(x) for x in labels],
    ).plot(ax=ax, colorbar=False)
    ax.set_title("Confusion matrix (test)")
    fig.tight_layout()
    fig.savefig(out_dir / f"lab4_confusion_matrix_test_{artifact_prefix}.png", dpi=120)
    plt.close(fig)

    coef = model.named_steps["model"].coef_
    if coef.shape[0] == 1:
        s = pd.Series(coef[0], index=X.columns).sort_values(key=np.abs, ascending=False)
        print("\nTop |coef| (binary):\n", s.head(12))
    else:
        print("\nMultinomial model: see coef_.shape", coef.shape)

    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Cervical cancer risk analysis (PWR-SAD lab style).")
    parser.add_argument(
        "--csv",
        type=Path,
        default=_project_root() / "risk_factors_cervical_cancer.csv",
        help="Path to risk_factors_cervical_cancer.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_project_root() / "artifacts",
        help="Directory for figures and saved model",
    )
    parser.add_argument(
        "--target",
        choices=["biopsy", "severity"],
        default="biopsy",
        help="biopsy: predict Biopsy; severity: 3-class ordinal screening outcome",
    )
    args = parser.parse_args()

    if not args.csv.is_file():
        raise SystemExit(f"CSV not found: {args.csv}")

    df = load_raw(args.csv)
    df = lab2_data_quality(df)
    df = build_targets(df)

    feature_cols = [c for c in df.columns if c not in OUTCOME_COLS + ["y_biopsy", "y_severity"]]
    X = df[feature_cols]

    if args.target == "biopsy":
        y = df["y_biopsy"]
        multiclass = False
    else:
        y = df["y_severity"]
        multiclass = True

    mask = y.notna()
    X, y = X.loc[mask], y.loc[mask].astype(int)

    lab3_eda_clustering_pca(X, y.to_numpy(), args.out_dir, artifact_prefix=args.target)

    if args.target == "biopsy":
        hypo = df.loc[mask, ["Age", "y_biopsy"]].copy()
        lab5_hypothesis_age(hypo.rename(columns={"y_biopsy": "grp"}), "grp", args.out_dir)
    else:
        sub = df.loc[mask & df["y_severity"].isin([0, 2]), ["Age", "y_severity"]].copy()
        if len(sub) > 10 and sub["y_severity"].nunique() >= 2:
            sub["grp"] = (sub["y_severity"] == 2).astype(int)
            lab5_hypothesis_age(sub.drop(columns=["y_severity"]), "grp", args.out_dir)
        else:
            print("Skipping Lab5 Age test for severity (insufficient 0 vs 2).")

    if y.nunique() < 2:
        raise SystemExit("Target has fewer than 2 classes after cleaning; cannot train.")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        model = lab4_train_logistic(X, y, args.out_dir, multiclass=multiclass, artifact_prefix=args.target)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.out_dir / f"cervical_risk_logreg_{args.target}.joblib"
    joblib.dump(model, model_path)
    print(f"\n=== Lab 6: saved pipeline to {model_path} ===")


if __name__ == "__main__":
    main()
