"""Train and compare 4 classifiers that predict a customer's preferred package tier.

Run:  python -m src.train_models
Outputs: models/preference_model.joblib, models/model_meta.json,
         reports/model_comparison.csv, reports/*.png
"""
import json

import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report,
                             f1_score, precision_score, recall_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src import config
from src.data_loader import load_customers


def build_preprocessor():
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num", numeric, config.NUMERIC_FEATURES),
                              ("cat", categorical, config.CATEGORICAL_FEATURES)])


def get_models():
    rs = config.RANDOM_STATE
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=rs),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=rs, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(random_state=rs),
    }


def main():
    config.MODEL_DIR.mkdir(exist_ok=True)
    config.REPORT_DIR.mkdir(exist_ok=True)

    df = load_customers()
    X = df[config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES]
    y = df[config.TARGET]
    print(f"Rows: {len(df)}  |  class balance:\n{y.value_counts()}\n")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=config.RANDOM_STATE)

    rows, fitted = [], {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)
    for name, model in get_models().items():
        pipe = Pipeline([("prep", build_preprocessor()), ("model", model)])
        cv_f1 = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="f1_macro").mean()
        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_te)
        rows.append({
            "model": name,
            "cv_f1_macro": round(cv_f1, 4),
            "test_accuracy": round(accuracy_score(y_te, pred), 4),
            "test_precision_macro": round(precision_score(y_te, pred, average="macro", zero_division=0), 4),
            "test_recall_macro": round(recall_score(y_te, pred, average="macro", zero_division=0), 4),
            "test_f1_macro": round(f1_score(y_te, pred, average="macro"), 4),
        })
        fitted[name] = pipe
        print(f"{name:20s} acc={rows[-1]['test_accuracy']:.3f}  f1_macro={rows[-1]['test_f1_macro']:.3f}")

    results = pd.DataFrame(rows).sort_values("test_f1_macro", ascending=False)
    results.to_csv(config.REPORT_DIR / "model_comparison.csv", index=False)

    best_name = results.iloc[0]["model"]
    best = fitted[best_name]
    pred = best.predict(X_te)
    report = classification_report(y_te, pred, output_dict=True, zero_division=0)
    pd.DataFrame(report).T.round(3).to_csv(config.REPORT_DIR / "best_model_classification_report.csv")

    # Confusion matrix for best model
    labels = [t for t in config.TIER_ORDER if t in set(y)]
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(y_te, pred, labels=labels, ax=ax, xticks_rotation=30, cmap="Blues")
    ax.set_title(f"Confusion matrix - {best_name}")
    fig.tight_layout(); fig.savefig(config.REPORT_DIR / "confusion_matrix.png", dpi=120); plt.close(fig)

    # Model comparison chart
    fig, ax = plt.subplots(figsize=(7, 4))
    r = results.set_index("model")[["test_accuracy", "test_f1_macro"]]
    r.plot.barh(ax=ax, color=["#2a78b5", "#e8913a"])
    ax.set_xlim(0, 1); ax.set_xlabel("score"); ax.set_title("Model comparison (test set)")
    fig.tight_layout(); fig.savefig(config.REPORT_DIR / "model_comparison.png", dpi=120); plt.close(fig)

    # Feature importance from the Random Forest (interpretable, always available)
    rf = fitted["Random Forest"]
    names = rf.named_steps["prep"].get_feature_names_out()
    imp = pd.Series(rf.named_steps["model"].feature_importances_, index=names).sort_values().tail(12)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    imp.plot.barh(ax=ax, color="#2a78b5"); ax.set_title("Random Forest - top feature importances")
    fig.tight_layout(); fig.savefig(config.REPORT_DIR / "feature_importance.png", dpi=120); plt.close(fig)

    # test accuracy when a field is left empty ("Prefer not to say"), shown in the app
    hidden_acc = {}
    for cols in [[c] for c in config.OPTIONAL_FIELDS] + [config.OPTIONAL_FIELDS]:
        X_hidden = X_te.copy()
        X_hidden[cols] = np.nan
        key = cols[0] if len(cols) == 1 else "all"
        hidden_acc[key] = round(accuracy_score(y_te, best.predict(X_hidden)), 3)
    print("Accuracy with a field hidden:", hidden_acc)

    # Retrain the best model on all data and save it
    best.fit(X, y)
    joblib.dump(best, config.MODEL_PATH)
    meta = {
        "best_model": best_name,
        "numeric_features": config.NUMERIC_FEATURES,
        "categorical_features": config.CATEGORICAL_FEATURES,
        "categories": {c: sorted(df[c].dropna().astype(str).unique().tolist()) for c in config.CATEGORICAL_FEATURES},
        "classes": [str(c) for c in best.classes_],
        "train_rows": int(len(df)),
        "test_accuracy": float(results.iloc[0]["test_accuracy"]),
        "hidden_field_accuracy": hidden_acc,
    }
    config.MODEL_META_PATH.write_text(json.dumps(meta, indent=2))
    print(f"\nBest model: {best_name}  -> saved to {config.MODEL_PATH}")
    return results


if __name__ == "__main__":
    main()
