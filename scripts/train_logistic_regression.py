"""Train and evaluate a logistic regression baseline on ASSISTments."""

import argparse
import sys
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd
import yaml
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import OneHotEncoder

# Ensure local imports work when executing as a script
sys.path.append(str(Path(__file__).parent.parent))

from src.data import ASSISTmentsProcessor, train_val_test_split  # noqa: E402


def load_config(config_path: str) -> Dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def prepare_features(df: pd.DataFrame, encoder: OneHotEncoder, fit: bool = False):
    features = df[["student_idx", "skill_idx"]]
    if fit:
        return encoder.fit_transform(features)
    return encoder.transform(features)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    y_pred = (y_prob >= 0.5).astype(int)
    rmse = float(np.sqrt(np.mean((y_true - y_prob) ** 2)))
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "AUC-ROC": float(roc_auc_score(y_true, y_prob)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "RMSE": rmse,
    }


def main():
    parser = argparse.ArgumentParser(description="Train logistic regression baseline")
    parser.add_argument("--config", required=True, help="Path to baseline YAML config")
    parser.add_argument(
        "--output_metrics", default="results/logistic_regression_metrics.csv", help="Where to store metrics"
    )
    parser.add_argument(
        "--output_predictions",
        default="results/logistic_predictions.csv",
        help="Where to store test predictions",
    )
    parser.add_argument(
        "--model_path",
        default="models/baseline/logistic_regression.joblib",
        help="Where to persist trained classifier",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    processor = ASSISTmentsProcessor(config["data"]["data_path"])

    df = processor.preprocess()
    train_df, val_df, test_df = train_val_test_split(
        df,
        config["data"]["train_split"],
        config["data"]["val_split"],
        config["data"]["test_split"],
        config["data"]["random_seed"],
    )

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)

    X_train = prepare_features(train_df, encoder, fit=True)
    y_train = train_df["correct"].values

    log_cfg = config.get("logistic_regression", {})
    clf = LogisticRegression(
        penalty=log_cfg.get("penalty", "l2"),
        C=log_cfg.get("C", 1.0),
        max_iter=log_cfg.get("max_iter", 200),
        solver=log_cfg.get("solver", "saga"),
        class_weight=log_cfg.get("class_weight", "balanced"),
        n_jobs=log_cfg.get("n_jobs", -1),
        verbose=1,
    )

    clf.fit(X_train, np.asarray(y_train).ravel())
    joblib.dump({"model": clf, "encoder": encoder}, args.model_path)

    metrics = {}
    for split_name, split_df in {"train": train_df, "val": val_df, "test": test_df}.items():
        X_split = prepare_features(split_df, encoder)
        probs = clf.predict_proba(X_split)[:, 1]
        split_metrics = compute_metrics(np.asarray(split_df["correct"]).ravel(), probs)
        metrics[split_name] = split_metrics
        print(f"\n=== {split_name.upper()} METRICS ===")
        for k, v in split_metrics.items():
            print(f"{k}: {v:.4f}")

        if split_name == "test":
            preds_df = pd.DataFrame({"prediction": probs, "target": split_df["correct"].values})
            Path(args.output_predictions).parent.mkdir(parents=True, exist_ok=True)
            preds_df.to_csv(args.output_predictions, index=False)

    metrics_df = pd.DataFrame(
        [
            {"Split": split.upper(), **values}
            for split, values in metrics.items()
        ]
    )
    Path(args.output_metrics).parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(args.output_metrics, index=False)
    print(f"Saved metrics to {args.output_metrics}")
    print(f"Saved predictions to {args.output_predictions}")
    print(f"Saved model to {args.model_path}")


if __name__ == "__main__":
    main()
