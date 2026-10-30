import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

TASKS = {
    "A": {
        "dir": "taskA_binary",
        "validation": "hastika_binary_validation.csv",
        "label_col": "Label",
        "labels": ["Non-Hate", "Hate"],
        "output_labels": ["Non-hate", "Hate"],
    },
    "B": {
        "dir": "taskB_finegrained",
        "validation": "hastika_multiclass_validation.csv",
        "label_col": "Hate Category",
        "labels": ["Gender", "Political", "Religion", "Geo-political", "Violence", "Others"],
        "output_labels": ["Gender", "Political", "Religion", "Geo-political", "Violence", "Others"],
    },
}


def normalize_label(value):
    return str(value).strip().lower().replace("_", "-").replace(" ", "-")


def main():
    parser = argparse.ArgumentParser(description="Verify saved HASTIKA prediction artifacts.")
    parser.add_argument("--data-dir", required=True, help="Directory containing the official labelled validation CSVs.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    results_dir = root / "hastika_all_predictions"
    data_dir = Path(args.data_dir)
    summary = pd.read_csv(results_dir / "results_summary.csv", encoding="utf-8-sig")

    failures = []
    rows = []

    for _, record in summary.iterrows():
        model = record["model"]
        task = record["task"]
        cfg = TASKS[task]
        run_dir = results_dir / model.replace("/", "__") / cfg["dir"]

        validation = pd.read_csv(data_dir / cfg["validation"], encoding="utf-8-sig")
        label_to_index = {normalize_label(label): index for index, label in enumerate(cfg["labels"])}
        y_true = validation[cfg["label_col"]].map(lambda value: label_to_index[normalize_label(value)]).to_numpy()

        if pd.isna(y_true).any():
            failures.append(f"{model} task {task}: unknown validation label")

        val_proba = np.load(run_dir / "val_proba.npy")
        test_proba = np.load(run_dir / "test_proba.npy")
        predictions = pd.read_csv(run_dir / "predictions.csv", encoding="utf-8-sig")

        expected_shape = (len(validation), len(cfg["labels"]))
        if val_proba.shape != expected_shape:
            failures.append(f"{model} task {task}: val_proba shape {val_proba.shape}, expected {expected_shape}")
        if test_proba.shape[1] != len(cfg["labels"]):
            failures.append(f"{model} task {task}: test_proba has {test_proba.shape[1]} columns")
        if len(predictions) != len(test_proba):
            failures.append(f"{model} task {task}: predictions rows {len(predictions)}, test rows {len(test_proba)}")
        if list(predictions.columns) != ["id", "label"]:
            failures.append(f"{model} task {task}: prediction columns {list(predictions.columns)}")
        if not predictions["id"].is_unique:
            failures.append(f"{model} task {task}: duplicate prediction IDs")
        if not set(predictions["label"]).issubset(set(cfg["output_labels"])):
            failures.append(f"{model} task {task}: invalid predicted label")
        if not np.allclose(val_proba.sum(axis=1), 1.0, atol=1e-5):
            failures.append(f"{model} task {task}: validation probabilities do not sum to 1")
        if not np.allclose(test_proba.sum(axis=1), 1.0, atol=1e-5):
            failures.append(f"{model} task {task}: test probabilities do not sum to 1")

        expected_test_labels = np.array([cfg["output_labels"][index] for index in test_proba.argmax(axis=1)])
        if len(predictions) == len(expected_test_labels):
            if not np.array_equal(expected_test_labels, predictions["label"].to_numpy()):
                failures.append(f"{model} task {task}: predictions.csv does not match argmax(test_proba)")

        val_pred = val_proba.argmax(axis=1)
        macro_f1 = f1_score(y_true, val_pred, average="macro", zero_division=0)
        weighted_f1 = f1_score(y_true, val_pred, average="weighted", zero_division=0)
        accuracy = accuracy_score(y_true, val_pred)

        expected_metrics = {
            "macro-F1": float(record["val_macro_f1"]),
            "weighted-F1": float(record["val_weighted_f1"]),
            "accuracy": float(record["val_acc"]),
        }
        actual_metrics = {
            "macro-F1": macro_f1,
            "weighted-F1": weighted_f1,
            "accuracy": accuracy,
        }

        for metric, expected in expected_metrics.items():
            if abs(actual_metrics[metric] - expected) > 5e-4:
                failures.append(
                    f"{model} task {task}: {metric} {actual_metrics[metric]:.6f}, reported {expected:.6f}"
                )

        best_epoch = int(record["best_epoch"])
        if not 1 <= best_epoch <= 6:
            failures.append(f"{model} task {task}: unexpected best epoch {best_epoch}")

        rows.append((model, task, macro_f1, weighted_f1, accuracy, best_epoch))

    print(f"{'Model':45s} {'Task':4s} {'Macro-F1':>9s} {'Weighted':>9s} {'Accuracy':>9s} {'Epoch':>5s}")
    print("-" * 90)
    for model, task, macro_f1, weighted_f1, accuracy, best_epoch in rows:
        print(f"{model:45s} {task:4s} {macro_f1:9.4f} {weighted_f1:9.4f} {accuracy:9.4f} {best_epoch:5d}")

    if failures:
        print("\nFAILED CHECKS")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)

    print(f"\nAll checks passed for {len(rows)} runs.")


if __name__ == "__main__":
    main()
