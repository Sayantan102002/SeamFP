"""
STEP 4: Leakage-aware ablation study for FVC2002 seam-carving detection.

Requirements for each .npz feature file:
    X          : feature matrix, shape (N, D)
    y          : labels, 0 = pristine, 1 = carved
    ratios     : 0 for pristine rows; 5/10/20 for carved rows
    source_id  : original pristine fingerprint ID for each row.
                 The pristine image and every carved version derived from it
                 MUST share the same source_id.

The script uses StratifiedGroupKFold so each source fingerprint is entirely
inside one fold. Metrics in the CSV are means across outer folds; confusion
matrices use pooled out-of-fold predictions.
"""

import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
plt.rcParams.update({
    "font.size": 16,          # base size for text, tick labels, annotations
    "axes.titlesize": 14,
    "axes.labelsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 16,
})
# ---------------------------------------------------------------------------
# Paths and settings
# ---------------------------------------------------------------------------
COMBINED_FILE = "data_FVC2002/features_all_v2.npz"
LBP_FILE = "data_FVC2002/features_only_LBP.npz"
GRAD_FILE = "data_FVC2002/features_only_HOG.npz"

ABLATION_CSV = "data_FVC2002/ablation_results_grouped_cv.csv"
FIGURES_DIR = "data_FVC2002/figures/ablation_grouped_cv"
MODEL_DIR = "artifacts"  # Saved LBP + Gradient deployment models

RATIOS = [5, 10, 20]
OUTER_FOLDS = 5
RANDOM_STATE = 42

# These are copied from the BEST_PARAMS dictionary in the supplied script.
# IMPORTANT: verify/reselect these using inner CV on each outer training fold
# if they were selected by inspecting the same folds used for final reporting.
BEST_PARAMS = {
    5:  {"C": 100, "gamma": 0.001},
    10: {"C": 100, "gamma": 0.001},
    20: {"C": 10,  "gamma": "scale"},
}

CONFIGS = [
    {"name": "LBP only",       "file": LBP_FILE,      "dim": 256},
    {"name": "Gradient only",  "file": GRAD_FILE,    "dim": 64},
    {"name": "LBP + Gradient", "file": COMBINED_FILE, "dim": 320},
]

os.makedirs(FIGURES_DIR, exist_ok=True)


def load_feature_file(path, config_name):
    """Load features and required metadata; fail loudly if source IDs are absent."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Feature file not found: {path}")

    with np.load(path, allow_pickle=False) as data:
        required = {"X", "y", "ratios", "source_id"}
        missing = required.difference(data.files)
        if missing:
            raise KeyError(
                f"{path} ({config_name}) is missing {sorted(missing)}. "
                "Regenerate this NPZ with source_id for every row. source_id "
                "must be the original fingerprint ID shared by pristine and "
                "all seam-carved derivatives; do not use a unique row index."
            )

        X = np.asarray(data["X"])
        y = np.asarray(data["y"]).reshape(-1)
        ratios = np.asarray(data["ratios"]).reshape(-1)
        source_id = np.asarray(data["source_id"]).astype(str).reshape(-1)

    n = len(y)
    if X.ndim != 2:
        raise ValueError(f"{path}: X must be 2-D; got shape {X.shape}.")
    if not (len(X) == len(ratios) == len(source_id) == n):
        raise ValueError(
            f"{path}: inconsistent lengths: X={len(X)}, y={len(y)}, "
            f"ratios={len(ratios)}, source_id={len(source_id)}."
        )
    if len(np.unique(y)) < 2 or not set(np.unique(y)).issubset({0, 1}):
        raise ValueError(f"{path}: y must contain binary labels 0 and 1.")
    if np.any(source_id == ""):
        raise ValueError(f"{path}: source_id contains empty values.")

    return {"X": X, "y": y.astype(int), "ratios": ratios,
            "source_id": source_id, "name": config_name, "path": path}


def validate_feature_file_alignment(datasets):
    """
    Check that feature variants describe the same rows in the same order.
    This is essential for a valid ablation comparison.
    """
    base = datasets[0]
    for other in datasets[1:]:
        for key in ("y", "ratios", "source_id"):
            if not np.array_equal(base[key], other[key]):
                raise ValueError(
                    f"Feature files are not row-aligned: {key} differs between "
                    f"{base['path']} and {other['path']}. Rebuild/reorder the "
                    "NPZ files so each row refers to the same sample."
                )
        if len(base["X"]) != len(other["X"]):
            raise ValueError(
                f"Different row counts: {base['path']}={len(base['X'])}, "
                f"{other['path']}={len(other['X'])}."
            )


def validate_ratio_pairs(y, ratios, source_id, ratio, config_name):
    """Require one pristine and one selected-ratio carved row per source ID."""
    neg_idx = np.flatnonzero(y == 0)
    pos_idx = np.flatnonzero((y == 1) & (ratios == ratio))

    neg_ids = source_id[neg_idx]
    pos_ids = source_id[pos_idx]

    if len(neg_idx) != len(pos_idx):
        raise ValueError(
            f"{config_name}, {ratio}%: expected equal pristine/carved counts; "
            f"got {len(neg_idx)} pristine and {len(pos_idx)} carved."
        )
    if len(np.unique(neg_ids)) != len(neg_ids):
        raise ValueError(
            f"{config_name}: duplicate pristine rows/source IDs detected."
        )
    if len(np.unique(pos_ids)) != len(pos_ids):
        raise ValueError(
            f"{config_name}, {ratio}%: duplicate carved source IDs detected."
        )
    if set(neg_ids) != set(pos_ids):
        missing_carved = sorted(set(neg_ids) - set(pos_ids))[:5]
        missing_pristine = sorted(set(pos_ids) - set(neg_ids))[:5]
        raise ValueError(
            f"{config_name}, {ratio}%: pristine/carved source IDs do not match. "
            f"Examples without carved counterpart: {missing_carved}; "
            f"examples without pristine counterpart: {missing_pristine}."
        )
    return neg_idx, pos_idx


def evaluate_grouped(X_neg, ids_neg, X_pos, ids_pos, ratio, config_name, C, gamma):
    """Run group-aware OOF evaluation and return fold metrics plus OOF predictions."""
    X = np.vstack([X_neg, X_pos])
    y = np.concatenate([
        np.zeros(len(X_neg), dtype=int),
        np.ones(len(X_pos), dtype=int),
    ])
    groups = np.concatenate([ids_neg, ids_pos])

    cv = StratifiedGroupKFold(
        n_splits=OUTER_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    metrics = {"acc": [], "prec": [], "rec": [], "f1": []}
    all_y_true = np.full(len(y), -1, dtype=int)
    all_y_pred = np.full(len(y), -1, dtype=int)

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups=groups), 1):
        train_groups = set(groups[train_idx])
        test_groups = set(groups[test_idx])
        overlap = train_groups.intersection(test_groups)
        if overlap:
            raise RuntimeError(
                f"Leakage detected in {config_name}, {ratio}%, fold {fold}: "
                f"{len(overlap)} source IDs occur in both train and test."
            )

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", C=C, gamma=gamma, random_state=RANDOM_STATE)),
        ])
        pipe.fit(X[train_idx], y[train_idx])
        y_pred = pipe.predict(X[test_idx])

        metrics["acc"].append(accuracy_score(y[test_idx], y_pred))
        metrics["prec"].append(
            precision_score(y[test_idx], y_pred, zero_division=0)
        )
        metrics["rec"].append(
            recall_score(y[test_idx], y_pred, zero_division=0)
        )
        metrics["f1"].append(f1_score(y[test_idx], y_pred, zero_division=0))

        all_y_true[test_idx] = y[test_idx]
        all_y_pred[test_idx] = y_pred

        print(
            f"    Fold {fold}: train groups={len(train_groups)}, "
            f"test groups={len(test_groups)}, "
            f"accuracy={metrics['acc'][-1] * 100:.2f}%"
        )

    if np.any(all_y_true < 0) or np.any(all_y_pred < 0):
        raise RuntimeError("Some samples did not receive an out-of-fold prediction.")

    result = {key: float(np.mean(values)) for key, values in metrics.items()}
    result.update({
        "ratio": ratio,
        "config": config_name,
        "C": C,
        "gamma": str(gamma),
        "n_samples": len(y),
        "n_source_ids": len(np.unique(groups)),
        "oof_acc": float(accuracy_score(all_y_true, all_y_pred)),
        "oof_prec": float(precision_score(all_y_true, all_y_pred, zero_division=0)),
        "oof_rec": float(recall_score(all_y_true, all_y_pred, zero_division=0)),
        "oof_f1": float(f1_score(all_y_true, all_y_pred, zero_division=0)),
    })

    cm = confusion_matrix(all_y_true, all_y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Pristine", f"Carved {ratio}%"],
    )
    safe_tag = config_name.lower().replace(" ", "_").replace("+", "plus")
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=False)
    ax.set_title(f"Confusion Matrix — {ratio}% Carving Ratio", fontsize = 14)
    fig.tight_layout()
    out_path = os.path.join(
        FIGURES_DIR, f"cm_{safe_tag}_ratio{ratio}.png"
    )
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

    return result



def train_and_save_deployment_model(X_neg, X_pos, ratio, C, gamma):
    """Fit the final LBP + Gradient scaler/SVM on all source groups for a ratio."""
    X = np.vstack([X_neg, X_pos])
    y = np.concatenate([
        np.zeros(len(X_neg), dtype=int),
        np.ones(len(X_pos), dtype=int),
    ])

    # Save preprocessing and classifier together so inference uses the same scaler.
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(
            kernel="rbf",
            C=C,
            gamma=gamma,
            random_state=RANDOM_STATE,
            probability=True,
        )),
    ])
    pipeline.fit(X, y)

    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, f"svm_{ratio}.joblib")
    joblib.dump(pipeline, model_path)
    print(
        f"  Saved deployment model: {model_path} "
        f"(features={X.shape[1]}, samples={len(y)}, C={C}, gamma={gamma})"
    )
    return model_path

def plot_ablation_bar(results):
    config_names = [cfg["name"] for cfg in CONFIGS]
    x = np.arange(len(RATIOS))
    width = 0.25
    colors = ["#4C72B0", "#DD8452", "#55A868"]

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, config_name in enumerate(config_names):
        accs = [
            next(r for r in results
                 if r["ratio"] == ratio and r["config"] == config_name)["oof_acc"] * 100
            for ratio in RATIOS
        ]
        bars = ax.bar(
            x + (i - 1) * width, accs, width, label=config_name,
            color=colors[i], edgecolor="white", linewidth=0.8
        )
        for bar, value in zip(bars, accs):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.1,
                f"{value:.2f}",
                ha="center", va="bottom", fontsize=12
            )

    ax.set_xlabel("Carving Ratio (%)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Ablation Study: Detection Accuracy by Feature Configuration", fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{ratio}%" for ratio in RATIOS])
    ax.set_ylim(60, 110)
    ax.legend(fontsize=14, loc='lower right')
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "ablation_bar_grouped_cv.png")
    fig.savefig(path, dpi=250)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_ablation_heatmap(results):
    config_names = [cfg["name"] for cfg in CONFIGS]
    matrix = np.zeros((len(config_names), len(RATIOS)))

    for i, config_name in enumerate(config_names):
        for j, ratio in enumerate(RATIOS):
            row = next(
                r for r in results
                if r["ratio"] == ratio and r["config"] == config_name
            )
            matrix[i, j] = row["oof_acc"] * 100

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(
        matrix, annot=True, fmt=".2f",
        xticklabels=[f"{ratio}%" for ratio in RATIOS],
        yticklabels=config_names, cmap="YlGnBu",
        vmin=0, vmax=100, linewidths=0.5, ax=ax,
    )
    ax.set_xlabel("Carving Ratio")
    ax.set_ylabel("Feature Configuration")
    ax.set_title("Grouped out-of-fold ablation accuracy (%)")
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "ablation_heatmap_grouped_cv.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"Saved: {path}")


def main():
    datasets = [
        load_feature_file(cfg["file"], cfg["name"])
        for cfg in CONFIGS
    ]
    validate_feature_file_alignment(datasets)

    results = []
    for cfg, dataset in zip(CONFIGS, datasets):
        print(f"\n{'=' * 70}\nConfig: {cfg['name']} ({cfg['dim']}-D)\n{'=' * 70}")

        if dataset["X"].shape[1] != cfg["dim"]:
            raise ValueError(
                f"{cfg['name']}: expected {cfg['dim']} features, "
                f"found {dataset['X'].shape[1]}."
            )

        y = dataset["y"]
        ratios = dataset["ratios"]
        source_id = dataset["source_id"]
        X = dataset["X"]

        for ratio in RATIOS:
            neg_idx, pos_idx = validate_ratio_pairs(
                y, ratios, source_id, ratio, cfg["name"]
            )
            C = BEST_PARAMS[ratio]["C"]
            gamma = BEST_PARAMS[ratio]["gamma"]

            print(f"  Ratio={ratio}% | C={C}, gamma={gamma}")
            result = evaluate_grouped(
                X_neg=X[neg_idx],
                ids_neg=source_id[neg_idx],
                X_pos=X[pos_idx],
                ids_pos=source_id[pos_idx],
                ratio=ratio,
                config_name=cfg["name"],
                C=C,
                gamma=gamma,
            )
            results.append(result)

            # Export only the combined 320-D LBP + Gradient model for deployment.
            if cfg["name"] == "LBP + Gradient":
                train_and_save_deployment_model(
                    X_neg=X[neg_idx],
                    X_pos=X[pos_idx],
                    ratio=ratio,
                    C=C,
                    gamma=gamma,
                )

            print(
                f"  Fold-mean Acc={result['acc'] * 100:.2f}% | "
                f"OOF Acc={result['oof_acc'] * 100:.2f}% | "
                f"OOF F1={result['oof_f1'] * 100:.2f}%"
            )

    fieldnames = [
        "config", "dim", "ratio", "acc", "prec", "rec", "f1",
        "oof_acc", "oof_prec", "oof_rec", "oof_f1",
        "C", "gamma", "n_samples", "n_source_ids",
    ]
    with open(ABLATION_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            dim = next(cfg["dim"] for cfg in CONFIGS if cfg["name"] == result["config"])
            row = dict(result)
            row["dim"] = dim
            for key in ("acc", "prec", "rec", "f1",
                        "oof_acc", "oof_prec", "oof_rec", "oof_f1"):
                row[key] = f"{row[key] * 100:.4f}"
            writer.writerow({key: row[key] for key in fieldnames})

    print(f"\nSaved results: {ABLATION_CSV}")
    plot_ablation_bar(results)
    plot_ablation_heatmap(results)

    print("\nGrouped-CV summary (fold-mean metrics; pooled OOF accuracy also shown)")
    print("-" * 112)
    print(f"{'Config':<20} {'Dim':>5} {'Ratio':>7} {'Acc':>9} "
          f"{'Prec':>9} {'Recall':>9} {'F1':>9} {'OOF Acc':>10}")
    print("-" * 112)
    for cfg in CONFIGS:
        for ratio in RATIOS:
            r = next(x for x in results
                     if x["config"] == cfg["name"] and x["ratio"] == ratio)
            print(
                f"{cfg['name']:<20} {cfg['dim']:>5} {ratio:>6}% "
                f"{r['acc'] * 100:>8.2f}% {r['prec'] * 100:>8.2f}% "
                f"{r['rec'] * 100:>8.2f}% {r['f1'] * 100:>8.2f}% "
                f"{r['oof_acc'] * 100:>9.2f}%"
            )
        print("-" * 112)


if __name__ == "__main__":
    main()
