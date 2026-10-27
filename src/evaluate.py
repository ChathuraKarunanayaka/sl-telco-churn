"""Comparison figures from the trained models. Run:  python src/evaluate.py  (after train.py)"""
import json
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score

import config
from features import RAW_FEATURES
from plot_style import BLUE, FAMILY_COLORS, GREY, INK, MODEL_FAMILY, ORANGE, apply_style


def main():
    apply_style()
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    res = pd.read_csv(config.REPORTS_DIR / "model_comparison.csv")
    best_name = pd.read_json(config.FEATURE_INFO_FILE, typ="series")["model_name"]

    # 1. Metric comparison, weak -> strong
    order = res["model"].tolist()
    colors = [FAMILY_COLORS[MODEL_FAMILY[m]] for m in order]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax, (col, title) in zip(axes, [("test_f1", "F1 for the churn class"),
                                       ("test_pr_auc", "PR-AUC"), ("test_roc_auc", "ROC-AUC")]):
        ax.barh(order, res[col], color=colors, height=0.65)
        for y, v in enumerate(res[col]):
            ax.text(v + 0.01, y, f"{v:.3f}", va="center", fontsize=9, color=INK)
        ax.set_xlim(0, 1.12); ax.set_title(title); ax.grid(axis="y", visible=False)
    axes[0].invert_yaxis()
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in FAMILY_COLORS.values()]
    fig.legend(handles, ["EBM (new algorithm)", "Baseline / weak", "Single tree", "Bagging", "Boosting", "Voting / stacking"],
               loc="lower center", ncols=6, frameon=False)
    fig.suptitle("Test-set comparison: the new algorithm (EBM) against models from the lectures (13,000 customers)",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(config.FIGURES_DIR / "model_comparison.png"); plt.close(fig)

    # 2. Weak vs best: confusion matrices and precision-recall curves
    test = pd.read_csv(config.TEST_LABELS).merge(pd.read_csv(config.FEATURES_FILE), on="customer_id")
    holdout = pd.read_csv(config.INFERENCE_SAMPLE_FILE)
    test = test[~test["customer_id"].isin(holdout["customer_id"])]
    X, y = test[RAW_FEATURES], test[config.TARGET]
    models = {"Logistic Regression (weak)": pickle.load(open(config.MODELS_DIR / "logistic_regression.pkl", "rb")),
              f"{best_name.split(' (')[0]} (selected, new algorithm)": pickle.load(open(config.MODEL_FILE, "rb"))}
    cms = {}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), gridspec_kw={"width_ratios": [1, 1, 1.3]})
    for ax, (name, m) in zip(axes[:2], models.items()):
        cm = confusion_matrix(y, m.predict(X))
        key = "Logistic Regression" if name.startswith("Logistic") else best_name
        cms[key] = {"tn": int(cm[0, 0]), "fp": int(cm[0, 1]), "fn": int(cm[1, 0]), "tp": int(cm[1, 1])}
        share = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}\n{share[i, j]:.0%}", ha="center", va="center",
                        color="white" if share[i, j] > 0.55 else INK)
        ax.set_xticks([0, 1], ["Stays", "Churns"]); ax.set_yticks([0, 1], ["Stays", "Churns"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(name, fontsize=10); ax.grid(False)
    for (name, m), c in zip(models.items(), [GREY, ORANGE]):
        p = m.predict_proba(X)[:, 1]
        prec, rec, _ = precision_recall_curve(y, p)
        axes[2].plot(rec, prec, color=c, label=f"{name.split(' (')[0]} (PR-AUC {average_precision_score(y, p):.3f})")
    axes[2].axhline(y.mean(), color=INK, linestyle="--", linewidth=1, label=f"Random guess ({y.mean():.2f})")
    axes[2].set_xlabel("Recall (churners found)"); axes[2].set_ylabel("Precision (alerts that are right)")
    axes[2].set_title("Precision-recall curve"); axes[2].legend(loc="lower left"); axes[2].set_ylim(0, 1.02)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "weak_vs_best.png"); plt.close(fig)
    (config.REPORTS_DIR / "confusion_matrices.json").write_text(json.dumps(cms, indent=1))
    print(json.dumps(cms))
    print("Saved model_comparison.png, weak_vs_best.png and confusion_matrices.json")


if __name__ == "__main__":
    main()
