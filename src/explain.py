"""Explain the selected model with SHAP (global + one customer) and LIME (one customer).

Run:  python src/explain.py      (after train.py)

SHAP  = how much each feature pushed a prediction up or down, based on game theory
        (Shapley values). Gives a global ranking and per-customer explanations.
LIME  = fits a small, simple (linear) model around ONE customer, by testing slightly
        changed copies of that customer, and reports which features mattered locally.
"""
import json
import pickle
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer

import config
from features import RAW_FEATURES, USAGE_TYPES, readable
from plot_style import BLUE, INK, ORANGE, apply_style

warnings.filterwarnings("ignore")


def split_pipeline(pipe):
    """The saved pipeline = feature steps + model. SHAP needs the engineered features."""
    return pipe[:-1], pipe[-1]


def tree_model_for_shap(model):
    """For a voting / stacking ensemble, explain its strongest tree member instead."""
    if hasattr(model, "named_estimators_"):
        for key in ("lgbm", "xgb", "cat"):
            if key in model.named_estimators_:
                return model.named_estimators_[key], key
    return model, None


def main():
    apply_style()
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    pipe = pickle.load(open(config.MODEL_FILE, "rb"))
    prep, model = split_pipeline(pipe)
    explain_model, member = tree_model_for_shap(model)
    test = pd.read_csv(config.TEST_LABELS).merge(pd.read_csv(config.FEATURES_FILE), on="customer_id")
    holdout = pd.read_csv(config.INFERENCE_SAMPLE_FILE)
    test = test[~test["customer_id"].isin(holdout["customer_id"])]
    sample = test.sample(2000, random_state=config.RANDOM_STATE)           # SHAP on 2,000 test customers
    Xs = prep.transform(sample[RAW_FEATURES])
    Xh = prep.transform(holdout[RAW_FEATURES])
    names = [readable(c) for c in Xs.columns]

    explainer = shap.TreeExplainer(explain_model.get_booster() if hasattr(explain_model, 'get_booster') else explain_model)
    sv = explainer(Xs)
    if sv.values.ndim == 3:                                                 # some models give one set per class
        sv = sv[:, :, 1]
    sv.feature_names = names
    note = f" (explaining the {member.upper()} member of the ensemble)" if member else ""

    # 1. Global importance: mean |SHAP| per feature
    plt.figure()
    shap.plots.bar(sv, max_display=15, show=False)
    plt.title("Which features matter most overall (mean |SHAP value|)" + note, loc="left", fontsize=11)
    plt.savefig(config.FIGURES_DIR / "shap_importance.png", bbox_inches="tight"); plt.close()

    # 2. Beeswarm: direction of the effect
    plt.figure()
    shap.plots.beeswarm(sv, max_display=15, show=False)
    plt.title("How each feature pushes towards churn (right) or staying (left)", loc="left", fontsize=11)
    plt.savefig(config.FIGURES_DIR / "shap_beeswarm.png", bbox_inches="tight"); plt.close()

    # 3. Dependence plot for the top feature
    top = int(np.abs(sv.values).mean(0).argmax())
    plt.figure()
    shap.plots.scatter(sv[:, top], color=sv, show=False)
    plt.title(f"Effect of '{names[top]}' on churn risk", loc="left", fontsize=11)
    plt.savefig(config.FIGURES_DIR / "shap_dependence_top.png", bbox_inches="tight"); plt.close()

    # 4. One customer: waterfall
    sh = explainer(Xh)
    if sh.values.ndim == 3:
        sh = sh[:, :, 1]
    sh.feature_names = names
    plt.figure()
    shap.plots.waterfall(sh[0], max_display=12, show=False)
    plt.title("Why the model flags the held-out customer (SHAP, log-odds)", loc="left", fontsize=11)
    plt.savefig(config.FIGURES_DIR / "shap_waterfall_holdout.png", bbox_inches="tight"); plt.close()

    # 3b. Many features describe the same usage type (levels, change, ratio), so the signal is
    #     spread thinly. Summing |SHAP| per usage type gives a clearer business picture.
    def usage_of(col):
        for u in USAGE_TYPES:
            if col.startswith(u + "_"):
                return USAGE_TYPES[u].split(" (")[0]
        return {"apps": "All apps (group totals)", "packs": "All packs (group totals)",
                "voice": "All calls (group totals)"}[col.split("_")[0]]
    per_customer = pd.DataFrame(np.abs(sv.values), columns=list(Xs.columns))
    grouped = per_customer.T.groupby([usage_of(c) for c in Xs.columns]).sum().T.mean().sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(grouped.index, grouped.values, color=ORANGE, height=0.6)
    for y_pos, v in enumerate(grouped.values):
        ax.text(v, y_pos, f" {v:.2f}", va="center", fontsize=8, color=INK)
    ax.set_title("Importance by usage type (sum of mean |SHAP| of its features)", loc="left")
    ax.set_xlabel("Total contribution to the churn prediction"); ax.grid(axis="y", visible=False)
    fig.savefig(config.FIGURES_DIR / "shap_by_usage_type.png", bbox_inches="tight"); plt.close(fig)
    grouped.sort_values(ascending=False).round(3).to_csv(config.REPORTS_DIR / "shap_by_usage_type.csv",
                                                         header=["sum_mean_abs_shap"])

    global_rank = (pd.Series(np.abs(sv.values).mean(0), index=names)
                   .sort_values(ascending=False).round(4))
    global_rank.to_csv(config.REPORTS_DIR / "shap_global_importance.csv", header=["mean_abs_shap"])

    # 5. LIME for the same customer, on the full pipeline (raw inputs -> probability)
    train = pd.read_csv(config.TRAIN_LABELS).merge(pd.read_csv(config.FEATURES_FILE), on="customer_id")
    background = train[RAW_FEATURES].sample(5000, random_state=config.RANDOM_STATE)
    lime_exp = LimeTabularExplainer(background.values, feature_names=[readable(c) for c in RAW_FEATURES],
                                    class_names=["Stays", "Churns"], mode="classification",
                                    discretize_continuous=True, random_state=config.RANDOM_STATE)
    predict = lambda arr: pipe.predict_proba(pd.DataFrame(arr, columns=RAW_FEATURES))
    exp = lime_exp.explain_instance(holdout[RAW_FEATURES].values[0], predict, num_features=10,
                                    num_samples=3000)
    weights = exp.as_list(label=1)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    labels = [w[0] for w in weights][::-1]
    vals = [w[1] for w in weights][::-1]
    ax.barh(labels, vals, color=[ORANGE if v > 0 else BLUE for v in vals], height=0.6)
    ax.axvline(0, color=INK, linewidth=0.8)
    ax.set_xlabel("Effect on churn probability (positive = towards churn)")
    ax.set_title("LIME: local explanation for the held-out customer", loc="left")
    ax.grid(axis="y", visible=False)
    fig.savefig(config.FIGURES_DIR / "lime_holdout.png", bbox_inches="tight"); plt.close(fig)

    prob = float(pipe.predict_proba(holdout[RAW_FEATURES])[0, 1])
    summary = {"model": json.loads(config.FEATURE_INFO_FILE.read_text())["model_name"],
               "explained_member": member, "holdout_churn_probability": round(prob, 4),
               "holdout_true_label": int(holdout[config.TARGET].iloc[0]),
               "shap_top10": global_rank.head(10).to_dict(),
               "shap_by_usage_type": grouped.sort_values(ascending=False).round(3).to_dict(),
               "lime_top10": [[w[0], round(w[1], 4)] for w in weights]}
    (config.REPORTS_DIR / "explainability_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
