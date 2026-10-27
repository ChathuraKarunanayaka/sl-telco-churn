"""Explain the selected model (EBM) in four ways and check them against domain knowledge.

Run:  python src/explain.py      (after train.py)

1. EBM's own (glassbox) explanation
   - Global term importance: mean |contribution| of each feature / pair.
   - Shape functions: the exact curve the model uses for a feature. For an additive
     model this curve IS the feature's partial dependence, read straight from the model.
2. SHAP values. EBM is additive, so SHAP values come straight from its terms:
   a main term's SHAP value = its contribution minus its average contribution;
   a pairwise term is split half and half between its two features.
3. Partial Dependence (PDP + ICE) on the RAW inputs, through the whole pipeline
   (scikit-learn, model-agnostic).
4. LIME for one customer (a small local linear model around that customer).
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
from sklearn.inspection import PartialDependenceDisplay

import config
from features import RAW_FEATURES, USAGE_TYPES, readable
from plot_style import BLUE, INK, ORANGE, apply_style

warnings.filterwarnings("ignore")
FIG = config.FIGURES_DIR


def term_label(name):
    """'a & b' for pairwise terms, readable names for the rest."""
    return " x ".join(readable(p.strip()) for p in name.split(" & "))


def usage_of(col):
    for u in USAGE_TYPES:
        if col.startswith(u + "_"):
            return USAGE_TYPES[u].split(" (")[0]
    return {"apps": "All apps (group totals)", "packs": "All packs (group totals)",
            "voice": "All calls (group totals)"}[col.split("_")[0]]


def ebm_shap(ebm, Xf, background):
    """SHAP values for an additive model with pairwise terms (log-odds units)."""
    names = list(Xf.columns)
    contrib = ebm.eval_terms(Xf)                    # rows x terms, log-odds
    base = ebm.eval_terms(background).mean(0)
    centred = contrib - base
    values = np.zeros((len(Xf), len(names)))
    for t, feats in enumerate(ebm.term_features_):
        for f in feats:
            values[:, f] += centred[:, t] / len(feats)
    base_value = float(ebm.intercept_[0] if np.ndim(ebm.intercept_) else ebm.intercept_) + float(base.sum())
    return shap.Explanation(values=values, base_values=np.full(len(Xf), base_value),
                            data=Xf.values, feature_names=[readable(c) for c in names])


def save(fig_or_none, name):
    plt.savefig(FIG / name, bbox_inches="tight", dpi=130)
    plt.close("all")


def main():
    apply_style()
    FIG.mkdir(parents=True, exist_ok=True)
    pipe = pickle.load(open(config.MODEL_FILE, "rb"))
    prep, ebm = pipe[:-1], pipe[-1]
    feats = pd.read_csv(config.FEATURES_FILE)
    test = pd.read_csv(config.TEST_LABELS).merge(feats, on="customer_id")
    train = pd.read_csv(config.TRAIN_LABELS).merge(feats, on="customer_id")
    holdout = pd.read_csv(config.INFERENCE_SAMPLE_FILE)
    test = test[~test["customer_id"].isin(holdout["customer_id"])]
    sample = test.sample(2000, random_state=config.RANDOM_STATE)
    Xs, Xh = prep.transform(sample[RAW_FEATURES]), prep.transform(holdout[RAW_FEATURES])
    Xb = prep.transform(train[RAW_FEATURES].sample(2000, random_state=config.RANDOM_STATE))

    # 1a. Global term importance (EBM glassbox)
    imp = pd.Series(ebm.term_importances(), index=[term_label(n) for n in ebm.term_names_])
    imp = imp.sort_values(ascending=False)
    imp.round(4).to_csv(config.REPORTS_DIR / "ebm_term_importance.csv", header=["mean_abs_contribution"])
    top = imp.head(15)[::-1]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(top.index, top.values, color=[INK if " x " in n else ORANGE for n in top.index], height=0.6)
    ax.set_title("EBM global importance (mean |contribution|, log-odds)\norange = single feature, dark = pair",
                 loc="left", fontsize=11)
    ax.grid(axis="y", visible=False)
    save(fig, "ebm_importance.png")

    # 1b. Importance by usage type (main terms summed; pairs split half and half)
    by_type = {}
    for name, value in zip(ebm.term_names_, ebm.term_importances()):
        parts = [p.strip() for p in name.split(" & ")]
        for p in parts:
            by_type[usage_of(p)] = by_type.get(usage_of(p), 0) + value / len(parts)
    by_type = pd.Series(by_type).sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(by_type.index, by_type.values, color=ORANGE, height=0.6)
    for y_pos, v in enumerate(by_type.values):
        ax.text(v, y_pos, f" {v:.2f}", va="center", fontsize=8, color=INK)
    ax.set_title("EBM importance by usage type (sum of its terms)", loc="left")
    ax.set_xlabel("Total contribution to the churn score"); ax.grid(axis="y", visible=False)
    save(fig, "ebm_by_usage_type.png")
    by_type.sort_values(ascending=False).round(3).to_csv(config.REPORTS_DIR / "ebm_by_usage_type.csv",
                                                         header=["importance"])

    # 1c. Shape functions of the four most important single features
    glob = ebm.explain_global()
    mains = [i for i, f in enumerate(ebm.term_features_) if len(f) == 1]
    mains = sorted(mains, key=lambda i: -ebm.term_importances()[i])[:4]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    shapes = {}
    for ax, i in zip(axes.ravel(), mains):
        d = glob.data(i)
        edges, scores = np.asarray(d["names"], dtype=float), np.asarray(d["scores"], dtype=float)
        x = np.repeat(edges, 2)[1:-1]; yv = np.repeat(scores, 2)
        ax.plot(x, yv, color=ORANGE, linewidth=2)
        if d.get("lower_bounds") is not None:
            ax.fill_between(x, np.repeat(d["lower_bounds"], 2), np.repeat(d["upper_bounds"], 2),
                            color=ORANGE, alpha=0.18, linewidth=0)
        ax.axhline(0, color=INK, linewidth=0.8)
        hi = np.percentile(Xb.iloc[:, ebm.term_features_[i][0]], 99)
        ax.set_xlim(edges[0], max(hi, edges[1]))
        ax.set_title(readable(ebm.term_names_[i]), loc="left", fontsize=10)
        ax.set_ylabel("Effect on churn score (log-odds)", fontsize=8)
        lo_val, hi_val = float(scores[: len(scores) // 4].mean()), float(scores[-len(scores) // 4:].mean())
        shapes[readable(ebm.term_names_[i])] = {"low_values_effect": round(lo_val, 3),
                                                "high_values_effect": round(hi_val, 3)}
    fig.suptitle("EBM shape functions: how each value moves the churn score (above 0 = towards churn)",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    save(fig, "ebm_shape_functions.png")

    # 2. SHAP (from the additive terms)
    sv = ebm_shap(ebm, Xs, Xb)
    plt.figure(); shap.plots.beeswarm(sv, max_display=15, show=False)
    plt.title("SHAP: how each feature pushes towards churn (right) or staying (left)", loc="left", fontsize=11)
    save(None, "shap_beeswarm.png")
    plt.figure(); shap.plots.bar(sv, max_display=15, show=False)
    plt.title("SHAP global importance (mean |SHAP value|)", loc="left", fontsize=11)
    save(None, "shap_importance.png")
    shap_rank = pd.Series(np.abs(sv.values).mean(0), index=sv.feature_names).sort_values(ascending=False)
    shap_rank.round(4).to_csv(config.REPORTS_DIR / "shap_global_importance.csv", header=["mean_abs_shap"])

    # 1d + 2b. One customer: EBM local contributions (= SHAP-style waterfall)
    sh = ebm_shap(ebm, Xh, Xb)
    plt.figure(); shap.plots.waterfall(sh[0], max_display=12, show=False)
    plt.title("Why the model flags the held-out customer (log-odds)", loc="left", fontsize=11)
    save(None, "shap_waterfall_holdout.png")
    local = pd.Series(ebm.eval_terms(Xh)[0], index=[term_label(n) for n in ebm.term_names_])
    local = local.reindex(local.abs().sort_values(ascending=False).index)

    # 3. PDP + ICE on raw inputs for the strongest usage type, through the full pipeline
    top_type = [u for u in USAGE_TYPES if USAGE_TYPES[u].split(" (")[0] == by_type.idxmax()]
    top_type = top_type[0] if top_type else "app_whatsapp"
    pdp_feats = [f"{top_type}_d01_10", f"{top_type}_d21_30", f"{top_type}_active"]
    Xraw = sample[RAW_FEATURES].sample(400, random_state=config.RANDOM_STATE)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    disp = PartialDependenceDisplay.from_estimator(
        pipe, Xraw, pdp_feats, kind="both", subsample=60, random_state=config.RANDOM_STATE,
        percentiles=(0.02, 0.98), ax=axes, ice_lines_kw={"color": BLUE, "alpha": 0.15},
        pd_line_kw={"color": ORANGE, "linewidth": 2.5})
    for ax, f in zip(disp.axes_.ravel(), pdp_feats):
        ax.set_xlabel(readable(f)); ax.set_ylabel("Churn probability")
        ax.get_legend() and ax.get_legend().remove()
    fig.suptitle(f"Partial dependence (orange) and ICE lines (blue): {by_type.idxmax()}",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    save(fig, "pdp_top_usage_type.png")
    pdp_summary = {}
    for f, avg in zip(pdp_feats, disp.pd_results):
        a = np.asarray(avg["average"]).ravel()
        pdp_summary[readable(f)] = {"low": round(float(a[0]), 3), "high": round(float(a[-1]), 3)}

    # 4. LIME for the same customer
    background = train[RAW_FEATURES].sample(5000, random_state=config.RANDOM_STATE)
    lime_exp = LimeTabularExplainer(background.values, feature_names=[readable(c) for c in RAW_FEATURES],
                                    class_names=["Stays", "Churns"], mode="classification",
                                    discretize_continuous=True, random_state=config.RANDOM_STATE)
    predict = lambda arr: pipe.predict_proba(pd.DataFrame(arr, columns=RAW_FEATURES))
    exp = lime_exp.explain_instance(holdout[RAW_FEATURES].values[0], predict, num_features=10,
                                    num_samples=3000)
    weights = exp.as_list(label=1)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    labels, vals = [w[0] for w in weights][::-1], [w[1] for w in weights][::-1]
    ax.barh(labels, vals, color=[ORANGE if v > 0 else BLUE for v in vals], height=0.6)
    ax.axvline(0, color=INK, linewidth=0.8)
    ax.set_xlabel("Effect on churn probability (positive = towards churn)")
    ax.set_title("LIME: local explanation for the held-out customer", loc="left")
    ax.grid(axis="y", visible=False)
    save(fig, "lime_holdout.png")

    prob = float(pipe.predict_proba(holdout[RAW_FEATURES])[0, 1])
    summary = {"model": "EBM (new algorithm)", "holdout_churn_probability": round(prob, 4),
               "holdout_true_label": int(holdout[config.TARGET].iloc[0]),
               "n_terms": len(ebm.term_names_),
               "n_pairs": int(sum(len(f) == 2 for f in ebm.term_features_)),
               "ebm_top10": imp.head(10).round(4).to_dict(),
               "ebm_by_usage_type": by_type.sort_values(ascending=False).round(3).to_dict(),
               "shape_functions": shapes,
               "shap_top10": shap_rank.head(10).round(4).to_dict(),
               "pdp": pdp_summary,
               "holdout_ebm_local_top10": local.head(10).round(4).to_dict(),
               "lime_top10": [[w[0], round(w[1], 4)] for w in weights]}
    (config.REPORTS_DIR / "explainability_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
