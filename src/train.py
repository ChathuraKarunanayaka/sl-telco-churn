"""Train the NEW algorithm (Explainable Boosting Machine, EBM) and compare it with the
standard models from the lectures (weak baseline up to stacked ensembles).

Run:  python src/train.py      (about 45-60 minutes on a 2-core laptop)

EBM (InterpretML, Microsoft Research) was not covered in the lectures. It is a
Generalized Additive Model (GAM) whose shape function for each feature is learned
with many rounds of tiny boosted trees, one feature at a time (cyclic boosting),
plus a few automatically detected pairwise interactions. The final model is a sum
of readable curves, so it is a "glassbox": accurate like boosting, readable like
logistic regression.

Steps
1. Load first-month usage + churn labels (train/test split given by the dataset).
2. Clean: duplicates, missing values, negative values.
3. Keep ONE test customer aside for the single-sample inference demo.
4. Tune EBM on a validation split; tune Random Forest, XGBoost and LightGBM with RandomizedSearchCV.
5. Compare every model on a VALIDATION part of the training data (never on the test set).
6. Refit every model on the full training data and score it once on the test set.
7. Save the EBM pipeline as models/model.pkl (the assignment's chosen new algorithm).
"""
import json
import pickle
import sys
import time
import warnings

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (AdaBoostClassifier, RandomForestClassifier, StackingClassifier,
                              VotingClassifier)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from interpret.glassbox import ExplainableBoostingClassifier

import config
from features import (RAW_FEATURES, USAGE_TYPES, PortableXGBClassifier, build_features,
                      engineered_names, signed_log)

warnings.filterwarnings("ignore")
SEED = config.RANDOM_STATE


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def load_data():
    feats = pd.read_csv(config.FEATURES_FILE)
    train = pd.read_csv(config.TRAIN_LABELS).merge(feats, on="customer_id", how="left")
    test = pd.read_csv(config.TEST_LABELS).merge(feats, on="customer_id", how="left")
    for name, part in [("train", train), ("test", test)]:
        print(f"{name}: {len(part):,} customers | churn rate {part[config.TARGET].mean():.1%} | "
              f"missing values {int(part[RAW_FEATURES].isna().sum().sum())} | "
              f"duplicate ids {int(part['customer_id'].duplicated().sum())} | "
              f"negative values {int((part[RAW_FEATURES] < 0).sum().sum())}")
    overlap = set(train.customer_id) & set(test.customer_id)
    print("Customers in both train and test:", len(overlap))
    return train, test


def pipeline(model, scale=False):
    """Feature engineering + imputation (+ log and scaling for linear models) + model.

    Trees split on thresholds, so they need no scaling. Logistic Regression is
    sensitive to scale and to very large values, so it gets a log transform
    and standard scaling.
    """
    steps = [("features", FunctionTransformer(build_features, feature_names_out=engineered_names)),
             ("impute", SimpleImputer(strategy="median"))]   # median: robust to heavy users
    if scale:
        steps += [("log", FunctionTransformer(signed_log, feature_names_out="one-to-one")),
                  ("scale", StandardScaler())]
    steps.append(("model", model))
    return Pipeline(steps).set_output(transform="pandas")


# --------------------------------------------------------------------------
# Models with designed (non-default) hyperparameters
# --------------------------------------------------------------------------
def build_models(spw, tuned):
    """spw = scale_pos_weight = stayers / churners (about 3.3), used by the boosters."""
    rf = RandomForestClassifier(**tuned["Random Forest"], class_weight="balanced",
                                random_state=SEED, n_jobs=config.N_JOBS)
    xgb = PortableXGBClassifier(**tuned["XGBoost"], scale_pos_weight=spw, objective="binary:logistic",
                        eval_metric="logloss", tree_method="hist", random_state=SEED, n_jobs=config.N_JOBS)
    lgbm = LGBMClassifier(**tuned["LightGBM"], scale_pos_weight=spw, random_state=SEED,
                          n_jobs=config.N_JOBS, verbose=-1)
    cat = CatBoostClassifier(iterations=500, depth=6, learning_rate=0.05, l2_leaf_reg=3,
                             auto_class_weights="Balanced", random_seed=SEED, verbose=0,
                             thread_count=2, allow_writing_files=False)
    return {
        # The NEW algorithm (not covered in lectures): glassbox additive model
        "EBM (new algorithm)": (make_ebm(tuned.get("EBM", EBM_DESIGNED)), False),
        # Baselines: what you get without real learning, and the classic weak model
        "Majority baseline": (DummyClassifier(strategy="most_frequent"), False),
        "Logistic Regression": (LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000,
                                                   random_state=SEED), True),
        # One tree: easy to read but high variance
        "Decision Tree": (DecisionTreeClassifier(max_depth=6, min_samples_leaf=50,
                                                 class_weight="balanced", random_state=SEED), False),
        # Bagging: many deep trees on random samples, averaged
        "Random Forest": (rf, False),
        # Boosting family: trees built one after another, each fixing earlier errors
        "AdaBoost": (AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=2), n_estimators=300,
                                        learning_rate=0.5, random_state=SEED), False),
        "XGBoost": (xgb, False),
        "LightGBM": (lgbm, False),
        "CatBoost": (cat, False),
        # Ensembles of the strongest boosters
        "Soft Voting (XGB + LGBM + CatBoost)": (VotingClassifier(
            [("xgb", xgb), ("lgbm", lgbm), ("cat", cat)], voting="soft"), False),
        "Stacking (RF + XGB + LGBM -> LR)": (StackingClassifier(
            [("rf", rf), ("xgb", xgb), ("lgbm", lgbm)],
            final_estimator=LogisticRegression(max_iter=2000, random_state=SEED),
            cv=StratifiedKFold(5, shuffle=True, random_state=SEED), stack_method="predict_proba"), False),
    }


# --------------------------------------------------------------------------
# EBM: designed values and a small validation search
# --------------------------------------------------------------------------
EBM_NAME = "EBM (new algorithm)"
EBM_DESIGNED = dict(learning_rate=0.015, interactions=10, max_bins=256, max_leaves=2,
                    min_samples_leaf=4, outer_bags=8)
EBM_CANDIDATES = [   # the number of pairwise interactions is the main choice; one setting changes the step size
    {**EBM_DESIGNED, "interactions": 0},                  # purely additive (no pairs)
    dict(EBM_DESIGNED),                                   # 10 pairs
    {**EBM_DESIGNED, "interactions": 20},
    {**EBM_DESIGNED, "interactions": 50},
    {**EBM_DESIGNED, "interactions": 100},
    {**EBM_DESIGNED, "interactions": 200},
    {**EBM_DESIGNED, "interactions": 100, "learning_rate": 0.03, "max_leaves": 3},   # bigger steps
]


def make_ebm(params):
    """EBM has no class_weight, so imbalance is handled with sample weights at fit time."""
    return ExplainableBoostingClassifier(**params, random_state=SEED, n_jobs=2)


def fit(pipe, X, y):
    """Fit a pipeline; EBM gets sample weights so churners count ~3.3 times more."""
    if isinstance(pipe[-1], ExplainableBoostingClassifier):
        w = np.where(y == 1, (y == 0).sum() / (y == 1).sum(), 1.0)
        return pipe.fit(X, y, model__sample_weight=w)
    return pipe.fit(X, y)


def tune_ebm(X, y):
    """EBM is slow to cross-validate, so each candidate is scored once on a stratified 20%
    validation split of the training data (PR-AUC)."""
    X_fit, X_val, y_fit, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED + 1)
    best, best_score, rows = None, -1, []
    for params in EBM_CANDIDATES:
        start = time.time()
        pipe = fit(pipeline(make_ebm(params)), X_fit, y_fit)
        score = average_precision_score(y_val, pipe.predict_proba(X_val)[:, 1])
        rows.append({**params, "val_pr_auc": round(score, 4), "seconds": round(time.time() - start)})
        print(f"  EBM {params} -> val PR-AUC {score:.4f} ({time.time() - start:.0f} s)")
        if score > best_score + 0.0005:                     # only switch for a real gain
            best, best_score = params, score
    pd.DataFrame(rows).to_csv(config.REPORTS_DIR / "ebm_tuning.csv", index=False)
    return best


DESIGNED = {   # starting values, each chosen for ~52,000 customers and 90 features
    "Random Forest": dict(n_estimators=200, max_depth=None, min_samples_leaf=5, max_features="sqrt",
                          max_samples=0.7),
    "XGBoost": dict(n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.8,
                    colsample_bytree=0.8, min_child_weight=5, gamma=0.1, reg_lambda=1.5),
    "LightGBM": dict(n_estimators=400, num_leaves=31, learning_rate=0.05, subsample=0.8,
                     subsample_freq=1, colsample_bytree=0.8, min_child_samples=40, reg_lambda=1.0),
}
TUNED_FILE = config.REPORTS_DIR / "tuned_params.json"
SEARCH = {
    "Random Forest": {"model__n_estimators": [150, 250], "model__max_depth": [None, 15, 25],
                      "model__min_samples_leaf": [2, 5, 10], "model__max_features": ["sqrt", 0.3]},
    "XGBoost": {"model__n_estimators": [300, 500, 700], "model__max_depth": [3, 4, 5, 6, 8],
                "model__learning_rate": [0.03, 0.05, 0.1], "model__subsample": [0.7, 0.8, 0.9],
                "model__colsample_bytree": [0.6, 0.8, 1.0], "model__min_child_weight": [1, 5, 10],
                "model__gamma": [0, 0.1, 0.5], "model__reg_lambda": [1.0, 1.5, 3.0]},
    "LightGBM": {"model__n_estimators": [300, 500, 700], "model__num_leaves": [15, 31, 63],
                 "model__learning_rate": [0.03, 0.05, 0.1], "model__min_child_samples": [20, 40, 80],
                 "model__colsample_bytree": [0.6, 0.8, 1.0], "model__reg_lambda": [0.0, 1.0, 5.0]},
}


def tune(X, y, spw):
    """RandomizedSearchCV with stratified folds; scored by PR-AUC (average precision),
    which focuses on the smaller 'churn' class better than accuracy does."""
    tuned, rows = {}, []
    base = build_models(spw, DESIGNED)
    for name, n_iter in [("XGBoost", 10), ("LightGBM", 10), ("Random Forest", 5)]:
        start = time.time()
        search = RandomizedSearchCV(pipeline(base[name][0]), SEARCH[name], n_iter=n_iter,
                                    scoring="average_precision", random_state=SEED, n_jobs=1,
                                    cv=StratifiedKFold(3, shuffle=True, random_state=SEED))
        search.fit(X, y)
        best = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
        tuned[name] = {**DESIGNED[name], **best}
        print(f"  {name}: best CV PR-AUC {search.best_score_:.4f} ({time.time() - start:.0f} s)")
        rows += [{"model": name, "parameter": k, "designed": DESIGNED[name].get(k),
                  "tuned": tuned[name][k]} for k in tuned[name]]
    table = pd.DataFrame(rows)
    table["changed"] = table["designed"].astype(str) != table["tuned"].astype(str)
    table.to_csv(config.REPORTS_DIR / "hyperparameters.csv", index=False)
    TUNED_FILE.write_text(json.dumps(tuned, indent=1))
    return tuned


def scores(y, prob, threshold=0.5):
    pred = (prob >= threshold).astype(int)
    return {"accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred), "f1": f1_score(y, pred),
            "roc_auc": roc_auc_score(y, prob) if len(set(prob)) > 1 else 0.5,
            "pr_auc": average_precision_score(y, prob)}


def main():
    for d in (config.MODELS_DIR, config.REPORTS_DIR, config.PROCESSED_DIR):
        d.mkdir(parents=True, exist_ok=True)
    train, test = load_data()

    # One test customer is kept completely aside for the inference demo.
    holdout = test[test[config.TARGET] == 1].sample(1, random_state=SEED)
    test = test.drop(holdout.index)
    holdout.to_csv(config.INFERENCE_SAMPLE_FILE, index=False)
    print("Held-out customer:", holdout["customer_id"].iloc[0][:12], "... true label =",
          int(holdout[config.TARGET].iloc[0]))

    X, y = train[RAW_FEATURES], train[config.TARGET]
    X_test, y_test = test[RAW_FEATURES], test[config.TARGET]
    spw = float((y == 0).sum() / (y == 1).sum())

    if TUNED_FILE.exists() and "--retune" not in sys.argv:
        tuned = json.loads(TUNED_FILE.read_text())       # reuse the last search (add --retune to search again)
        print("\nUsing tuned parameters from", TUNED_FILE.name)
    else:
        print("\nTuning (RandomizedSearchCV, 3 stratified folds, PR-AUC):")
        tuned = tune(X, y, spw)
    if "EBM" not in tuned or "--retune" in sys.argv:
        print("\nTuning EBM on a validation split:")
        tuned["EBM"] = tune_ebm(X, y)
        TUNED_FILE.write_text(json.dumps(tuned, indent=1))

    # Validation split inside the training data, used ONLY to choose the final model
    X_fit, X_val, y_fit, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    rows = []
    print("\nFit on 80% of train, score on 20% validation, then refit on all train and score on test:")
    for name, (model, scale) in build_models(spw, tuned).items():
        val_pipe = fit(pipeline(model, scale), X_fit, y_fit)
        val = scores(y_val, val_pipe.predict_proba(X_val)[:, 1])
        start = time.time()
        pipe = fit(pipeline(build_models(spw, tuned)[name][0], scale), X, y)
        seconds = time.time() - start
        test_s = scores(y_test, pipe.predict_proba(X_test)[:, 1])
        train_f1 = f1_score(y, pipe.predict(X))
        rows.append({"model": name, "val_pr_auc": val["pr_auc"], "val_f1": val["f1"],
                     **{f"test_{k}": v for k, v in test_s.items()},
                     "train_f1": train_f1, "train_time_s": seconds})
        pickle.dump(pipe, open(config.MODELS_DIR / f"_tmp_{len(rows)}.pkl", "wb"))
        print(f"  {name:38s} val PR-AUC {val['pr_auc']:.4f} | test F1 {test_s['f1']:.4f} "
              f"ROC-AUC {test_s['roc_auc']:.4f} | {seconds:.0f} s")

    results = pd.DataFrame(rows)
    results.round(4).to_csv(config.REPORTS_DIR / "model_comparison.csv", index=False)

    # The assignment asks for a NEW algorithm, so EBM is the deployed model. The other models
    # are benchmarks: they show whether EBM is competitive with the methods from the lectures.
    best_i = results.index[results["model"] == EBM_NAME][0]
    best_name = EBM_NAME
    gap = results["val_pr_auc"].max() - results.loc[best_i, "val_pr_auc"]
    print(f"EBM validation PR-AUC is {gap:.4f} below the best benchmark")
    for i in range(len(rows)):
        tmp = config.MODELS_DIR / f"_tmp_{i + 1}.pkl"
        if i == best_i:
            tmp.replace(config.MODEL_FILE)                       # pickle.dump(best_model, 'model.pkl')
        elif results.loc[i, "model"] == "Logistic Regression":
            tmp.replace(config.MODELS_DIR / "logistic_regression.pkl")
        else:
            tmp.unlink()
    print(f"\nSelected model: {best_name}")

    info = {
        "model_name": best_name, "class_labels": {str(k): v for k, v in config.CLASS_LABELS.items()},
        "usage_types": USAGE_TYPES, "raw_features": RAW_FEATURES,
        "ranges": {c: {"min": float(X[c].min()), "max": float(X[c].max()),
                       "median": float(X[c].median())} for c in RAW_FEATURES},
        "examples": {
            "Typical customer who stayed": X[y == 0].median().round(2).to_dict(),
            "Typical customer who churned": X[y == 1].median().round(2).to_dict(),
        },
    }
    config.FEATURE_INFO_FILE.write_text(json.dumps(info, indent=1))
    print("Saved models/model.pkl, models/logistic_regression.pkl, reports/model_comparison.csv")


if __name__ == "__main__":
    main()
