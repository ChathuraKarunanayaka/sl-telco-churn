"""Telco churn early-warning app. Run from the project folder:  streamlit run app/app.py"""
import json
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import config                                           # noqa: E402
from features import RAW_FEATURES, USAGE_TYPES, readable  # noqa: E402
from plot_style import BLUE, INK, ORANGE, apply_style     # noqa: E402

st.set_page_config(page_title="Telco Churn Early Warning", page_icon="📶", layout="wide")
apply_style()
WINDOW_LABELS = {"d01_10": "Days 1-10", "d11_20": "Days 11-20", "d21_30": "Days 21-30"}


@st.cache_resource
def load():
    """Pickle loader: the full pipeline (feature engineering + EBM model) and its metadata."""
    model = pickle.load(open(config.MODEL_FILE, "rb"))
    info = json.loads(config.FEATURE_INFO_FILE.read_text())
    return model, info


def term_label(name):
    return " x ".join(readable(p.strip()) for p in name.split(" & "))


def fill(values: dict):
    for c in RAW_FEATURES:
        st.session_state[c] = float(values[c])


def load_holdout():
    s = pd.read_csv(config.INFERENCE_SAMPLE_FILE).iloc[0]
    fill(s)
    st.session_state["truth"] = int(s[config.TARGET])


try:
    model, info = load()
except Exception as error:
    st.error("models/model.pkl could not be loaded on this machine. Pickle files depend on the "
             "operating system and library versions. Run `python src/train.py` once, then reload.")
    st.caption(f"Technical detail: {type(error).__name__}")
    st.stop()

if RAW_FEATURES[0] not in st.session_state:
    fill(info["examples"]["Typical customer who stayed"])
    st.session_state["truth"] = None

st.title("📶 Telco Churn Early Warning")
st.markdown(
    "Predicts whether a mobile customer will **leave the network (churn)**, using only their "
    "**first month** of usage. Built for an MSc Applied Machine Learning assignment on a Sri Lankan "
    f"telco dataset (65,005 real anonymised customers, Kaggle).  \n**Model:** {info['model_name']}: "
    "an Explainable Boosting Machine, a glassbox model that is as accurate as XGBoost but shows its own "
    "reasons. Loaded from `models/model.pkl`.")

c1, c2, c3 = st.columns(3)
c1.button("Load typical customer who stayed", on_click=lambda: (fill(info["examples"]["Typical customer who stayed"]),
                                                                 st.session_state.update(truth=None)))
c2.button("Load typical customer who churned", on_click=lambda: (fill(info["examples"]["Typical customer who churned"]),
                                                                  st.session_state.update(truth=None)))
c3.button("Load held-out test customer", on_click=load_holdout,
          help="One real customer kept out of training and testing.")

with st.form("usage"):
    st.markdown("**Average usage per day in the first month** (three 10-day periods) and the number of "
                "active days out of 30.")
    groups = {"📱 Apps": [u for u in USAGE_TYPES if u.startswith("app_")],
              "💳 Packs": [u for u in USAGE_TYPES if u.startswith("pack_")],
              "📞 Calls": [u for u in USAGE_TYPES if u.startswith("voice_")]}
    for title, members in groups.items():
        with st.expander(title, expanded=(title == "📱 Apps")):
            for u in members:
                cols = st.columns([2.2, 1, 1, 1, 1])
                cols[0].markdown(f"<div style='padding-top:30px'>{USAGE_TYPES[u]}</div>", unsafe_allow_html=True)
                for col, w in zip(cols[1:4], WINDOW_LABELS):
                    col.number_input(WINDOW_LABELS[w], min_value=0.0, step=1.0, key=f"{u}_{w}", format="%.1f")
                cols[4].number_input("Active days", min_value=0.0, max_value=30.0, step=1.0, key=f"{u}_active",
                                     format="%.0f")
    go = st.form_submit_button("Predict churn risk", type="primary")

if go:
    row = pd.DataFrame([{c: st.session_state[c] for c in RAW_FEATURES}])
    prob = float(model.predict_proba(row)[0, 1])
    churn = prob >= 0.5
    colour = ORANGE if churn else BLUE
    left, right = st.columns([1, 1.6])
    left.markdown(f"<div style='background:{colour};color:white;padding:14px 18px;border-radius:10px;"
                  f"font-size:1.4rem;font-weight:700;display:inline-block'>"
                  f"{'⚠️ Likely to churn' if churn else '✅ Likely to stay'}</div>", unsafe_allow_html=True)
    shown = ">99.9%" if prob > 0.999 else "<0.1%" if prob < 0.001 else f"{prob:.1%}"
    left.metric("Churn probability", shown)
    left.caption("Suggested action: " + ("contact this customer with a retention offer."
                                          if churn else "no action needed."))
    if st.session_state.get("truth") is not None:
        truth = info["class_labels"][str(st.session_state["truth"])]
        left.info(f"Held-out customer: the real outcome was **{truth}**.")

    # EBM is a glassbox: the score is the sum of one contribution per feature (or feature pair),
    # so the reasons below are read straight from the model, not approximated.
    ebm = model[-1]
    Xf = model[:-1].transform(row)
    contrib = pd.Series(ebm.eval_terms(Xf)[0], index=[term_label(n) for n in ebm.term_names_])
    top = contrib.reindex(contrib.abs().sort_values(ascending=False).index)[:10][::-1]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.barh(top.index, top.values, color=[ORANGE if v > 0 else BLUE for v in top.values], height=0.6)
    ax.axvline(0, color=INK, linewidth=0.8)
    ax.set_title("Why? Top 10 reasons (EBM contributions)", loc="left")
    ax.set_xlabel("Pushes towards churn (orange) or staying (blue), log-odds")
    ax.grid(axis="y", visible=False)
    right.pyplot(fig)
    plt.close(fig)
    right.caption("EBM adds one contribution per feature to get the final score, so these reasons "
                  "are exact, not estimated.")

st.caption("Data: Real World Customer Churn Dataset in Telco Domain (Kaggle, CC0). Only days 1-30 are used, "
           "to avoid leakage from later days when churned customers have already stopped using the service.")
