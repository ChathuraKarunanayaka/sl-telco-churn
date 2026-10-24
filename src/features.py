"""Feature engineering shared by training AND the app, so they never drift apart."""
import numpy as np
import pandas as pd

# The 12 usage types in the dataset and how a person would describe them
USAGE_TYPES = {
    "app_facebook": "Facebook data (MB per day)",
    "app_tiktok": "TikTok data (MB per day)",
    "app_youtube": "YouTube data (MB per day)",
    "app_whatsapp": "WhatsApp data (MB per day)",
    "app_helakuru": "Helakuru app data (MB per day)",
    "app_other": "Other apps data (MB per day)",
    "pack_data": "Spend on data packs (LKR per day)",
    "pack_vas": "Spend on value-added services (LKR per day)",
    "voice_d2d_outgoing": "Outgoing calls, same network (min per day)",
    "voice_d2nd_outgoing": "Outgoing calls, other networks (min per day)",
    "voice_d2d_incoming": "Incoming calls, same network (min per day)",
    "voice_nd2d_incoming": "Incoming calls, other networks (min per day)",
}
WINDOWS = ["d01_10", "d11_20", "d21_30"]          # 10-day averages inside the first month
GROUPS = {"apps": [u for u in USAGE_TYPES if u.startswith("app_")],
          "packs": [u for u in USAGE_TYPES if u.startswith("pack_")],
          "voice": [u for u in USAGE_TYPES if u.startswith("voice_")]}

# Raw inputs: 12 usage types x (3 window averages + active days) = 48 columns
RAW_FEATURES = [f"{u}_{w}" for u in USAGE_TYPES for w in WINDOWS + ["active"]]


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Raw first-month usage -> model features.

    Domain idea: a customer who is about to leave usually uses the service LESS
    as the month goes on. So besides the raw levels, we add the change between
    the first and the last 10 days, for each usage type and for each group.
    """
    df = raw[RAW_FEATURES].astype(float).copy()
    new = {}
    for u in USAGE_TYPES:
        start, end = df[f"{u}_d01_10"], df[f"{u}_d21_30"]
        new[f"{u}_change"] = end - start                         # drop (negative) or growth
        new[f"{u}_ratio"] = (end + 1) / (start + 1)              # +1 avoids dividing by zero
    for g, members in GROUPS.items():
        for w in WINDOWS:
            new[f"{g}_total_{w}"] = df[[f"{u}_{w}" for u in members]].sum(axis=1)
        new[f"{g}_ratio"] = (new[f"{g}_total_d21_30"] + 1) / (new[f"{g}_total_d01_10"] + 1)
        new[f"{g}_active_mean"] = df[[f"{u}_active" for u in members]].mean(axis=1)
    return pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)


def feature_names() -> list:
    sample = pd.DataFrame([[1.0] * len(RAW_FEATURES)], columns=RAW_FEATURES)
    return list(build_features(sample).columns)


def readable(name: str) -> str:
    """Short human-readable feature name for charts."""
    for u, label in USAGE_TYPES.items():
        if name.startswith(u + "_"):
            short = label.split(" (")[0].replace("Spend on ", "").replace(" data", "")
            part = name[len(u) + 1:]
            return f"{short}: " + {"d01_10": "days 1-10", "d11_20": "days 11-20", "d21_30": "days 21-30",
                                   "active": "active days", "change": "change", "ratio": "end/start ratio"}[part]
    return name.replace("_total_", " total ").replace("d01_10", "days 1-10").replace(
        "d11_20", "days 11-20").replace("d21_30", "days 21-30").replace("_", " ")


def engineered_names(transformer, input_features):
    """Column names produced by build_features (used by the sklearn pipeline)."""
    return feature_names()


def signed_log(x):
    """log1p that also works for negative 'change' features: sign(x) * log(1 + |x|)."""
    return np.sign(x) * np.log1p(np.abs(x))


# --------------------------------------------------------------------------
# XGBoost model that pickles portably between Windows, Linux and macOS
# --------------------------------------------------------------------------
from xgboost import Booster, XGBClassifier  # noqa: E402


class PortableXGBClassifier(XGBClassifier):
    """Same as XGBClassifier, but inside the pickle the trees are stored as JSON text.

    A normal XGBoost pickle made on Linux can fail on Windows with
    "input stream corrupted". JSON is plain text, so it loads on any system.
    """

    def __getstate__(self):
        state = self.__dict__.copy()
        booster = state.pop("_Booster", None)
        if booster is not None:
            state["_booster_json"] = bytes(booster.save_raw("json")).decode("utf-8")
        return state

    def __setstate__(self, state):
        booster_json = state.pop("_booster_json", None)
        self.__dict__.update(state)
        if booster_json is not None:
            booster = Booster()
            booster.load_model(bytearray(booster_json.encode("utf-8")))
            self._Booster = booster
