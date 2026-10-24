"""Build data/raw/customer_first30.csv from the original Kaggle file (optional).

Dataset: "Real World Customer Churn Dataset in Telco Domain" (Kaggle, lasaljaywardena, CC0).
Download the zip, extract main.csv into data/kaggle/, then run:
    python src/build_dataset.py

main.csv has one row per customer and usage type, with 90 daily values (Day_1 .. Day_90).

LEAKAGE NOTE: only Day_1 .. Day_30 are used. Customers who churn stop using the
service later in the 90-day window, so later days already "contain the answer".
A model trained on all 90 days scores ROC-AUC 1.000 because it simply sees that
usage fell to zero. Using only the first month turns the task into a real
early-warning problem: predict churn from first-month behaviour.
"""
import numpy as np
import pandas as pd

import config

SOURCE = config.ROOT / "data" / "kaggle" / "main.csv"
DAYS = [f"Day_{i}" for i in range(1, 31)]          # first 30 days only


def main():
    parts = []
    for chunk in pd.read_csv(SOURCE, chunksize=60000, usecols=["customer_id", "usage_type"] + DAYS):
        x = chunk[DAYS].to_numpy(dtype="float32")
        parts.append(pd.DataFrame({
            "customer_id": chunk["customer_id"].values,
            "usage_type": chunk["usage_type"].values,
            "d01_10": x[:, :10].mean(axis=1),       # average per day, days 1-10
            "d11_20": x[:, 10:20].mean(axis=1),     # days 11-20
            "d21_30": x[:, 20:30].mean(axis=1),     # days 21-30
            "active": (x > 0).sum(axis=1),          # number of days with any use
        }))
    long = pd.concat(parts)
    long["usage_type"] = (long["usage_type"].str.replace("usage_", "", regex=False)
                          .str.replace("_daily", "", regex=False))
    wide = long.pivot_table(index="customer_id", columns="usage_type",
                            values=["d01_10", "d11_20", "d21_30", "active"])
    wide.columns = [f"{usage}_{window}" for window, usage in wide.columns]
    wide = wide[sorted(wide.columns)].reset_index()
    wide.to_csv(config.FEATURES_FILE, index=False, float_format="%.3f")
    print("Saved", config.FEATURES_FILE, wide.shape)


if __name__ == "__main__":
    main()
