"""Load models/model.pkl and predict the held-out customer. Run:  python src/predict.py"""
import pickle

import pandas as pd

import config
from features import RAW_FEATURES, USAGE_TYPES


def main():
    model = pickle.load(open(config.MODEL_FILE, "rb"))          # pickle loader
    sample = pd.read_csv(config.INFERENCE_SAMPLE_FILE)
    prob = model.predict_proba(sample[RAW_FEATURES])[0]
    pred, true = int(prob[1] >= 0.5), int(sample[config.TARGET].iloc[0])
    print("Held-out customer (never used for training or testing):", sample["customer_id"].iloc[0][:16], "...")
    print("\nFirst-month usage (average per day in days 1-10 -> 21-30):")
    for u, label in USAGE_TYPES.items():
        print(f"  {label:48s} {sample[f'{u}_d01_10'].iloc[0]:8.1f} -> {sample[f'{u}_d21_30'].iloc[0]:8.1f}")
    print(f"\nPredicted: {config.CLASS_LABELS[pred]}  (churn probability {prob[1]:.3f})")
    print(f"Actual   : {config.CLASS_LABELS[true]}  ->  {'CORRECT' if pred == true else 'INCORRECT'}")


if __name__ == "__main__":
    main()
