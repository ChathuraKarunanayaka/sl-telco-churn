"""Central settings shared by every script and the app."""
from pathlib import Path

RANDOM_STATE = 42          # one seed everywhere, so results can be reproduced
N_JOBS = -1                # use all CPU cores

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

FEATURES_FILE = RAW_DIR / "customer_first30.csv"   # usage in the first 30 days, one row per customer
TRAIN_LABELS = RAW_DIR / "train_cxid.csv"           # customer_id, churn (split given by the dataset author)
TEST_LABELS = RAW_DIR / "test_cxid.csv"

MODEL_FILE = MODELS_DIR / "model.pkl"               # best model, used by the app
FEATURE_INFO_FILE = MODELS_DIR / "feature_info.json"
INFERENCE_SAMPLE_FILE = PROCESSED_DIR / "inference_sample.csv"

TARGET = "churn"
CLASS_LABELS = {0: "Stays", 1: "Churns"}
