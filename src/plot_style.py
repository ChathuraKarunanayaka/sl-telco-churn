"""One shared look for every chart in the project."""
import matplotlib.pyplot as plt

# Fixed colour for each model, used in the same order in every figure.
# Colour by model family, used the same way in every figure.
FAMILY_COLORS = {"baseline": "#898781", "single": "#e87ba4", "bagging": "#2a78d6",
                 "boosting": "#eb6834", "ensemble": "#1baf7a"}
MODEL_FAMILY = {
    "Majority baseline": "baseline", "Logistic Regression": "baseline", "Decision Tree": "single",
    "Random Forest": "bagging", "AdaBoost": "boosting", "XGBoost": "boosting", "LightGBM": "boosting",
    "CatBoost": "boosting", "Soft Voting (XGB + LGBM + CatBoost)": "ensemble",
    "Stacking (RF + XGB + LGBM -> LR)": "ensemble",
}
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#898781"
INK, INK_SOFT, GRID = "#0b0b0b", "#52514e", "#e1e0d9"


def apply_style():
    plt.rcParams.update({
        "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
        "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK_SOFT, "text.color": INK,
        "xtick.color": INK_SOFT, "ytick.color": INK_SOFT,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
        "legend.frameon": False, "lines.linewidth": 2, "figure.dpi": 110, "savefig.dpi": 150,
        "savefig.bbox": "tight", "font.family": "DejaVu Sans",
    })
