"""Generate the research-paper architecture for all nine experiments."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "results" / "architecture_diagram.png"


def box(ax, x, y, w, h, title, body, color, size=8.3):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                                facecolor="white", edgecolor=color, linewidth=1.7))
    ax.text(x + w / 2, y + h - 0.012, title, ha="center", va="top",
            fontsize=9.7, fontweight="bold", color=color)
    ax.text(x + w / 2, y + h * 0.34, body, ha="center", va="center",
            fontsize=size, color="#263238", linespacing=1.35)


def arrow(ax, start, end, color="#607d8b"):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=13,
                                 linewidth=1.4, color=color))


def build_diagram():
    fig, ax = plt.subplots(figsize=(16, 11), dpi=180)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(0.5, 0.975, "PROPOSED ARCHITECTURE OF THE ML BASED CROP RECOMMENDATION SYSTEM",
            ha="center", va="top", fontsize=15, fontweight="bold", color="#17324d")
    ax.text(0.5, 0.948, "Three feature-selection strategies × three classifiers = nine fair experiments",
            ha="center", va="top", fontsize=10, color="#607d8b")

    box(ax, 0.16, 0.865, 0.68, 0.06, "AGRICULTURAL DATASET",
        "N | P | K | Temperature | Humidity | Soil pH | Rainfall → Crop label", "#1f5a7a")
    arrow(ax, (0.5, 0.865), (0.5, 0.815))
    box(ax, 0.16, 0.755, 0.68, 0.06, "DATA PREPROCESSING",
        "Schema and missing/duplicate checks | EDA | same stratified 80/20 split", "#2f6f4f")
    arrow(ax, (0.5, 0.755), (0.5, 0.70))
    box(ax, 0.10, 0.625, 0.80, 0.075, "DOMAIN-BASED AGRICULTURAL FEATURE ENGINEERING",
        "N/P Ratio | N/K Ratio | P/K Ratio | Total/Mean NPK | Nutrient Balance | Temperature-Humidity Index | Rainfall-Humidity Index",
        "#8a5a2b", 7.8)
    arrow(ax, (0.5, 0.625), (0.5, 0.565), "#8a5a2b")

    centers = [0.18, 0.50, 0.82]
    colors = ["#536dfe", "#7a4a9d", "#0b6865"]
    titles = ["MUTUAL INFORMATION", "RFE", "MUTUAL INFORMATION + RFE"]
    bodies = ["MI scores\nSelected features", "Tree-estimator RFE\nSelected features", "MI candidates\nFinal RFE features"]
    for x, color, title, body in zip(centers, colors, titles, bodies):
        box(ax, x - 0.13, 0.47, 0.26, 0.075, title, body, color, 7.7)
        arrow(ax, (x, 0.47), (x, 0.405), color)
        box(ax, x - 0.13, 0.31, 0.26, 0.075, "SELECTED FEATURES",
            "Training data only\nSame features for 3 classifiers", "#455a64", 7.2)
        arrow(ax, (x, 0.31), (x, 0.255), "#455a64")

    box(ax, 0.07, 0.15, 0.86, 0.07, "THREE CLASSIFIERS",
        "XGBoost  |  LightGBM  |  Random Forest", "#a65e13", 10)
    for x in centers:
        arrow(ax, (x, 0.15), (x, 0.22), "#a65e13")
    arrow(ax, (0.5, 0.15), (0.5, 0.105), "#607d8b")
    box(ax, 0.20, 0.035, 0.60, 0.06, "NINE EXPERIMENTAL RESULTS → MODEL COMPARISON → BEST COMBINATION",
        "Accuracy | Precision | Recall | F1 | confusion matrices | feature importance", "#0b6865", 8)
    ax.text(0.5, 0.005, "Best model receives the same engineered terminal input and produces crop + confidence.",
            ha="center", va="bottom", fontsize=7.5, color="#78909c")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Architecture diagram saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    build_diagram()
