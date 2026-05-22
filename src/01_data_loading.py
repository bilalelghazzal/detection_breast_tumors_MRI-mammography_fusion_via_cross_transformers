"""
01_data_loading.py
==================
Water Potability — Chargement & exploration initiale des données.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

# ─── CONFIG (chemins relatifs à la racine du projet, pas au CWD) ─────────────
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def _resolve_input_csv() -> Path:
    """Fichier attendu dans data/ ; sinon copie courante à côté des scripts (src/)."""
    in_data = DATA_DIR / "water_potability.csv"
    in_src = SRC_DIR / "water_potability.csv"
    if in_data.is_file():
        return in_data
    if in_src.is_file():
        return in_src
    raise FileNotFoundError(
        "Fichier water_potability.csv introuvable. "
        f"Attendu : {in_data} ou {in_src}"
    )


DATA_PATH = _resolve_input_csv()
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── 1. CHARGEMENT ────────────────────────────────────────────────────────────
def load_data(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"✅ Dataset chargé : {df.shape[0]} lignes × {df.shape[1]} colonnes")
    return df


# ─── 2. RÉSUMÉ RAPIDE ─────────────────────────────────────────────────────────
def summarize(df: pd.DataFrame) -> None:
    print("\n── Colonnes ──")
    print(df.columns.tolist())

    print("\n── Aperçu (5 premières lignes) ──")
    print(df.head())

    print("\n── Statistiques descriptives ──")
    print(df.describe().round(3))

    print("\n── Valeurs manquantes ──")
    missing = df.isnull().sum()
    pct     = (missing / len(df) * 100).round(2)
    print(pd.DataFrame({"count": missing, "pct (%)": pct})[missing > 0])

    print("\n── Distribution de la cible ──")
    vc = df["Potability"].value_counts()
    print(vc)
    ratio = vc[0] / vc[1]
    print(f"   → Ratio Non-Potable / Potable : {ratio:.2f}  (déséquilibre à corriger)")


# ─── 3. VISUALISATIONS EDA ────────────────────────────────────────────────────
def plot_eda(df: pd.DataFrame) -> None:

    # 3a. Distribution de la cible
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.countplot(data=df, x="Potability", palette="Set2", ax=axes[0])
    axes[0].set_title("Distribution de la Potabilité")
    df["Potability"].value_counts().plot.pie(
        autopct="%1.1f%%", labels=["Non-Potable", "Potable"],
        colors=["#E74C3C", "#2ECC71"], ax=axes[1]
    )
    axes[1].set_ylabel("")
    axes[1].set_title("Répartition (%) — Potabilité")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "01_target_distribution.png", dpi=150)
    plt.close()
    print("📊 Graphe target sauvegardé.")

    # 3b. Box plots par variable × potabilité
    features = [c for c in df.columns if c != "Potability"]
    n = len(features)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 4))
    axes = axes.flatten()
    for i, col in enumerate(features):
        sns.boxplot(data=df, x="Potability", y=col,
                    palette="Set2", ax=axes[i])
        axes[i].set_title(f"Box Plot — {col}")
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "01_boxplots.png", dpi=150)
    plt.close()
    print("📊 Box plots sauvegardés.")

    # 3c. Heatmap de corrélation
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", linewidths=.5)
    plt.title("Matrice de Corrélation — Water Potability")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "01_correlation_heatmap.png", dpi=150)
    plt.close()
    print("📊 Heatmap corrélation sauvegardée.")

    # 3d. Histogrammes des features
    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 4))
    axes = axes.flatten()
    colors = sns.color_palette("viridis", n)
    for i, col in enumerate(features):
        sns.histplot(df[col].dropna(), kde=True, color=colors[i], ax=axes[i])
        axes[i].set_title(f"Distribution — {col}")
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "01_feature_distributions.png", dpi=150)
    plt.close()
    print("📊 Histogrammes sauvegardés.")


# ─── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = load_data(DATA_PATH)
    summarize(df)
    plot_eda(df)
    # Sauvegarder le raw pour la suite
    out_raw = DATA_DIR / "raw_loaded.csv"
    df.to_csv(out_raw, index=False)
    print(f"\n✅ Données brutes sauvegardées → {out_raw}")
