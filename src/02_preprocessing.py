"""
02_preprocessing.py
===================
Water Potability — Nettoyage, feature engineering & préparation des splits.

Améliorations clés vs le notebook original :
  • Imputation par médiane (robuste aux outliers) au lieu d'un dropna brutal
  • Suppression des outliers via IQR sur percentiles 20-80 (conservée)
  • Feature engineering : ratios et interactions chimiques
  • SMOTE uniquement sur le TRAIN set (pas de fuite de données)
  • StandardScaler ajusté sur le TRAIN uniquement
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import os, joblib
from pathlib import Path

# ─── CONFIG (racine projet, indépendant du CWD) ───────────────────────────────
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
INPUT_PATH = DATA_DIR / "raw_loaded.csv"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE    = 0.20


# ─── 1. IMPUTATION (médiane par groupe de potabilité) ─────────────────────────
def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Impute les NaN par la médiane de chaque classe."""
    df = df.copy()
    features = [c for c in df.columns if c != "Potability"]
    for col in features:
        medians = df.groupby("Potability")[col].transform("median")
        df[col] = df[col].fillna(medians)
        # Cas où toute une classe est NaN → fallback médiane globale
        df[col] = df[col].fillna(df[col].median())

    remaining = df.isnull().sum().sum()
    print(f"✅ Imputation terminée — valeurs manquantes restantes : {remaining}")
    return df


# ─── 2. SUPPRESSION DES OUTLIERS (IQR 20-80) ─────────────────────────────────
def remove_outliers_iqr(df: pd.DataFrame,
                         lower_q: float = 0.20,
                         upper_q: float = 0.80,
                         iqr_mult: float = 1.5) -> pd.DataFrame:
    df_clean = df.copy()
    before = len(df_clean)
    for col in df_clean.columns:
        if df_clean[col].nunique() >= 12:
            Q1  = df_clean[col].quantile(lower_q)
            Q3  = df_clean[col].quantile(upper_q)
            IQR = Q3 - Q1
            df_clean = df_clean[
                (df_clean[col] >= Q1 - iqr_mult * IQR) &
                (df_clean[col] <= Q3 + iqr_mult * IQR)
            ]
    df_clean = df_clean.reset_index(drop=True)
    after = len(df_clean)
    print(f"✅ Outliers supprimés : {before - after} lignes retirées "
          f"({(before - after)/before*100:.1f}%) — reste {after} lignes")
    return df_clean


# ─── 3. FEATURE ENGINEERING ───────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crée des features chimiques pertinentes pour la potabilité.
    Source : domaine eau potable (WHO standards).
    """
    df = df.copy()

    # Ratio pH/Hardness — l'équilibre acide-minéraux impacte la potabilité
    df["ph_hardness_ratio"]       = df["ph"] / (df["Hardness"] + 1e-6)

    # Rapport Chloramines/Sulfate — traitement chimique vs minéraux dissous
    df["chloramines_sulfate_ratio"] = df["Chloramines"] / (df["Sulfate"] + 1e-6)

    # Produit Turbidité × Solides — indicateur de charge totale en suspension
    df["turbidity_solids"]        = df["Turbidity"] * df["Solids"]

    # Conductivité normalisée par les solides totaux dissous
    df["conductivity_per_solid"]  = df["Conductivity"] / (df["Solids"] + 1e-6)

    # Score de "dureté totale" : dureté + TOC + sulfate
    df["hardness_score"]          = df["Hardness"] + df["Organic_carbon"] + df["Sulfate"]

    print(f"✅ Feature engineering : +5 nouvelles features → {df.shape[1] - 1} features au total")
    return df


# ─── 4. SPLIT + SMOTE + SCALING ───────────────────────────────────────────────
def prepare_splits(df: pd.DataFrame):
    X = df.drop("Potability", axis=1)
    y = df["Potability"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\n── Split train/test ──")
    print(f"   Train : {X_train.shape}  |  Test : {X_test.shape}")
    print(f"   Train classes : {y_train.value_counts().to_dict()}")

    # SMOTE appliqué UNIQUEMENT sur le train (pas de data leakage)
    smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=5)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f"\n── Après SMOTE ──")
    print(f"   Train rééquilibré : {X_train_res.shape}")
    print(f"   Classes : {pd.Series(y_train_res).value_counts().to_dict()}")

    # Scaling ajusté uniquement sur le TRAIN rééquilibré
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train_res)
    X_test_sc  = scaler.transform(X_test)

    # Sauvegarder le scaler pour l'inférence future
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    print(f"\n✅ Scaler sauvegardé → {MODELS_DIR / 'scaler.pkl'}")

    return (X_train_res, X_test, y_train_res, y_test,
            X_train_sc, X_test_sc, X.columns.tolist())


# ─── 5. PERSISTANCE ───────────────────────────────────────────────────────────
def save_splits(X_train, X_test, y_train, y_test,
                X_train_sc, X_test_sc, feature_names):
    """Sauvegarde tous les splits pour le module modeling."""

    pd.DataFrame(X_train, columns=feature_names).assign(Potability=y_train.values)\
      .to_csv(DATA_DIR / "train_raw.csv", index=False)
    pd.DataFrame(X_test, columns=feature_names).assign(Potability=y_test.values)\
      .to_csv(DATA_DIR / "test_raw.csv", index=False)
    pd.DataFrame(X_train_sc, columns=feature_names).assign(Potability=y_train.values)\
      .to_csv(DATA_DIR / "train_scaled.csv", index=False)
    pd.DataFrame(X_test_sc, columns=feature_names).assign(Potability=y_test.values)\
      .to_csv(DATA_DIR / "test_scaled.csv", index=False)

    print(f"\n✅ Splits sauvegardés dans → {DATA_DIR}/")


# ─── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = pd.read_csv(INPUT_PATH)
    print(f"📥 Données chargées : {df.shape}")

    df = impute_missing(df)
    df = remove_outliers_iqr(df)
    df = engineer_features(df)

    (X_train, X_test, y_train, y_test,
     X_train_sc, X_test_sc, feature_names) = prepare_splits(df)

    save_splits(X_train, X_test, y_train, y_test,
                X_train_sc, X_test_sc, feature_names)

    print("\n🎯 Preprocessing terminé avec succès !")
