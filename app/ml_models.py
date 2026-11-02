
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error


# ==========================================
# KONFIGURASI DIREKTORI
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

STATIC_DIR = os.path.join(BASE_DIR, "static")
IMAGE_DIR = os.path.join(STATIC_DIR, "images")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)


# ==========================================
# KONFIGURASI DATASET
# ==========================================

SLR_FILE = os.path.join(DATA_DIR, "data_penjualan.csv")
MLR_FILE = os.path.join(DATA_DIR, "data_listrik.csv")

# Sesuaikan dengan nama kolom CSV MLR kamu
MLR_FEATURES = [
    "jumlah_penghuni",
    "jam_penggunaan_ac",
    "jam_penggunaan_elektronik"
]

MLR_TARGET = "konsumsi_listrik"


# ==========================================
# FUNGSI EVALUASI MODEL
# ==========================================

def evaluate_model(model, X, y):

    predictions = model.predict(X)

    return {
        "r2": round(r2_score(y, predictions), 4),
        "mae": round(mean_absolute_error(y, predictions), 4)
    }


# ==========================================
# SIMPLE LINEAR REGRESSION
# ==========================================

def train_slr():

    dataset = pd.read_csv(SLR_FILE)

    X = dataset[["biaya_iklan"]]
    y = dataset["penjualan"]

    model = LinearRegression()
    model.fit(X, y)

    predictions = model.predict(X)

    metrics = evaluate_model(model, X, y)

    # Menyimpan model
    joblib.dump(
        model,
        os.path.join(MODEL_DIR, "slr_model.pkl")
    )

    # Membuat grafik regresi
    plt.figure(figsize=(9, 5))

    plt.scatter(
        X["biaya_iklan"],
        y,
        label="Data Aktual"
    )

    plt.plot(
        X["biaya_iklan"],
        predictions,
        color="red",
        label="Garis Regresi"
    )

    plt.title("Simple Linear Regression: Biaya Iklan dan Penjualan")
    plt.xlabel("Biaya Iklan (ribuan rupiah)")
    plt.ylabel("Jumlah Penjualan (unit)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        os.path.join(IMAGE_DIR, "slr_regression.png"),
        dpi=150
    )

    plt.close()

    return {
        "intercept": round(float(model.intercept_), 4),
        "coefficient": round(float(model.coef_[0]), 4),
        "metrics": metrics,
        "rows": len(dataset)
    }


# ==========================================
# MULTIPLE LINEAR REGRESSION
# ==========================================

def train_mlr():

    dataset = pd.read_csv(MLR_FILE)

    X = dataset[MLR_FEATURES]
    y = dataset[MLR_TARGET]

    model = LinearRegression()
    model.fit(X, y)

    predictions = model.predict(X)

    metrics = evaluate_model(model, X, y)

    # Menyimpan model
    joblib.dump(
        model,
        os.path.join(MODEL_DIR, "mlr_model.pkl")
    )

    # Membuat heatmap korelasi
    plt.figure(figsize=(9, 6))

    sns.heatmap(
        dataset.corr(numeric_only=True),
        annot=True,
        cmap="coolwarm",
        fmt=".2f"
    )

    plt.title("Correlation Heatmap - Konsumsi Listrik")
    plt.tight_layout()

    plt.savefig(
        os.path.join(IMAGE_DIR, "mlr_correlation.png"),
        dpi=150
    )

    plt.close()

    return {
        "intercept": round(float(model.intercept_), 4),
        "coefficients": {
            feature: round(float(coef), 4)
            for feature, coef in zip(MLR_FEATURES, model.coef_)
        },
        "metrics": metrics,
        "rows": len(dataset)
    }


# ==========================================
# MENJALANKAN SELURUH MODEL
# ==========================================

def train_all_models():

    print("\n" + "=" * 50)
    print("TRAINING MACHINE LEARNING PORTFOLIO")
    print("=" * 50)

    slr_result = train_slr()

    print("\nSIMPLE LINEAR REGRESSION")
    print(slr_result)

    mlr_result = train_mlr()

    print("\nMULTIPLE LINEAR REGRESSION")
    print(mlr_result)

    print("\nSemua model berhasil dilatih!")
    print("Model tersimpan di:", MODEL_DIR)
    print("Grafik tersimpan di:", IMAGE_DIR)

    return {
        "slr": slr_result,
        "mlr": mlr_result
    }


if __name__ == "__main__":
    train_all_models()
