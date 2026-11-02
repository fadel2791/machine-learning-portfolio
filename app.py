from flask import Flask, render_template, request
from sklearn.metrics import r2_score, mean_absolute_error
import pandas as pd
import joblib
import os

app = Flask(
    __name__,
    template_folder="app/templates",
    static_folder="app/static"
)

# ==========================================
# DIREKTORI
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "app", "data")
MODEL_DIR = os.path.join(BASE_DIR, "app", "models")

# ==========================================
# MEMUAT DATASET DAN MODEL
# ==========================================

slr_data = pd.read_csv(
    os.path.join(DATA_DIR, "data_penjualan.csv")
)

mlr_data = pd.read_csv(
    os.path.join(DATA_DIR, "data_listrik.csv")
)

slr_model = joblib.load(
    os.path.join(MODEL_DIR, "slr_model.pkl")
)

mlr_model = joblib.load(
    os.path.join(MODEL_DIR, "mlr_model.pkl")
)

# ==========================================
# EVALUASI MODEL SLR
# ==========================================

X_slr = slr_data[["biaya_iklan"]]
y_slr = slr_data["penjualan"]

slr_predictions = slr_model.predict(X_slr)

slr_r2 = r2_score(y_slr, slr_predictions)
slr_mae = mean_absolute_error(y_slr, slr_predictions)


# ==========================================
# EVALUASI MODEL MLR
# ==========================================

X_mlr = mlr_data[
    [
        "jumlah_penghuni",
        "jam_penggunaan_ac",
        "jam_penggunaan_elektronik"
    ]
]

y_mlr = mlr_data["konsumsi_listrik"]

mlr_predictions = mlr_model.predict(X_mlr)

mlr_r2 = r2_score(y_mlr, mlr_predictions)
mlr_mae = mean_absolute_error(y_mlr, mlr_predictions)

# ==========================================
# HALAMAN HOME
# ==========================================

@app.route("/")
def home():
    return render_template("index.html")


# ==========================================
# HALAMAN SLR
# ==========================================

@app.route("/practicum/slr", methods=["GET", "POST"])
def slr():

    prediction = None
    error = None

    if request.method == "POST":
        try:
            biaya_iklan = float(
                request.form["biaya_iklan"]
            )

            if biaya_iklan < 0:
                raise ValueError(
                    "Biaya iklan tidak boleh negatif."
                )

            result = slr_model.predict(
                pd.DataFrame({
                    "biaya_iklan": [biaya_iklan]
                })
            )

            prediction = round(float(result[0]), 2)

        except (ValueError, TypeError):
            error = "Masukkan biaya iklan berupa angka yang valid."

    return render_template(
        "slr.html",
        dataset=slr_data.to_dict("records"),
        total_data=len(slr_data),
        mean_sales=round(slr_data["penjualan"].mean(), 2),
        mean_ad=round(slr_data["biaya_iklan"].mean(), 2),
        intercept=round(float(slr_model.intercept_), 4),
        coefficient=round(float(slr_model.coef_[0]), 4),
        prediction=prediction,
        error=error,
        r2_score=round(float(slr_r2), 4),
        mae=round(float(slr_mae), 4)
    )


# ==========================================
# HALAMAN MLR
# ==========================================

@app.route("/practicum/mlr", methods=["GET", "POST"])
def mlr():

    prediction = None
    error = None

    if request.method == "POST":
        try:
            penghuni = float(
                request.form["jumlah_penghuni"]
            )

            ac = float(
                request.form["jam_penggunaan_ac"]
            )

            elektronik = float(
                request.form["jam_penggunaan_elektronik"]
            )

            if penghuni < 0 or ac < 0 or elektronik < 0:
                raise ValueError(
                    "Nilai variabel tidak boleh negatif."
                )

            input_data = pd.DataFrame([{
                "jumlah_penghuni": penghuni,
                "jam_penggunaan_ac": ac,
                "jam_penggunaan_elektronik": elektronik
            }])

            result = mlr_model.predict(input_data)

            prediction = round(float(result[0]), 2)

        except (ValueError, TypeError):
            error = "Pastikan seluruh input berupa angka yang valid."

    return render_template(
        "mlr.html",
        dataset=mlr_data.to_dict("records"),
        total_data=len(mlr_data),
        intercept=round(float(mlr_model.intercept_), 4),
        coefficients=mlr_model.coef_,
        prediction=prediction,
        error=error,
        r2_score=round(float(mlr_r2), 4),
        mae=round(float(mlr_mae), 4)
    )


# ==========================================
# MENJALANKAN APLIKASI
# ==========================================

if __name__ == "__main__":
    app.run(debug=True)
