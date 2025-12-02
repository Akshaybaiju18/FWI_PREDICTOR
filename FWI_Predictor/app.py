from flask import Flask, render_template, request
import os
import joblib
import numpy as np
import pandas as pd


app = Flask(__name__)


# Project root (where ridge_model.pkl and scaler.pkl live)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, "ridge_model.pkl")
SCALER_PATH = os.path.join(PROJECT_ROOT, "scaler.pkl")


# Feature order used during training
FEATURE_COLS = [
    "temperature", "rh", "ws", "rain",
    "dmc", "ffmc", "dc", "isi",
]

# Threshold for "high fire risk" warning
FIRE_WARNING_THRESHOLD = 20.0


# Load model (and scaler, even though the model was trained on raw features)
ridge_model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Collect form inputs
        temperature = float(request.form.get("temperature"))
        rh = float(request.form.get("rh"))
        ws = float(request.form.get("ws"))
        rain = float(request.form.get("rain"))
        dmc = float(request.form.get("dmc"))
        ffmc = float(request.form.get("ffmc"))
        dc = float(request.form.get("dc"))
        isi = float(request.form.get("isi"))

        # Build a DataFrame with the same columns used during training
        input_df = pd.DataFrame(
            [[temperature, rh, ws, rain, dmc, ffmc, dc, isi]],
            columns=FEATURE_COLS,
        )

        # IMPORTANT:
        # In ridge_regression.py, the model (ridge_model.pkl) was trained on RAW features,
        # not scaled ones. To keep predictions consistent with training, we pass
        # raw features to the model. scaler.pkl is loaded but not applied here.
        predicted_fwi = float(ridge_model.predict(input_df)[0])

        is_high_risk = predicted_fwi >= FIRE_WARNING_THRESHOLD

        return render_template(
            "home.html",
            predicted_fwi=round(predicted_fwi, 3),
            raw_fwi=predicted_fwi,
            is_high_risk=is_high_risk,
            threshold=FIRE_WARNING_THRESHOLD,
        )
    except Exception as e:
        return render_template(
            "home.html",
            predicted_fwi=None,
            raw_fwi=None,
            is_high_risk=False,
            threshold=FIRE_WARNING_THRESHOLD,
            error=str(e),
        )


if __name__ == "__main__":
    app.run(debug=True)



