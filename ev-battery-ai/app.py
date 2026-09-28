import os
import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

MODEL_FILE = "ev_battery_model.joblib"
DATASET_FILE = "dataset/ev battery_failure prediction Dataset.csv"

model = None
default_values = {}

def init_app():
    global model, default_values
    if os.path.exists(MODEL_FILE):
        model = joblib.load(MODEL_FILE)
    
    if os.path.exists(DATASET_FILE):
        df = pd.read_csv(DATASET_FILE)
        X_base = df.drop(columns=["battery_failure", "vehicle_id", "battery_serial", "predicted_remaining_life_cycles"], errors="ignore")
        default_values = X_base.median(numeric_only=True).to_dict()
        for col in X_base.select_dtypes(include=['object', 'category', 'string']).columns:
            default_values[col] = X_base[col].mode()[0] if not X_base[col].mode().empty else "N/A"

init_app()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model bulunamadı"}), 500

    user_input = request.json
    row_data = default_values.copy()
    row_data.update(user_input)

    df_row = pd.DataFrame([row_data])
    prob = model.predict_proba(df_row)[:, 1][0]
    
    return jsonify({
        "probability": round(float(prob) * 100, 2),
        "is_high_risk": bool(prob >= 0.20)
    })

if __name__ == "__main__":
    print("🚀 EV Battery AI Web Sunucusu Başlatılıyor: http://127.0.0.1:5050")
    app.run(host="0.0.0.0", port=5050, debug=True)
