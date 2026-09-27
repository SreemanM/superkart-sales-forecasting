
from flask import Flask, request, jsonify
import pandas as pd
import joblib

superkart_api = Flask(__name__)
model = joblib.load("superkart_model.joblib")

EXPECTED_COLUMNS = [
    "Product_Weight",
    "Product_Sugar_Content",
    "Product_Allocated_Area",
    "Product_MRP",
    "Store_Size",
    "Store_Location_City_Type",
    "Store_Type",
    "Product_Id_char",
    "Store_Age_Years",
    "Product_Type_Category",
]

def validate_columns(df):
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df[EXPECTED_COLUMNS]

@superkart_api.get("/health")
def health():
    return jsonify({"status": "ok"}), 200

@superkart_api.post("/v1/predict")
def predict():
    try:
        payload = request.get_json(force=True)
        frame = pd.DataFrame([payload])
        frame = validate_columns(frame)
        prediction = float(model.predict(frame)[0])
        return jsonify({"predicted_sales": round(prediction, 2)}), 200
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400

@superkart_api.post("/v1/predictbatch")
def predict_batch():
    try:
        if "file" not in request.files:
            return jsonify({"error": "Upload a CSV file using form field 'file'."}), 400

        frame = pd.read_csv(request.files["file"])
        frame = validate_columns(frame)
        predictions = model.predict(frame)

        result = {
            str(i): round(float(pred), 2)
            for i, pred in enumerate(predictions)
        }
        return jsonify(result), 200
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400

if __name__ == "__main__":
    superkart_api.run(host="0.0.0.0", port=7860)
