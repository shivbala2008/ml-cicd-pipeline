# src/predict.py
# chaged to black test
import joblib
import os
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

# --- Configuration ---
MODEL_PATH = 'models/model.joblib'
APP = Flask(__name__)
MODEL = None

def load_model():
    """Load the model artifact from the file system."""
    global MODEL
    if os.path.exists(MODEL_PATH):
        try:
            MODEL = joblib.load(MODEL_PATH)
            print(f"INFO: Model loaded successfully from {MODEL_PATH}")
        except Exception as e:
            print(f"ERROR: Failed to load model: {e}")
            MODEL = None
    else:
        print(f"WARNING: Model file not found at {MODEL_PATH}")

load_model()

# --- API Endpoints ---

@APP.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint."""
    return jsonify({"status": "ok", "model_loaded": MODEL is not None}), 200

@APP.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint."""
    if MODEL is None:
        return jsonify({"error": "Model is not loaded"}), 503
    
    try:
        data = request.get_json(force=True)
        features = data.get('features')

        if not features or not isinstance(features, list) or len(features) < 3:
            return jsonify({"error": "Invalid or missing 'features' list"}), 400

        # Create a DataFrame for prediction (matching the training data structure)
        # NOTE: Assumes 3 features from the training script
        input_data = pd.DataFrame([features[:3]], columns=['feature_1', 'feature_2', 'feature_3'])
        
        prediction = MODEL.predict(input_data)[0]
        
        return jsonify({
            "prediction": int(prediction),
            "status": "success"
        }), 200

    except Exception as e:
        return jsonify({"error": str(e), "status": "failure"}), 500

if __name__ == '__main__':
    # Flask defaults to port 5000
    print(f"INFO: Starting Flask API on port 5000...")
    APP.run(host='0.0.0.0', port=5000)