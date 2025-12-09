import logging
import os
from datetime import datetime

import joblib
import numpy as np
from flask import Flask, jsonify, request

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)


class ModelPredictor:
    def __init__(self, model_path="models/model.joblib"):
        """Initialize model predictor"""
        self.model_path = model_path
        self.model = None
        self.model_info = {}
        self.load_model()

    def load_model(self):
        """Load trained model"""
        try:
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)

                # Load model metrics if available
                metrics_path = self.model_path.replace(".joblib", "_metrics.json")
                if os.path.exists(metrics_path):
                    import json

                    with open(metrics_path, "r") as f:
                        self.model_info = json.load(f)

                logger.info(f"✅ Model loaded from {self.model_path}")
                return True
            else:
                logger.error(f"❌ Model file not found: {self.model_path}")
                return False
        except Exception as e:
            logger.error(f"❌ Error loading model: {str(e)}")
            return False

    def predict(self, features):
        """Make prediction"""
        if self.model is None:
            raise ValueError("Model not loaded")

        # Validate input
        if len(features) != 10:
            raise ValueError(f"Expected 10 features, got {len(features)}")

        # Make prediction
        features_array = np.array(features).reshape(1, -1)
        prediction = self.model.predict(features_array)[0]
        probabilities = self.model.predict_proba(features_array)[0]

        return {
            "prediction": int(prediction),
            "probabilities": probabilities.tolist(),
            "confidence": float(max(probabilities)),
        }


# Initialize predictor
predictor = ModelPredictor()


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify(
        {
            "status": "healthy" if predictor.model is not None else "unhealthy",
            "model_loaded": predictor.model is not None,
            "model_info": predictor.model_info,
            "timestamp": datetime.now().isoformat(),
        }
    )


@app.route("/predict", methods=["POST"])
def predict():
    """Prediction endpoint"""
    try:
        # Get input data
        data = request.get_json()

        if "features" not in data:
            return jsonify({"error": "Missing features in request"}), 400

        # Make prediction
        result = predictor.predict(data["features"])

        # Add metadata
        result["timestamp"] = datetime.now().isoformat()
        result["model_version"] = predictor.model_info.get("timestamp", "unknown")

        logger.info(f"Prediction made: {result['prediction']}")
        return jsonify(result)

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({"error": str(e)}), 400


@app.route("/model/info", methods=["GET"])
def model_info():
    """Get model information"""
    return jsonify(
        {
            "model_path": predictor.model_path,
            "model_loaded": predictor.model is not None,
            "model_metrics": predictor.model_info,
        }
    )


if __name__ == "__main__":
    if predictor.model is None:
        logger.error("❌ Cannot start API - model not loaded")
        exit(1)

    logger.info("🚀 Starting ML prediction API...")
    app.run(host="0.0.0.0", port=5000, debug=False)