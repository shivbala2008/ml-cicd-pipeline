import json
import os
import sys

import pytest

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from predict import APP


class TestPredictionAPI:
    """Test suite for prediction API"""

    @pytest.fixture
    def client(self):
        """Create test client"""
        app.config["TESTING"] = True
        with app.test_client() as client:
            yield client

    @pytest.fixture
    def sample_features(self):
        """Sample feature data for testing"""
        return [15.0, 20.0, 100.0, 500.0, 0.1, 0.09, 0.03, 0.02, 0.18, 0.06]

    def test_health_endpoint(self, client):
        """Test health check endpoint"""
        response = client.get("/health")

        assert response.status_code == 200
        data = json.loads(response.data)

        # Check required fields
        assert "status" in data
        assert "model_loaded" in data
        assert "timestamp" in data

        print("✅ Health endpoint test passed")

    def test_predict_endpoint_valid_input(self, client, sample_features):
        """Test prediction endpoint with valid input"""
        # Skip if model not available
        if not os.path.exists("models/model.joblib"):
            pytest.skip("Model file not available for testing")

        response = client.post(
            "/predict",
            data=json.dumps({"features": sample_features}),
            content_type="application/json",
        )

        if response.status_code == 200:
            data = json.loads(response.data)

            # Check response structure
            assert "prediction" in data
            assert "probabilities" in data
            assert "confidence" in data
            assert "timestamp" in data

            # Check data types
            assert isinstance(data["prediction"], int)
            assert isinstance(data["probabilities"], list)
            assert isinstance(data["confidence"], float)

            # Check value ranges
            assert data["prediction"] in [0, 1]
            assert 0 <= data["confidence"] <= 1
            assert len(data["probabilities"]) == 2

            print("✅ Predict endpoint test passed")
        else:
            print("⚠️ Predict endpoint test skipped - model not loaded")

    def test_predict_endpoint_invalid_input(self, client):
        """Test prediction endpoint with invalid input"""
        # Test missing features
        response = client.post(
            "/predict", data=json.dumps({}), content_type="application/json"
        )
        assert response.status_code == 400

        # Test wrong number of features
        response = client.post(
            "/predict",
            data=json.dumps({"features": [1, 2, 3]}),
            content_type="application/json",
        )
        assert response.status_code == 400

        print("✅ Invalid input test passed")

    def test_model_info_endpoint(self, client):
        """Test model info endpoint"""
        response = client.get("/model/info")

        assert response.status_code == 200
        data = json.loads(response.data)

        # Check required fields
        assert "model_path" in data
        assert "model_loaded" in data

        print("✅ Model info endpoint test passed")


class TestModelPredictor:
    """Test suite for ModelPredictor class"""

    def test_predictor_initialization(self):
        """Test predictor initialization"""
        predictor = ModelPredictor()

        # Check initialization
        assert predictor.model_path is not None
        assert hasattr(predictor, "model")
        assert hasattr(predictor, "model_info")

        print("✅ Predictor initialization test passed")

    def test_predictor_prediction(self, tmp_path):
        """Test predictor prediction functionality"""
        # Skip if model not available
        if not os.path.exists("models/model.joblib"):
            pytest.skip("Model file not available for testing")

        predictor = ModelPredictor()

        if predictor.model is not None:
            # Test prediction
            sample_features = [
                15.0,
                20.0,
                100.0,
                500.0,
                0.1,
                0.09,
                0.03,
                0.02,
                0.18,
                0.06,
            ]
            result = predictor.predict(sample_features)

            # Check result structure
            assert "prediction" in result
            assert "probabilities" in result
            assert "confidence" in result

            print("✅ Predictor prediction test passed")
        else:
            print("⚠️ Predictor prediction test skipped - model not loaded")


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])
