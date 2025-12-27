"""
Component/Integration Tests for Prediction Service

==============================================================================
WHY THESE ARE COMPONENT/INTEGRATION TESTS (NOT UNIT TESTS):
==============================================================================

These tests qualify as COMPONENT/INTEGRATION TESTS because they:

1. ARE SLOWER THAN UNIT TESTS:
   - Involve multiple components working together
   - May include file system access or mock database operations
   - Test the interaction between modules

2. TEST COMPONENT INTERACTIONS:
   - Feature Engineering + Model Prediction working together
   - API endpoints with actual request/response flow
   - Data consistency across the pipeline

3. MAY HAVE EXTERNAL DEPENDENCIES:
   - File system (loading model files)
   - HTTP server (TestClient for API testing)
   - Mock databases (if applicable)

4. VERIFY END-TO-END BEHAVIOR:
   - From input to output through multiple layers
   - Integration between model serving logic and data sources

MLOps Context:
- These tests verify the complete prediction pipeline works
- Ensure feature engineering and model prediction are integrated correctly
- Must pass before deployment but after unit tests
==============================================================================
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path

# FastAPI test client
from fastapi.testclient import TestClient

# Import application components
from src.api import app
from src.feature_engineering import FeatureEngineer, FeatureHasher
from src.model import PredictionModel, ModelRegistry


# ============================================================================
# Test Client Setup
# ============================================================================

@pytest.fixture
def client():
    """
    Create a test client for the FastAPI application.

    This is an INTEGRATION test fixture - it sets up the full application.
    """
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def temp_model_file():
    """
    Create a temporary model file for testing.

    This tests file system interaction (not a unit test).
    """
    import joblib
    from sklearn.linear_model import LogisticRegression

    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = Path(temp_dir) / "test_model.joblib"

        # Create and save a simple model
        model = LogisticRegression()
        X = np.random.randn(100, 1028)
        y = np.random.randint(0, 2, 100)
        model.fit(X, y)

        joblib.dump(model, model_path)

        yield str(model_path)


# ============================================================================
# API Integration Tests
# ============================================================================

class TestAPIIntegration:
    """
    Integration tests for the FastAPI prediction service.

    These tests verify the complete request/response cycle through the API.
    """

    def test_health_endpoint_returns_200(self, client):
        """
        Test that the health endpoint returns 200 OK.

        This is the same check the smoke test will perform.
        """
        response = client.get("/health")

        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert "model_loaded" in data
        assert "version" in data

    def test_root_endpoint_returns_service_info(self, client):
        """
        Test that the root endpoint returns service information.
        """
        response = client.get("/")

        assert response.status_code == 200

        data = response.json()
        assert "service" in data
        assert "version" in data

    def test_predict_endpoint_with_valid_input(self, client):
        """
        Test prediction endpoint with valid input data.

        This is a COMPONENT TEST because it tests:
        - API request parsing
        - Feature engineering transformation
        - Model prediction
        - Response serialization
        """
        request_data = {
            "user_id": "user_12345",
            "product_id": "prod_67890",
            "category": "electronics",
            "price": 299.99,
            "quantity": 1,
            "user_age": 35,
            "session_duration": 120.5
        }

        response = client.post("/predict", json=request_data)

        assert response.status_code == 200

        data = response.json()
        assert "prediction" in data
        assert "confidence" in data
        assert "model_version" in data

        # Prediction should be 0 or 1
        assert data["prediction"] in [0, 1]

        # Confidence should be between 0 and 1
        assert 0 <= data["confidence"] <= 1

    def test_predict_endpoint_is_deterministic(self, client):
        """
        Test that same input produces same prediction.

        This verifies the integration between feature engineering and model.
        """
        request_data = {
            "user_id": "deterministic_user",
            "product_id": "deterministic_product",
            "category": "test_category",
            "price": 100.0,
            "quantity": 1,
            "user_age": 30,
            "session_duration": 60.0
        }

        response_1 = client.post("/predict", json=request_data)
        response_2 = client.post("/predict", json=request_data)

        assert response_1.status_code == 200
        assert response_2.status_code == 200

        # Same input should produce same output
        assert response_1.json() == response_2.json()

    def test_predict_endpoint_with_missing_optional_fields(self, client):
        """
        Test prediction with only required fields.
        """
        request_data = {
            "user_id": "user_123",
            "product_id": "prod_456",
            "category": "books",
            "price": 19.99,
            "quantity": 2
            # user_age and session_duration are optional
        }

        response = client.post("/predict", json=request_data)

        assert response.status_code == 200

    def test_predict_endpoint_with_invalid_input(self, client):
        """
        Test that invalid input returns 422 validation error.
        """
        # Missing required fields
        request_data = {
            "user_id": "user_123"
            # Missing product_id, category, price, quantity
        }

        response = client.post("/predict", json=request_data)

        assert response.status_code == 422  # Validation error

    def test_predict_endpoint_with_invalid_price(self, client):
        """
        Test that negative price returns validation error.
        """
        request_data = {
            "user_id": "user_123",
            "product_id": "prod_456",
            "category": "test",
            "price": -10.0,  # Invalid: negative price
            "quantity": 1
        }

        response = client.post("/predict", json=request_data)

        assert response.status_code == 422

    def test_batch_predict_endpoint(self, client):
        """
        Test batch prediction endpoint.
        """
        request_data = {
            "requests": [
                {
                    "user_id": "user_1",
                    "product_id": "prod_1",
                    "category": "cat_1",
                    "price": 10.0,
                    "quantity": 1
                },
                {
                    "user_id": "user_2",
                    "product_id": "prod_2",
                    "category": "cat_2",
                    "price": 20.0,
                    "quantity": 2
                }
            ]
        }

        response = client.post("/predict/batch", json=request_data)

        assert response.status_code == 200

        data = response.json()
        assert "predictions" in data
        assert "batch_size" in data
        assert data["batch_size"] == 2
        assert len(data["predictions"]) == 2

    def test_model_info_endpoint(self, client):
        """
        Test model info endpoint.
        """
        response = client.get("/model/info")

        assert response.status_code == 200

        data = response.json()
        assert "is_loaded" in data
        assert "feature_dimension" in data

    def test_hash_feature_endpoint(self, client):
        """
        Test the hash feature utility endpoint.
        """
        response = client.get("/features/hash", params={
            "value": "test_value",
            "n_buckets": 1024
        })

        assert response.status_code == 200

        data = response.json()
        assert data["value"] == "test_value"
        assert 0 <= data["bucket"] < 1024


# ============================================================================
# Model + Feature Engineering Integration Tests
# ============================================================================

class TestModelFeatureIntegration:
    """
    Integration tests for Model and Feature Engineering working together.

    These verify that the complete prediction pipeline functions correctly.
    """

    def test_feature_engineering_produces_correct_dimension(self):
        """
        Test that feature engineering output matches model input dimension.
        """
        engineer = FeatureEngineer(
            categorical_features=["user_id", "product_id", "category"],
            numerical_features=["price", "quantity", "user_age", "session_duration"],
            n_hash_buckets=1024
        )

        input_data = {
            "user_id": "user_123",
            "product_id": "prod_456",
            "category": "electronics",
            "price": 99.99,
            "quantity": 1,
            "user_age": 25,
            "session_duration": 120.0
        }

        features = engineer.transform(input_data)

        # Model expects 1028 features (1024 hash + 4 numerical)
        model = PredictionModel()
        model.load()

        # Features should work with the model
        result = model.predict(features)

        assert "prediction" in result
        assert "confidence" in result

    def test_end_to_end_prediction_pipeline(self):
        """
        Test the complete prediction pipeline from raw input to prediction.

        This is the most comprehensive integration test.
        """
        # Setup components
        engineer = FeatureEngineer(
            categorical_features=["user_id", "product_id", "category"],
            numerical_features=["price", "quantity", "user_age", "session_duration"],
            n_hash_buckets=1024
        )

        model = PredictionModel()
        model.load()

        # Test data
        test_inputs = [
            {
                "user_id": "user_001",
                "product_id": "prod_001",
                "category": "electronics",
                "price": 299.99,
                "quantity": 1,
                "user_age": 30,
                "session_duration": 150.0
            },
            {
                "user_id": "user_002",
                "product_id": "prod_002",
                "category": "clothing",
                "price": 49.99,
                "quantity": 3,
                "user_age": 25,
                "session_duration": 80.0
            },
            {
                "user_id": "high_cardinality_user_12345678",
                "product_id": "rare_product_xyz",
                "category": "misc",
                "price": 9.99,
                "quantity": 10,
                "user_age": 40,
                "session_duration": 300.0
            }
        ]

        for input_data in test_inputs:
            # Transform features
            features = engineer.transform(input_data)

            # Make prediction
            result = model.predict(features)

            # Verify result structure
            assert "prediction" in result
            assert "confidence" in result
            assert "model_version" in result

            # Verify result values
            assert result["prediction"] in [0, 1]
            assert 0 <= result["confidence"] <= 1

    def test_batch_predictions_consistency(self):
        """
        Test that batch and individual predictions are consistent.
        """
        engineer = FeatureEngineer(
            categorical_features=["user_id"],
            numerical_features=["price"],
            n_hash_buckets=512
        )

        model = PredictionModel()
        model.load()

        input_data = {"user_id": "batch_test_user", "price": 100.0}

        # Single prediction
        features = engineer.transform(input_data)
        single_result = model.predict(features)

        # Batch prediction (single item)
        batch_results = model.predict_batch([features])

        # Should be identical
        assert single_result["prediction"] == batch_results[0]["prediction"]


# ============================================================================
# Model Loading Integration Tests
# ============================================================================

class TestModelLoading:
    """
    Integration tests for model loading from file system.
    """

    def test_load_model_from_file(self, temp_model_file):
        """
        Test loading a model from a file.

        This is a FILE SYSTEM integration test.
        """
        model = PredictionModel(temp_model_file)

        assert model.load() is True
        assert model.is_loaded is True

    def test_model_prediction_after_file_load(self, temp_model_file):
        """
        Test that model works correctly after loading from file.
        """
        model = PredictionModel(temp_model_file)
        model.load()

        # Create dummy features
        features = np.random.randn(1028)

        result = model.predict(features)

        assert "prediction" in result
        assert result["prediction"] in [0, 1]

    def test_mock_model_creation(self):
        """
        Test that mock model is created when no file exists.
        """
        model = PredictionModel(model_path=None)

        assert model.load() is True
        assert model.is_loaded is True
        assert model.model is not None


# ============================================================================
# Data Consistency Tests
# ============================================================================

class TestDataConsistency:
    """
    Integration tests for data consistency across the pipeline.
    """

    def test_feature_hash_consistency_across_calls(self):
        """
        Test that feature hashing is consistent across multiple service calls.

        This is critical for ML system correctness.
        """
        hasher = FeatureHasher(n_buckets=1024)

        # Simulate multiple "requests" with same value
        user_id = "consistent_user_id_12345"

        buckets = [hasher.hash_value(user_id) for _ in range(1000)]

        # All should be identical
        assert len(set(buckets)) == 1

    def test_numerical_feature_ordering_consistency(self):
        """
        Test that numerical features maintain consistent ordering.
        """
        engineer = FeatureEngineer(
            categorical_features=["cat"],
            numerical_features=["a", "b", "c"],
            n_hash_buckets=100
        )

        input_1 = {"cat": "x", "a": 1.0, "b": 2.0, "c": 3.0}
        input_2 = {"cat": "x", "a": 1.0, "b": 2.0, "c": 3.0}

        features_1 = engineer.transform(input_1)
        features_2 = engineer.transform(input_2)

        np.testing.assert_array_equal(features_1, features_2)


# ============================================================================
# Error Handling Integration Tests
# ============================================================================

class TestErrorHandling:
    """
    Integration tests for error handling across components.
    """

    def test_model_not_loaded_error(self):
        """
        Test that calling predict without loading raises error.
        """
        model = PredictionModel()
        # Don't call load()

        features = np.random.randn(1028)

        with pytest.raises(RuntimeError):
            model.predict(features)

    def test_api_handles_malformed_json(self, client):
        """
        Test that API handles malformed JSON gracefully.
        """
        response = client.post(
            "/predict",
            content="this is not json",
            headers={"Content-Type": "application/json"}
        )

        assert response.status_code == 422


# ============================================================================
# Model Registry Integration Tests
# ============================================================================

class TestModelRegistry:
    """
    Integration tests for the model registry.
    """

    def test_list_models_empty_directory(self):
        """
        Test listing models in non-existent directory.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            registry = ModelRegistry(models_dir=temp_dir)

            # Directory exists but is empty
            models = registry.list_models()
            assert models == []

    def test_get_default_model(self):
        """
        Test getting the default model.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            registry = ModelRegistry(models_dir=temp_dir)

            model = registry.get_default_model()

            assert model is not None
            assert model.is_loaded is True
