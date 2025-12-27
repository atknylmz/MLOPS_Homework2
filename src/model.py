"""
Model Module for High-Cardinality Prediction Service

This module handles model loading and prediction logic.
It provides a clean interface between the API layer and the ML model.
"""

import os
import joblib
import numpy as np
from typing import Optional, List, Dict, Any
from pathlib import Path


class PredictionModel:
    """
    Wrapper class for the ML prediction model.

    Handles model loading from disk and provides prediction interface.
    """

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the prediction model.

        Args:
            model_path: Path to the serialized model file (.joblib)
                       If None, uses a mock model for testing.
        """
        self.model = None
        self.model_path = model_path
        self.is_loaded = False
        self.feature_dimension = 1028  # 1024 hash buckets + 4 numerical features

    def load(self) -> bool:
        """
        Load the model from disk.

        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        if self.model_path and os.path.exists(self.model_path):
            try:
                self.model = joblib.load(self.model_path)
                self.is_loaded = True
                return True
            except Exception as e:
                print(f"Error loading model: {e}")
                return False
        else:
            # Use mock model if no model file exists
            self._create_mock_model()
            self.is_loaded = True
            return True

    def _create_mock_model(self):
        """
        Create a simple mock model for testing purposes.

        This creates a simple linear model that can be used when
        no trained model is available.
        """
        from sklearn.linear_model import LogisticRegression

        # Create a simple mock model
        mock_model = LogisticRegression()

        # Fit with dummy data
        n_samples = 100
        X_dummy = np.random.randn(n_samples, self.feature_dimension)
        y_dummy = np.random.randint(0, 2, n_samples)

        mock_model.fit(X_dummy, y_dummy)
        self.model = mock_model

    def predict(self, features: np.ndarray) -> Dict[str, Any]:
        """
        Make a prediction using the loaded model.

        Args:
            features: Feature vector of shape (feature_dimension,) or (n_samples, feature_dimension)

        Returns:
            Dict containing prediction and confidence

        Raises:
            RuntimeError: If model is not loaded
        """
        if not self.is_loaded or self.model is None:
            raise RuntimeError("Model is not loaded. Call load() first.")

        # Ensure features is 2D
        if len(features.shape) == 1:
            features = features.reshape(1, -1)

        # Pad or truncate features to match expected dimension
        if features.shape[1] < self.feature_dimension:
            padding = np.zeros((features.shape[0], self.feature_dimension - features.shape[1]))
            features = np.concatenate([features, padding], axis=1)
        elif features.shape[1] > self.feature_dimension:
            features = features[:, :self.feature_dimension]

        # Make prediction
        prediction = self.model.predict(features)

        # Get probability if available
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(features)
            confidence = float(np.max(probabilities[0]))
        else:
            confidence = 1.0

        return {
            "prediction": int(prediction[0]),
            "confidence": confidence,
            "model_version": "1.0.0"
        }

    def predict_batch(self, features_list: List[np.ndarray]) -> List[Dict[str, Any]]:
        """
        Make predictions for a batch of feature vectors.

        Args:
            features_list: List of feature vectors

        Returns:
            List of prediction results
        """
        results = []
        for features in features_list:
            result = self.predict(features)
            results.append(result)
        return results

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.

        Returns:
            Dict with model metadata
        """
        return {
            "is_loaded": self.is_loaded,
            "model_path": self.model_path,
            "feature_dimension": self.feature_dimension,
            "model_type": type(self.model).__name__ if self.model else None
        }


class ModelRegistry:
    """
    Simple model registry for managing multiple model versions.
    """

    def __init__(self, models_dir: str = "./models"):
        """
        Initialize the model registry.

        Args:
            models_dir: Directory containing model files
        """
        self.models_dir = Path(models_dir)
        self.models: Dict[str, PredictionModel] = {}

    def list_models(self) -> List[str]:
        """
        List available model files.

        Returns:
            List of model file names
        """
        if not self.models_dir.exists():
            return []

        return [f.name for f in self.models_dir.glob("*.joblib")]

    def load_model(self, model_name: str) -> Optional[PredictionModel]:
        """
        Load a specific model by name.

        Args:
            model_name: Name of the model file

        Returns:
            Loaded PredictionModel or None if not found
        """
        model_path = self.models_dir / model_name

        if not model_path.exists():
            return None

        model = PredictionModel(str(model_path))
        if model.load():
            self.models[model_name] = model
            return model

        return None

    def get_default_model(self) -> PredictionModel:
        """
        Get the default model (creates mock if none available).

        Returns:
            PredictionModel instance
        """
        if "default" not in self.models:
            model = PredictionModel()
            model.load()
            self.models["default"] = model

        return self.models["default"]
