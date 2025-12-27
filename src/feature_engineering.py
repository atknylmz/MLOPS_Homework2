"""
Feature Engineering Module for High-Cardinality Categorical Features

This module implements hash-based feature encoding for high-cardinality categorical
variables. The hashing trick allows us to handle millions of unique categories
without creating an explosion in feature dimensionality.

MLOps Context:
- This is a PURE function with NO external dependencies (no database, no network)
- Deterministic: Same input always produces same output
- Fast: Simple mathematical operations
- Testable: Easy to write isolated unit tests
"""

import hashlib
from typing import List, Dict, Any
import numpy as np


class FeatureHasher:
    """
    Feature Hasher for High-Cardinality Categorical Variables.

    Uses the hashing trick to map categorical values to a fixed number of buckets.
    This is particularly useful for:
    - User IDs (millions of unique users)
    - Product IDs (large product catalogs)
    - IP addresses
    - URLs
    - Any high-cardinality categorical feature

    Attributes:
        n_buckets (int): Number of hash buckets (feature dimensions)
        hash_algorithm (str): Hash algorithm to use (default: 'md5')
    """

    def __init__(self, n_buckets: int = 1024, hash_algorithm: str = 'md5'):
        """
        Initialize the FeatureHasher.

        Args:
            n_buckets: Number of hash buckets. Should be a power of 2 for efficiency.
                      Default is 1024.
            hash_algorithm: Hash algorithm to use. Options: 'md5', 'sha256', 'sha1'

        Raises:
            ValueError: If n_buckets is not positive or hash_algorithm is invalid.
        """
        if n_buckets <= 0:
            raise ValueError("n_buckets must be a positive integer")

        if hash_algorithm not in ['md5', 'sha256', 'sha1']:
            raise ValueError(
                f"Invalid hash algorithm: {hash_algorithm}. "
                f"Supported: md5, sha256, sha1"
            )

        self.n_buckets = n_buckets
        self.hash_algorithm = hash_algorithm

    def hash_value(self, value: str) -> int:
                    this will crash the code
        """
        Hash a single categorical value to a bucket index.

        This is a PURE FUNCTION:
        - No side effects
        - Deterministic (same input -> same output)
        - No external dependencies

        Args:
            value: The categorical value to hash (string)

        Returns:
            int: Bucket index in range [0, n_buckets)

        Example:
            >>> hasher = FeatureHasher(n_buckets=1024)
            >>> hasher.hash_value("user_12345")
            742  # Always returns the same value for "user_12345"
        """
        if not isinstance(value, str):
            value = str(value)

        # Encode string to bytes
        encoded_value = value.encode('utf-8')

        # Create hash object based on algorithm
        if self.hash_algorithm == 'md5':
            hash_obj = hashlib.md5(encoded_value)
        elif self.hash_algorithm == 'sha256':
            hash_obj = hashlib.sha256(encoded_value)
        else:  # sha1
            hash_obj = hashlib.sha1(encoded_value)

        # Convert hash to integer and take modulo n_buckets
        hash_int = int(hash_obj.hexdigest(), 16)
        bucket_index = hash_int % self.n_buckets

        return bucket_index

    def hash_value_signed(self, value: str) -> int:
        """
        Hash a value and return a signed indicator (+1 or -1).

        This is used for signed feature hashing which can reduce
        collision effects in some ML models.

        Args:
            value: The categorical value to hash

        Returns:
            int: Either +1 or -1
        """
        _ = self.hash_value(value)  # Ensure value is hashable
        # Use second hash or different computation for sign
        sign_hash = self.hash_value(value + "_sign")
        return 1 if sign_hash % 2 == 0 else -1

    def transform_single(self, value: str) -> np.ndarray:
        """
        Transform a single categorical value to a sparse vector.

        Creates a one-hot-like encoding in the hashed space.

        Args:
            value: The categorical value to transform

        Returns:
            np.ndarray: Sparse vector of shape (n_buckets,)
        """
        vector = np.zeros(self.n_buckets, dtype=np.float32)
        bucket_index = self.hash_value(value)
        vector[bucket_index] = 1.0
        return vector

    def transform_multiple(self, values: List[str]) -> np.ndarray:
        """
        Transform multiple categorical values to a combined feature vector.

        This handles the case where you have multiple categorical features
        that need to be hashed together (e.g., user_id + product_id + category).

        Args:
            values: List of categorical values to transform

        Returns:
            np.ndarray: Combined feature vector of shape (n_buckets,)
        """
        vector = np.zeros(self.n_buckets, dtype=np.float32)
        for value in values:
            bucket_index = self.hash_value(value)
            sign = self.hash_value_signed(value)
            vector[bucket_index] += sign
        return vector

    def transform_with_prefix(self, feature_name: str, value: str) -> int:
        """
        Hash a value with a feature name prefix.

        This prevents collisions between different features that might
        have the same values (e.g., user_id=123 vs product_id=123).

        Args:
            feature_name: Name of the feature (e.g., "user_id")
            value: The categorical value

        Returns:
            int: Bucket index
        """
        combined = f"{feature_name}:{value}"
        return self.hash_value(combined)


class FeatureEngineer:
    """
    Complete Feature Engineering Pipeline for High-Cardinality Prediction Service.

    This class orchestrates the transformation of raw input features into
    a format suitable for ML model inference.
    """

    def __init__(
        self,
        categorical_features: List[str],
        numerical_features: List[str],
        n_hash_buckets: int = 1024
    ):
        """
        Initialize the Feature Engineer.

        Args:
            categorical_features: List of categorical feature names to hash
            numerical_features: List of numerical feature names to pass through
            n_hash_buckets: Number of hash buckets for categorical features
        """
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        self.hasher = FeatureHasher(n_buckets=n_hash_buckets)

    def transform(self, input_data: Dict[str, Any]) -> np.ndarray:
        """
        Transform raw input data into a feature vector.

        Args:
            input_data: Dictionary with feature names as keys

        Returns:
            np.ndarray: Combined feature vector
        """
        # Initialize feature vector for categorical features
        categorical_vector = np.zeros(self.hasher.n_buckets, dtype=np.float32)

        # Hash categorical features
        for feature_name in self.categorical_features:
            if feature_name in input_data:
                value = input_data[feature_name]
                bucket = self.hasher.transform_with_prefix(feature_name, str(value))
                categorical_vector[bucket] += 1.0

        # Extract numerical features
        numerical_vector = []
        for feature_name in self.numerical_features:
            if feature_name in input_data:
                numerical_vector.append(float(input_data[feature_name]))
            else:
                numerical_vector.append(0.0)

        numerical_array = np.array(numerical_vector, dtype=np.float32)

        # Combine categorical and numerical features
        combined_vector = np.concatenate([categorical_vector, numerical_array])

        return combined_vector

    def get_feature_dimension(self) -> int:
        """
        Get the total dimension of the feature vector.

        Returns:
            int: Total number of features
        """
        return self.hasher.n_buckets + len(self.numerical_features)


def compute_hash_bucket(value: str, n_buckets: int = 1024) -> int:
    """
    Standalone function to compute hash bucket for a value.

    This is a convenience function that can be used without
    instantiating the FeatureHasher class.

    This is a PURE FUNCTION - perfect for unit testing:
    - Deterministic
    - No side effects
    - No external dependencies

    Args:
        value: String value to hash
        n_buckets: Number of buckets

    Returns:
        int: Bucket index in range [0, n_buckets)

    Example:
        >>> compute_hash_bucket("test_value", 1024)
        742  # Deterministic result
    """
    hasher = FeatureHasher(n_buckets=n_buckets)
    return hasher.hash_value(value)


def normalize_numerical_feature(value: float, min_val: float, max_val: float) -> float:
    """
    Normalize a numerical feature to [0, 1] range.

    PURE FUNCTION for unit testing.

    Args:
        value: The value to normalize
        min_val: Minimum value in the feature range
        max_val: Maximum value in the feature range

    Returns:
        float: Normalized value in [0, 1]

    Raises:
        ValueError: If min_val >= max_val
    """
    if min_val >= max_val:
        raise ValueError("min_val must be less than max_val")

    normalized = (value - min_val) / (max_val - min_val)
    return max(0.0, min(1.0, normalized))


def create_feature_interactions(
    feature_a: str,
    feature_b: str,
    n_buckets: int = 1024
) -> int:
    """
    Create interaction feature by hashing combined features.

    PURE FUNCTION for unit testing.

    Args:
        feature_a: First feature value
        feature_b: Second feature value
        n_buckets: Number of hash buckets

    Returns:
        int: Bucket index for the interaction
    """
    combined = f"{feature_a}__x__{feature_b}"
    return compute_hash_bucket(combined, n_buckets)
