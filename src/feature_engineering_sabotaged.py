"""
SABOTAGED Feature Engineering Module - INTENTIONAL BUG FOR TESTING

==============================================================================
⚠️ WARNING: THIS FILE CONTAINS INTENTIONAL BUGS FOR CI/CD DEMONSTRATION ⚠️
==============================================================================

This file is a COPY of feature_engineering.py with an intentional bug
introduced to demonstrate the "Stop the Line" principle in CI/CD.

Purpose:
- To demonstrate that the CI pipeline DETECTS errors
- To show that deployment is BLOCKED when tests fail
- To verify the "Stop the Line" principle works correctly

How to use:
1. Rename original: feature_engineering.py → feature_engineering_backup.py
2. Rename this: feature_engineering_sabotaged.py → feature_engineering.py
3. Commit and push
4. Observe the CI pipeline FAIL
5. Restore: feature_engineering_backup.py → feature_engineering.py

The bugs introduced:
1. SYNTAX ERROR: Missing colon in function definition (line ~60)
2. LOGIC BUG: Wrong modulo operation causes bucket overflow (line ~80)

==============================================================================
"""

import hashlib
from typing import List, Dict, Any, Optional, Union
import numpy as np


class FeatureHasher:
    """
    Feature Hasher for High-Cardinality Categorical Variables.
    """
    
    def __init__(self, n_buckets: int = 1024, hash_algorithm: str = 'md5'):
        if n_buckets <= 0:
            raise ValueError("n_buckets must be a positive integer")
        
        if hash_algorithm not in ['md5', 'sha256', 'sha1']:
            raise ValueError(f"Invalid hash algorithm: {hash_algorithm}. "
                           f"Supported: md5, sha256, sha1")
        
        self.n_buckets = n_buckets
        self.hash_algorithm = hash_algorithm
    
    # =========================================================================
    # ⚠️ INTENTIONAL BUG #1: SYNTAX ERROR - Missing colon
    # =========================================================================
    # The function definition below is missing a colon at the end
    # This will cause a SyntaxError and FAIL the linting stage
    # =========================================================================
    def hash_value(self, value: str) -> int  # ❌ MISSING COLON HERE!
        """
        Hash a single categorical value to a bucket index.
        
        THIS FUNCTION HAS AN INTENTIONAL SYNTAX ERROR!
        """
        if not isinstance(value, str):
            value = str(value)
        
        encoded_value = value.encode('utf-8')
        
        if self.hash_algorithm == 'md5':
            hash_obj = hashlib.md5(encoded_value)
        elif self.hash_algorithm == 'sha256':
            hash_obj = hashlib.sha256(encoded_value)
        else:
            hash_obj = hashlib.sha1(encoded_value)
        
        hash_int = int(hash_obj.hexdigest(), 16)
        
        # =====================================================================
        # ⚠️ INTENTIONAL BUG #2: LOGIC ERROR - Wrong calculation
        # =====================================================================
        # The line below adds n_buckets instead of using modulo
        # This causes bucket indices to be OUT OF RANGE
        # This will cause UNIT TESTS to FAIL
        # =====================================================================
        bucket_index = hash_int + self.n_buckets  # ❌ WRONG! Should be % not +
        
        return bucket_index
    
    def hash_value_signed(self, value: str) -> int:
        bucket = self.hash_value(value)
        sign_hash = self.hash_value(value + "_sign")
        return 1 if sign_hash % 2 == 0 else -1
    
    def transform_single(self, value: str) -> np.ndarray:
        vector = np.zeros(self.n_buckets, dtype=np.float32)
        bucket_index = self.hash_value(value)
        vector[bucket_index] = 1.0  # ❌ This will cause IndexError due to bug above
        return vector
    
    def transform_multiple(self, values: List[str]) -> np.ndarray:
        vector = np.zeros(self.n_buckets, dtype=np.float32)
        for value in values:
            bucket_index = self.hash_value(value)
            sign = self.hash_value_signed(value)
            vector[bucket_index] += sign
        return vector
    
    def transform_with_prefix(self, feature_name: str, value: str) -> int:
        combined = f"{feature_name}:{value}"
        return self.hash_value(combined)


class FeatureEngineer:
    """
    Complete Feature Engineering Pipeline for High-Cardinality Prediction Service.
    """
    
    def __init__(
        self,
        categorical_features: List[str],
        numerical_features: List[str],
        n_hash_buckets: int = 1024
    ):
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        self.hasher = FeatureHasher(n_buckets=n_hash_buckets)
    
    def transform(self, input_data: Dict[str, Any]) -> np.ndarray:
        categorical_vector = np.zeros(self.hasher.n_buckets, dtype=np.float32)
        
        for feature_name in self.categorical_features:
            if feature_name in input_data:
                value = input_data[feature_name]
                bucket = self.hasher.transform_with_prefix(feature_name, str(value))
                categorical_vector[bucket] += 1.0
        
        numerical_vector = []
        for feature_name in self.numerical_features:
            if feature_name in input_data:
                numerical_vector.append(float(input_data[feature_name]))
            else:
                numerical_vector.append(0.0)
        
        numerical_array = np.array(numerical_vector, dtype=np.float32)
        combined_vector = np.concatenate([categorical_vector, numerical_array])
        
        return combined_vector
    
    def get_feature_dimension(self) -> int:
        return self.hasher.n_buckets + len(self.numerical_features)


def compute_hash_bucket(value: str, n_buckets: int = 1024) -> int:
    hasher = FeatureHasher(n_buckets=n_buckets)
    return hasher.hash_value(value)


def normalize_numerical_feature(value: float, min_val: float, max_val: float) -> float:
    if min_val >= max_val:
        raise ValueError("min_val must be less than max_val")
    
    normalized = (value - min_val) / (max_val - min_val)
    return max(0.0, min(1.0, normalized))


def create_feature_interactions(
    feature_a: str,
    feature_b: str,
    n_buckets: int = 1024
) -> int:
    combined = f"{feature_a}__x__{feature_b}"
    return compute_hash_bucket(combined, n_buckets)
