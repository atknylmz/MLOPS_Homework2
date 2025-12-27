#!/usr/bin/env python3
"""
Smoke Test Script for High-Cardinality Prediction Service

==============================================================================
WHY THIS IS A SMOKE TEST (END-TO-END DEPLOYMENT TEST):
==============================================================================

This script qualifies as a SMOKE TEST / DEPLOYMENT TEST because it:

1. IS END-TO-END:
   - Tests the COMPLETE deployed system, not individual components
   - Sends actual HTTP requests to the running container
   - Verifies the entire stack works (API -> Feature Engineering -> Model)

2. TESTS DEPLOYMENT HEALTH:
   - Verifies the service starts correctly
   - Confirms the service responds to requests
   - Validates HTTP 200 OK response

3. IS A CRITICAL DEPLOYMENT GATE:
   - If this test fails, deployment MUST be blocked
   - It's the final verification before production release
   - Ensures the container is actually usable

4. TESTS FROM USER PERSPECTIVE:
   - Makes the same type of request a real user would make
   - Doesn't have special access to internals
   - Uses public API endpoints

MLOps Context:
- This is the CRITICAL "Deployment Test" from the assignment
- Failure of this test STOPS the deployment pipeline
- Verifies the container is deployable and functional
==============================================================================

Usage:
    # Default: test against localhost:8000
    python scripts/smoke_test.py

    # Custom host/port
    python scripts/smoke_test.py --host localhost --port 8080

    # With timeout
    python scripts/smoke_test.py --timeout 30 --retries 5
"""

import argparse
import sys
import time
import json
from typing import Optional, Dict, Any

try:
    import requests
except ImportError:
    print("ERROR: 'requests' library is required. Install with: pip install requests")
    sys.exit(1)


class SmokeTestRunner:
    """
    Smoke test runner for the prediction service.
    
    Performs end-to-end tests to verify the deployed service is healthy.
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 8000,
        timeout: int = 10,
        retries: int = 3,
        retry_delay: int = 5
    ):
        """
        Initialize the smoke test runner.
        
        Args:
            host: Service host
            port: Service port
            timeout: Request timeout in seconds
            retries: Number of retries for failed requests
            retry_delay: Delay between retries in seconds
        """
        self.base_url = f"http://{host}:{port}"
        self.timeout = timeout
        self.retries = retries
        self.retry_delay = retry_delay
        self.results: Dict[str, Any] = {}
    
    def _make_request(
        self,
        method: str,
        endpoint: str,
        json_data: Optional[Dict] = None
    ) -> Optional[requests.Response]:
        """
        Make an HTTP request with retries.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint
            json_data: JSON body for POST requests
        
        Returns:
            Response object or None if all retries failed
        """
        url = f"{self.base_url}{endpoint}"
        
        for attempt in range(self.retries):
            try:
                if method.upper() == "GET":
                    response = requests.get(url, timeout=self.timeout)
                elif method.upper() == "POST":
                    response = requests.post(url, json=json_data, timeout=self.timeout)
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")
                
                return response
                
            except requests.exceptions.ConnectionError as e:
                print(f"  [Attempt {attempt + 1}/{self.retries}] Connection failed: {e}")
            except requests.exceptions.Timeout as e:
                print(f"  [Attempt {attempt + 1}/{self.retries}] Request timed out: {e}")
            except Exception as e:
                print(f"  [Attempt {attempt + 1}/{self.retries}] Unexpected error: {e}")
            
            if attempt < self.retries - 1:
                print(f"  Retrying in {self.retry_delay} seconds...")
                time.sleep(self.retry_delay)
        
        return None
    
    def test_health_endpoint(self) -> bool:
        """
        Test 1: Health Check Endpoint
        
        Verifies:
        - Service is running
        - Health endpoint returns 200 OK
        - Response contains expected fields
        
        This is the PRIMARY smoke test.
        """
        print("\n" + "=" * 60)
        print("TEST 1: Health Check Endpoint")
        print("=" * 60)
        
        print(f"Sending GET request to {self.base_url}/health ...")
        
        response = self._make_request("GET", "/health")
        
        if response is None:
            print("❌ FAILED: Could not connect to service")
            self.results["health"] = {"passed": False, "error": "Connection failed"}
            return False
        
        print(f"Response Status Code: {response.status_code}")
        
        if response.status_code != 200:
            print(f"❌ FAILED: Expected 200 OK, got {response.status_code}")
            self.results["health"] = {
                "passed": False,
                "status_code": response.status_code,
                "error": "Non-200 status code"
            }
            return False
        
        try:
            data = response.json()
            print(f"Response Body: {json.dumps(data, indent=2)}")
            
            # Verify response structure
            required_fields = ["status", "model_loaded", "version"]
            missing_fields = [f for f in required_fields if f not in data]
            
            if missing_fields:
                print(f"❌ FAILED: Missing required fields: {missing_fields}")
                self.results["health"] = {
                    "passed": False,
                    "error": f"Missing fields: {missing_fields}"
                }
                return False
            
            if data["status"] != "healthy":
                print(f"❌ FAILED: Service status is not 'healthy': {data['status']}")
                self.results["health"] = {"passed": False, "error": "Unhealthy status"}
                return False
            
            print("✅ PASSED: Health check successful")
            self.results["health"] = {
                "passed": True,
                "status_code": 200,
                "response": data
            }
            return True
            
        except json.JSONDecodeError as e:
            print(f"❌ FAILED: Invalid JSON response: {e}")
            self.results["health"] = {"passed": False, "error": "Invalid JSON"}
            return False
    
    def test_prediction_endpoint(self) -> bool:
        """
        Test 2: Prediction Endpoint
        
        Verifies:
        - Prediction endpoint accepts valid input
        - Returns 200 OK
        - Response contains prediction result
        
        This tests the core ML functionality.
        """
        print("\n" + "=" * 60)
        print("TEST 2: Prediction Endpoint")
        print("=" * 60)
        
        # Sample prediction request
        test_payload = {
            "user_id": "smoke_test_user_12345",
            "product_id": "smoke_test_product_67890",
            "category": "test_category",
            "price": 99.99,
            "quantity": 1,
            "user_age": 30,
            "session_duration": 120.0
        }
        
        print(f"Sending POST request to {self.base_url}/predict ...")
        print(f"Request Body: {json.dumps(test_payload, indent=2)}")
        
        response = self._make_request("POST", "/predict", test_payload)
        
        if response is None:
            print("❌ FAILED: Could not connect to service")
            self.results["prediction"] = {"passed": False, "error": "Connection failed"}
            return False
        
        print(f"Response Status Code: {response.status_code}")
        
        if response.status_code != 200:
            print(f"❌ FAILED: Expected 200 OK, got {response.status_code}")
            self.results["prediction"] = {
                "passed": False,
                "status_code": response.status_code,
                "error": "Non-200 status code"
            }
            return False
        
        try:
            data = response.json()
            print(f"Response Body: {json.dumps(data, indent=2)}")
            
            # Verify response structure
            required_fields = ["prediction", "confidence", "model_version"]
            missing_fields = [f for f in required_fields if f not in data]
            
            if missing_fields:
                print(f"❌ FAILED: Missing required fields: {missing_fields}")
                self.results["prediction"] = {
                    "passed": False,
                    "error": f"Missing fields: {missing_fields}"
                }
                return False
            
            # Verify prediction value
            if data["prediction"] not in [0, 1]:
                print(f"❌ FAILED: Invalid prediction value: {data['prediction']}")
                self.results["prediction"] = {
                    "passed": False,
                    "error": "Invalid prediction value"
                }
                return False
            
            # Verify confidence is in valid range
            if not (0 <= data["confidence"] <= 1):
                print(f"❌ FAILED: Confidence out of range: {data['confidence']}")
                self.results["prediction"] = {
                    "passed": False,
                    "error": "Confidence out of range"
                }
                return False
            
            print("✅ PASSED: Prediction endpoint successful")
            self.results["prediction"] = {
                "passed": True,
                "status_code": 200,
                "response": data
            }
            return True
            
        except json.JSONDecodeError as e:
            print(f"❌ FAILED: Invalid JSON response: {e}")
            self.results["prediction"] = {"passed": False, "error": "Invalid JSON"}
            return False
    
    def test_root_endpoint(self) -> bool:
        """
        Test 3: Root Endpoint
        
        Verifies basic service information is available.
        """
        print("\n" + "=" * 60)
        print("TEST 3: Root Endpoint")
        print("=" * 60)
        
        print(f"Sending GET request to {self.base_url}/ ...")
        
        response = self._make_request("GET", "/")
        
        if response is None:
            print("❌ FAILED: Could not connect to service")
            self.results["root"] = {"passed": False, "error": "Connection failed"}
            return False
        
        print(f"Response Status Code: {response.status_code}")
        
        if response.status_code != 200:
            print(f"❌ FAILED: Expected 200 OK, got {response.status_code}")
            self.results["root"] = {"passed": False, "status_code": response.status_code}
            return False
        
        print(f"Response Body: {response.text}")
        print("✅ PASSED: Root endpoint successful")
        self.results["root"] = {"passed": True, "status_code": 200}
        return True
    
    def run_all_tests(self) -> bool:
        """
        Run all smoke tests.
        
        Returns:
            bool: True if ALL tests pass, False otherwise
        
        Critical: If ANY test fails, deployment should be blocked!
        """
        print("\n" + "#" * 60)
        print("#  SMOKE TEST SUITE - HIGH-CARDINALITY PREDICTION SERVICE")
        print("#" * 60)
        print(f"\nTarget: {self.base_url}")
        print(f"Timeout: {self.timeout}s | Retries: {self.retries}")
        
        start_time = time.time()
        
        # Run all tests
        test_results = [
            self.test_health_endpoint(),
            self.test_prediction_endpoint(),
            self.test_root_endpoint()
        ]
        
        elapsed_time = time.time() - start_time
        
        # Summary
        print("\n" + "#" * 60)
        print("#  SMOKE TEST SUMMARY")
        print("#" * 60)
        
        passed = sum(test_results)
        total = len(test_results)
        
        print(f"\nTests Passed: {passed}/{total}")
        print(f"Time Elapsed: {elapsed_time:.2f}s")
        
        if all(test_results):
            print("\n" + "=" * 60)
            print("🎉 ALL SMOKE TESTS PASSED - DEPLOYMENT CAN PROCEED")
            print("=" * 60)
            return True
        else:
            print("\n" + "=" * 60)
            print("❌ SMOKE TESTS FAILED - DEPLOYMENT MUST BE BLOCKED")
            print("=" * 60)
            return False
    
    def get_results_json(self) -> str:
        """Get test results as JSON string."""
        return json.dumps(self.results, indent=2)


def main():
    """Main entry point for smoke test script."""
    parser = argparse.ArgumentParser(
        description="Smoke Test for High-Cardinality Prediction Service",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Test default localhost:8000
    python scripts/smoke_test.py

    # Test custom endpoint
    python scripts/smoke_test.py --host api.example.com --port 443

    # Increase retries for slow startup
    python scripts/smoke_test.py --retries 10 --retry-delay 10
        """
    )
    
    parser.add_argument(
        "--host",
        default="localhost",
        help="Service host (default: localhost)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Service port (default: 8000)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=10,
        help="Request timeout in seconds (default: 10)"
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Number of retries for failed requests (default: 3)"
    )
    parser.add_argument(
        "--retry-delay",
        type=int,
        default=5,
        help="Delay between retries in seconds (default: 5)"
    )
    parser.add_argument(
        "--output-json",
        action="store_true",
        help="Output results as JSON"
    )
    
    args = parser.parse_args()
    
    runner = SmokeTestRunner(
        host=args.host,
        port=args.port,
        timeout=args.timeout,
        retries=args.retries,
        retry_delay=args.retry_delay
    )
    
    success = runner.run_all_tests()
    
    if args.output_json:
        print("\n--- JSON Results ---")
        print(runner.get_results_json())
    
    # Exit with appropriate code (0 = success, 1 = failure)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
