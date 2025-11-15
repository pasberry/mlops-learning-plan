"""
API test client for model serving.

Tests the model serving API endpoints.
"""
import requests
import json
import time
from typing import Dict, Any, List


class ModelAPIClient:
    """Client for interacting with the model serving API."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        """Initialize API client.

        Args:
            base_url: Base URL of the API server
        """
        self.base_url = base_url
        self.session = requests.Session()

    def health_check(self) -> Dict[str, Any]:
        """Check API health."""
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()

    def predict(
        self,
        features: Dict[str, Any],
        model_version: str = "latest"
    ) -> Dict[str, Any]:
        """Make a single prediction.

        Args:
            features: Feature dictionary
            model_version: Model version to use

        Returns:
            Prediction response
        """
        payload = {
            "features": features,
            "model_version": model_version
        }
        response = self.session.post(
            f"{self.base_url}/predict",
            json=payload
        )
        response.raise_for_status()
        return response.json()

    def batch_predict(
        self,
        instances: List[Dict[str, Any]],
        model_version: str = "latest"
    ) -> Dict[str, Any]:
        """Make batch predictions.

        Args:
            instances: List of feature dictionaries
            model_version: Model version to use

        Returns:
            Batch prediction response
        """
        payload = {
            "instances": instances,
            "model_version": model_version
        }
        response = self.session.post(
            f"{self.base_url}/batch_predict",
            json=payload
        )
        response.raise_for_status()
        return response.json()

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information.

        Returns:
            Model information
        """
        response = self.session.get(f"{self.base_url}/model/info")
        response.raise_for_status()
        return response.json()


def test_health_check(client: ModelAPIClient):
    """Test health check endpoint."""
    print("\n=== Testing Health Check ===")
    try:
        health = client.health_check()
        print(f"Status: {health['status']}")
        print(f"Model Loaded: {health['model_loaded']}")
        print(f"Model Version: {health.get('model_version', 'N/A')}")
        print("✓ Health check passed")
        return True
    except Exception as e:
        print(f"✗ Health check failed: {str(e)}")
        return False


def test_single_prediction(client: ModelAPIClient):
    """Test single prediction endpoint."""
    print("\n=== Testing Single Prediction ===")
    try:
        # Sample features
        features = {
            "age": 35,
            "income": 75000,
            "credit_score": 720,
            "num_purchases": 5,
            "account_age_days": 365,
            "avg_transaction": 150.0,
            "num_returns": 1,
            "is_premium": 1,
            "region": 2,
            "category_preference": 3
        }

        # Make prediction
        start_time = time.time()
        result = client.predict(features)
        latency = (time.time() - start_time) * 1000

        print(f"Prediction: {result['prediction']:.4f}")
        print(f"Prediction Class: {result['prediction_class']}")
        print(f"Model Version: {result['model_version']}")
        print(f"API Latency: {latency:.2f}ms")
        print(f"Server Latency: {result.get('latency_ms', 'N/A')}")
        print("✓ Single prediction passed")
        return True

    except Exception as e:
        print(f"✗ Single prediction failed: {str(e)}")
        return False


def test_batch_prediction(client: ModelAPIClient):
    """Test batch prediction endpoint."""
    print("\n=== Testing Batch Prediction ===")
    try:
        # Sample batch instances
        instances = [
            {
                "age": 25,
                "income": 45000,
                "credit_score": 650,
                "num_purchases": 2,
                "account_age_days": 180,
                "avg_transaction": 80.0,
                "num_returns": 0,
                "is_premium": 0,
                "region": 1,
                "category_preference": 1
            },
            {
                "age": 45,
                "income": 95000,
                "credit_score": 780,
                "num_purchases": 15,
                "account_age_days": 730,
                "avg_transaction": 250.0,
                "num_returns": 2,
                "is_premium": 1,
                "region": 3,
                "category_preference": 2
            },
            {
                "age": 35,
                "income": 65000,
                "credit_score": 700,
                "num_purchases": 8,
                "account_age_days": 450,
                "avg_transaction": 120.0,
                "num_returns": 1,
                "is_premium": 0,
                "region": 2,
                "category_preference": 4
            }
        ]

        # Make batch prediction
        start_time = time.time()
        result = client.batch_predict(instances)
        latency = (time.time() - start_time) * 1000

        print(f"Total Predictions: {result['total_count']}")
        print(f"Model Version: {result['model_version']}")
        print(f"Batch Latency: {latency:.2f}ms")
        print(f"Avg Latency per Instance: {latency / len(instances):.2f}ms")

        print("\nIndividual Predictions:")
        for i, pred in enumerate(result['predictions']):
            print(f"  Instance {i}: {pred['prediction']:.4f} (class: {pred['prediction_class']})")

        print("✓ Batch prediction passed")
        return True

    except Exception as e:
        print(f"✗ Batch prediction failed: {str(e)}")
        return False


def test_model_info(client: ModelAPIClient):
    """Test model info endpoint."""
    print("\n=== Testing Model Info ===")
    try:
        info = client.get_model_info()
        print(f"Model Name: {info['model_name']}")
        print(f"Model Version: {info['model_version']}")
        print(f"Model Type: {info['model_type']}")
        print(f"Input Features: {len(info.get('input_features', []))}")
        if info.get('input_features'):
            print(f"  Features: {', '.join(info['input_features'][:5])}...")
        print("✓ Model info passed")
        return True

    except Exception as e:
        print(f"✗ Model info failed: {str(e)}")
        return False


def test_error_handling(client: ModelAPIClient):
    """Test error handling."""
    print("\n=== Testing Error Handling ===")

    # Test 1: Empty features
    try:
        client.predict({})
        print("✗ Empty features should fail")
        return False
    except requests.exceptions.HTTPError:
        print("✓ Empty features rejected correctly")

    # Test 2: Invalid batch (empty instances)
    try:
        client.batch_predict([])
        print("✗ Empty batch should fail")
        return False
    except requests.exceptions.HTTPError:
        print("✓ Empty batch rejected correctly")

    print("✓ Error handling tests passed")
    return True


def run_all_tests(base_url: str = "http://localhost:8000"):
    """Run all API tests.

    Args:
        base_url: Base URL of the API server
    """
    print("=" * 60)
    print("MLOps Model Serving API Tests")
    print("=" * 60)
    print(f"Target: {base_url}")

    # Create client
    client = ModelAPIClient(base_url)

    # Run tests
    results = []
    results.append(("Health Check", test_health_check(client)))
    results.append(("Single Prediction", test_single_prediction(client)))
    results.append(("Batch Prediction", test_batch_prediction(client)))
    results.append(("Model Info", test_model_info(client)))
    results.append(("Error Handling", test_error_handling(client)))

    # Print summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:.<40} {status}")

    total_passed = sum(1 for _, passed in results if passed)
    total_tests = len(results)
    print(f"\nTotal: {total_passed}/{total_tests} tests passed")

    if total_passed == total_tests:
        print("\n🎉 All tests passed!")
    else:
        print(f"\n⚠️  {total_tests - total_passed} test(s) failed")

    return total_passed == total_tests


def run_load_test(base_url: str = "http://localhost:8000", num_requests: int = 100):
    """Run a simple load test.

    Args:
        base_url: Base URL of the API server
        num_requests: Number of requests to send
    """
    print("\n" + "=" * 60)
    print(f"Load Test: {num_requests} requests")
    print("=" * 60)

    client = ModelAPIClient(base_url)

    features = {
        "age": 35,
        "income": 75000,
        "credit_score": 720,
        "num_purchases": 5,
        "account_age_days": 365,
        "avg_transaction": 150.0,
        "num_returns": 1,
        "is_premium": 1,
        "region": 2,
        "category_preference": 3
    }

    latencies = []
    errors = 0

    print(f"Sending {num_requests} requests...")
    start_time = time.time()

    for i in range(num_requests):
        try:
            req_start = time.time()
            client.predict(features)
            latency = (time.time() - req_start) * 1000
            latencies.append(latency)

            if (i + 1) % 10 == 0:
                print(f"  Completed {i + 1}/{num_requests}")

        except Exception as e:
            errors += 1

    total_time = time.time() - start_time

    # Calculate statistics
    if latencies:
        avg_latency = sum(latencies) / len(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)
        p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]

        print(f"\nResults:")
        print(f"  Total Time: {total_time:.2f}s")
        print(f"  Requests/sec: {num_requests / total_time:.2f}")
        print(f"  Successful: {len(latencies)}")
        print(f"  Errors: {errors}")
        print(f"\nLatency:")
        print(f"  Average: {avg_latency:.2f}ms")
        print(f"  Min: {min_latency:.2f}ms")
        print(f"  Max: {max_latency:.2f}ms")
        print(f"  P95: {p95_latency:.2f}ms")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test model serving API")
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="API base URL"
    )
    parser.add_argument(
        "--load-test",
        action="store_true",
        help="Run load test"
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=100,
        help="Number of requests for load test"
    )

    args = parser.parse_args()

    # Wait for server to be ready
    print("Checking if server is ready...")
    client = ModelAPIClient(args.url)
    max_retries = 10
    for i in range(max_retries):
        try:
            client.health_check()
            print("Server is ready!")
            break
        except Exception:
            if i < max_retries - 1:
                print(f"Waiting for server... ({i+1}/{max_retries})")
                time.sleep(2)
            else:
                print("Server is not responding. Please start the server first.")
                exit(1)

    # Run tests
    success = run_all_tests(args.url)

    # Run load test if requested
    if args.load_test:
        run_load_test(args.url, args.num_requests)

    exit(0 if success else 1)
