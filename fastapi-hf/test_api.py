import requests
import json

BASE_URL = "http://localhost:7860"

def test_endpoints():
    """Test all API endpoints"""
    
    print("Testing FLAN-T5 API...")
    print("=" * 50)
    
    # Test health endpoint
    try:
        response = requests.get(f"{BASE_URL}/health")
        print("1. Health Check:")
        print(json.dumps(response.json(), indent=2))
    except Exception as e:
        print(f"Health check failed: {e}")
    
    # Test root endpoint
    try:
        response = requests.get(f"{BASE_URL}/")
        print("\n2. Root Endpoint:")
        print(json.dumps(response.json(), indent=2))
    except Exception as e:
        print(f"Root endpoint failed: {e}")
    
    # Test generation endpoint
    try:
        params = {"prompt": "Explain quantum computing in simple terms:"}
        response = requests.get(f"{BASE_URL}/generate", params=params)
        print("\n3. Generation Test:")
        print(json.dumps(response.json(), indent=2))
    except Exception as e:
        print(f"Generation test failed: {e}")
    
    print("\n" + "=" * 50)
    print("Testing complete!")

if __name__ == "__main__":
    test_endpoints()