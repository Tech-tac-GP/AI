import requests
import json

URL = "http://127.0.0.1:8000/predict"

# The test payload containing 3 sequential clicks
payload = {
    "clicks": [
        {
            "event_time": "2026-08-15 20:00:00 UTC",
            "event_type": "view",
            "product_id": 4804056,
            "category_id": 2053013554658804075,
            "category_code": "electronics.audio.headphone",
            "brand": "apple",
            "price": 25.50,
            "user_id": 515384420,
            "user_session": "test-session-123"
        },
        {
            "event_time": "2026-08-15 20:01:00 UTC",
            "event_type": "view",
            "product_id": 4804056,
            "category_id": 2053013554658804075,
            "category_code": "electronics.audio.headphone",
            "brand": "apple",
            "price": 25.50,
            "user_id": 515384420,
            "user_session": "test-session-123"
        },
        {
            "event_time": "2026-08-15 20:01:30 UTC",
            "event_type": "cart",
            "product_id": 4804056,
            "category_id": 2053013554658804075,
            "category_code": "electronics.audio.headphone",
            "brand": "apple",
            "price": 25.50,
            "user_id": 515384420,
            "user_session": "test-session-123"
        }
    ]
}

def run_test():
    print(f"Sending POST request to {URL}...")
    
    try:
        # Send the POST request with the JSON payload
        response = requests.post(URL, json=payload)
        
        # Check if the request was successful
        if response.status_code == 200:
            print("\nSuccess! API Response:")
            print(json.dumps(response.json(), indent=4))
        else:
            print(f"\nError {response.status_code}:")
            print(response.text)
            
    except requests.exceptions.ConnectionError:
        print("\nConnection Error: Is your FastAPI server running?")

if __name__ == "__main__":
    run_test()