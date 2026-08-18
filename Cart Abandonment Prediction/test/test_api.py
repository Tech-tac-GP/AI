import requests
import json

# The URL of your local FastAPI server
URL = "http://127.0.0.1:8000/predict"

# A synthetic payload
payload = {
  "clicks": [
    {
      "event_time": "2021-02-04T12:00:00",
      "event_type": "view",
      "product_id": "4500981",
      "category_id": "2053013555563954111",
      "brand": "msi",
      "category_code": "computers.components.graphic_card",
      "price": 950.00
    },
    {
      "event_time": "2021-02-04T12:01:00",
      "event_type": "cart",
      "product_id": "4500981",
      "category_id": "2053013555563954111",
      "brand": "msi",
      "category_code": "computers.components.graphic_card",
      "price": 950.00
    },
    {
      "event_time": "2021-02-04T12:01:45",
      "event_type": "view",
      "product_id": "4500981",
      "category_id": "2053013555563954111",
      "brand": "msi",
      "category_code": "computers.components.graphic_card",
      "price": 950.00
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
            print("\n✅ Success! API Response:")
            print(json.dumps(response.json(), indent=4))
        else:
            print(f"\n❌ Error {response.status_code}:")
            print(response.text)
            
    except requests.exceptions.ConnectionError:
        print("\n❌ Connection Error: Is your FastAPI server running?")

if __name__ == "__main__":
    run_test()