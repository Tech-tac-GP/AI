import requests
import pandas as pd

URL = "http://127.0.0.1:8000/api/v1/forecast/predict"

def test_demand_forecasting(csv_path):
    print(f"Loading historical data from {csv_path}...\n" + "-"*40)
    
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: Could not find the file at {csv_path}")
        return

    # Sort the dataframe to find the items with the highest actual historical purchases
    if 'purchases' in df.columns:
        df = df.sort_values(by='purchases', ascending=False)
        print("Dataset sorted to test High-Demand items first!\n")

    # Loop through the top 5 highest-demand rows
    for index, row in df.head(5).iterrows():
        payload = row.to_dict()
        
        print(f"Testing Product ID: {payload.get('product_id')} for Date: {payload.get('date')}")
        
        try:
            response = requests.post(URL, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                # Print the forecasted units clearly
                print(f"Success: Forecasted Demand = {result['forecasted_units']:.2f} units\n")
            else:
                print(f"Error {response.status_code}: {response.text}\n")
                
        except requests.exceptions.ConnectionError:
            print("Connection Error: Is your FastAPI server currently running?")
            break

if __name__ == "__main__":
    test_demand_forecasting("data/test_data_processed.csv")


