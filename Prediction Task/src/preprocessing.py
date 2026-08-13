# src/preprocessing.py
import pandas as pd
import numpy as np
from typing import List, Dict

def preprocess_session_data(clicks: List[Dict], metadata: dict) -> pd.DataFrame:
    df = pd.DataFrame(clicks)
    df['event_time'] = pd.to_datetime(df['event_time'])
    
    # Core session counts
    view_count = (df['event_type'] == 'view').sum()
    cart_count = (df['event_type'] == 'cart').sum()
    unique_products = df['product_id'].nunique()
    unique_brands = df['brand'].nunique()
    unique_categories = df['category_id'].nunique() # Added
    
    average_price = df['price'].mean()
    max_price = df['price'].max()
    min_price = df['price'].min()
    price_range = max_price - min_price # Added
    
    cart_to_view_ratio = cart_count / (view_count + 1e-5) # Added
    
    # Time metrics based on the last event
    last_time = df['event_time'].max()
    hour = last_time.hour
    weekday = last_time.weekday()
    
    hour_sin = np.sin(2 * np.pi * hour / 24) # Added
    hour_cos = np.cos(2 * np.pi * hour / 24) # Added
    weekday_sin = np.sin(2 * np.pi * weekday / 7) # Added
    weekday_cos = np.cos(2 * np.pi * weekday / 7) # Added
    is_weekend = 1 if weekday >= 5 else 0 # Added
    
    duration = (last_time - df['event_time'].min()).total_seconds()
    time_gap_1_2 = (df.iloc[1]['event_time'] - df.iloc[0]['event_time']).total_seconds()
    time_gap_2_3 = (df.iloc[2]['event_time'] - df.iloc[1]['event_time']).total_seconds()
    
    last_event = df.iloc[2]
    
    features = {
        'view_count': view_count,
        'cart_count': cart_count,
        'unique_products': unique_products,
        'unique_brands': unique_brands,
        'unique_categories': unique_categories,
        'average_price': average_price,
        'max_price': max_price,
        'min_price': min_price,
        'price_range': price_range,
        'cart_to_view_ratio': cart_to_view_ratio,
        'pre_prediction_duration_seconds': duration,
        'time_gap_1_2': time_gap_1_2,
        'time_gap_2_3': time_gap_2_3,
        'hour_sin': hour_sin,
        'hour_cos': hour_cos,
        'weekday_sin': weekday_sin,
        'weekday_cos': weekday_cos,
        'is_weekend': is_weekend,
        'user_prior_sessions': 0, # Default fallback
        'last_event_type': last_event.get('event_type', 'view'),
        'last_product_id': str(last_event.get('product_id', 'missing')),
        'last_category_id': str(last_event.get('category_id', 'missing')),
        'last_brand': str(last_event.get('brand', 'missing')),
        'last_category_l1': str(last_event.get('category_l1', 'missing')), # Added
        'last_category_l2': str(last_event.get('category_l2', 'missing')),
        'last_category_l3': str(last_event.get('category_l3', 'missing')),
        'last_price_band': str(last_event.get('price_band', 'mid'))
    }
    
    processed_df = pd.DataFrame([features])
    
    # Handle missing categorical values natively
    for col in metadata["categorical_columns"]:
        if col in processed_df.columns:
            processed_df[col] = processed_df[col].fillna('missing').astype(str)
            
    # Reindex to match the exact columns the model was trained on
    processed_df = processed_df.reindex(columns=metadata["features"], fill_value=0)
            
    return processed_df