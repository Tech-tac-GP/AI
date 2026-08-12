import pandas as pd
import numpy as np
from typing import List, Dict

def preprocess_session_data(clicks: List[Dict], metadata: dict) -> pd.DataFrame:
    df = pd.DataFrame(clicks)
    df['event_time'] = pd.to_datetime(df['event_time'])
    
    # Core session metrics
    view_count = (df['event_type'] == 'view').sum()
    cart_count = (df['event_type'] == 'cart').sum()
    unique_products = df['product_id'].nunique()
    unique_brands = df['brand'].nunique()
    
    average_price = df['price'].mean()
    max_price = df['price'].max()
    min_price = df['price'].min()
    
    duration = (df['event_time'].max() - df['event_time'].min()).total_seconds()
    
    # Last event details
    last_event = df.iloc[-1]
    
    features = {
        'view_count': view_count,
        'cart_count': cart_count,
        'unique_products': unique_products,
        'unique_brands': unique_brands,
        'average_price': average_price,
        'max_price': max_price,
        'min_price': min_price,
        'session_duration': duration,
        'last_event_type': str(last_event.get('event_type', 'view')),
        'last_product_id': str(last_event.get('product_id', 'missing')),
        'last_category_id': str(last_event.get('category_id', 'missing')),
        'last_brand': str(last_event.get('brand', 'missing')),
        'last_category_code': str(last_event.get('category_code', 'missing'))
    }
    
    processed_df = pd.DataFrame([features])
    
    # Handle missing categorical values natively
    for col in metadata["categorical_columns"]:
        if col in processed_df.columns:
            processed_df[col] = processed_df[col].fillna('missing').astype(str)
            
    # Reindex to match the exact columns the model was trained on
    processed_df = processed_df.reindex(columns=metadata["features"], fill_value=0)
            
    return processed_df