from pydantic import BaseModel, Field

class ForecastRequest(BaseModel):
    product_id: int = Field(..., description="The unique ID of the product")
    
    # 1-day lags
    views_lag_1: int
    carts_lag_1: int
    purchases_lag_1: int
    
    # 7-day lags
    views_lag_7: int
    carts_lag_7: int
    purchases_lag_7: int
    
    # Rolling averages
    views_rolling_7d_mean: float
    carts_rolling_7d_mean: float
    purchases_rolling_7d_mean: float
    
    # Calendar features
    day_of_week: int = Field(..., description="0 for Monday, 6 for Sunday")
    is_weekend: int = Field(..., description="0 for weekday, 1 for weekend")
    month: int = Field(..., description="Month number (1-12)")

class ForecastResponse(BaseModel):
    product_id: int
    forecasted_units: float
    message: str