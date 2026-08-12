from pydantic import BaseModel, Field

class CustomerFeatures(BaseModel):
    purchase_rate: float
    cart_rate: float
    view_rate: float
    events_per_session: float
    products_per_session: float
    total_spending: float
    average_purchase_value: float
    days_since_last_activity: float
    activity_days: float

class SegmentResponse(BaseModel):
    cluster: int
    segment: str
