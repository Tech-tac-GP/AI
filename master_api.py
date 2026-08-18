from fastapi import FastAPI

# Import the routers from sub-folders
from cart_abandonment.src.main import router as cart_router
from purchase_prediction.src.main import router as purchase_router
from demand_forecasting.api.routers.forecast import router as forecast_router
from customer_segmentation.api.main import router as segment_router

# Initialize the master application
app = FastAPI(
    title="E-Commerce AI Suite",
    description="Unified API gateway for all machine learning microservices.",
    version="1.0.0",
    docs_url="/"
)

# Mount the individual routers with specific prefixes
app.include_router(cart_router, prefix="/api/v1/cart", tags=["Cart Abandonment"])
app.include_router(purchase_router, prefix="/api/v1/purchase", tags=["Purchase Prediction"])
app.include_router(forecast_router, prefix="/api/v1/forecast", tags=["Demand Forecasting"])
app.include_router(segment_router, prefix="/api/v1/segmentation", tags=["Customer Segmentation"])
