from fastapi import FastAPI
from api.routers import forecast

# Initialize the application
app = FastAPI(
    title="E-Commerce Demand Forecasting API",
    description="Machine Learning backend for predicting product demand.",
    version="1.0.0",
    docs_url="/" 
)

# Connect the forecasting router
app.include_router(forecast.router, prefix="/api/v1/forecast", tags=["Forecasting"])

@app.get("/", tags=["Health"])
def health_check():
    """
    Root endpoint to verify the API is running.
    """
    return {
        "status": "online",
        "message": "Welcome to the Demand Forecasting API. Navigate to /docs for the Swagger UI."
    }
