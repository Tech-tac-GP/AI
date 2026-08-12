# E-Commerce Cart Abandonment Prediction API

## Overview
This repository houses an academic team project designed to apply machine learning concepts to a simulated e-commerce environment. The core of this project is a predictive AI system that evaluates user session behavior to determine the probability of **cart abandonment**. 

Rather than a live production real-time system, this acts as a proof-of-concept where our cross-functional team (AI, Data Science, Backend, Frontend) applies learned concepts to bridge the gap between predictive modeling and web integration. 

By identifying high-risk sessions dynamically, the API outputs a probability score and an actionable trigger (e.g., offering a discount) to help retain potential lost sales.

---

## Model Performance & Evaluation
The core prediction engine is powered by a **CatBoost Classifier**, trained on historical e-commerce session data. The model evaluates behavioral footprints (such as price sensitivity, session duration, and click sequences) to separate converting users from abandoners.

**Test Set Results:**
* **ROC-AUC:** 0.9590
* **PR-AUC:** 0.9623
* **Precision:** 88.05%
* **Recall:** 86.01%
* **Overall Accuracy:** 86.94%

These metrics indicate an exceptionally stable model with a high capacity to capture minority class events (abandoners) while maintaining a low false-positive rate.


## The FastAPI Backend
FastAPI serves as the critical bridge between the machine learning model and the web application. It was chosen for this project to seamlessly serve the predictive model to the rest of the team's software ecosystem.

Why FastAPI?

Speed & Performance: It is one of the fastest Python web frameworks available, allowing the API to process session clicks and return predictions with minimal latency.

Automatic Interactive Docs: It automatically generates the Swagger UI (/docs), giving frontend and backend developers an instant, visual way to test JSON payloads without needing external tools like Postman.

Strict Data Validation: Using Pydantic models, FastAPI automatically validates incoming request payloads. If the web platform sends malformed JSON, FastAPI catches it before it crashes the machine learning pipeline.

How the Pipeline Works:

Ingestion: The /predict endpoint receives a JSON array of raw user clicks.

Transformation: The raw JSON is routed to preprocessing.py, where it is converted into a Pandas DataFrame and engineered into the exact behavioral features the model expects (e.g., session_duration, view_count).

Inference: The processed DataFrame is fed into the localized CatBoost .cbm model.

Actionable Response: The API returns the raw probability score alongside a translated business action (whether to trigger a discount or do nothing).

## Installation & Setup

Prerequisites
Ensure you have Python 3.9+ installed on your local machine.

Install Dependencies
pip install -r requirements.txt

Run the Development Server
Launch the FastAPI backend locally using Uvicorn.

python -m uvicorn src.main:app --reload


## API Usage

Endpoint: POST /predict
Evaluates an active user session and returns the probability of cart abandonment.

Example Request Payload
The frontend or backend should send a JSON payload containing an array of click events from the user's current session.

{
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

Example Response

{
  "abandonment_probability": 0.8786,
  "predicted_action": "ABANDONED (Trigger Discount!)"
}

Actions:
ABANDONED (Trigger Discount!): High risk of abandonment. Frontend should intercept the user with a promotional offer, free shipping, or a pop-up.

CONVERTED (Do Nothing): Low risk of abandonment. Let the user proceed through the checkout funnel naturally.

---

## Project Architecture
The repository is structured to separate data science workflows from the FastAPI application serving the model.

```text
cart-abandonment-api/
│
├── models/
│   ├── cart_abandonment_model.cbm      # Serialized CatBoost model
│   └── metadata.json                   # Feature names and categorical column mapping
│
├── notebooks/
│   └── cart_abandonment_training.ipynb # Training, evaluation, and visualizations
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py                # Feature engineering and data alignment logic
│   └── main.py                         # FastAPI application and prediction endpoints
│
├── test/                               # Testing artifacts and response logs
├── vis/                                # Evaluation charts (PR Curve, ROC, Confusion Matrix)
├── .gitignore
├── requirements.txt
└── README.md


