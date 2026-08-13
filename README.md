# AI — Electronics E-Commerce Machine Learning Suite

Machine learning components powering the AI layer of an electronics e-commerce platform. This repository covers the full pipeline for each task — data preparation, feature engineering, modeling, evaluation, and production-ready artifacts — for four core business problems built on the platform's event-log data (views, cart adds, and purchases).

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Modules](#modules)
  - [Purchase Prediction](#purchase-prediction)
  - [Cart Abandonment Prediction](#cart-abandonment-prediction)
  - [Customer Segmentation](#customer-segmentation)
  - [Demand Forecasting](#demand-forecasting)
- [Dataset](#dataset)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
- [Methodology Principles](#methodology-principles)
- [Contributing](#contributing)

## Overview

The platform's event-log data (user views, cart additions, and purchases across an electronics/computers catalog) is used to power four ML-driven capabilities:

1. **Predict** whether a browsing session will end in a purchase, early enough to act on it.
2. **Predict** whether a customer is likely to abandon items left in their cart.
3. **Segment** customers by behavior for targeted marketing and personalization.
4. **Forecast** product/category-level demand to support inventory and planning decisions.

Each module is self-contained — its own data preparation, feature engineering, modeling, and evaluation — but all share a common dataset and a common set of engineering standards (see [Methodology Principles](#methodology-principles)) to keep results comparable and trustworthy across the suite.

## Repository Structure

```
AI/
├── Prediction Task/              # Purchase prediction pipeline
├── Cart Abandonment Prediction/  # Cart abandonment classification pipeline
├── Customer Segmentation/        # Behavioral clustering / RFM segmentation
├── Demand Forecasting/           # Product/category-level demand forecasting
└── README.md
```

Each folder contains the notebooks, scripts, and saved model artifacts for that task. See each module's section below for details.

## Modules

### Purchase Prediction

**Business question**: Will this session end in a purchase — early enough in the visit that the platform can still act on the answer (e.g., trigger a personalized offer)?

- **Unit of prediction**: session
- **Approach**: prefix-based prediction using only the first *N* events of a session (configurable), so the model reflects what's actually known at the moment of prediction — never information from later in the session
- **Leakage controls**: target is derived from the complete session outcome, but every feature is built exclusively from the pre-prediction-point event window; verified programmatically, not just assumed
- **Models compared**: Logistic Regression (baseline), XGBoost, CatBoost
- **Validation**: time-based train/validation/test split (never random, given the temporal nature of session data)
- **Primary metric**: PR-AUC, given class imbalance in purchase outcomes

### Cart Abandonment Prediction

**Business question**: Given a session that has added an item to cart, will the customer complete the purchase or abandon it?

- **Unit of prediction**: cart-containing session
- **Note**: this dataset has no explicit "remove from cart" event, so abandonment is defined as *a cart addition with no subsequent purchase in that session* — a slightly weaker signal than an explicit removal action, documented wherever this model's output is used downstream
- **Use case**: powers cart-recovery interventions (reminder emails, incentives, retargeting)

### Customer Segmentation

**Business question**: What distinct behavioral groups exist within the customer base, and how should each be treated differently?

- **Approach**: unsupervised clustering (e.g., K-Means) over RFM-style and engagement features (recency, frequency, monetary value, session/purchase behavior)
- **Use case**: informs targeted marketing, loyalty programs, and personalized merchandising

### Demand Forecasting

**Business question**: What demand should be expected for products/categories going forward, to support inventory and planning?

- **Approach**: time-series forecasting at the product-group/category level, chosen based on data volume and reliability at each granularity
- **Validation**: chronological (walk-forward) validation, consistent with standard time-series practice

## Dataset

All modules share a common source: an event-level e-commerce log where each row represents a single user action.

**Source**: [electronicsnew](https://www.kaggle.com/datasets/omarhusnye/electronicsnew) on Kaggle


| Column | Description |
|---|---|
| `event_time` | Timestamp of the event (UTC) |
| `event_type` | `view`, `cart`, or `purchase` |
| `product_id`, `category_id`, `category_code`, `brand`, `price` | Product attributes |
| `user_id`, `user_session` | User and session identifiers |
| `category_l1` / `category_l2` / `category_l3` | Parsed category hierarchy |
| `is_pc_component`, `component_type`, `product_group` | Derived product taxonomy |

The raw data is cleaned and filtered upstream (duplicate removal, missing-value handling, category/brand normalization, bot-session flagging) before being used by any module — see the data preparation steps within each module's notebooks for the exact logic applied.

## Tech Stack

- **Language**: Python 3.12+
- **Data manipulation**: pandas, NumPy
- **Modeling**: scikit-learn, XGBoost, CatBoost, Kmeans
- **Visualization**: Matplotlib, Seaborn
- **Serving**: FastAPI, joblib
- **Environment**: Jupyter notebooks

## Getting Started

```bash
# Clone the repository
git clone https://github.com/Tech-tac-GP/AI.git
cd AI

# Install dependencies (per module, if requirements differ)
pip install -r requirements.txt

# Open a module's notebook(s) in Jupyter
jupyter notebook "Prediction Task"
```

Each module's notebooks are designed to run top-to-bottom against the cleaned dataset. Adjust the data-loading path at the top of each notebook to point at your local copy of the cleaned CSV.

## Methodology Principles

Every module in this repository follows the same non-negotiable standards:

- **No target leakage** — features never use information that wouldn't be available at the actual prediction point; every prediction task explicitly separates its feature-generation window from its target-observation window.
- **Time-aware validation** — any temporal data is split chronologically, never randomly, to reflect how the model will actually be used in production.
- **Preprocessing fit on training data only** — imputers, scalers, and encoders are never fit on validation or test data.
- **Imbalance-aware evaluation** — PR-AUC and F1 are reported alongside (not instead of) ROC-AUC and accuracy, since purchase/conversion events are a minority class throughout this dataset.
- **Explainability** — feature importance and error analysis are part of some modeling notebook, not an afterthought.

## Contributing

1. Create a feature branch from `main`.
2. Keep new modules/notebooks self-contained and consistent with the methodology principles above.
3. Open a pull request with a clear description of the change.

---

**Team**: Tech-tac-GP — Graduation Project
