# SuperKart — Sales Prediction and Containerized ML Deployment

## Project Overview

SuperKart is an end-to-end machine learning and deployment project that predicts **product-store sales** from product and store attributes.

The project covers the complete ML lifecycle:

- exploratory data analysis,
- feature engineering,
- outlier analysis,
- preprocessing pipelines,
- Random Forest and XGBoost regression,
- hyperparameter tuning,
- model comparison,
- final-model serialization,
- Flask API development,
- Streamlit frontend development,
- Docker containerization,
- GitHub Codespaces deployment,
- single-record inference,
- and batch inference.

Unlike a notebook-only project, this repository also contains the actual deployment application.

---

## Business Problem

Retailers need reliable sales estimates to support:

- inventory planning,
- replenishment,
- stock allocation,
- pricing decisions,
- and store-level demand planning.

The objective is to predict:

```text
Product_Store_Sales_Total
```

for a given product-store combination.

A stronger forecast can help reduce both:

- **stock-outs** caused by underestimating demand,
- and **excess inventory** caused by overestimating demand.

---

## Dataset

The training dataset contains:

```text
Rows:    8,763
Columns: 12
```

The project also includes a separate batch-inference file with:

```text
10 rows
10 model input features
```

### Original columns

- `Product_Id`
- `Product_Weight`
- `Product_Sugar_Content`
- `Product_Allocated_Area`
- `Product_Type`
- `Product_MRP`
- `Store_Id`
- `Store_Establishment_Year`
- `Store_Size`
- `Store_Location_City_Type`
- `Store_Type`
- `Product_Store_Sales_Total`

### Target

```text
Product_Store_Sales_Total
```

Summary of the target in the supplied data:

- Mean: approximately **3464**
- Median: approximately **3452**
- Minimum: **33**
- Maximum: **8000**

No missing values were found in the supplied dataset.

---

## Exploratory Data Analysis

The notebook includes:

- dataset shape and datatype inspection,
- missing-value checks,
- duplicate checks,
- descriptive statistics,
- univariate distributions,
- boxplots,
- store/category frequency analysis,
- bivariate sales analysis,
- correlation analysis,
- and written EDA observations.

Examples of observed patterns include:

- `Low Sugar` is the most common sugar-content category.
- Tier 2 stores form the largest share of records.
- Departmental stores showed the highest average sales among the store types in this dataset.
- Product MRP, store type, allocated area, and other store/product attributes are examined against total sales.

---

## Feature Engineering

The project creates three deployment-friendly features.

### Product ID prefix

```text
Product_Id_char
```

The first characters of `Product_Id` are used instead of the full high-cardinality identifier.

### Store age

```text
Store_Age_Years
```

Derived from the store establishment year.

### Product category grouping

```text
Product_Type_Category
```

Detailed product types are consolidated into:

- `Perishables`
- `Non Perishables`

This reduces category complexity and makes deployment inputs easier to manage.

---

## Outlier Analysis

The notebook uses the **IQR method** to identify potential outliers.

Examples of detected outlier proportions include approximately:

- Product Weight: **0.6%**
- Product Allocated Area: **1.2%**

The project retains plausible extreme observations rather than automatically removing them because:

- they may represent legitimate business cases,
- the proportion is small,
- and tree-based ensemble models are relatively robust to outliers.

---

## Preprocessing Pipeline

The model uses a reusable `ColumnTransformer`.

### Numerical features

Numerical variables are standardized using:

```python
StandardScaler
```

### Categorical features

Categorical variables are encoded with:

```python
OneHotEncoder(handle_unknown="ignore")
```

The preprocessing step is included inside each machine-learning pipeline so that training and inference use exactly the same transformations.

---

## Model Evaluation Metric

### Primary metric: RMSE

The project uses **Root Mean Squared Error (RMSE)** as the main model-selection metric.

RMSE is appropriate because:

- the target is continuous,
- it is expressed in the same units as the sales target,
- and it penalizes large forecasting errors more strongly.

Supporting metrics include:

- MAE
- MAPE
- R²

---

## Models

Two regression models are compared:

### Random Forest Regressor

Baseline cross-validated RMSE:

```text
286.44
```

### XGBoost Regressor

Baseline cross-validated RMSE:

```text
298.29
```

Both models are constructed as complete preprocessing + estimator pipelines.

---

## Hyperparameter Tuning

`GridSearchCV` is used with 3-fold cross-validation and RMSE optimization.

### Tuned Random Forest

Best cross-validated RMSE:

```text
285.68
```

Best parameters in the completed run included:

```text
n_estimators: 350
max_depth: None
max_features: 0.8
min_samples_leaf: 2
min_samples_split: 5
```

### Tuned XGBoost

Best cross-validated RMSE:

```text
292.56
```

Best parameters included:

```text
n_estimators: 350
learning_rate: 0.03
max_depth: 5
subsample: 0.8
colsample_bytree: 1.0
```

---

## Final Model

The project selected:

```text
Tuned Random Forest
```

because it achieved the lowest cross-validated RMSE among the baseline and tuned candidates.

### Held-Out Test Performance

The completed run achieved approximately:

| Metric | Result |
|---|---:|
| RMSE | **277.32** |
| MAE | **105.91** |
| MAPE | **3.84%** |
| R² | **0.9326** |

An R² of about **0.93** indicates that the model explains a large proportion of the observed sales variation in this test split.

---

## Model Serialization

The complete trained pipeline is serialized using:

```python
joblib
```

The deployment artifact is:

```text
backend/superkart_model.joblib
```

Because preprocessing is stored inside the same pipeline, the API does not need to manually recreate encoding or scaling logic.

The notebook also reloads the serialized model and confirms that its predictions match the pre-serialization predictions.

---

# Deployment Architecture

```text
                    ┌────────────────────┐
                    │   User / Browser   │
                    └─────────┬──────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │ Streamlit Frontend │
                    │     Port 8501      │
                    └─────────┬──────────┘
                              │ HTTP
                              ▼
                    ┌────────────────────┐
                    │     Flask API      │
                    │     Port 7860      │
                    └─────────┬──────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │ Serialized ML      │
                    │ Pipeline / Model   │
                    └────────────────────┘
```

The frontend and backend run in separate Docker containers and communicate over the Docker Compose network.

---

## Flask Backend

The Flask application exposes three endpoints.

### Health check

```http
GET /health
```

Expected response:

```json
{
  "status": "ok"
}
```

### Single prediction

```http
POST /v1/predict
```

Example JSON payload:

```json
{
  "Product_Weight": 12.66,
  "Product_Sugar_Content": "Low Sugar",
  "Product_Allocated_Area": 0.027,
  "Product_MRP": 117.08,
  "Store_Size": "Small",
  "Store_Location_City_Type": "Tier 1",
  "Store_Type": "Supermarket Type1",
  "Product_Id_char": "FD",
  "Store_Age_Years": 16,
  "Product_Type_Category": "Perishables"
}
```

Example response format:

```json
{
  "predicted_sales": 2840.15
}
```

### Batch prediction

```http
POST /v1/predictbatch
```

The endpoint accepts a CSV using multipart form field:

```text
file
```

The included file:

```text
data/Batch_Data_SuperKart.csv
```

can be used for testing.

---

## Streamlit Frontend

The Streamlit interface supports:

### Single prediction

Users enter product and store attributes through an interactive form.

### Batch prediction

Users upload a CSV and receive predictions for multiple records.

The frontend communicates with the Flask backend using:

```text
API_ROOT
```

Inside Docker Compose:

```text
API_ROOT=http://backend:7860
```

---

## Repository Structure

```text
superkart-sales-forecasting/
├── README.md
├── SuperKart_Sales_Prediction_and_Deployment.ipynb
├── requirements.txt
├── .gitignore
├── docker-compose.yml
│
├── data/
│   ├── SuperKart.csv
│   └── Batch_Data_SuperKart.csv
│
├── backend/
│   ├── app.py
│   ├── Dockerfile
│   ├── requirements.txt
│   └── superkart_model.joblib
│
└── frontend/
    ├── streamlit_app.py
    ├── Dockerfile
    └── requirements.txt
```

---

# Running the Notebook

## Option 1 — Google Colab

Upload or clone the repository and open:

```text
SuperKart_Sales_Prediction_and_Deployment.ipynb
```

The notebook expects:

```text
data/SuperKart.csv
data/Batch_Data_SuperKart.csv
```

Install the main dependencies:

```python
!pip install -r requirements.txt
```

Then use:

```text
Runtime → Run all
```

The notebook covers:

1. Data overview
2. EDA
3. Feature engineering
4. Outlier analysis
5. Train/test split
6. Preprocessing
7. Baseline model building
8. Hyperparameter tuning
9. Model comparison
10. Final test evaluation
11. Serialization
12. Flask backend generation
13. Streamlit frontend generation
14. Docker setup
15. Online inference
16. Batch inference

---

# Running Locally with Docker

This is the easiest way to run the full deployed application.

## 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/superkart-sales-forecasting.git
cd superkart-sales-forecasting
```

## 2. Build and start both services

```bash
docker compose up --build
```

Docker Compose starts:

```text
Backend  → http://localhost:7860
Frontend → http://localhost:8501
```

## 3. Test backend health

Open:

```text
http://localhost:7860/health
```

or run:

```bash
curl http://localhost:7860/health
```

## 4. Open the frontend

Navigate to:

```text
http://localhost:8501
```

You can now perform both single and batch predictions.

## 5. Stop the application

```bash
docker compose down
```

---

# Run Without Docker

## Backend

```bash
cd backend
pip install -r requirements.txt
python app.py
```

Backend:

```text
http://localhost:7860
```

## Frontend

Open a second terminal:

```bash
cd frontend
pip install -r requirements.txt
```

Set the backend URL.

### Windows PowerShell

```powershell
$env:API_ROOT="http://localhost:7860"
streamlit run streamlit_app.py
```

### macOS / Linux

```bash
export API_ROOT="http://localhost:7860"
streamlit run streamlit_app.py
```

Frontend:

```text
http://localhost:8501
```

---

# GitHub Codespaces Deployment

The project can also be demonstrated using GitHub Codespaces.

## 1. Create/open a Codespace

Open the repository in GitHub and select:

```text
Code → Codespaces → Create codespace
```

## 2. Start the containers

Inside the Codespaces terminal:

```bash
docker compose up --build
```

## 3. Forward ports

Open the **Ports** panel.

Make these ports public:

```text
7860 — Flask backend
8501 — Streamlit frontend
```

GitHub Codespaces will generate temporary forwarded URLs.

Example format:

```text
https://YOUR-CODESPACE-7860.app.github.dev
https://YOUR-CODESPACE-8501.app.github.dev
```

Codespaces URLs are temporary and may change when the Codespace is recreated, so they are intentionally not hard-coded in this public repository.

---

# Testing the API

## Single inference with Python

```python
import requests

api = "http://localhost:7860"

payload = {
    "Product_Weight": 12.66,
    "Product_Sugar_Content": "Low Sugar",
    "Product_Allocated_Area": 0.027,
    "Product_MRP": 117.08,
    "Store_Size": "Small",
    "Store_Location_City_Type": "Tier 1",
    "Store_Type": "Supermarket Type1",
    "Product_Id_char": "FD",
    "Store_Age_Years": 16,
    "Product_Type_Category": "Perishables"
}

response = requests.post(
    f"{api}/v1/predict",
    json=payload
)

print(response.status_code)
print(response.json())
```

Expected HTTP status:

```text
200
```

## Batch inference with Python

```python
import requests

api = "http://localhost:7860"

with open("data/Batch_Data_SuperKart.csv", "rb") as f:
    response = requests.post(
        f"{api}/v1/predictbatch",
        files={"file": f}
    )

print(response.status_code)
print(response.json())
```

Expected HTTP status:

```text
200
```

The completed course run successfully returned HTTP 200 for both single and batch inference.

---

## Technologies Used

### Data Science

- Python
- pandas
- NumPy
- scikit-learn
- XGBoost
- Matplotlib
- Seaborn
- joblib

### Backend

- Flask
- Gunicorn

### Frontend

- Streamlit
- Requests

### Deployment

- Docker
- Docker Compose
- GitHub
- GitHub Codespaces

---

## Business Recommendations

- Use predicted demand to improve replenishment and safety-stock decisions.
- Compare forecast error across store types, city tiers, and product categories.
- Combine sales predictions with pricing and inventory policies rather than treating price as the only driver.
- Monitor production RMSE and MAE for model drift.
- Retrain when product assortment, pricing, store mix, or buying behavior changes.
- Use batch inference for regular inventory-planning cycles and the API for real-time prediction workflows.

---

## Important Notes

- The `backend/superkart_model.joblib` file is large because it contains the complete trained preprocessing + Random Forest pipeline.
- GitHub currently supports files below its normal per-file size limit; if the model later becomes larger, use Git LFS or external artifact storage.
- The Codespaces forwarded URL in the original course notebook was temporary and has been replaced with a placeholder in this portfolio version.
- Model metrics can vary slightly if the notebook is retrained with different library versions or random seeds.

---

## Author

**Sreeman Mandava**

Machine Learning | Data Engineering | Software Engineering | AI Engineering
