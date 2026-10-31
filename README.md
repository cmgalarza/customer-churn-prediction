# Customer Churn Prediction
 
**Course:** IN503 – Unit 10 Assignment
**Author:** Marie Galarza
 
## Project Description
 
This project predicts whether a telecom customer is likely to cancel their service (churn) using machine learning. It uses the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) from Kaggle (7,043 customers, 21 columns) and a **Logistic Regression** classifier wrapped in a scikit-learn `Pipeline`.
 
The notebook walks through the full workflow:
 
1. Environment setup and library imports
2. Data acquisition, cleaning, and exploration (5 visualizations)
3. Preprocessing (`StandardScaler` for numeric features, `OneHotEncoder` for categorical features)
4. Model training with an 80/20 stratified train/test split
5. Evaluation and coefficient interpretation
6. Saving the trained pipeline as a `.pkl` file and demonstrating how to load it for predictions
## Repository Contents
 
| File | Description |
|------|-------------|
| `Galarza_Marie_IN503_Unit10_Assignment.ipynb` | Jupyter notebook with the full analysis |
| `customer_churn_model.pkl` | Trained pipeline (preprocessing + Logistic Regression) |
| `requirements.txt` | Python dependencies |
| `README.md` | Project documentation |
 
## Installation
 
1. Clone the repository:
```bash
   git clone https://github.com/<your-username>/customer-churn-prediction.git
   cd customer-churn-prediction
```
2. (Optional) Create and activate a virtual environment:
```bash
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
```
3. Install dependencies:
```bash
   pip install -r requirements.txt
```
 
> **Note:** The notebook downloads the dataset with `kagglehub`, so an internet connection is required the first time you run it. It was originally developed in Google Colab.
 
## Usage
 
### Run the notebook
 
```bash
jupyter notebook Galarza_Marie_IN503_Unit10_Assignment.ipynb
```
 
### Load the trained model and predict churn
 
The saved file is a complete pipeline, so raw customer data goes in directly. Scaling and encoding are handled automatically.
 
```python
import joblib
import pandas as pd
 
model = joblib.load("customer_churn_model.pkl")
 
new_customer = pd.DataFrame({
    "gender": ["Male"],
    "SeniorCitizen": [0],
    "Partner": ["No"],
    "Dependents": ["No"],
    "tenure": [1],
    "PhoneService": ["Yes"],
    "MultipleLines": ["No"],
    "InternetService": ["Fiber optic"],
    "OnlineSecurity": ["No"],
    "OnlineBackup": ["No"],
    "DeviceProtection": ["No"],
    "TechSupport": ["No"],
    "StreamingTV": ["No"],
    "StreamingMovies": ["No"],
    "Contract": ["Month-to-month"],
    "PaperlessBilling": ["Yes"],
    "PaymentMethod": ["Electronic check"],
    "MonthlyCharges": [75.75],
    "TotalCharges": [75.75],
})
 
prediction = model.predict(new_customer)
print("Predicted Churn:", "Yes" if prediction[0] == 1 else "No")   # -> Yes
 
print(model.predict_proba(new_customer))   # [[P(no churn), P(churn)]]
```
 
All 19 feature columns shown above are required, using the same column names and category values as the original dataset (`customerID` and `Churn` are not used as inputs).
 
## Key Findings
 
**Data**
- After removing 11 rows with blank `TotalCharges` values, 7,032 customers remained.
- Overall churn rate: **26.6%** (1,869 churned vs. 5,163 stayed), so the classes are imbalanced.
**Model performance (test set, 1,407 customers)**
 
| Metric | No Churn | Churn |
|--------|----------|-------|
| Precision | 0.85 | 0.65 |
| Recall | 0.89 | 0.57 |
| F1-score | 0.87 | 0.61 |
 
- Overall accuracy: **80.38%**
- Confusion matrix: 917 true negatives, 116 false positives, 160 false negatives, 214 true positives.
- The model identifies non-churners well but catches only about 57% of customers who actually churn.
**Most influential features (logistic regression coefficients)**
- **Tenure** is the strongest predictor: longer-tenured customers are much less likely to churn.
- **Two-year contracts** strongly reduce churn risk, while **month-to-month contracts** increase it.
- **Fiber optic internet** is associated with higher churn, while **DSL** is associated with lower churn.
- Customers **without** paperless billing are less likely to churn (so paperless billing is linked to higher churn risk).
- `TotalCharges` and `MonthlyCharges` show opposite coefficient signs. These variables are correlated with tenure and each other, so their individual coefficients should be interpreted with caution.
**Business takeaway:** New customers on month-to-month contracts with fiber optic service are the highest-risk group. Contract incentives and early-tenure retention efforts are the most promising places to focus.
 
## Dependencies
 
- Python 3.9+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- joblib
- kagglehub
- jupyter
Install everything with `pip install -r requirements.txt`.
