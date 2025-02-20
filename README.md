# Bengaluru House Price Prediction

![Image_Alt](House_Prediction_Bangalore.png)

## 📌Overview
This project predicts house prices in Bengaluru, India, using a machine learning model based on the Bengaluru House Data dataset. The notebook demonstrates data preprocessing, exploratory data analysis (EDA), feature engineering, and the implementation of a Linear Regression model to estimate house prices based on location, square footage, number of bathrooms, and bedrooms (BHK). A prediction function is also provided for practical use.

## 📂Dataset

### Description
The dataset contains housing data for Bengaluru, sourced from Kaggle, with details about house features and their corresponding prices.

- **Source:** Bengaluru_House_Data.csv (available on Kaggle or stored in Google Drive).
- **Size:** 13,320 rows, 9 columns (initially); reduced to 7,325 rows after preprocessing.
- **Columns (Original):**
  - **area_type:** Type of area (e.g., Super built-up Area, Plot Area).
  - **availability:** Readiness status (e.g., Ready To Move, 19-Dec).
  - **location:** Locality in Bengaluru (e.g., Electronic City Phase II).
  - **size:** Size of the house (e.g., 2 BHK, 4 Bedroom).
  - **society:** Housing society name (e.g., Coomee).
  - **total_sqft:** Total area in square feet.
  - **bath:** Number of bathrooms.
  - **balcony:** Number of balconies.
  - **price:** Price of the house in lakhs (target variable).
- **Columns (After Preprocessing):**
  - **location:** One-hot encoded into 243 columns (e.g., 1st Phase JP Nagar, Indira Nagar).
  - **total_sqft:** Converted to numeric (float).
  - **bath:** Number of bathrooms (float).
  - **BHK:** Number of bedrooms extracted from size (integer).

## 📊Preprocessing Notes
- Dropped columns: area_type, society, availability.
- Extracted BHK from size and converted total_sqft to numeric.
- Applied one-hot encoding to location, resulting in 243 location-specific binary columns.
- Final dataset shape: (7325 rows, 243 columns).

## 🔍Acknowledgements
The dataset is sourced from Kaggle: Bengaluru House Prices.

## 🛠Methodology

### Dependencies
The project uses the following Python libraries:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import *
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
```

## 📊Results
### Model: Linear Regression with StandardScaler preprocessing.
##### Sample Predictions:
- predict_price('1st Phase JP Nagar', 1000, 2, 2): ₹82.15 Lakhs.
- predict_price('1st Phase JP Nagar', 1000, 3, 3): ₹84.23 Lakhs.
- predict_price('Indira Nagar', 1000, 2, 2): ₹185.14 Lakhs.
- predict_price('Indira Nagar', 1000, 3, 3): ₹187.23 Lakhs.
- Insights: Prices vary significantly by location (e.g., Indira Nagar is more expensive than 1st Phase JP Nagar), with marginal increases for additional bathrooms 
   and bedrooms.


## 🚀Future Work
- Add evaluation metrics (e.g., R², RMSE) to assess model performance.
- Include EDA visualizations (e.g., price distribution by location) using seaborn or plotly.
- Experiment with advanced models (e.g., Random Forest, XGBoost) for better accuracy.
- Handle irregular total_sqft values (e.g., ranges like "1200-1500") more robustly.

