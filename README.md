
# House Price Prediction - Data Preprocessing and Feature Engineering

This README summarizes the data preprocessing and feature engineering steps performed on the Ames Housing dataset to prepare it for machine learning models.

---

## 1. Dataset Overview
- **Training Set:** `train.csv` (1460 rows, 81 columns)
- **Test Set:** `test.csv` (1459 rows, 80 columns)
- **Target Variable:** `SalePrice`

The dataset consists of numerical and categorical features related to residential home sales.

---

## 2. Data Preprocessing Steps

### 2.1 Missing Value Handling
- Categorical features where NA indicates absence were filled with `'None'`. These include features like `PoolQC`, `Alley`, `Fence`, `FireplaceQu`, `GarageType`, etc.
- Numerical features with missing values were filled with the **median** of the respective column. Features include `LotFrontage`, `GarageYrBlt`, `MasVnrArea`, `TotalBsmtSF`, etc.
- Remaining categorical features with few missing values were filled with the **mode** (most frequent value).

### 2.2 Target Transformation
- The target variable `SalePrice` was log-transformed using `log1p` to reduce skewness and stabilize variance.

### 2.3 Outlier Removal
- Outliers were removed from training data based on `GrLivArea` and `SalePrice` to improve model generalization.

---

## 3. Feature Engineering Steps

### 3.1 Ordinal Encoding
- Several categorical features with clear ordered relationships were mapped to numerical scales:
    - `ExterQual`, `ExterCond`, `BsmtQual`, `BsmtCond`, `HeatingQC`, `KitchenQual`, `FireplaceQu`, `GarageQual`, `GarageCond`, `PoolQC`, `BsmtExposure`, `BsmtFinType1`, `BsmtFinType2`, `GarageFinish`.
    - Mapping: `{Ex: 5, Gd: 4, TA: 3, Fa: 2, Po: 1, None: 0}`

### 3.2 One-Hot Encoding
- Remaining nominal categorical variables were one-hot encoded to convert them into binary features.

### 3.3 New Features Created
- `TotalSF`: Sum of `TotalBsmtSF`, `1stFlrSF`, and `2ndFlrSF` (total house area).
- `TotalBath`: Sum of full and half bathrooms (including basement) accounting for half baths.
- `HouseAge`: Years since house was built.
- `RemodAge`: Years since last remodeling.

---

## 4. Exploratory Data Analysis (EDA)
- Descriptive statistics were printed for numerical features.
- Correlation heatmap was plotted to observe highly correlated features.
- Distribution plot of `SalePrice` before and after log transformation.

---

## 5. Modeling Approach (Demonstration)
- An advanced model using **XGBoost Regressor** was trained on the processed data.
- 5-Fold cross-validation was performed to evaluate model performance.
- Achieved a cross-validation RMSE score of approximately **0.11 (log scale)**, which translates to roughly **11% average prediction error**.

---


