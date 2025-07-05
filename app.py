import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ─── 1. Load resources ────────────────────────────────────────────────────────
@st.cache(allow_output_mutation=True)
def load_model_and_data():
    model = joblib.load("house_price_model.joblib")
    # Load a bit of training data to build distributions
    df = pd.read_csv("train.csv")
    return model, df

model, df = load_model_and_data()

# select a handful of key numeric features used during training
FEATURES = [
    "OverallQual",    # Overall quality (1-10)
    "GrLivArea",      # Living area
    "YearBuilt",      # Year built
    "TotalBsmtSF",    # Total basement square footage
    "GarageCars",     # Garage capacity
    "FullBath",       # Full bathrooms
    "LotArea",        # Lot area
    "1stFlrSF",       # First floor square footage
    "2ndFlrSF",       # Second floor square footage
    "Fireplaces"      # Number of fireplaces
]

def preprocess_input(user_input):
    """Transform user input to match model's expected features"""
    # Create a DataFrame with all expected features, filled with default values
    all_features = pd.DataFrame(index=pd.Index([0]), columns=model.feature_names_in_)
    all_features = all_features.fillna(0).infer_objects(copy=False)  # Fill with zeros as default
    
    # Map user inputs to their corresponding features
    feature_mapping = {
        'OverallQual': 'OverallQual',
        'GrLivArea': 'GrLivArea', 
        'YearBuilt': 'YearBuilt',
        'TotalBsmtSF': 'TotalBsmtSF',
        'GarageCars': 'GarageCars',
        'FullBath': 'FullBath',
        'LotArea': 'LotArea',
        '1stFlrSF': '1stFlrSF',
        '2ndFlrSF': '2ndFlrSF',
        'Fireplaces': 'Fireplaces'
    }
    
    for user_feature, model_feature in feature_mapping.items():
        if model_feature in all_features.columns:
            all_features[model_feature] = user_input[user_feature]
    
    return all_features

# ─── 2. Sidebar for user inputs ───────────────────────────────────────────────
st.sidebar.header("🏠 Input House Features")

def user_input_features():
    data = {}
    data["OverallQual"] = st.sidebar.slider("Overall Quality (1-10)", 1, 10, 5)
    data["GrLivArea"] = st.sidebar.slider("Living Area (sq ft)", 300, 6000, 1500)
    data["YearBuilt"] = st.sidebar.slider("Year Built", 1870, 2020, 1970)
    data["TotalBsmtSF"] = st.sidebar.slider("Basement Area (sq ft)", 0, 3000, 800)
    data["GarageCars"] = st.sidebar.slider("Garage Capacity (# cars)", 0, 5, 2)
    data["FullBath"] = st.sidebar.slider("Full Bathrooms", 0, 4, 2)
    data["LotArea"] = st.sidebar.slider("Lot Area (sq ft)", 500, 100000, 8000)
    data["1stFlrSF"] = st.sidebar.slider("1st Floor Area (sq ft)", 300, 4000, 1000)
    data["2ndFlrSF"] = st.sidebar.slider("2nd Floor Area (sq ft)", 0, 3000, 0)
    data["Fireplaces"] = st.sidebar.slider("Fireplaces", 0, 4, 1)
    return pd.DataFrame(data, index=pd.Index([0]))

input_df = user_input_features()

# ─── 3. Main panel ───────────────────────────────────────────────────────────
st.title("🏡 House Price Prediction")
st.write(
    """
    This app loads a pre-trained model to predict the sale price of a house 
    based on selected features. Adjust the sliders in the sidebar to see how 
    the predicted price changes.
    """
)

# Show user inputs
st.subheader("Input parameters")
st.table(input_df)

# ─── 4. Prediction ────────────────────────────────────────────────────────────
# Preprocess input to match model's expected features
processed_input = preprocess_input(input_df.iloc[0])
prediction_log = model.predict(processed_input)[0]
# Reverse log transformation to get actual dollar amount
prediction = np.expm1(prediction_log)
st.subheader("🔮 Predicted Sale Price")
st.write(f"**${prediction:,.0f}**")

# ─── 5. Price Distribution Visualization ──────────────────────────────────────
st.subheader("Price distribution in training set")
fig, ax = plt.subplots()
ax.hist(df["SalePrice"], bins=50, alpha=0.7)
ax.axvline(prediction, color="red", linestyle="--", linewidth=2)
ax.set_xlabel("SalePrice")
ax.set_ylabel("Count")
st.pyplot(fig)

# ─── 6. Feature Importance ───────────────────────────────────────────────────
if hasattr(model, "feature_importances_"):
    st.subheader("Feature Importances")
    importances = model.feature_importances_
    feature_names = model.feature_names_in_
    
    # Get importance for the features we're using in the app
    app_feature_importances = []
    for feature in FEATURES:
        if feature in feature_names:
            idx = list(feature_names).index(feature)
            app_feature_importances.append(importances[idx])
        else:
            app_feature_importances.append(0)
    
    # sort for display
    imp_df = pd.DataFrame({
        "feature": FEATURES,
        "importance": app_feature_importances
    }).sort_values(by="importance", ascending=False)
    fig2, ax2 = plt.subplots()
    ax2.barh(imp_df["feature"], imp_df["importance"])
    ax2.invert_yaxis()
    ax2.set_xlabel("Importance")
    st.pyplot(fig2)
else:
    st.info("Model does not support feature importances.")

# ─── 7. Optional: Show raw training data ───────────────────────────────────────
with st.expander("Show raw training data"):
    st.dataframe(df[FEATURES + ["SalePrice"]].head(100))
