import streamlit as st
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore")

# --- Load Model and Feature Metadata ---
MODEL_PATH = "rf_model.pkl"
FEATURES_PATH = "model_features.pkl"
SCALER_PATH = "scaler.pkl"

@st.cache_resource
def load_resources():
    model = joblib.load(MODEL_PATH)
    rent_features = joblib.load(FEATURES_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, rent_features, scaler

model, rent_features, scaler = load_resources()

# Categorical and Numerical Columns
NUMERICAL_COLS_PRESENT = [
    "Size", "Bathroom", "BHK", "Security_Deposit"
]
CAT_FEATURES = [
    "City", "Area_Type", "Furnishing_Status", "Parking", "Power_Backup",
    "Water_Supply", "Lift_Available", "Gym", "Swimming_Pool", "Garden",
    "Pet_Allowed", "Fire_Support", "Road_Connectivity"
]

# --- Preprocessing Function ---
def preprocess_new_data(new_df, original_df_columns, numerical_cols, categorical_cols, scaler):
    st.subheader("🔍 Debugging Info")

    # Strip spaces and lowercase categorical values
    for col in categorical_cols:
        if col in new_df:
            new_df[col] = new_df[col].astype(str).str.strip().str.lower()

    st.write("Initial Input Data:", new_df)

    # One-hot encode
    new_df = pd.get_dummies(new_df, drop_first=True)

    # Add missing columns as 0
    missing_cols = set(original_df_columns) - set(new_df.columns)
    for col in missing_cols:
        new_df[col] = 0

    # Drop extra columns
    extra_cols = set(new_df.columns) - set(original_df_columns)
    if extra_cols:
        st.write("⚠️ Dropping extra columns:", extra_cols)
    new_df = new_df[original_df_columns]

    # Scale numerical columns that actually exist
    cols_to_scale = [col for col in numerical_cols if col in new_df.columns]
    if cols_to_scale:
        new_df[cols_to_scale] = scaler.transform(new_df[cols_to_scale])

    st.write("Final Processed DataFrame:", new_df)
    return new_df

# --- Streamlit App UI ---
st.title("🏠 House Rent Prediction App")

st.sidebar.header("Enter House Details")

city = st.sidebar.selectbox("City", ["mumbai", "delhi", "bangalore"])
area_type = st.sidebar.selectbox("Area Type", ["super built-up  area", "built-up area", "plot area"])
bhk = st.sidebar.number_input("BHK", min_value=1, max_value=10, value=2)
size = st.sidebar.number_input("Size (sqft)", min_value=100, max_value=10000, value=1000)
bathroom = st.sidebar.number_input("Number of Bathrooms", min_value=1, max_value=10, value=2)
security_deposit = st.sidebar.number_input("Security Deposit", min_value=0, max_value=1000000, value=50000)
furnishing_status = st.sidebar.selectbox("Furnishing Status", ["unfurnished", "semi-furnished", "furnished"])
parking = st.sidebar.selectbox("Parking", ["yes", "no"])
power_backup = st.sidebar.selectbox("Power Backup", ["yes", "no"])
water_supply = st.sidebar.selectbox("Water Supply", ["corporation", "borewell"])
lift_available = st.sidebar.selectbox("Lift Available", ["yes", "no"])
gym = st.sidebar.selectbox("Gym", ["yes", "no"])
swimming_pool = st.sidebar.selectbox("Swimming Pool", ["yes", "no"])
garden = st.sidebar.selectbox("Garden", ["yes", "no"])
pet_allowed = st.sidebar.selectbox("Pet Allowed", ["yes", "no"])
fire_support = st.sidebar.selectbox("Fire Support", ["yes", "no"])
road_connectivity = st.sidebar.selectbox("Road Connectivity", ["good", "average", "poor"])

if st.sidebar.button("Predict Rent"):
    input_data = pd.DataFrame([{
        "City": city,
        "Area_Type": area_type,
        "BHK": bhk,
        "Size": size,
        "Bathroom": bathroom,
        "Security_Deposit": security_deposit,
        "Furnishing_Status": furnishing_status,
        "Parking": parking,
        "Power_Backup": power_backup,
        "Water_Supply": water_supply,
        "Lift_Available": lift_available,
        "Gym": gym,
        "Swimming_Pool": swimming_pool,
        "Garden": garden,
        "Pet_Allowed": pet_allowed,
        "Fire_Support": fire_support,
        "Road_Connectivity": road_connectivity
    }])

    processed_data = preprocess_new_data(input_data, rent_features, NUMERICAL_COLS_PRESENT, CAT_FEATURES, scaler)
    prediction = model.predict(processed_data)
    st.success(f"💰 Estimated Monthly Rent: ₹{prediction[0]:,.2f}")
