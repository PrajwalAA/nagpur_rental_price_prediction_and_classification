import streamlit as st
import pandas as pd
import joblib # For loading the trained model and scaler
import numpy as np # For numerical operations like np.expm1

# Suppress warnings for cleaner output in Streamlit
import warnings
warnings.simplefilter('ignore')

# --- 1. Model and Scaler Loading ---
# Ensure these files are in the same directory as your Streamlit app.py.
# You will need to replace 'rent_model.pkl', 'rent_scaler.pkl', and 'rent_features.pkl'
# with the actual names of your saved model, scaler, and feature list files.
MODEL_PATH = 'rf_model.pkl'
SCALER_PATH = 'scaler.pkl'
FEATURES_PATH = 'model_features.pkl' # This file should contain the list of column names the model expects

loaded_model = None
loaded_scaler = None
loaded_features = None

try:
    loaded_model = joblib.load(MODEL_PATH)
    loaded_scaler = joblib.load(SCALER_PATH)
    loaded_features = joblib.load(FEATURES_PATH)
    st.success("Rent prediction model, scaler, and feature names loaded successfully!")
except FileNotFoundError:
    st.error(f"Error: One or more model files not found. Please ensure '{MODEL_PATH}', '{SCALER_PATH}', and '{FEATURES_PATH}' are in the correct directory.")
    st.stop() # Stop the app if essential files are missing
except Exception as e:
    st.error(f"An error occurred while loading model files: {e}")
    st.stop()

# Define categorical features and numerical columns for scaling, consistent with training
# These lists should match how they were defined and used in your training script.
# Based on the user's provided 'predict_new_rent_from_input' function:
CATEGORICAL_FEATURES = [
    'Area', 'Zone', 'Furnishing Status', 'Recommended For', 'Water Supply'
]
NUMERICAL_COLS_PRESENT = [
    'Size_In_Sqft', 'Carpet_Area_Sqft', 'Bedrooms', 'Bathrooms', 'Balcony',
    'Number_Of_Amenities', 'Floor_No', 'Total_floors_In_Building', #'Road_Connectivity',
    'gym', #'gated_community',
    'intercom', 'lift', #'pet_allowed',
    'pool', 'security', 'wifi', 'gas_pipeline', 'sports_facility', 'kids_area',
    'power_backup', 'Garden', #'Fire_Support',
    'Parking', 'ATM_Near_me',
    'Airport_Near_me', 'Bus_Stop__Near_me', 'Hospital_Near_me', 'Mall_Near_me',
    'Market_Near_me', 'Metro_Station_Near_me', 'Park_Near_me', 'School_Near_me',
    #'Property_Age'
]

# --- 2. Preprocessing Function for New Data ---
def preprocess_new_data(input_data: dict, original_df_columns: list, scaler, categorical_features: list) -> pd.DataFrame:
    """
    Preprocesses new input data to match the format expected by the trained model.
    This replicates the feature engineering steps from the training script, specifically for
    one-hot encoding and column alignment.
    """
    # Create a DataFrame from the new input data
    new_df = pd.DataFrame([input_data])

    # Apply one-hot encoding
    for feature in categorical_features:
        if feature in new_df.columns:
            # Using get_dummies with drop_first=True to avoid multicollinearity
            new_df = pd.get_dummies(new_df, columns=[feature], drop_first=True)

    # Align columns with the training data (CRITICAL for consistent input to model)
    # Add missing columns with 0, and drop extra columns that weren't in training data
    missing_cols = set(original_df_columns) - set(new_df.columns)
    for c in missing_cols:
        new_df[c] = 0
    # Ensure the order of columns is exactly the same as during training
    new_df = new_df[original_df_columns]

    # Scale numerical features - this step should happen AFTER column alignment
    # because the scaler was fitted on data that already had the correct columns.
    # We only transform the numerical columns, which are now correctly aligned.
    new_df[NUMERICAL_COLS_PRESENT] = scaler.transform(new_df[NUMERICAL_COLS_PRESENT])

    return new_df

# --- 3. Streamlit Application Layout ---
st.set_page_config(
    page_title="House Rent Predictor",
    page_icon="🏠",
    layout="wide", # Use wide layout for more space
    initial_sidebar_state="auto"
)

st.title("🏠 House Rent Prediction")
st.markdown("Enter the details of a house to get an estimated monthly rent.")

st.subheader("Property Details")

# Input widgets organized into columns for better layout
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("##### Basic Property Information")
    size_sqft = st.number_input("Size (in Sqft)", min_value=100, max_value=50000, value=1000, step=50)
    carpet_area_sqft = st.number_input("Carpet Area (in Sqft)", min_value=50, max_value=40000, value=800, step=50)
    bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=2, step=1)
    bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10, value=2, step=1)
    balcony = st.number_input("Balcony", min_value=0, max_value=5, value=1, step=1)
    property_age = st.number_input("Property Age (years)", min_value=0, max_value=100, value=5, step=1)

with col2:
    st.markdown("##### Location & Structure")
    floor_no = st.number_input("Floor Number", min_value=0, max_value=50, value=1, step=1)
    total_floors_building = st.number_input("Total Floors in Building", min_value=1, max_value=100, value=5, step=1)
    #road_connectivity = st.number_input("Road Connectivity (0-10)", min_value=0, max_value=10, value=7, step=1)
    number_of_amenities = st.number_input("Number of Amenities", min_value=0, max_value=30, value=5, step=1)

    # Categorical Inputs
    area_options = [
        'Hingna', 'Trimurti Nagar', 'Ashirwad Nagar', 'Beltarodi', 'Besa',
        'Bharatwada', 'Boriyapura', 'Chandrakiran Nagar', 'Dabha', 'Dhantoli',
        'Dharampeth', 'Dighori', 'Duttawadi', 'Gandhibagh', 'Ganeshpeth',
        'Godhni', 'Godhni', 'Gotal Panjri', 'Hingna', 'Hudkeswar', 'Itwari',
        'Jaitala', 'Jaripatka', 'Kalamna', 'Kalmeshwar', 'Khamla', 'Kharbi',
        'Koradi Colony', 'Kotewada', 'Mahal', 'Manewada', 'Manish Nagar',
        'Mankapur', 'Medical Square', 'MIHAN', 'Nandanwan',
        'Narendra Nagar Extension', 'Nari Village', 'Narsala', 'Omkar Nagar',
        'Parvati Nagar', 'Pratap Nagar', 'Ram Nagar', 'Rameshwari',
        'Reshim Bagh', 'Sadar', 'Sanmarga Nagar', 'Seminary Hills',
        'Shatabdi Square', 'Shatabdi Square', 'Sitabuldi', 'Somalwada',
        'Sonegaon', 'Teka Naka', 'Vayusena Nagar', 'Wanadongri',
        'Wardsman Nagar', 'Wathoda', 'Zingabai Takli'
    ]
    area = st.selectbox("Area", area_options)

    zone_options = ['East Zone', 'North Zone', 'South Zone', 'West Zone', 'Rural']
    zone = st.selectbox("Zone", zone_options)

with col3:
    st.markdown("##### Features & Preferences")
    furnishing_status_options = ['Fully Furnished', 'Semi Furnished', 'Unfurnished']
    furnishing_status = st.selectbox("Furnishing Status", furnishing_status_options)

    recommended_for_options = ['Anyone', 'Bachelors', 'Family', 'Family and Bachelors', 'Family and Company']
    recommended_for = st.selectbox("Recommended For", recommended_for_options)

    water_supply_options = ['Borewell', 'Both', 'Municipal']
    water_supply = st.selectbox("Water Supply", water_supply_options)

    st.markdown("##### Amenities (0 = No, 1 = Yes)")
    gym = st.radio("Gym", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No", key='gym')
   # gated_community = st.radio("Gated Community", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No", key='gated_community')
    intercom = st.radio("Intercom", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No", key='intercom')
    lift = st.radio("Lift", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No", key='lift')
    # pet_allowed = st.radio("Pet Allowed", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No", key='pet_allowed')
    pool = st.radio("Pool", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No", key='pool')
    security = st.radio("Security", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No", key='security')
    wifi = st.radio("Wifi", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No", key='wifi')
    gas_pipeline = st.radio("Gas Pipeline", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No", key='gas_pipeline')
    sports_facility = st.radio("Sports Facility", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No", key='sports_facility')
    kids_area = st.radio("Kids Area", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No", key='kids_area')
    power_backup = st.radio("Power Backup", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No", key='power_backup')
    garden = st.radio("Garden", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No", key='garden')
    #fire_support = st.radio("Fire Support", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No", key='fire_support')
    parking = st.radio("Parking", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No", key='parking')
    atm_near_me = st.radio("ATM Near Me", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No", key='atm_near_me')
    airport_near_me = st.radio("Airport Near Me", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No", key='airport_near_me')
    bus_stop_near_me = st.radio("Bus Stop Near Me", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No", key='bus_stop_near_me')
    hospital_near_me = st.radio("Hospital Near Me", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No", key='hospital_near_me')
    mall_near_me = st.radio("Mall Near Me", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No", key='mall_near_me')
    market_near_me = st.radio("Market Near Me", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No", key='market_near_me')
    metro_station_near_me = st.radio("Metro Station Near Me", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No", key='metro_station_near_me')
    park_near_me = st.radio("Park Near Me", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No", key='park_near_me')
    school_near_me = st.radio("School Near Me", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No", key='school_near_me')

# Button to trigger prediction
if st.button("Predict Rent"):
    if loaded_model and loaded_scaler and loaded_features:
        input_data = {
            'Size_In_Sqft': size_sqft,
            'Carpet_Area_Sqft': carpet_area_sqft,
            'Bedrooms': bedrooms,
            'Bathrooms': bathrooms,
            'Balcony': balcony,
            'Number_Of_Amenities': number_of_amenities,
            'Floor_No': floor_no,
            'Total_floors_In_Building': total_floors_building,
            #'Road_Connectivity': road_connectivity,
            'gym': gym,
            #'gated_community': gated_community,
            'intercom': intercom,
            'lift': lift,
            #'pet_allowed': pet_allowed,
            'pool': pool,
            'security': security,
            'wifi': wifi,
            'gas_pipeline': gas_pipeline,
            'sports_facility': sports_facility,
            'kids_area': kids_area,
            'power_backup': power_backup,
            'Garden': garden,
            #'Fire_Support': fire_support,
            'Parking': parking,
            'ATM_Near_me': atm_near_me,
            'Airport_Near_me': airport_near_me,
            'Bus_Stop__Near_me': bus_stop_near_me,
            'Hospital_Near_me': hospital_near_me,
            'Mall_Near_me': mall_near_me,
            'Market_Near_me': market_near_me,
            'Metro_Station_Near_me': metro_station_near_me,
            'Park_Near_me': park_near_me,
            'School_Near_me': school_near_me,
            #'Property_Age': property_age,
            'Area': area,
            'Zone': zone,
            'Furnishing Status': furnishing_status,
            'Recommended For': recommended_for,
            'Water Supply': water_supply
        }

        try:
            # Preprocess the input data
            processed_input = preprocess_new_data(
                input_data,
                loaded_features, # Use the feature names saved during training
                loaded_scaler,
                CATEGORICAL_FEATURES
            )

            # Make prediction
            log_predicted_rent = loaded_model.predict(processed_input)[0]
            predicted_rent = np.expm1(log_predicted_rent) # Inverse transform log rent

            st.subheader("Prediction Result:")
            st.success(f"Estimated Monthly Rent: **₹{predicted_rent:,.2f}**") # Using INR symbol

            # --- Price Classification ---
            st.markdown("---")
            st.subheader("Price Comparison")
            st.markdown("Enter a listed price to see if the property is underpriced, overpriced, or fairly priced based on our prediction.")

            listed_price = st.number_input("Enter Listed Price (₹)", min_value=0, value=int(predicted_rent * 1.05), step=100, key='listed_price_input')

            FAIR_PRICE_TOLERANCE = 0.10 # 10% tolerance

            lower_bound = predicted_rent * (1 - FAIR_PRICE_TOLERANCE)
            upper_bound = predicted_rent * (1 + FAIR_PRICE_TOLERANCE)

            st.info(f"A fair price for this property would typically be between **₹{lower_bound:,.2f}** and **₹{upper_bound:,.2f}**.")

            if listed_price < lower_bound:
                st.warning(f"**₹{listed_price:,.2f}**  This property appears to be **Underpriced**! Great deal!")
            elif listed_price > upper_bound:
                st.error(f"**₹{listed_price:,.2f}**  This property appears to be **Overpriced**! Consider negotiating.")
            else:
                st.success(f"**₹{listed_price:,.2f}**  This property appears to be **Fairly Priced**.")

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            st.warning("Please check your input values and ensure the model files are correct.")
    else:
        st.warning("Model not loaded. Please check the file paths and restart the app.")

# --- How to Run Instructions ---
st.markdown(
    """
    ---
    ### How to Run This Application:
    1.  **Save** the code above as a Python file (e.g., `rent_prediction_app.py`).
    2.  **Ensure** you have trained your rent prediction model and saved the following files:
        * `rent_model.pkl` (your trained model)
        * `rent_scaler.pkl` (your fitted scaler, e.g., StandardScaler)
        * `rent_features.pkl` (a list of feature names in the exact order your model expects, saved during training, e.g., `joblib.dump(X_train.columns.tolist(), 'rent_features.pkl')`)
    3.  **Place** these three `.pkl` files in the **same directory** as `rent_prediction_app.py`.
    4.  **Install** the necessary libraries:
        ```bash
        pip install streamlit numpy pandas scikit-learn joblib
        ```
    5.  **Run** the app from your terminal:
        ```bash
        streamlit run rent_prediction_app.py
        ```
    6.  Your browser will automatically open to the Streamlit app!
    """
)





