import streamlit as st
import pandas as pd
import numpy as np
import joblib # For loading and saving the trained model and scaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error

# Suppress warnings for cleaner output in Streamlit
import warnings
warnings.simplefilter('ignore')

# --- Global Paths for Model and Scaler ---
MODEL_PATH = 'rf_model.pkl'
SCALER_PATH = 'scaler.pkl'
FEATURES_PATH = 'model_features.pkl'

# --- 1. Data Preprocessing & Model Training (Run Once for Model Generation) ---
# This part is typically run once to generate and save the model files.
# In a deployed app, you'd only load the pre-trained model.
# However, for a self-contained script to demonstrate, we'll include it.

@st.cache_resource
def train_and_save_model():
    """
    Performs data preprocessing, trains the Random Forest Regressor,
    and saves the model, scaler, and feature columns.
    This function uses st.cache_resource to run only once.
    """
    st.write("Training model and preprocessing data (this runs once)...")

    # Define columns - 'Security_Deposite' removed as requested
    numerical_cols_raw = [
        'Size_In_Sqft', 'Carpet_Area_Sqft', 'Bedrooms', 'Bathrooms', 'Balcony',
        'Number_Of_Amenities', 'Floor_No', 'Total_floors_In_Building',
        'Road_Connectivity', 'gym', 'gated_community', 'intercom', 'lift',
        'pet_allowed', 'pool', 'security', 'water_supply', 'wifi', 'gas_pipeline',
        'sports_facility', 'kids_area', 'power_backup', 'Garden', 'Fire_Support',
        'Parking', 'ATM_Near_me', 'Airport_Near_me', 'Bus_Stop__Near_me',
        'Hospital_Near_me', 'Mall_Near_me', 'Market_Near_me', 'Metro_Station_Near_me',
        'Park_Near_me', 'School_Near_me', 'Property_Age'
    ]

    categorical_cols = [
        'Area', 'Zone', 'Furnishing Status', 'Recommended For',
        'Water Supply', 'Type of Society'
    ]

    # Create dummy data for demonstration (1000 data points for better training)
    data_rows = 1000
    dummy_data = {
        'Size_In_Sqft': np.random.randint(500, 3000, data_rows),
        'Carpet_Area_Sqft': np.random.randint(400, 2500, data_rows),
        'Bedrooms': np.random.randint(1, 5, data_rows),
        'Bathrooms': np.random.randint(1, 4, data_rows),
        'Balcony': np.random.randint(0, 3, data_rows),
        'Number_Of_Amenities': np.random.randint(0, 15, data_rows),
        # 'Security_Deposite' is intentionally removed here
        'Floor_No': np.random.randint(1, 20, data_rows),
        'Total_floors_In_Building': np.random.randint(5, 30, data_rows),
        'Road_Connectivity': np.random.randint(0, 2, data_rows), # Assuming binary (0 or 1)
        'gym': np.random.randint(0, 2, data_rows),
        'gated_community': np.random.randint(0, 2, data_rows),
        'intercom': np.random.randint(0, 2, data_rows),
        'lift': np.random.randint(0, 2, data_rows),
        'pet_allowed': np.random.randint(0, 2, data_rows),
        'pool': np.random.randint(0, 2, data_rows),
        'security': np.random.randint(0, 2, data_rows),
        'water_supply': np.random.randint(0, 2, data_rows),
        'wifi': np.random.randint(0, 2, data_rows),
        'gas_pipeline': np.random.randint(0, 2, data_rows),
        'sports_facility': np.random.randint(0, 2, data_rows),
        'kids_area': np.random.randint(0, 2, data_rows),
        'power_backup': np.random.randint(0, 2, data_rows),
        'Garden': np.random.randint(0, 2, data_rows),
        'Fire_Support': np.random.randint(0, 2, data_rows),
        'Parking': np.random.randint(0, 2, data_rows),
        'ATM_Near_me': np.random.randint(0, 2, data_rows),
        'Airport_Near_me': np.random.randint(0, 2, data_rows),
        'Bus_Stop__Near_me': np.random.randint(0, 2, data_rows),
        'Hospital_Near_me': np.random.randint(0, 2, data_rows),
        'Mall_Near_me': np.random.randint(0, 2, data_rows),
        'Market_Near_me': np.random.randint(0, 2, data_rows),
        'Metro_Station_Near_me': np.random.randint(0, 2, data_rows),
        'Park_Near_me': np.random.randint(0, 2, data_rows),
        'School_Near_me': np.random.randint(0, 2, data_rows),
        'Property_Age': np.random.randint(0, 20, data_rows),
        'Area': np.random.choice(['Hingna', 'Trimurti Nagar', 'Besa', 'Jaitala', 'Dharampeth', 'Manish Nagar'], data_rows),
        'Zone': np.random.choice(['East Zone', 'North Zone', 'South Zone', 'West Zone', 'Rural'], data_rows),
        'Furnishing Status': np.random.choice(['Fully Furnished', 'Semi Furnished', 'Unfurnished'], data_rows),
        'Recommended For': np.random.choice(['Family', 'Bachelors', 'Family and Bachelors', 'Anyone'], data_rows),
        'Water Supply': np.random.choice(['Borewell', 'Municipal', 'Both'], data_rows),
        'Type of Society': np.random.choice(['Gated', 'Non-Gated', 'Township'], data_rows),
        'Rent': np.random.randint(5000, 50000, data_rows) # Target variable
    }

    df = pd.DataFrame(dummy_data)

    # Apply log transformation to the target variable 'Rent'
    df['Rent_log'] = np.log1p(df['Rent'])
    y = df['Rent_log']

    # Drop original 'Rent' and 'Rent_log' columns from features
    X = df.drop(['Rent', 'Rent_log'], axis=1)

    # Apply one-hot encoding
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    # Identify numerical columns for scaling after potential categorical features are removed
    numerical_cols_present = [col for col in numerical_cols_raw if col in X.columns]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and fit the scaler on training data
    scaler = MinMaxScaler()
    X_train[numerical_cols_present] = scaler.fit_transform(X_train[numerical_cols_present])
    X_test[numerical_cols_present] = scaler.transform(X_test[numerical_cols_present])

    # Store final column names for consistent prediction
    original_df_columns = X_train.columns.tolist()

    # Train the Random Forest Regressor model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)

    # Save model, scaler, and feature names
    try:
        joblib.dump(rf_model, MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)
        joblib.dump(original_df_columns, FEATURES_PATH)
        st.success("Model, scaler, and feature names trained and saved successfully!")
    except Exception as e:
        st.error(f"Error saving model/scaler: {e}")
        st.stop() # Stop the app if saving fails

    return rf_model, scaler, original_df_columns, categorical_cols, numerical_cols_present

# --- 2. Model Loading (for the Streamlit app) ---
# This function loads the trained components when the app starts.
@st.cache_resource
def load_model_components():
    """Loads the pre-trained model, scaler, and feature list."""
    try:
        loaded_model = joblib.load(MODEL_PATH)
        loaded_scaler = joblib.load(SCALER_PATH)
        loaded_features = joblib.load(FEATURES_PATH)
        st.success("Rental price prediction model loaded successfully!")
        return loaded_model, loaded_scaler, loaded_features
    except FileNotFoundError:
        st.error("Error: Model files not found. Please ensure the model has been trained and saved.")
        st.info("The model will attempt to train automatically on first run. If this fails, check console for errors.")
        return None, None, None
    except Exception as e:
        st.error(f"An error occurred while loading model components: {e}")
        return None, None, None

# Run training and loading
try:
    rf_model, scaler, original_df_columns, categorical_features_for_prediction, numerical_cols_present = train_and_save_model()
    loaded_rf_model, loaded_scaler, loaded_features = load_model_components()
except Exception as e:
    st.error(f"Failed to initialize model components: {e}")
    st.stop() # Stop the app if initialization fails

# --- 3. Streamlit Application Layout ---
st.set_page_config(
    page_title="Rental Price Predictor",
    page_icon="🏠",
    layout="centered",
    initial_sidebar_state="auto"
)

st.title("🏠 Rental Price Predictor for Nagpur Properties")
st.markdown("Enter property details to get an estimated rental price and fair price range.")
st.markdown("---")

st.subheader("Property Details")

# Input widgets for numerical features
col1, col2 = st.columns(2)

with col1:
    size_in_sqft = st.number_input("Size In Sqft", min_value=100, max_value=10000, value=1200, step=100)
    carpet_area_sqft = st.number_input("Carpet Area Sqft", min_value=100, max_value=9000, value=900, step=100)
    bedrooms = st.number_input("Number of Bedrooms", min_value=1, max_value=10, value=2, step=1)
    bathrooms = st.number_input("Number of Bathrooms", min_value=1, max_value=10, value=2, step=1)
    balcony = st.number_input("Number of Balcony", min_value=0, max_value=5, value=1, step=1)
    number_of_amenities = st.number_input("Number of Amenities", min_value=0, max_value=30, value=5, step=1)
    floor_no = st.number_input("Floor No", min_value=0, max_value=100, value=3, step=1)
    total_floors_in_building = st.number_input("Total Floors In Building", min_value=1, max_value=200, value=10, step=1)
    road_connectivity = st.selectbox("Road Connectivity (0: Bad, 1: Good)", options=[0, 1], format_func=lambda x: "Good" if x == 1 else "Bad")
    property_age = st.number_input("Property Age (years)", min_value=0, max_value=100, value=5, step=1)

with col2:
    st.write("### Amenities & Nearby Facilities (0: No, 1: Yes)")
    gym = st.selectbox("Gym", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    gated_community = st.selectbox("Gated Community", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    intercom = st.selectbox("Intercom", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    lift = st.selectbox("Lift", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    pet_allowed = st.selectbox("Pet Allowed", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    pool = st.selectbox("Pool", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    security_feature = st.selectbox("Security", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No") # Renamed to avoid conflict
    wifi = st.selectbox("WiFi", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    gas_pipeline = st.selectbox("Gas Pipeline", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    sports_facility = st.selectbox("Sports Facility", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    kids_area = st.selectbox("Kids Area", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    power_backup = st.selectbox("Power Backup", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    garden = st.selectbox("Garden", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    fire_support = st.selectbox("Fire Support", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    parking = st.selectbox("Parking", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    atm_near_me = st.selectbox("ATM Near Me", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    airport_near_me = st.selectbox("Airport Near Me", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    bus_stop_near_me = st.selectbox("Bus Stop Near Me", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    hospital_near_me = st.selectbox("Hospital Near Me", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    mall_near_me = st.selectbox("Mall Near Me", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    market_near_me = st.selectbox("Market Near Me", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    metro_station_near_me = st.selectbox("Metro Station Near Me", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    park_near_me = st.selectbox("Park Near Me", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    school_near_me = st.selectbox("School Near Me", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")


st.markdown("---")
st.subheader("Categorical Details")

# Define options for categorical inputs (these should match your training data)
area_options = [
    'Hingna', 'Trimurti Nagar', 'Ashirwad Nagar', 'Beltarodi', 'Besa',
    'Bharatwada', 'Boriyapura', 'Chandrakiran Nagar', 'Dabha', 'Dhantoli',
    'Dharampeth', 'Dighori', 'Duttawadi', 'Gandhibagh', 'Ganeshpeth',
    'Godhni', 'Gotal Panjri', 'Hudkeswar', 'Itwari',
    'Jaitala', 'Jaripatka', 'Kalamna', 'Kalmeshwar', 'Khamla', 'Kharbi',
    'Koradi Colony', 'Kotewada', 'Mahal', 'Manewada', 'Manish Nagar',
    'Mankapur', 'Medical Square', 'MIHAN', 'Nandanwan',
    'Narendra Nagar Extension', 'Nari Village', 'Narsala', 'Omkar Nagar',
    'Parvati Nagar', 'Pratap Nagar', 'Ram Nagar', 'Rameshwari',
    'Reshim Bagh', 'Sadar', 'Sanmarga Nagar', 'Seminary Hills',
    'Shatabdi Square', 'Sitabuldi', 'Somalwada',
    'Sonegaon', 'Teka Naka', 'Vayusena Nagar', 'Wanadongri',
    'Wardsman Nagar', 'Wathoda', 'Zingabai Takli'
]
zone_options = ['East Zone', 'North Zone', 'South Zone', 'West Zone', 'Rural']
furnishing_status_options = ['Fully Furnished', 'Semi Furnished', 'Unfurnished']
recommended_for_options = ['Anyone', 'Bachelors', 'Family', 'Family and Bachelors', 'Family and Company']
water_supply_options = ['Borewell', 'Both', 'Municipal']
type_of_society_options = ['Gated', 'Non-Gated', 'Township']


area = st.selectbox("Area", options=area_options)
zone = st.selectbox("Zone", options=zone_options)
furnishing_status = st.selectbox("Furnishing Status", options=furnishing_status_options)
recommended_for = st.selectbox("Recommended For", options=recommended_for_options)
water_supply_type = st.selectbox("Water Supply Type", options=water_supply_options)
type_of_society = st.selectbox("Type of Society", options=type_of_society_options)

st.markdown("---")

# Button to trigger prediction
if st.button("Predict Rental Price"):
    if loaded_rf_model and loaded_scaler and loaded_features:
        # Create a dictionary of user inputs
        new_data_dict = {
            'Size_In_Sqft': size_in_sqft,
            'Carpet_Area_Sqft': carpet_area_sqft,
            'Bedrooms': bedrooms,
            'Bathrooms': bathrooms,
            'Balcony': balcony,
            'Number_Of_Amenities': number_of_amenities,
            # 'Security_Deposite' is excluded here
            'Floor_No': floor_no,
            'Total_floors_In_Building': total_floors_in_building,
            'Road_Connectivity': road_connectivity,
            'gym': gym,
            'gated_community': gated_community,
            'intercom': intercom,
            'lift': lift,
            'pet_allowed': pet_allowed,
            'pool': pool,
            'security': security_feature, # Use the renamed variable
            'water_supply': water_supply_type, # Use the renamed variable
            'wifi': wifi,
            'gas_pipeline': gas_pipeline,
            'sports_facility': sports_facility,
            'kids_area': kids_area,
            'power_backup': power_backup,
            'Garden': garden,
            'Fire_Support': fire_support,
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
            'Property_Age': property_age,
            'Area': area,
            'Zone': zone,
            'Furnishing Status': furnishing_status,
            'Recommended For': recommended_for,
            'Water Supply': water_supply_type,
            'Type of Society': type_of_society
        }

        # Create DataFrame from new data
        new_df = pd.DataFrame([new_data_dict])

        # Apply one-hot encoding for categorical features
        # Ensure 'categorical_features_for_prediction' from training is used
        new_df = pd.get_dummies(new_df, columns=categorical_features_for_prediction, drop_first=True)

        # Align columns with the training data's columns (loaded_features)
        # Add missing columns with 0, and drop extra columns
        missing_cols = set(loaded_features) - set(new_df.columns)
        for c in missing_cols:
            new_df[c] = 0
        new_df = new_df[loaded_features] # Ensure order is the same

        # Identify numerical columns to scale (excluding 'Security_Deposite')
        # This list must match the numerical_cols_present used during training.
        current_numerical_cols_present = [col for col in numerical_cols_present if col in new_df.columns]

        # Scale numerical features
        new_df[current_numerical_cols_present] = loaded_scaler.transform(new_df[current_numerical_cols_present])

        try:
            # Make prediction
            log_predicted_rent = loaded_rf_model.predict(new_df)[0]
            predicted_rent = np.expm1(log_predicted_rent) # Inverse transform

            st.subheader("Prediction Result:")
            st.success(f"Estimated Monthly Rent: **Rs {predicted_rent:.2f}**")

            # --- Price Classification ---
            FAIR_PRICE_TOLERANCE = 0.10 # 10%
            lower_bound = predicted_rent * (1 - FAIR_PRICE_TOLERANCE)
            upper_bound = predicted_rent * (1 + FAIR_PRICE_TOLERANCE)

            st.info(f"A fair price for this property would typically be between Rs {lower_bound:.2f} and Rs {upper_bound:.2f}.")

            # Optional: Allow user to input a listed price for comparison
            listed_price = st.number_input("Enter the listed price of the property for comparison (e.g., 25000):", min_value=0.0, value=float(predicted_rent))

            if listed_price < lower_bound:
                st.warning("This property appears to be **Underpriced**!")
            elif listed_price > upper_bound:
                st.warning("This property appears to be **Overpriced**!")
            else:
                st.success("This property appears to be **Fairly Priced**.")


        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            st.warning("Please check your input values. Ensure the model files are correctly loaded.")
    else:
        st.warning("Model components not loaded. Please restart the app or check for errors during initialization.")

st.markdown("---")
st.markdown(
    """
    ### How to Run This Application:
    1.  **Save** this code as a Python file (e.g., `rental_app.py`).
    2.  **Ensure** you have the necessary libraries installed:
        ```bash
        pip install streamlit pandas numpy scikit-learn joblib
        ```
    3.  **Run** the app from your terminal:
        ```bash
        streamlit run rental_app.py
        ```
    4.  Your browser will automatically open to the Streamlit app!
    *(Note: The model will train and save its files (`rf_model.pkl`, `scaler.pkl`, `model_features.pkl`)
    in the same directory when you run the app for the first time.)*
    """
)

