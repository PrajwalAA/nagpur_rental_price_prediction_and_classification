import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime  # Added this import

# --- Constants for features ---
# These lists define the features used in your model for proper alignment.
# Ensure these lists accurately reflect the features your model was trained on.

CATEGORICAL_FEATURES = [
    'City', 'Area', 'Zone', 'Frurnishing_Status', 'Brokerage', 'Maintenance_Charge',
    'Recomened for', 'Muncipla Water Or Bore Water', 'Type of Society', 'Room', 'Type'
]

NUMERICAL_FEATURES = [
    'Size_In_Sqft', 'Carpet_Area_Sqft', 'Bedrooms', 'Bathrooms', 'Balcony',
    'Number_Of_Amenities', 'Security_Deposite', 'Floor_No', 'Total_floors_In_Building',
    'Road_Connectivity', 'gated_community', 'gym', 'intercom', 'lift', 'pet_allowed', 'pool',
    'security', 'water_supply', 'wifi', 'gas_pipeline', 'sports_facility', 'kids_area',
    'power_backup', 'Garden', 'Fire_Support', 'Parking', 'ATM_Near_me', 'Airport_Near_me',
    'Bus_Stop__Near_me', 'Hospital_Near_me', 'Mall_Near_me', 'Market_Near_me',
    'Metro_Station_Near_me', 'Park_Near_me', 'School_Near_me', 'Property_Age'
]

# --- Load Model Resources ---
@st.cache_resource
def load_resources():
    """Loads the model, scaler, and feature file."""
    try:
        # Load Model 1 files
        rf_model = joblib.load('m.pkl')
        scaler = joblib.load('s.pkl')
        features = joblib.load('f.pkl')
        st.success("Model (m.pkl) and its resources loaded successfully.")
        return rf_model, scaler, features
    except FileNotFoundError as e:
        st.error(f"Error: A required file was not found. Please ensure 'm.pkl', 's.pkl', and 'f.pkl' are in the same directory.")
        st.info(f"Details: {e}")
        return None, None, None

rf_model, scaler, features = load_resources()

# --- Prediction Function ---
def predict_rent_with_model(model, scaler, original_df_columns, data_dict):
    """
    Makes a prediction using the model and its associated resources.
    Handles data preprocessing (one-hot encoding, column alignment, scaling).
    """
    if model is None or scaler is None or original_df_columns is None:
        return None

    # Create a DataFrame from the new data dictionary
    new_df = pd.DataFrame([data_dict])
    
    # Apply one-hot encoding
    for feature in CATEGORICAL_FEATURES:
        if feature in new_df.columns:
            # Create a temporary DataFrame for one-hot encoding
            temp_df = pd.get_dummies(new_df[[feature]], prefix=feature)
            
            # Drop the original categorical column from new_df and join the one-hot encoded columns
            new_df = new_df.drop(columns=[feature])
            new_df = pd.concat([new_df.reset_index(drop=True), temp_df.reset_index(drop=True)], axis=1)

    # Align columns with the training data (important for one-hot encoding)
    missing_cols = set(original_df_columns) - set(new_df.columns)
    for c in missing_cols:
        new_df[c] = 0
    new_df = new_df[original_df_columns] # Ensure order is the same

    # Scale numerical features
    numerical_cols_for_current_model = [col for col in NUMERICAL_FEATURES if col in original_df_columns]
    
    if not new_df[numerical_cols_for_current_model].empty:
        new_df[numerical_cols_for_current_model] = scaler.transform(new_df[numerical_cols_for_current_model])

    # Make prediction
    try:
        log_predicted_rent = model.predict(new_df)[0]
        predicted_rent = np.expm1(log_predicted_rent) # Inverse transform
        return predicted_rent
    except Exception as e:
        st.error(f"Prediction failed for model. Error: {e}")
        return None

# --- Streamlit UI ---
st.title("Rental Price Prediction App")
st.markdown("Enter the details of the property to predict its fair rental price.")

if rf_model is not None and scaler is not None and features is not None:
    
    col1, col2 = st.columns(2)

    with col1:
        st.header("Property Details")
        size = st.number_input("Size In Sqft", min_value=0, max_value=20000, value=1000, key='size')
        carpet_area = st.number_input("Carpet Area Sqft", min_value=0, max_value=20000, value=1000, key='carpet_area')
        bedrooms = st.number_input("Number of Bedrooms", min_value=0, max_value=10, value=2, key='bedrooms')
        bathrooms = st.number_input("Number of Bathrooms", min_value=0, max_value=10, value=2, key='bathrooms')
        balcony = st.number_input("Number of Balconies", min_value=0, max_value=10, value=1, key='balcony')
        total_floors = st.number_input("Total Floors In Building", min_value=0, max_value=50, value=4, key='total_floors')
        floor_no = st.number_input("Floor No", min_value=0, max_value=total_floors, value=1, key='floor_no')
        property_age = st.number_input("Property Age (in years)", min_value=0, max_value=100, value=5, key='property_age')
        amenities_count = st.number_input("Number of Amenities", min_value=0, max_value=30, value=5, key='amenities_count')
        security_deposite = st.number_input("Security Deposite", min_value=0, value=20000, key='security_deposite')
        road_connectivity = st.slider("Road Connectivity (1-10)", min_value=0, max_value=10, value=5, key='road_connectivity')
        
    with col2:
        st.header("Categorical & Binary Features")
        
        area_options = ['Hingna', 'Trimurti Nagar', 'Ashirwad Nagar', 'Beltarodi', 'Besa', 'Bharatwada', 'Boriyapura', 'Chandrakiran Nagar', 'Dabha', 'Dhantoli', 'Dharampeth', 'Dighori', 'Duttawadi', 'Gandhibagh', 'Ganeshpeth', 'Godhni', 'Gotal Panjri', 'Hudkeswar', 'Itwari', 'Jaitala', 'Jaripatka', 'Kalamna', 'Kalmeshwar', 'Khamla', 'Kharbi', 'Koradi Colony', 'Kotewada', 'Mahal', 'Manewada', 'Manish Nagar', 'Mankapur', 'Medical Square', 'MIHAN', 'Nandanwan', 'Narendra Nagar Extension', 'Nari Village', 'Narsala', 'Omkar Nagar', 'Parvati Nagar', 'Pratap Nagar', 'Ram Nagar', 'Rameshwari', 'Reshim Bagh', 'Sadar', 'Sanmarga Nagar', 'Seminary Hills', 'Shatabdi Square', 'Sitabuldi', 'Somalwada', 'Sonegaon', 'Teka Naka', 'Vayusena Nagar', 'Wanadongri', 'Wardsman Nagar', 'Wathoda', 'Zingabai Takli']
        area = st.selectbox("Select Area:", area_options, key='area')
        
        zone_options = ['East Zone', 'North Zone', 'South Zone', 'West Zone', 'Rural']
        zone = st.selectbox("Select Zone:", zone_options, key='zone')
        
        furnishing_status_options = ['Fully Furnished', 'Semi Furnished', 'Unfurnished']
        furnishing_status = st.selectbox("Select Furnishing Status:", furnishing_status_options, key='furnishing_status')
        
        recommended_for_options = ['Anyone', 'Bachelors', 'Family', 'Family and Bachelors', 'Family and Company']
        recommended_for = st.selectbox("Recommended For:", recommended_for_options, key='recommended_for')
        
        water_supply_options_categorical = ['Borewell', 'Both', 'Municipal']
        municipal_bore_water = st.selectbox("Municipal Water Or Bore Water:", water_supply_options_categorical, key='municipal_bore_water')

        type_of_society_options = ['Gated','Non-Gated','Township']
        type_of_society = st.selectbox("Type of Society:", type_of_society_options, key='type_of_society')

        room_options = ['1 RK', '1 BHK', '2 BHK', '3 BHK', '4 BHK', '5+ BHK']
        room_type = st.selectbox("Room Type:", room_options, key='room_type')

        property_type_options = ['Flat','Studio Apartment','Independent House','Independent Builder Floor','Villa','Duplex']
        property_type = st.selectbox("Property Type:", property_type_options, key='property_type')

        brokerage_options = ['No Brokerage', 'With Brokerage']
        brokerage = st.selectbox("Brokerage:", brokerage_options, key='brokerage')

        maintenance_charge_options = ['Maintenance Not Included', 'Maintenance Included']
        maintenance_charge = st.selectbox("Maintenance Charge:", maintenance_charge_options, key='maintenance_charge')


        st.subheader("Amenities & Proximity (Check if available)")
        gym = st.checkbox("Gym", key='gym')
        gated_community = st.checkbox("Gated Community", key='gated_community')
        intercom = st.checkbox("Intercom", key='intercom')
        lift = st.checkbox("Lift", key='lift')
        pet_allowed = st.checkbox("Pet Allowed", key='pet_allowed')
        pool = st.checkbox("Pool", key='pool')
        security = st.checkbox("Security", key='security')
        water_supply_amenity = st.checkbox("Water Supply (as amenity)", help="Check if this specific water supply amenity is available", key='water_supply_amenity')
        wifi = st.checkbox("WiFi", key='wifi')
        gas_pipeline = st.checkbox("Gas Pipeline", key='gas_pipeline')
        sports_facility = st.checkbox("Sports Facility", key='sports_facility')
        kids_area = st.checkbox("Kids Area", key='kids_area')
        power_backup = st.checkbox("Power Backup", key='power_backup')
        garden = st.checkbox("Garden", key='garden')
        fire_support = st.checkbox("Fire Support", key='fire_support')
        parking = st.checkbox("Parking", key='parking')
        atm_near_me = st.checkbox("ATM Near Me", key='atm_near_me')
        airport_near_me = st.checkbox("Airport Near Me", key='airport_near_me')
        bus_stop_near_me = st.checkbox("Bus Stop Near Me", key='bus_stop_near_me')
        hospital_near_me = st.checkbox("Hospital Near Me", key='hospital_near_me')
        mall_near_me = st.checkbox("Mall Near Me", key='mall_near_me')
        market_near_me = st.checkbox("Market Near Me", key='market_near_me')
        metro_station_near_me = st.checkbox("Metro Station Near Me", key='metro_station_near_me')
        park_near_me = st.checkbox("Park Near Me", key='park_near_me')
        school_near_me = st.checkbox("School Near Me", key='school_near_me')

    # When the user clicks the predict button
    if st.button("Predict Rent"):
        user_input_data = {
            'Size_In_Sqft': size, 'Carpet_Area_Sqft': carpet_area, 'Bedrooms': bedrooms, 'Bathrooms': bathrooms,
            'Balcony': balcony, 'Number_Of_Amenities': amenities_count, 'Security_Deposite': security_deposite,
            'Floor_No': floor_no, 'Total_floors_In_Building': total_floors, 'Road_Connectivity': road_connectivity,
            'gym': 1 if gym else 0, 'gated_community': 1 if gated_community else 0, 'intercom': 1 if intercom else 0,
            'lift': 1 if lift else 0, 'pet_allowed': 1 if pet_allowed else 0, 'pool': 1 if pool else 0,
            'security': 1 if security else 0, 'water_supply': 1 if water_supply_amenity else 0, 'wifi': 1 if wifi else 0,
            'gas_pipeline': 1 if gas_pipeline else 0, 'sports_facility': 1 if sports_facility else 0, 'kids_area': 1 if kids_area else 0,
            'power_backup': 1 if power_backup else 0, 'Garden': 1 if garden else 0, 'Fire_Support': 1 if fire_support else 0,
            'Parking': 1 if parking else 0, 'ATM_Near_me': 1 if atm_near_me else 0, 'Airport_Near_me': 1 if airport_near_me else 0,
            'Bus_Stop__Near_me': 1 if bus_stop_near_me else 0, 'Hospital_Near_me': 1 if hospital_near_me else 0,
            'Mall_Near_me': 1 if mall_near_me else 0, 'Market_Near_me': 1 if market_near_me else 0,
            'Metro_Station_Near_me': 1 if metro_station_near_me else 0, 'Park_Near_me': 1 if park_near_me else 0,
            'School_Near_me': 1 if school_near_me else 0, 'Property_Age': property_age,
            'City': 'Nagpur', 'Area': area, 'Zone': zone, 'Frurnishing_Status': furnishing_status,
            'Recomened for': recommended_for, 'Muncipla Water Or Bore Water': municipal_bore_water,
            'Type of Society': type_of_society, 'Room': room_type, 'Type': property_type,
            'Brokerage': brokerage, 'Maintenance_Charge': maintenance_charge
        }

        st.markdown("---")
        st.subheader("Prediction Results")

        # Get and display the current date
        today = datetime.date.today()
        st.info(f"Prediction for property on: **{today.strftime('%B %d, %Y')}**")

        # Predict with the single Model
        predicted_rent = predict_rent_with_model(rf_model, scaler, features, user_input_data)
        if predicted_rent is not None:
            st.success(f"Predicted Rent: **Rs {predicted_rent:,.2f}**")
            st.warning(f"**The predicted rent is: Rs {predicted_rent:,.2f}**") # Since there's only one model, its prediction is the median.

        # --- Price Classification ---
        FAIR_PRICE_TOLERANCE = 0.5
        
        st.markdown("---")
        st.subheader("Price Comparison")
        listed_price = st.number_input("Enter the listed price of the property for comparison:", min_value=0, value=25000, key='listed_price_comp')

        if predicted_rent is not None:
            st.markdown(f"**Comparison based on Model's Prediction (Rs {predicted_rent:,.2f}):**")
            lower_bound = predicted_rent * (1 - FAIR_PRICE_TOLERANCE)
            upper_bound = predicted_rent * (1 + FAIR_PRICE_TOLERANCE)
            st.text(f"Fair range: Rs {lower_bound:,.2f} - Rs {upper_bound:,.2f}")
            if listed_price < lower_bound:
                st.warning(f"Listed price {listed_price} appears to be **Underpriced**!")
            elif listed_price > upper_bound:
                st.warning(f"Listed price {listed_price} appears to be **Overpriced**!")
            else:
                st.success(f"Listed price {listed_price} appears to be **Fairly Priced**.")
else:
    st.warning("Cannot run prediction. Please ensure all model files ('m.pkl', 's.pkl', and 'f.pkl') are available in the same directory.")
