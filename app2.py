import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- Constants ---
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

FAIR_PRICE_TOLERANCE = 0.07  # 7% tolerance

# --- Load Both Models ---
@st.cache_resource
def load_models():
    loaded_models = {}
    model_files = {
        "RandomForest_1": {"model": "m.pkl", "scaler": "s.pkl", "features": "f.pkl"},
        "RandomForest_2": {"model": "m3.pkl", "scaler": "s3.pkl", "features": "f3.pkl"}
    }
    for name, files in model_files.items():
        try:
            loaded_models[name] = {
                "model": joblib.load(files["model"]),
                "scaler": joblib.load(files["scaler"]),
                "features": joblib.load(files["features"])
            }
            st.success(f"{name} loaded successfully.")
        except FileNotFoundError:
            st.error(f"{name} files not found. Please check the directory.")
        except Exception as e:
            st.error(f"Error loading {name}: {e}")
    return loaded_models

models = load_models()

# --- Prediction Function ---
def predict_rent_all_models(models, data_dict):
    results = {}
    for model_name, resources in models.items():
        model = resources["model"]
        scaler = resources["scaler"]
        original_features = resources["features"]
        
        new_df = pd.DataFrame([data_dict])
        # One-hot encode categorical features
        for feature in CATEGORICAL_FEATURES:
            if feature in new_df.columns:
                temp_df = pd.get_dummies(new_df[[feature]], prefix=feature)
                new_df = new_df.drop(columns=[feature])
                new_df = pd.concat([new_df.reset_index(drop=True), temp_df.reset_index(drop=True)], axis=1)
        
        # Align columns
        missing_cols = set(original_features) - set(new_df.columns)
        for c in missing_cols:
            new_df[c] = 0
        new_df = new_df[original_features]
        
        # Scale numerical features
        num_cols = [col for col in NUMERICAL_FEATURES if col in original_features]
        if not new_df[num_cols].empty:
            new_df[num_cols] = scaler.transform(new_df[num_cols])
        
        # Make prediction
        try:
            log_pred = model.predict(new_df)[0]
            results[model_name] = np.expm1(log_pred)
        except Exception as e:
            results[model_name] = None
            st.error(f"Prediction failed for {model_name}: {e}")
    return results

# --- Streamlit UI ---
st.title("🏠 Rental Price Prediction App")
st.markdown("Enter the details of the property to predict its fair rental price using **two models**.")

if models:
    col1, col2 = st.columns(2)

    with col1:
        st.header("Property Details")
        size = st.number_input("Size In Sqft", min_value=100, max_value=20000, value=1000)
        carpet_area = st.number_input("Carpet Area Sqft", min_value=100, max_value=20000, value=1000)
        bedrooms = st.number_input("Number of Bedrooms", min_value=1, max_value=10, value=2)
        bathrooms = st.number_input("Number of Bathrooms", min_value=1, max_value=10, value=2)
        balcony = st.number_input("Number of Balconies", min_value=0, max_value=10, value=1)
        total_floors = st.number_input("Total Floors In Building", min_value=1, max_value=50, value=4)
        floor_no = st.number_input("Floor No", min_value=0, max_value=total_floors, value=1)
        property_age = st.number_input("Property Age (years)", min_value=0, max_value=100, value=5)
        amenities_count = st.number_input("Number of Amenities", min_value=0, max_value=30, value=5)
        security_deposite = st.number_input("Security Deposite", min_value=0, value=20000)
        road_connectivity = st.slider("Road Connectivity (1-10)", min_value=1, max_value=10, value=5)

    with col2:
        st.header("Categorical & Binary Features")
        area_options = ['Hingna', 'Trimurti Nagar', 'Ashirwad Nagar', 'Beltarodi', 'Besa']
        area = st.selectbox("Select Area:", area_options)
        
        zone_options = ['East Zone', 'North Zone', 'South Zone', 'West Zone', 'Rural']
        zone = st.selectbox("Select Zone:", zone_options)
        
        furnishing_status_options = ['Fully Furnished', 'Semi Furnished', 'Unfurnished']
        furnishing_status = st.selectbox("Furnishing Status:", furnishing_status_options)
        
        recommended_for_options = ['Anyone', 'Bachelors', 'Family', 'Family and Bachelors', 'Family and Company']
        recommended_for = st.selectbox("Recommended For:", recommended_for_options)
        
        water_supply_options_categorical = ['Borewell', 'Both', 'Municipal']
        municipal_bore_water = st.selectbox("Municipal/Bore Water:", water_supply_options_categorical)

        type_of_society_options = ['Gated','Non-Gated','Township']
        type_of_society = st.selectbox("Type of Society:", type_of_society_options)

        room_options = ['1 RK', '1 BHK', '2 BHK', '3 BHK', '4 BHK', '5+ BHK']
        room_type = st.selectbox("Room Type:", room_options)

        property_type_options = ['Flat','Studio Apartment','Independent House','Independent Builder Floor','Villa','Duplex']
        property_type = st.selectbox("Property Type:", property_type_options)

        brokerage_options = ['No Brokerage', 'With Brokerage']
        brokerage = st.selectbox("Brokerage:", brokerage_options)

        maintenance_charge_options = ['Maintenance Not Included', 'Maintenance Included']
        maintenance_charge = st.selectbox("Maintenance Charge:", maintenance_charge_options)

        st.subheader("Amenities & Proximity")
        gym = st.checkbox("Gym")
        gated_community = st.checkbox("Gated Community")
        intercom = st.checkbox("Intercom")
        lift = st.checkbox("Lift")
        pet_allowed = st.checkbox("Pet Allowed")
        pool = st.checkbox("Pool")
        security = st.checkbox("Security")
        water_supply_amenity = st.checkbox("Water Supply (amenity)")
        wifi = st.checkbox("WiFi")
        gas_pipeline = st.checkbox("Gas Pipeline")
        sports_facility = st.checkbox("Sports Facility")
        kids_area = st.checkbox("Kids Area")
        power_backup = st.checkbox("Power Backup")
        garden = st.checkbox("Garden")
        fire_support = st.checkbox("Fire Support")
        parking = st.checkbox("Parking")
        atm_near_me = st.checkbox("ATM Near Me")
        airport_near_me = st.checkbox("Airport Near Me")
        bus_stop_near_me = st.checkbox("Bus Stop Near Me")
        hospital_near_me = st.checkbox("Hospital Near Me")
        mall_near_me = st.checkbox("Mall Near Me")
        market_near_me = st.checkbox("Market Near Me")
        metro_station_near_me = st.checkbox("Metro Station Near Me")
        park_near_me = st.checkbox("Park Near Me")
        school_near_me = st.checkbox("School Near Me")

    # --- Predict Button ---
    if st.button("Predict Rent"):
        user_input_data = {
            'Size_In_Sqft': size, 'Carpet_Area_Sqft': carpet_area, 'Bedrooms': bedrooms,
            'Bathrooms': bathrooms, 'Balcony': balcony, 'Number_Of_Amenities': amenities_count,
            'Security_Deposite': security_deposite, 'Floor_No': floor_no,
            'Total_floors_In_Building': total_floors, 'Road_Connectivity': road_connectivity,
            'gym': int(gym), 'gated_community': int(gated_community), 'intercom': int(intercom),
            'lift': int(lift), 'pet_allowed': int(pet_allowed), 'pool': int(pool),
            'security': int(security), 'water_supply': int(water_supply_amenity),
            'wifi': int(wifi), 'gas_pipeline': int(gas_pipeline), 'sports_facility': int(sports_facility),
            'kids_area': int(kids_area), 'power_backup': int(power_backup), 'Garden': int(garden),
            'Fire_Support': int(fire_support), 'Parking': int(parking), 'ATM_Near_me': int(atm_near_me),
            'Airport_Near_me': int(airport_near_me), 'Bus_Stop__Near_me': int(bus_stop_near_me),
            'Hospital_Near_me': int(hospital_near_me), 'Mall_Near_me': int(mall_near_me),
            'Market_Near_me': int(market_near_me), 'Metro_Station_Near_me': int(metro_station_near_me),
            'Park_Near_me': int(park_near_me), 'School_Near_me': int(school_near_me), 'Property_Age': property_age,
            'City': 'Nagpur', 'Area': area, 'Zone': zone, 'Frurnishing_Status': furnishing_status,
            'Recomened for': recommended_for, 'Muncipla Water Or Bore Water': municipal_bore_water,
            'Type of Society': type_of_society, 'Room': room_type, 'Type': property_type,
            'Brokerage': brokerage, 'Maintenance_Charge': maintenance_charge
        }

        st.markdown("---")
        st.subheader("Predictions from Both Models")
        predictions = predict_rent_all_models(models, user_input_data)

        listed_price = st.number_input("Enter Listed Price for Comparison:", min_value=0, value=25000)

        for model_name, rent in predictions.items():
            if rent is not None:
                st.success(f"{model_name} Predicted Rent: Rs {rent:,.2f}")
                lower_bound = rent * (1 - FAIR_PRICE_TOLERANCE)
                upper_bound = rent * (1 + FAIR_PRICE_TOLERANCE)
                st.text(f"Fair Range: Rs {lower_bound:,.2f} - Rs {upper_bound:,.2f}")
                if listed_price < lower_bound:
                    st.warning(f"{model_name}: Property appears **Underpriced**!")
                elif listed_price > upper_bound:
                    st.warning(f"{model_name}: Property appears **Overpriced**!")
                else:
                    st.success(f"{model_name}: Property appears **Fairly Priced**.")
else:
    st.warning("Cannot run prediction. Please ensure all model files are available.")
