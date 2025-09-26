import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime
import matplotlib.pyplot as plt

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

# --- Area to Zone Mapping ---
AREA_TO_ZONE = {
    'Hingna': 'Rural', 'Trimurti Nagar': 'West Zone', 'Ashirwad Nagar': 'West Zone',
    'Beltarodi': 'East Zone', 'Besa': 'South Zone', 'Bharatwada': 'East Zone',
    'Boriyapura': 'East Zone', 'Chandrakiran Nagar': 'West Zone', 'Dabha': 'East Zone',
    'Dhantoli': 'Central Zone', 'Dharampeth': 'Central Zone', 'Dighori': 'East Zone',
    'Duttawadi': 'Central Zone', 'Gandhibagh': 'Central Zone', 'Ganeshpeth': 'Central Zone',
    'Godhni': 'North Zone', 'Gotal Panjri': 'North Zone', 'Hudkeswar': 'East Zone',
    'Itwari': 'Central Zone', 'Jaitala': 'West Zone', 'Jaripatka': 'North Zone',
    'Kalamna': 'East Zone', 'Kalmeshwar': 'Rural', 'Khamla': 'West Zone',
    'Kharbi': 'East Zone', 'Koradi Colony': 'North Zone', 'Kotewada': 'North Zone',
    'Mahal': 'Central Zone', 'Manewada': 'South Zone', 'Manish Nagar': 'West Zone',
    'Mankapur': 'West Zone', 'Medical Square': 'West Zone', 'MIHAN': 'East Zone',
    'Nandanwan': 'East Zone', 'Narendra Nagar Extension': 'West Zone',
    'Nari Village': 'South Zone', 'Narsala': 'East Zone', 'Omkar Nagar': 'West Zone',
    'Parvati Nagar': 'West Zone', 'Pratap Nagar': 'West Zone', 'Ram Nagar': 'West Zone',
    'Rameshwari': 'North Zone', 'Reshim Bagh': 'Central Zone', 'Sadar': 'Central Zone',
    'Sanmarga Nagar': 'West Zone', 'Seminary Hills': 'Central Zone',
    'Shatabdi Square': 'West Zone', 'Sitabuldi': 'Central Zone', 'Somalwada': 'West Zone',
    'Sonegaon': 'East Zone', 'Teka Naka': 'East Zone', 'Vayusena Nagar': 'West Zone',
    'Wanadongri': 'North Zone', 'Wardsman Nagar': 'West Zone', 'Wathoda': 'South Zone',
    'Zingabai Takli': 'Central Zone'
}

# --- Room Type Size Guidelines ---
ROOM_SIZE_GUIDELINES = {
    '1 RK': {'min': 200, 'max': 400},
    '1 BHK': {'min': 400, 'max': 700},
    '2 BHK': {'min': 700, 'max': 1100},
    '3 BHK': {'min': 1100, 'max': 1500},
    '4 BHK': {'min': 1500, 'max': 2200},
    '5+ BHK': {'min': 2200, 'max': 10000}
}

# --- Property Type Room Configuration Rules ---
PROPERTY_ROOM_RULES = {
    'Studio Apartment': {
        'bedrooms': {'min': 0, 'max': 0},
        'bathrooms': {'min': 1, 'max': 1},
        'balconies': {'min': 0, 'max': 1}
    },
    'Flat': {
        'bedrooms': {'min': 0, 'max': 5},
        'bathrooms': {'min': 1, 'max': 6},
        'balconies': {'min': 0, 'max': 5}
    },
    'Independent House': {
        'bedrooms': {'min': 1, 'max': 10},
        'bathrooms': {'min': 1, 'max': 10},
        'balconies': {'min': 0, 'max': 10}
    },
    'Independent Builder Floor': {
        'bedrooms': {'min': 1, 'max': 6},
        'bathrooms': {'min': 1, 'max': 6},
        'balconies': {'min': 0, 'max': 5}
    },
    'Villa': {
        'bedrooms': {'min': 2, 'max': 10},
        'bathrooms': {'min': 2, 'max': 10},
        'balconies': {'min': 1, 'max': 10}
    },
    'Duplex': {
        'bedrooms': {'min': 2, 'max': 6},
        'bathrooms': {'min': 2, 'max': 6},
        'balconies': {'min': 1, 'max': 5}
    }
}

# --- Room Type Configuration Rules ---
ROOM_TYPE_RULES = {
    '1 RK': {
        'bedrooms': {'min': 0, 'max': 0},
        'bathrooms': {'min': 1, 'max': 1},
        'balconies': {'min': 0, 'max': 1}
    },
    '1 BHK': {
        'bedrooms': {'min': 1, 'max': 1},
        'bathrooms': {'min': 1, 'max': 2},
        'balconies': {'min': 0, 'max': 2}
    },
    '2 BHK': {
        'bedrooms': {'min': 2, 'max': 2},
        'bathrooms': {'min': 1, 'max': 3},
        'balconies': {'min': 0, 'max': 3}
    },
    '3 BHK': {
        'bedrooms': {'min': 3, 'max': 3},
        'bathrooms': {'min': 2, 'max': 4},
        'balconies': {'min': 1, 'max': 4}
    },
    '4 BHK': {
        'bedrooms': {'min': 4, 'max': 4},
        'bathrooms': {'min': 2, 'max': 5},
        'balconies': {'min': 1, 'max': 5}
    },
    '5+ BHK': {
        'bedrooms': {'min': 5, 'max': 10},
        'bathrooms': {'min': 3, 'max': 10},
        'balconies': {'min': 1, 'max': 10}
    }
}

# --- Amenity Impact Percentages ---
# Define the percentage impact of each amenity on the rent
AMENITY_IMPACT = {
    'gym': 2.5, 'gated_community': 5.0, 'intercom': 1.0, 'lift': 1.5, 
    'pet_allowed': 2.0, 'pool': 3.5, 'security': 3.0, 'water_supply_amenity': 1.25,
    'wifi': 1.5, 'gas_pipeline': 1.0, 'sports_facility': 2.0, 'kids_area': 0.75,
    'power_backup': 2.5, 'garden': 1.5, 'fire_support': 1.0, 'parking': 6.5,
    'atm_near_me': 0.5, 'airport_near_me': 1.0, 'bus_stop_near_me': 0.25, 
    'hospital_near_me': 0.75, 'mall_near_me': 1.25, 'market_near_me': 0.75,
    'metro_station_near_me': 1.0, 'park_near_me': 0.5, 'school_near_me': 0.75,
    'vastu': 3.0  # Added Vastu compliance
}

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

# --- Validation Functions ---
def validate_property_details(data_dict):
    """Validate property details and return warnings if any."""
    warnings = []
    
    # Check if built-up area is greater than total area
    if data_dict.get('area_type') == "Built-up Area" and data_dict.get('area_value', 0) > data_dict.get('size', 0):
        warnings.append("Built-up area cannot be greater than total area!")
    
    # Check if 1 RK has bedrooms
    if data_dict.get('room_type') == "1 RK" and data_dict.get('bedrooms', 0) > 0:
        warnings.append("1 RK should not have bedrooms!")
    
    # Check if duplex has exactly 2 floors
    if data_dict.get('property_type') == "Duplex" and data_dict.get('total_floors', 0) != 2:
        warnings.append("Duplex property should have exactly 2 floors!")
    
    # Check bedroom, bathroom, balcony limits based on property type and room type
    property_type = data_dict.get('property_type', '')
    room_type = data_dict.get('room_type', '')
    bedrooms = data_dict.get('bedrooms', 0)
    bathrooms = data_dict.get('bathrooms', 0)
    balcony = data_dict.get('balcony', 0)
    size = data_dict.get('size', 0)
    
    # Validate based on property type
    if property_type in PROPERTY_ROOM_RULES:
        rules = PROPERTY_ROOM_RULES[property_type]
        
        # Check bedrooms
        if bedrooms < rules['bedrooms']['min'] or bedrooms > rules['bedrooms']['max']:
            warnings.append(f"For {property_type}, bedrooms should be between {rules['bedrooms']['min']} and {rules['bedrooms']['max']}!")
        
        # Check bathrooms
        if bathrooms < rules['bathrooms']['min'] or bathrooms > rules['bathrooms']['max']:
            warnings.append(f"For {property_type}, bathrooms should be between {rules['bathrooms']['min']} and {rules['bathrooms']['max']}!")
        
        # Check balconies
        if balcony < rules['balconies']['min'] or balcony > rules['balconies']['max']:
            warnings.append(f"For {property_type}, balconies should be between {rules['balconies']['min']} and {rules['balconies']['max']}!")
    
    # Validate based on room type
    if room_type in ROOM_TYPE_RULES:
        rules = ROOM_TYPE_RULES[room_type]
        
        # Check bedrooms
        if bedrooms < rules['bedrooms']['min'] or bedrooms > rules['bedrooms']['max']:
            warnings.append(f"For {room_type}, bedrooms should be between {rules['bedrooms']['min']} and {rules['bedrooms']['max']}!")
        
        # Check bathrooms
        if bathrooms < rules['bathrooms']['min'] or bathrooms > rules['bathrooms']['max']:
            warnings.append(f"For {room_type}, bathrooms should be between {rules['bathrooms']['min']} and {rules['bathrooms']['max']}!")
        
        # Check balconies
        if balcony < rules['balconies']['min'] or balcony > rules['balconies']['max']:
            warnings.append(f"For {room_type}, balconies should be between {rules['balconies']['min']} and {rules['balconies']['max']}!")
    
    # Validate size based on room type
    if room_type in ROOM_SIZE_GUIDELINES:
        guidelines = ROOM_SIZE_GUIDELINES[room_type]
        if size < guidelines['min'] or size > guidelines['max']:
            warnings.append(f"For {room_type}, size should be between {guidelines['min']} and {guidelines['max']} sq ft!")
    
    # Check if flat has appropriate characteristics
    if data_dict.get('property_type') == "Flat":
        if data_dict.get('total_floors', 0) < 2:
            warnings.append("Flat should be in a building with at least 2 floors!")
        if data_dict.get('floor_no', 0) > data_dict.get('total_floors', 0):
            warnings.append("Floor number cannot exceed total floors in building!")
    
    # Check bathroom to bedroom ratio
    if bedrooms > 0 and bathrooms > bedrooms + 2:
        warnings.append(f"Having {bathrooms} bathrooms for {bedrooms} bedrooms is unusual!")
    
    # Check balcony to bedroom ratio
    if bedrooms > 0 and balcony > bedrooms + 2:
        warnings.append(f"Having {balcony} balconies for {bedrooms} bedrooms is unusual!")
    
    return warnings

# --- Streamlit UI ---
st.title("Rental Price Prediction App")
st.markdown("Enter the details of the property to predict its fair rental price.")

if rf_model is not None and scaler is not None and features is not None:
    
    col1, col2 = st.columns(2)

    with col1:
        st.header("Property Details")
        size = st.number_input("Size In Sqft", min_value=0, max_value=20000, value=1000, key='size')
        
        # New: Area Type and Value input inside an expander
        with st.expander("Area Details"):
            area_type_options = ["Carpet Area", "Built-up Area", "Super Area"]
            area_type = st.selectbox("Select Area Type:", area_type_options, key='area_type')
            area_value = st.number_input("Enter Area Value (Sqft)", min_value=0, max_value=50000, value=1500, key='area_value')
        
        bedrooms = st.number_input("Number of Bedrooms", min_value=0, max_value=10, value=2, key='bedrooms')
        bathrooms = st.number_input("Number of Bathrooms", min_value=0, max_value=10, value=2, key='bathrooms')
        balcony = st.number_input("Number of Balconies", min_value=0, max_value=10, value=1, key='balcony')
        total_floors = st.number_input("Total Floors In Building", min_value=0, max_value=50, value=4, key='total_floors')
        floor_no = st.number_input("Floor No", min_value=0, max_value=total_floors, value=1, key='floor_no')
        property_age = st.number_input("Property Age (in years)", min_value=0, max_value=100, value=5, key='property_age')
        
        # Removed manual amenities count input - will be calculated automatically
        security_deposite = st.number_input("Security Deposite", min_value=0, value=20000, key='security_deposite')
        road_connectivity = st.slider("Road Connectivity (1-10)", min_value=0, max_value=10, value=5, key='road_connectivity')
        
    with col2:
        st.header("Categorical & Binary Features")
        
        area_options = ['Hingna', 'Trimurti Nagar', 'Ashirwad Nagar', 'Beltarodi', 'Besa', 'Bharatwada', 'Boriyapura', 'Chandrakiran Nagar', 'Dabha', 'Dhantoli', 'Dharampeth', 'Dighori', 'Duttawadi', 'Gandhibagh', 'Ganeshpeth', 'Godhni', 'Gotal Panjri', 'Hudkeswar', 'Itwari', 'Jaitala', 'Jaripatka', 'Kalamna', 'Kalmeshwar', 'Khamla', 'Kharbi', 'Koradi Colony', 'Kotewada', 'Mahal', 'Manewada', 'Manish Nagar', 'Mankapur', 'Medical Square', 'MIHAN', 'Nandanwan', 'Narendra Nagar Extension', 'Nari Village', 'Narsala', 'Omkar Nagar', 'Parvati Nagar', 'Pratap Nagar', 'Ram Nagar', 'Rameshwari', 'Reshim Bagh', 'Sadar', 'Sanmarga Nagar', 'Seminary Hills', 'Shatabdi Square', 'Sitabuldi', 'Somalwada', 'Sonegaon', 'Teka Naka', 'Vayusena Nagar', 'Wanadongri', 'Wardsman Nagar', 'Wathoda', 'Zingabai Takli']
        area = st.selectbox("Select Area:", area_options, key='area')
        
        # Auto-set zone based on area
        default_zone = AREA_TO_ZONE.get(area, 'West Zone')
        zone_options = ['East Zone', 'North Zone', 'South Zone', 'West Zone', 'Central Zone', 'Rural']
        zone = st.selectbox("Select Zone:", zone_options, index=zone_options.index(default_zone) if default_zone in zone_options else 0, key='zone')
        
        furnishing_status_options = ['Fully Furnished', 'Semi Furnished', 'Unfurnished']
        furnishing_status = st.selectbox("Select Furnishing Status:", furnishing_status_options, key='furnishing_status')
        
        recommended_for_options = ['Anyone', 'Bachelors', 'Family', 'Family and Bachelors', 'Family and Company']
        recommended_for = st.selectbox("Recommended For:", recommended_for_options, key='recommended_for')
        
        water_supply_options_categorical = ['Borewell', 'Both', 'Municipal']
        municipal_bore_water = st.selectbox("Municipal Water Or Bore Water:", water_supply_options_categorical, key='municipal_bore_water')

        type_of_society_options = ['Gated','Non-Gated','Township']
        type_of_society = st.selectbox("Type of Society:", type_of_society_options, key='type_of_society')

        room_type_options = ['1 RK', '1 BHK', '2 BHK', '3 BHK', '4 BHK', '5+ BHK']
        room_type = st.selectbox("Room Type:", room_type_options, key='room_type')
        
        # Auto-set bedrooms to 0 for 1 RK
        if room_type == "1 RK":
            st.info("1 RK selected: Number of bedrooms automatically set to 0")
            bedrooms = 0

        property_type_options = ['Flat','Studio Apartment','Independent House','Independent Builder Floor','Villa','Duplex']
        property_type = st.selectbox("Property Type:", property_type_options, key='property_type')
        
        # Auto-set total floors to 2 for Duplex
        if property_type == "Duplex":
            st.info("Duplex selected: Total floors automatically set to 2")
            total_floors = 2

        brokerage_options = ['No Brokerage', 'With Brokerage']
        brokerage = st.selectbox("Brokerage:", brokerage_options, key='brokerage')

        maintenance_charge_options = ['Maintenance Not Included', 'Maintenance Included']
        maintenance_charge = st.selectbox("Maintenance Charge:", maintenance_charge_options, key='maintenance_charge')


        # --- Organized Amenities & Proximity using expanders ---
        st.subheader("Amenities & Proximity (Check if available)")

        # Store checkbox states in session state to access them after button click
        if 'amenity_states' not in st.session_state:
            st.session_state['amenity_states'] = {}
            for amenity_key in AMENITY_IMPACT.keys():
                st.session_state['amenity_states'][amenity_key] = False

        with st.expander("Property Amenities"):
            col_a, col_b = st.columns(2)
            with col_a:
                st.session_state['amenity_states']['gym'] = st.checkbox("Gym (+2.5%)", key='gym_cb')
                st.session_state['amenity_states']['intercom'] = st.checkbox("Intercom (+1.0%)", key='intercom_cb')
                st.session_state['amenity_states']['pet_allowed'] = st.checkbox("Pet Allowed (+2.0%)", key='pet_allowed_cb')
                st.session_state['amenity_states']['security'] = st.checkbox("Security (+3.0%)", key='security_cb')
                st.session_state['amenity_states']['gas_pipeline'] = st.checkbox("Gas Pipeline (+1.0%)", key='gas_pipeline_cb')
                st.session_state['amenity_states']['power_backup'] = st.checkbox("Power Backup (+2.5%)", key='power_backup_cb')
                st.session_state['amenity_states']['fire_support'] = st.checkbox("Fire Support (+1.0%)", key='fire_support_cb')
                st.session_state['amenity_states']['vastu'] = st.checkbox("Vastu Compliant (+3.0%)", key='vastu_cb')
            with col_b:
                st.session_state['amenity_states']['gated_community'] = st.checkbox("Gated Community (+5.0%)", key='gated_community_cb')
                st.session_state['amenity_states']['lift'] = st.checkbox("Lift (+1.5%)", key='lift_cb')
                st.session_state['amenity_states']['pool'] = st.checkbox("Pool (+3.5%)", key='pool_cb')
                st.session_state['amenity_states']['water_supply_amenity'] = st.checkbox("Water Supply (as amenity) (+1.25%)", help="Check if this specific water supply amenity is available", key='water_supply_amenity_cb')
                st.session_state['amenity_states']['wifi'] = st.checkbox("WiFi (+1.5%)", key='wifi_cb')
                st.session_state['amenity_states']['sports_facility'] = st.checkbox("Sports Facility (+2.0%)", key='sports_facility_cb')
                st.session_state['amenity_states']['kids_area'] = st.checkbox("Kids Area (+0.75%)", key='kids_area_cb')
                st.session_state['amenity_states']['garden'] = st.checkbox("Garden (+1.5%)", key='garden_cb')
                st.session_state['amenity_states']['parking'] = st.checkbox("Parking (+2.5%)", key='parking_cb')

        with st.expander("Proximity to Essential Services"):
            col_c, col_d = st.columns(2)
            with col_c:
                st.session_state['amenity_states']['atm_near_me'] = st.checkbox("ATM Near Me (+0.5%)", key='atm_near_me_cb')
                st.session_state['amenity_states']['bus_stop_near_me'] = st.checkbox("Bus Stop Near Me (+0.25%)", key='bus_stop_near_me_cb')
                st.session_state['amenity_states']['mall_near_me'] = st.checkbox("Mall Near Me (+1.25%)", key='mall_near_me_cb')
                st.session_state['amenity_states']['metro_station_near_me'] = st.checkbox("Metro Station Near Me (+1.0%)", key='metro_station_near_me_cb')
                st.session_state['amenity_states']['school_near_me'] = st.checkbox("School Near Me (+0.75%)", key='school_near_me_cb')
            with col_d:
                st.session_state['amenity_states']['airport_near_me'] = st.checkbox("Airport Near Me (+1.0%)", key='airport_near_me_cb')
                st.session_state['amenity_states']['hospital_near_me'] = st.checkbox("Hospital Near Me (+0.75%)", key='hospital_near_me_cb')
                st.session_state['amenity_states']['market_near_me'] = st.checkbox("Market Near Me (+0.75%)", key='market_near_me_cb')
                st.session_state['amenity_states']['park_near_me'] = st.checkbox("Park Near Me (+0.5%)", key='park_near_me_cb')


    # --- New User Inputs for Future Rate Prediction ---
    st.markdown("---")
    st.subheader("Future Rental Rate Projection")
    projection_years = st.slider("Years from now to project:", min_value=1, max_value=20, value=5, key='projection_years')
    annual_growth_rate = st.slider("Expected Annual Growth Rate (%):", min_value=0.0, max_value=10.0, value=3.5, step=0.1, key='annual_growth_rate')
    
    # This remains the user's input for the "actual" listed price for comparison.
    listed_price = st.number_input("Enter the Listed Price of the property for comparison:", min_value=0, value=25000, key='listed_price_comp')


    # When the user clicks the predict button
    if st.button("Predict Rent"):
        # Calculate the number of amenities based on checked boxes
        amenities_count = sum(1 for amenity_key, state in st.session_state['amenity_states'].items() if state)
        
        # Calculate amenity impact percentage
        total_amenity_impact = 0
        amenity_impact_details = {}
        for amenity_key, impact in AMENITY_IMPACT.items():
            if st.session_state['amenity_states'].get(amenity_key, False):
                total_amenity_impact += impact
                amenity_impact_details[amenity_key] = impact
        
        # Define conversion ratios (adjust as needed for your local market)
        built_up_to_carpet_ratio = 0.85 # Example: Carpet is 85% of Built-up
        super_to_carpet_ratio = 0.70    # Example: Carpet is 70% of Super Built-up

        # Convert the entered area to carpet area based on area_type
        converted_carpet_area = area_value
        if area_type == "Built-up Area":
            converted_carpet_area = area_value * built_up_to_carpet_ratio
        elif area_type == "Super Area":
            converted_carpet_area = area_value * super_to_carpet_ratio

        user_input_data = {
            'Size_In_Sqft': size,
            'Carpet_Area_Sqft': converted_carpet_area, # Use the converted carpet area
            'Bedrooms': bedrooms, 'Bathrooms': bathrooms,
            'Balcony': balcony, 'Number_Of_Amenities': amenities_count, # Now calculated automatically
            'Security_Deposite': security_deposite,
            'Floor_No': floor_no, 'Total_floors_In_Building': total_floors, 'Road_Connectivity': road_connectivity,
            # Pass the 0/1 status for the model based on the session state
            'gym': 1 if st.session_state['amenity_states']['gym'] else 0,
            'gated_community': 1 if st.session_state['amenity_states']['gated_community'] else 0,
            'intercom': 1 if st.session_state['amenity_states']['intercom'] else 0,
            'lift': 1 if st.session_state['amenity_states']['lift'] else 0,
            'pet_allowed': 1 if st.session_state['amenity_states']['pet_allowed'] else 0,
            'pool': 1 if st.session_state['amenity_states']['pool'] else 0,
            'security': 1 if st.session_state['amenity_states']['security'] else 0,
            'water_supply': 1 if st.session_state['amenity_states']['water_supply_amenity'] else 0, 
            'wifi': 1 if st.session_state['amenity_states']['wifi'] else 0,
            'gas_pipeline': 1 if st.session_state['amenity_states']['gas_pipeline'] else 0,
            'sports_facility': 1 if st.session_state['amenity_states']['sports_facility'] else 0,
            'kids_area': 1 if st.session_state['amenity_states']['kids_area'] else 0,
            'power_backup': 1 if st.session_state['amenity_states']['power_backup'] else 0,
            'Garden': 1 if st.session_state['amenity_states']['garden'] else 0, 
            'Fire_Support': 1 if st.session_state['amenity_states']['fire_support'] else 0, 
            'Parking': 1 if st.session_state['amenity_states']['parking'] else 0, 
            'ATM_Near_me': 1 if st.session_state['amenity_states']['atm_near_me'] else 0,
            'Airport_Near_me': 1 if st.session_state['amenity_states']['airport_near_me'] else 0,
            'Bus_Stop__Near_me': 1 if st.session_state['amenity_states']['bus_stop_near_me'] else 0,
            'Hospital_Near_me': 1 if st.session_state['amenity_states']['hospital_near_me'] else 0,
            'Mall_Near_me': 1 if st.session_state['amenity_states']['mall_near_me'] else 0,
            'Market_Near_me': 1 if st.session_state['amenity_states']['market_near_me'] else 0,
            'Metro_Station_Near_me': 1 if st.session_state['amenity_states']['metro_station_near_me'] else 0,
            'Park_Near_me': 1 if st.session_state['amenity_states']['park_near_me'] else 0,
            'School_Near_me': 1 if st.session_state['amenity_states']['school_near_me'] else 0,
            'Property_Age': property_age,
            'City': 'Nagpur', 'Area': area, 'Zone': zone, 'Frurnishing_Status': furnishing_status,
            'Recomened for': recommended_for, 'Muncipla Water Or Bore Water': municipal_bore_water,
            'Type of Society': type_of_society, 'Room': room_type, 'Type': property_type,
            'Brokerage': brokerage, 'Maintenance_Charge': maintenance_charge,
            # Add area_type and area_value for validation
            'area_type': area_type, 'area_value': area_value
        }

        # Validate property details
        validation_warnings = validate_property_details(user_input_data)
        
        st.markdown("---")
        st.subheader("Prediction Results")

        # Display validation warnings if any
        if validation_warnings:
            st.warning("Property Validation Warnings:")
            for warning in validation_warnings:
                st.warning(f"- {warning}")

        # Get and display the current date
        today = datetime.date.today()
        st.info(f"Prediction based on today's market conditions: **{today.strftime('%B %d, %Y')}**")

        # Predict with the single Model (base predicted rent)
        base_predicted_rent = predict_rent_with_model(rf_model, scaler, features, user_input_data)
        
        # Calculate Adjusted Predicted Rent by applying amenity impact percentages
        adjusted_predicted_rent = None
        if base_predicted_rent is not None:
            # Apply total amenity impact percentage
            adjusted_predicted_rent = base_predicted_rent * (1 + total_amenity_impact / 100)

        if base_predicted_rent is not None:
            st.success(f"Base Predicted Rent (without amenities): **Rs {base_predicted_rent:,.2f}**")
            st.info(f"**Total Amenity Impact:** +{total_amenity_impact:.2f}%")
            
            # Display amenity impact details
            with st.expander("View Amenity Impact Breakdown"):
                for amenity, impact in amenity_impact_details.items():
                    st.write(f"- {amenity.replace('_', ' ').title()}: +{impact:.2f}%")
            
            if adjusted_predicted_rent is not None:
                # Display Adjusted Predicted Rent in white and bigger size
                st.markdown(f"<span style='color:white; font-weight:bold; font-size: 3em;'>Rent ₹ {adjusted_predicted_rent:,.2f}</span>", unsafe_allow_html=True)

                # --- Future Rent Calculation (using adjusted_predicted_rent) ---
                future_predicted_rent_adjusted = adjusted_predicted_rent * (1 + annual_growth_rate / 100)**projection_years
                
                st.info(f"**Projected Adjusted Rent in {projection_years} years:**")
                st.success(f"Rs {future_predicted_rent_adjusted:,.2f} (assuming a {annual_growth_rate:.1f}% annual growth rate)")

                # --- Price Comparison (comparing Listed Price to Adjusted Predicted Rent) ---
                FAIR_PRICE_TOLERANCE = 0.3
                
                st.markdown("---")
                st.subheader("Price Comparison")

                st.markdown(f"**User Entered Listed Price:** Rs {listed_price:,.2f}")
                st.markdown(f"**Comparison based on Adjusted Predicted Rent (Rs {adjusted_predicted_rent:,.2f}):**")
                
                lower_bound = adjusted_predicted_rent * (1 - FAIR_PRICE_TOLERANCE)
                upper_bound = adjusted_predicted_rent * (1 + FAIR_PRICE_TOLERANCE)
                st.text(f"Fair range for Adjusted Predicted Rent: Rs {lower_bound:,.2f} - Rs {upper_bound:,.2f}")
                
                # Compare the user's listed price against the fair range of the adjusted predicted rent
                if listed_price < lower_bound:
                    st.warning(f"Listed price {listed_price:,.2f} appears to be **Underpriced** compared to Adjusted Predicted Rent!")
                elif listed_price > upper_bound:
                    st.warning(f"Listed price {listed_price:,.2f} appears to be **Overpriced** compared to Adjusted Predicted Rent!")
                else:
                    st.success(f"Listed price {listed_price:,.2f} appears to be **Fairly Priced** compared to Adjusted Predicted Rent!")

                # --- 15-Year Predicted Price Projection and Graph (using adjusted_predicted_rent) ---
                st.markdown("---")
                st.subheader("15-Year Adjusted Predicted Rent Projection")
                
                if adjusted_predicted_rent > 0:
                    st.info(f"Projecting the Adjusted Predicted Rent (Rs {adjusted_predicted_rent:,.2f}) with a {annual_growth_rate:.1f}% annual increase:")
                    
                    # Create lists for the full projection data
                    yearly_projections = []
                    prices_for_plot = []
                    
                    current_projected_price = adjusted_predicted_rent # Start with adjusted predicted rent
                    for year in range(1, 16):
                        current_projected_price *= (1 + annual_growth_rate / 100)
                        yearly_projections.append(f"**Year {year}:** Rs {current_projected_price:,.2f}")
                        prices_for_plot.append(current_projected_price)
                    
                    # Display the full list of projections
                    st.markdown("\n".join(yearly_projections))
                    
                    # Filter for odd years to plot
                    odd_years_to_plot = [y for y in range(1, 16) if y % 2 != 0]
                    odd_prices_to_plot = [prices_for_plot[y-1] for y in odd_years_to_plot if (y-1) < len(prices_for_plot)]

                    # Create the plot
                    plt.figure(figsize=(10, 6))
                    plt.plot(odd_years_to_plot, odd_prices_to_plot, marker='o', linestyle='-')
                    
                    # Add titles and labels
                    plt.title('15-Year Adjusted Predicted Rent Projection (Odd Years Only)')
                    plt.xlabel('Year')
                    plt.ylabel('Projected Rent (Rs)')
                    plt.xticks(odd_years_to_plot) # Set x-ticks to odd years for clarity
                    plt.grid(True)
                    plt.tight_layout()
                    
                    # Display the plot in the Streamlit app
                    st.pyplot(plt)
                    plt.clf() # Clear the current figure to prevent plots from overlapping

                else:
                    st.warning("Adjusted Predicted Rent is not positive. Cannot generate 15-year projection.")
            else:
                st.error("Could not calculate adjusted predicted rent.")

else:
    st.warning("Cannot run prediction. Please ensure all model files ('m.pkl', 's.pkl', and 'f.pkl') are available in the same directory.")
