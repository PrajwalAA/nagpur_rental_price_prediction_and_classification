# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Any

# Set a single page configuration globally.
st.set_page_config(page_title="Property Price Prediction Hub", layout="wide")

# --- Constants for features (keep these in sync with your saved features file) ---
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

# --- Room Size & Rules ---
ROOM_SIZE_GUIDELINES = {
    '1 RK': {'min': 150, 'max': 1000},
    '1 BHK': {'min': 350, 'max': 1500},
    '2 BHK': {'min': 500, 'max': 2500},
    '3 BHK': {'min': 1000, 'max': 4000},
    '4 BHK': {'min': 1500, 'max': 5000},
    '5+ BHK': {'min': 1500, 'max': 10000}
}

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
AMENITY_IMPACT = {
    'gym': 2.5, 'gated_community': 5.0, 'intercom': 1.0, 'lift': 1.5,
    'pet_allowed': 2.0, 'pool': 3.5, 'security': 3.0, 'water_supply_amenity': 1.25,
    'wifi': 1.5, 'gas_pipeline': 1.0, 'sports_facility': 2.0, 'kids_area': 0.75,
    'power_backup': 2.5, 'garden': 1.5, 'fire_support': 1.0, 'parking': 2.5,
    'atm_near_me': 0.5, 'airport_near_me': 1.0, 'bus_stop_near_me': 0.25,
    'hospital_near_me': 0.75, 'mall_near_me': 1.25, 'market_near_me': 0.75,
    'metro_station_near_me': 1.0, 'park_near_me': 0.5, 'school_near_me': 0.75,
    'vastu': 3.0
}

# --- Load Model Resources ---
@st.cache_resource(show_spinner=True)
def load_resources() -> Tuple[Any, Any, List[str]]:
    """
    Loads model, scaler, and features list. Returns (model, scaler, features_list).
    If resources aren't found or fail to load, returns (None, None, None).
    """
    try:
        rf_model = joblib.load('m.pkl')
        scaler = joblib.load('s.pkl')
        features = joblib.load('f.pkl')
        # Normalize features to a list of column names (if it's an index or array)
        if isinstance(features, (pd.Index, np.ndarray, list)):
            features_list = list(features)
        else:
            features_list = list(features)
        st.success("Model and resources loaded successfully.")
        return rf_model, scaler, features_list
    except FileNotFoundError as e:
        st.error("Required file(s) not found. Please place 'm.pkl', 's.pkl' and 'f.pkl' in the app directory.")
        st.info(str(e))
        return None, None, None
    except Exception as e:
        st.error("An error occurred while loading model resources.")
        st.info(str(e))
        return None, None, None


# --- Prediction Function ---
def predict_rent_with_model(model, scaler, original_df_columns: List[str], data_dict: Dict[str, Any]) -> float:
    """
    Prepare input, align columns, scale numeric features, and return predicted rent (inverse transformed).
    Returns None on failure.
    """
    if model is None or scaler is None or original_df_columns is None:
        return None

    # Make a DataFrame for the single sample
    new_df = pd.DataFrame([data_dict])

    # One-hot encode categorical features present in input
    for feature in CATEGORICAL_FEATURES:
        if feature in new_df.columns:
            temp_df = pd.get_dummies(new_df[[feature]], prefix=feature)
            new_df = new_df.drop(columns=[feature])
            new_df = pd.concat([new_df.reset_index(drop=True), temp_df.reset_index(drop=True)], axis=1)

    # Ensure all expected columns exist; fill missing with 0
    for c in original_df_columns:
        if c not in new_df.columns:
            new_df[c] = 0

    # Reorder columns to match the model's training columns
    new_df = new_df[original_df_columns]

    # Identify numerical columns that are present and scale them
    numerical_cols_for_current_model = [col for col in NUMERICAL_FEATURES if col in original_df_columns]
    if numerical_cols_for_current_model:
        try:
            # scaler expects 2D array with same order columns - use the same subframe
            new_df[numerical_cols_for_current_model] = scaler.transform(new_df[numerical_cols_for_current_model])
        except Exception as e:
            st.error("Scaling failed. Ensure the scaler matches the model training features.")
            st.info(str(e))
            return None

    # Make prediction using the model
    try:
        # model.predict expects 2D array
        log_pred = model.predict(new_df)[0]
        predicted_rent = np.expm1(log_pred)  # inverse of log1p
        # guard against negative / NaN
        if np.isnan(predicted_rent) or predicted_rent < 0:
            return None
        return float(predicted_rent)
    except Exception as e:
        st.error("Prediction failed. See details below.")
        st.info(str(e))
        return None


# --- Validation Functions ---
def validate_property_details(data_dict: Dict[str, Any]) -> List[str]:
    """Return warnings_list."""
    warnings = []

    area_type = data_dict.get('area_type', '')
    area_value = data_dict.get('area_value', 0)
    total_size = data_dict.get('Size_In_Sqft', data_dict.get('size', 0))

    # Area validations
    if area_type == "Super Area":
        if area_value != total_size:
            warnings.append(f"Super Area ({area_value} sq ft) must match the total size ({total_size} sq ft) exactly!")
    elif area_type == "Built-up Area":
        if area_value >= total_size:
            warnings.append(f"Built-up Area ({area_value} sq ft) must be less than total size ({total_size} sq ft)!")
        else:
            expected_min = total_size * 0.80
            expected_max = total_size * 0.90
            if area_value < expected_min or area_value > expected_max:
                warnings.append(f"Built-up Area ({area_value} sq ft) should be between {expected_min:.0f}-{expected_max:.0f} sq ft (80-90% of total size {total_size} sq ft)!")
    elif area_type == "Carpet Area":
        if area_value >= total_size:
            warnings.append(f"Carpet Area ({area_value} sq ft) must be less than total size ({total_size} sq ft)!")
        else:
            expected_min = total_size * 0.65
            expected_max = total_size * 0.80
            if area_value < expected_min or area_value > expected_max:
                warnings.append(f"Carpet Area ({area_value} sq ft) should be between {expected_min:.0f}-{expected_max:.0f} sq ft (65-80% of total size {total_size} sq ft)!")

    # 1 RK rule
    if data_dict.get('Room') == "1 RK" or data_dict.get('room_type') == "1 RK":
        bedrooms = data_dict.get('Bedrooms', data_dict.get('bedrooms', 0))
        if bedrooms > 0:
            warnings.append("1 RK should not have bedrooms!")

    # Duplex floors rule
    if data_dict.get('Type') == "Duplex" or data_dict.get('property_type') == "Duplex":
        if data_dict.get('Total_floors_In_Building', data_dict.get('total_floors', 0)) != 2:
            warnings.append("Duplex property should have exactly 2 floors!")

    # Property/room type checks
    property_type = data_dict.get('Type', data_dict.get('property_type', ''))
    room_type = data_dict.get('Room', data_dict.get('room_type', ''))
    bedrooms = data_dict.get('Bedrooms', data_dict.get('bedrooms', 0))
    bathrooms = data_dict.get('Bathrooms', data_dict.get('bathrooms', 0))
    balcony = data_dict.get('Balcony', data_dict.get('balcony', 0))
    size = total_size

    if property_type in PROPERTY_ROOM_RULES:
        rules = PROPERTY_ROOM_RULES[property_type]
        if bedrooms < rules['bedrooms']['min'] or bedrooms > rules['bedrooms']['max']:
            warnings.append(f"For {property_type}, bedrooms should be between {rules['bedrooms']['min']} and {rules['bedrooms']['max']}!")
        if bathrooms < rules['bathrooms']['min'] or bathrooms > rules['bathrooms']['max']:
            warnings.append(f"For {property_type}, bathrooms should be between {rules['bathrooms']['min']} and {rules['bathrooms']['max']}!")
        if balcony < rules['balconies']['min'] or balcony > rules['balconies']['max']:
            warnings.append(f"For {property_type}, balconies should be between {rules['balconies']['min']} and {rules['balconies']['max']}!")

    if room_type in ROOM_TYPE_RULES:
        rules = ROOM_TYPE_RULES[room_type]
        if bedrooms < rules['bedrooms']['min'] or bedrooms > rules['bedrooms']['max']:
            warnings.append(f"For {room_type}, bedrooms should be between {rules['bedrooms']['min']} and {rules['bedrooms']['max']}!")
        if bathrooms < rules['bathrooms']['min'] or bathrooms > rules['bathrooms']['max']:
            warnings.append(f"For {room_type}, bathrooms should be between {rules['bathrooms']['min']} and {rules['bathrooms']['max']}!")
        if balcony < rules['balconies']['min'] or balcony > rules['balconies']['max']:
            warnings.append(f"For {room_type}, balconies should be between {rules['balconies']['min']} and {rules['balconies']['max']}!")

    if room_type in ROOM_SIZE_GUIDELINES:
        guidelines = ROOM_SIZE_GUIDELINES[room_type]
        if size < guidelines['min'] or size > guidelines['max']:
            warnings.append(f"For {room_type}, size should be between {guidelines['min']} and {guidelines['max']} sq ft!")

    # Flat-specific checks
    if property_type == "Flat":
        if data_dict.get('Total_floors_In_Building', data_dict.get('total_floors', 0)) < 2:
            warnings.append("Flat should be in a building with at least 2 floors!")
        if data_dict.get('Floor_No', data_dict.get('floor_no', 0)) > data_dict.get('Total_floors_In_Building', data_dict.get('total_floors', 0)):
            warnings.append("Floor number cannot exceed total floors in building!")

    # Ratios and abnormal counts
    if bedrooms > 0 and bathrooms > bedrooms + 2:
        warnings.append(f"Having {bathrooms} bathrooms for {bedrooms} bedrooms is unusual!")
    if bedrooms > 0 and balcony > bedrooms + 2:
        warnings.append(f"Having {balcony} balconies for {bedrooms} bedrooms is unusual!")

    # Abnormal large counts
    if bedrooms >= 10:
        if property_type not in ['Independent House', 'Villa']:
            warnings.append(f"Having {bedrooms} bedrooms in a {property_type} is unusual!")
        if size < 3000:
            warnings.append(f"Having {bedrooms} bedrooms in a {size} sq ft property is unusual!")

    if bathrooms >= 10:
        if property_type not in ['Independent House', 'Villa']:
            warnings.append(f"Having {bathrooms} bathrooms in a {property_type} is unusual!")
        if size < 3000:
            warnings.append(f"Having {bathrooms} bathrooms in a {size} sq ft property is unusual!")

    return warnings


def show_rental_app(rf_model, scaler, features):
    """Contains the UI and logic for the Rental Price Prediction tab."""
    st.title("Rental Price Prediction App")
    st.markdown("Enter property details and predict a fair rental price.")

    if rf_model is None or scaler is None or features is None:
        st.warning("Cannot run prediction. Ensure 'm.pkl', 's.pkl' and 'f.pkl' are available in the app directory.")
        return

    col1, col2 = st.columns(2)

    # Use a unique key prefix for session state to avoid conflicts between tabs
    key_prefix = 'rental_'

    with col1:
        st.header("Property Details")
        size = st.number_input("Size In Sqft", min_value=0, max_value=20000, value=1000, key=f'{key_prefix}size')
        with st.expander("Area Details"):
            area_type_options = ["Carpet Area", "Built-up Area", "Super Area"]
            area_type = st.selectbox("Select Area Type:", area_type_options, key=f'{key_prefix}area_type')
            area_value = st.number_input("Enter Area Value (Sqft)", min_value=0, max_value=50000, value=1500, key=f'{key_prefix}area_value')

        bedrooms = st.number_input("Number of Bedrooms", min_value=0, max_value=10, value=2, key=f'{key_prefix}bedrooms')
        bathrooms = st.number_input("Number of Bathrooms", min_value=0, max_value=10, value=2, key=f'{key_prefix}bathrooms')
        balcony = st.number_input("Number of Balconies", min_value=0, max_value=10, value=1, key=f'{key_prefix}balcony')
        total_floors = st.number_input("Total Floors In Building", min_value=0, max_value=50, value=4, key=f'{key_prefix}total_floors')
        floor_no = st.number_input("Floor No", min_value=0, max_value=total_floors if total_floors > 0 else 50, value=1, key=f'{key_prefix}floor_no')
        property_age = st.number_input("Property Age (in years)", min_value=0, max_value=100, value=5, key=f'{key_prefix}property_age')

        security_deposite = st.number_input("Security Deposite", min_value=0, value=20000, key=f'{key_prefix}security_deposite')
        road_connectivity = st.slider("Road Connectivity (1-10)", min_value=1, max_value=10, value=5, key=f'{key_prefix}road_connectivity')

    with col2:
        st.header("Categorical & Binary Features")
        area_options = sorted(list(AREA_TO_ZONE.keys()))
        area = st.selectbox("Select Area:", area_options, index=0, key=f'{key_prefix}area')

        default_zone = AREA_TO_ZONE.get(area, 'West Zone')
        zone_options = ['East Zone', 'North Zone', 'South Zone', 'West Zone', 'Central Zone', 'Rural']
        try:
            zone_index = zone_options.index(default_zone)
        except ValueError:
            zone_index = 0
        zone = st.selectbox("Select Zone:", zone_options, index=zone_index, key=f'{key_prefix}zone')

        furnishing_status_options = ['Fully Furnished', 'Semi Furnished', 'Unfurnished']
        furnishing_status = st.selectbox("Select Furnishing Status:", furnishing_status_options, key=f'{key_prefix}furnishing_status')

        recommended_for_options = ['Anyone', 'Bachelors', 'Family', 'Family and Bachelors', 'Family and Company']
        recommended_for = st.selectbox("Recommended For:", recommended_for_options, key=f'{key_prefix}recommended_for')

        water_supply_options_categorical = ['Borewell', 'Both', 'Municipal']
        municipal_bore_water = st.selectbox("Municipal Water Or Bore Water:", water_supply_options_categorical, key=f'{key_prefix}municipal_bore_water')

        type_of_society_options = ['Gated', 'Non-Gated', 'Township']
        type_of_society = st.selectbox("Type of Society:", type_of_society_options, key=f'{key_prefix}type_of_society')

        room_type_options = ['1 RK', '1 BHK', '2 BHK', '3 BHK', '4 BHK', '5+ BHK']
        room_type = st.selectbox("Room Type:", room_type_options, key=f'{key_prefix}room_type')

        # Auto-set bedrooms for 1 RK
        if room_type == "1 RK":
            st.info("1 RK selected: Number of bedrooms automatically set to 0")
            bedrooms = 0

        property_type_options = ['Flat', 'Studio Apartment', 'Independent House', 'Independent Builder Floor', 'Villa', 'Duplex']
        property_type = st.selectbox("Property Type:", property_type_options, key=f'{key_prefix}property_type')

        if property_type == "Duplex":
            st.info("Duplex selected: Total floors automatically set to 2")
            total_floors = 2

        brokerage_options = ['No Brokerage', 'With Brokerage']
        brokerage = st.selectbox("Brokerage:", brokerage_options, key=f'{key_prefix}brokerage')

        maintenance_charge_options = ['Maintenance Not Included', 'Maintenance Included']
        maintenance_charge = st.selectbox("Maintenance Charge:", maintenance_charge_options, key=f'{key_prefix}maintenance_charge')

        # Amenities state initialization
        amenity_state_key = f'{key_prefix}amenity_states'
        if amenity_state_key not in st.session_state:
            st.session_state[amenity_state_key] = {k: False for k in AMENITY_IMPACT.keys()}

        st.subheader("Amenities & Proximity (Check if available)")
        with st.expander("Property Amenities"):
            col_a, col_b = st.columns(2)
            with col_a:
                st.session_state[amenity_state_key]['gym'] = st.checkbox("Gym (+2.5%)", key=f'{key_prefix}gym_cb', value=st.session_state[amenity_state_key].get('gym', False))
                st.session_state[amenity_state_key]['intercom'] = st.checkbox("Intercom (+1.0%)", key=f'{key_prefix}intercom_cb', value=st.session_state[amenity_state_key].get('intercom', False))
                st.session_state[amenity_state_key]['pet_allowed'] = st.checkbox("Pet Allowed (+2.0%)", key=f'{key_prefix}pet_allowed_cb', value=st.session_state[amenity_state_key].get('pet_allowed', False))
                st.session_state[amenity_state_key]['security'] = st.checkbox("Security (+3.0%)", key=f'{key_prefix}security_cb', value=st.session_state[amenity_state_key].get('security', False))
                st.session_state[amenity_state_key]['gas_pipeline'] = st.checkbox("Gas Pipeline (+1.0%)", key=f'{key_prefix}gas_pipeline_cb', value=st.session_state[amenity_state_key].get('gas_pipeline', False))
                st.session_state[amenity_state_key]['power_backup'] = st.checkbox("Power Backup (+2.5%)", key=f'{key_prefix}power_backup_cb', value=st.session_state[amenity_state_key].get('power_backup', False))
                st.session_state[amenity_state_key]['fire_support'] = st.checkbox("Fire Support (+1.0%)", key=f'{key_prefix}fire_support_cb', value=st.session_state[amenity_state_key].get('fire_support', False))
                st.session_state[amenity_state_key]['vastu'] = st.checkbox("Vastu Compliant (+3.0%)", key=f'{key_prefix}vastu_cb', value=st.session_state[amenity_state_key].get('vastu', False))
            with col_b:
                st.session_state[amenity_state_key]['gated_community'] = st.checkbox("Gated Community (+5.0%)", key=f'{key_prefix}gated_community_cb', value=st.session_state[amenity_state_key].get('gated_community', False))
                st.session_state[amenity_state_key]['lift'] = st.checkbox("Lift (+1.5%)", key=f'{key_prefix}lift_cb', value=st.session_state[amenity_state_key].get('lift', False))
                st.session_state[amenity_state_key]['pool'] = st.checkbox("Pool (+3.5%)", key=f'{key_prefix}pool_cb', value=st.session_state[amenity_state_key].get('pool', False))
                st.session_state[amenity_state_key]['water_supply_amenity'] = st.checkbox("Water Supply (amenity) (+1.25%)", help="Check if this specific water supply amenity is available", key=f'{key_prefix}water_supply_amenity_cb', value=st.session_state[amenity_state_key].get('water_supply_amenity', False))
                st.session_state[amenity_state_key]['wifi'] = st.checkbox("WiFi (+1.5%)", key=f'{key_prefix}wifi_cb', value=st.session_state[amenity_state_key].get('wifi', False))
                st.session_state[amenity_state_key]['sports_facility'] = st.checkbox("Sports Facility (+2.0%)", key=f'{key_prefix}sports_facility_cb', value=st.session_state[amenity_state_key].get('sports_facility', False))
                st.session_state[amenity_state_key]['kids_area'] = st.checkbox("Kids Area (+0.75%)", key=f'{key_prefix}kids_area_cb', value=st.session_state[amenity_state_key].get('kids_area', False))
                st.session_state[amenity_state_key]['garden'] = st.checkbox("Garden (+1.5%)", key=f'{key_prefix}garden_cb', value=st.session_state[amenity_state_key].get('garden', False))
                st.session_state[amenity_state_key]['parking'] = st.checkbox("Parking (+2.5%)", key=f'{key_prefix}parking_cb', value=st.session_state[amenity_state_key].get('parking', False))

        with st.expander("Proximity to Essential Services"):
            col_c, col_d = st.columns(2)
            with col_c:
                st.session_state[amenity_state_key]['atm_near_me'] = st.checkbox("ATM Near Me (+0.5%)", key=f'{key_prefix}atm_near_me_cb', value=st.session_state[amenity_state_key].get('atm_near_me', False))
                st.session_state[amenity_state_key]['bus_stop_near_me'] = st.checkbox("Bus Stop Near Me (+0.25%)", key=f'{key_prefix}bus_stop_near_me_cb', value=st.session_state[amenity_state_key].get('bus_stop_near_me', False))
                st.session_state[amenity_state_key]['mall_near_me'] = st.checkbox("Mall Near Me (+1.25%)", key=f'{key_prefix}mall_near_me_cb', value=st.session_state[amenity_state_key].get('mall_near_me', False))
                st.session_state[amenity_state_key]['metro_station_near_me'] = st.checkbox("Metro Station Near Me (+1.0%)", key=f'{key_prefix}metro_station_near_me_cb', value=st.session_state[amenity_state_key].get('metro_station_near_me', False))
                st.session_state[amenity_state_key]['school_near_me'] = st.checkbox("School Near Me (+0.75%)", key=f'{key_prefix}school_near_me_cb', value=st.session_state[amenity_state_key].get('school_near_me', False))
            with col_d:
                st.session_state[amenity_state_key]['airport_near_me'] = st.checkbox("Airport Near Me (+1.0%)", key=f'{key_prefix}airport_near_me_cb', value=st.session_state[amenity_state_key].get('airport_near_me', False))
                st.session_state[amenity_state_key]['hospital_near_me'] = st.checkbox("Hospital Near Me (+0.75%)", key=f'{key_prefix}hospital_near_me_cb', value=st.session_state[amenity_state_key].get('hospital_near_me', False))
                st.session_state[amenity_state_key]['market_near_me'] = st.checkbox("Market Near Me (+0.75%)", key=f'{key_prefix}market_near_me_cb', value=st.session_state[amenity_state_key].get('market_near_me', False))
                st.session_state[amenity_state_key]['park_near_me'] = st.checkbox("Park Near Me (+0.5%)", key=f'{key_prefix}park_near_me_cb', value=st.session_state[amenity_state_key].get('park_near_me', False))

        # Projection inputs and listed price
        st.markdown("---")
        st.subheader("Future Rental Rate Projection")
        projection_years = st.slider("Years from now to project:", min_value=1, max_value=20, value=5, key=f'{key_prefix}projection_years')
        annual_growth_rate = st.slider("Expected Annual Growth Rate (%):", min_value=0.0, max_value=15.0, value=3.5, step=0.1, key=f'{key_prefix}annual_growth_rate')
        listed_price = st.number_input("Enter the Listed Price of the property for comparison:", min_value=0, value=25000, key=f'{key_prefix}listed_price_comp')

        # Predict button
        if st.button("Predict Rent", key=f'{key_prefix}predict_button'):
            # Build input data dictionary for model
            # Convert area_value to carpet area based on area_type
            built_up_to_carpet_ratio = 0.85
            super_to_carpet_ratio = 0.70
            converted_carpet_area = area_value
            if area_type == "Built-up Area":
                converted_carpet_area = area_value * built_up_to_carpet_ratio
            elif area_type == "Super Area":
                converted_carpet_area = area_value * super_to_carpet_ratio

            # Count selected amenities
            amenities_count = sum(1 for k, v in st.session_state[amenity_state_key].items() if v)

            user_input_data = {
                'Size_In_Sqft': size,
                'Carpet_Area_Sqft': converted_carpet_area,
                'Bedrooms': bedrooms,
                'Bathrooms': bathrooms,
                'Balcony': balcony,
                'Number_Of_Amenities': amenities_count,
                'Security_Deposite': security_deposite,
                'Floor_No': floor_no,
                'Total_floors_In_Building': total_floors,
                'Road_Connectivity': road_connectivity,
                # Model boolean numeric flags
                'gym': 1 if st.session_state[amenity_state_key].get('gym', False) else 0,
                'gated_community': 1 if st.session_state[amenity_state_key].get('gated_community', False) else 0,
                'intercom': 1 if st.session_state[amenity_state_key].get('intercom', False) else 0,
                'lift': 1 if st.session_state[amenity_state_key].get('lift', False) else 0,
                'pet_allowed': 1 if st.session_state[amenity_state_key].get('pet_allowed', False) else 0,
                'pool': 1 if st.session_state[amenity_state_key].get('pool', False) else 0,
                'security': 1 if st.session_state[amenity_state_key].get('security', False) else 0,
                'water_supply': 1 if st.session_state[amenity_state_key].get('water_supply_amenity', False) else 0,
                'wifi': 1 if st.session_state[amenity_state_key].get('wifi', False) else 0,
                'gas_pipeline': 1 if st.session_state[amenity_state_key].get('gas_pipeline', False) else 0,
                'sports_facility': 1 if st.session_state[amenity_state_key].get('sports_facility', False) else 0,
                'kids_area': 1 if st.session_state[amenity_state_key].get('kids_area', False) else 0,
                'power_backup': 1 if st.session_state[amenity_state_key].get('power_backup', False) else 0,
                'Garden': 1 if st.session_state[amenity_state_key].get('garden', False) else 0,
                'Fire_Support': 1 if st.session_state[amenity_state_key].get('fire_support', False) else 0,
                'Parking': 1 if st.session_state[amenity_state_key].get('parking', False) else 0,
                'ATM_Near_me': 1 if st.session_state[amenity_state_key].get('atm_near_me', False) else 0,
                'Airport_Near_me': 1 if st.session_state[amenity_state_key].get('airport_near_me', False) else 0,
                'Bus_Stop__Near_me': 1 if st.session_state[amenity_state_key].get('bus_stop_near_me', False) else 0,
                'Hospital_Near_me': 1 if st.session_state[amenity_state_key].get('hospital_near_me', False) else 0,
                'Mall_Near_me': 1 if st.session_state[amenity_state_key].get('mall_near_me', False) else 0,
                'Market_Near_me': 1 if st.session_state[amenity_state_key].get('market_near_me', False) else 0,
                'Metro_Station_Near_me': 1 if st.session_state[amenity_state_key].get('metro_station_near_me', False) else 0,
                'Park_Near_me': 1 if st.session_state[amenity_state_key].get('park_near_me', False) else 0,
                'School_Near_me': 1 if st.session_state[amenity_state_key].get('school_near_me', False) else 0,
                'Property_Age': property_age,
                # Categorical fields (names match what you used earlier)
                'City': 'Nagpur', 'Area': area, 'Zone': zone, 'Frurnishing_Status': furnishing_status,
                'Recomened for': recommended_for, 'Muncipla Water Or Bore Water': municipal_bore_water,
                'Type of Society': type_of_society, 'Room': room_type, 'Type': property_type,
                'Brokerage': brokerage, 'Maintenance_Charge': maintenance_charge,
                # Validation-only fields
                'area_type': area_type, 'area_value': area_value
            }

            # Validate
            validation_warnings = validate_property_details(user_input_data)
            num_warnings = len(validation_warnings)

            st.markdown("---")
            st.subheader("Prediction Results")

            if validation_warnings:
                st.warning("Property Validation Warnings:")
                for w in validation_warnings:
                    st.warning(f"- {w}")

            today = datetime.date.today()
            st.info(f"Prediction based on market conditions as of: **{today.strftime('%B %d, %Y')}**")

            # Predict using the model
            base_pred = predict_rent_with_model(rf_model, scaler, features, user_input_data)

            # Amenity impact calculation
            total_amenity_impact = 0.0
            amenity_impact_details = {}
            for amenity_key, impact in AMENITY_IMPACT.items():
                # check both state key naming conventions used in the original code
                state_val = st.session_state[amenity_state_key].get(amenity_key, st.session_state[amenity_state_key].get(amenity_key.replace('near_me', '_near_me'), False))
                if state_val:
                    total_amenity_impact += impact
                    amenity_impact_details[amenity_key] = impact

            adjusted_pred = None
            if base_pred is not None:
                adjusted_pred = base_pred * (1 + total_amenity_impact / 100.0)
                # Apply warning deductions - 30% per warning
                if num_warnings > 0:
                    for _ in range(num_warnings):
                        adjusted_pred *= 0.7
                    st.error(f"Applied {num_warnings} warning deduction(s): Each warning reduces the rent by 30% (total reduction: {100*(1-0.7**num_warnings):.1f}%)")

            if base_pred is None:
                st.error("Model failed to produce a base prediction. Check model/scaler compatibility.")
            else:
                st.success(f"Base Predicted Rent (without amenities): Rs {base_pred:,.2f}")
                st.info(f"Total Amenity Impact: +{total_amenity_impact:.2f}%")

                with st.expander("Amenity Impact Breakdown"):
                    if amenity_impact_details:
                        for a, v in amenity_impact_details.items():
                            st.write(f"- {a.replace('_', ' ').title()}: +{v:.2f}%")
                    else:
                        st.write("No amenities selected.")

                if adjusted_pred is not None:
                    # Display adjusted rent
                    st.markdown(f"<div style='font-size:28px; font-weight:700;'>Adjusted Rent Estimate: Rs {adjusted_pred:,.2f}</div>", unsafe_allow_html=True)

                    # Price comparison
                    FAIR_PRICE_TOLERANCE = 0.3
                    lower_bound = adjusted_pred * (1 - FAIR_PRICE_TOLERANCE)
                    upper_bound = adjusted_pred * (1 + FAIR_PRICE_TOLERANCE)

                    st.markdown("---")
                    st.subheader("Price Comparison")
                    st.markdown(f"**User Entered Listed Price:** Rs {listed_price:,.2f}")
                    st.markdown(f"**Fair Range (±{int(FAIR_PRICE_TOLERANCE*100)}%):** Rs {lower_bound:,.2f} - Rs {upper_bound:,.2f}")

                    if listed_price < lower_bound:
                        st.warning("Listed price appears to be UNDERPRICED compared to the adjusted predicted rent.")
                    elif listed_price > upper_bound:
                        st.warning("Listed price appears to be OVERPRICED compared to the adjusted predicted rent.")
                    else:
                        st.success("Listed price appears FAIR compared to the adjusted predicted rent.")

                    # Future projection for projection_years and 15-year table/graph (odd years)
                    st.markdown("---")
                    st.subheader(f"{projection_years}-Year Projection (using adjusted rent and {annual_growth_rate:.1f}% annual growth)")

                    future_pred = adjusted_pred * ((1 + annual_growth_rate / 100.0) ** projection_years)
                    st.info(f"Projected Adjusted Rent in {projection_years} years: Rs {future_pred:,.2f}")

                    # 15-year projection list + plot (odd years)
                    st.markdown("### 15-Year Projection (odd years shown on plot)")
                    prices = []
                    current_price = adjusted_pred
                    year_labels = []
                    for y in range(1, 16):
                        current_price *= (1 + annual_growth_rate / 100.0)
                        prices.append(current_price)
                        year_labels.append(y)

                    # Display textual yearly projections
                    projection_texts = [f"Year {i+1}: Rs {prices[i]:,.2f}" for i in range(len(prices))]
                    st.markdown("\n".join(projection_texts))

                    # Plot odd years only (1,3,5,...,15)
                    odd_years = [y for y in year_labels if y % 2 != 0]
                    odd_prices = [prices[y-1] for y in odd_years]

                    # Create figure properly and show using st.pyplot
                    fig = plt.figure(figsize=(8, 4))
                    plt.plot(odd_years, odd_prices, marker='o', linestyle='-')
                    plt.title('15-Year Adjusted Predicted Rent Projection (Odd Years)')
                    plt.xlabel('Year')
                    plt.ylabel('Projected Rent (Rs)')
                    plt.xticks(odd_years)
                    plt.grid(True)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
                else:
                    st.error("Adjusted predicted rent not available.")


def show_commercial_app():
    """Placeholder for the Commercial Property Rent Predictor tab."""
    # The original document included the start of this section but not the main logic.
    # The initial setup and CSS were also included, but CSS is difficult to combine cleanly
    # with the rental app without the full commercial property logic.

    st.markdown("""
        <style>
            /* Apply custom dark theme/white text for this tab if needed, 
               but Streamlit should generally handle this via the global config/theme */
            h1, h2, h3, h4, h5, h6, p, span, div, label { color: white !important; }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("Commercial Property Rent Predictor")
    st.markdown("---")
    
    # Placeholder for missing logic
    st.warning("The full implementation for the Commercial Property Rent Predictor is missing in the provided document.")
    st.info("Please insert the Streamlit UI components, data processing, and prediction logic for the commercial model here.")
    
    st.header("Commercial Property Details (Placeholder)")
    st.number_input("Commercial Size In Sqft", min_value=0, max_value=50000, value=2000, key='commercial_size')
    st.selectbox("Commercial Property Type:", ['Office', 'Retail', 'Warehouse'], key='commercial_type')
    st.button("Predict Commercial Rent", key='commercial_predict')


# --- Main Application Execution ---

if __name__ == '__main__':
    # Load resources once
    rf_model, scaler, features = load_resources()

    # Create the tabs
    tab1, tab2 = st.tabs(["Rental Price", "Commercial Price"])

    with tab1:
        # Tab 1: Rental Price Prediction
        show_rental_app(rf_model, scaler, features)

    with tab2:
        # Tab 2: Commercial Price Prediction
        show_commercial_app()
