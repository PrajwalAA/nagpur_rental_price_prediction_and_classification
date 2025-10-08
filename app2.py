# app_commercial.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Any

st.set_page_config(page_title="Commercial Rental Price Prediction", layout="wide")

# --- Constants for features (adjust these to match your commercial model) ---
CATEGORICAL_FEATURES = [
    'City', 'Area', 'Zone', 'Property_Type', 'Business_Type', 'Furnishing_Status', 
    'Maintenance_Charge', 'Brokerage', 'Zone_Type', 'Facing'
]

NUMERICAL_FEATURES = [
    'Size_In_Sqft', 'Carpet_Area_Sqft', 'Floor_No', 'Total_floors_In_Building', 
    'Road_Connectivity', 'Security_Deposit', 'Property_Age', 'Number_Of_Parking_Slots',
    'Power_Backup', 'Water_Supply', 'ATM_Near_me', 'Airport_Near_me', 'Bus_Stop_Near_me',
    'Hospital_Near_me', 'Mall_Near_me', 'Market_Near_me', 'Metro_Station_Near_me',
    'Park_Near_me', 'School_Near_me'
]

# --- Area to Zone Mapping (keep if applicable) ---
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

# --- Commercial Property Size Guidelines ---
COMMERCIAL_SIZE_GUIDELINES = {
    'Office Space': {'min': 200, 'max': 50000},
    'Retail Shop': {'min': 100, 'max': 10000},
    'Restaurant': {'min': 500, 'max': 15000},
    'Warehouse': {'min': 1000, 'max': 100000},
    'Showroom': {'min': 1000, 'max': 20000},
    'Industrial Space': {'min': 2000, 'max': 200000},
    'Co-working Space': {'min': 500, 'max': 20000}
}

# --- Commercial Property Rules ---
COMMERCIAL_PROPERTY_RULES = {
    'Office Space': {
        'parking_slots': {'min': 0, 'max': 100},
        'power_backup': {'min': 0, 'max': 1},
        'water_supply': {'min': 0, 'max': 1}
    },
    'Retail Shop': {
        'parking_slots': {'min': 0, 'max': 50},
        'power_backup': {'min': 0, 'max': 1},
        'water_supply': {'min': 0, 'max': 1}
    },
    'Restaurant': {
        'parking_slots': {'min': 0, 'max': 100},
        'power_backup': {'min': 0, 'max': 1},
        'water_supply': {'min': 1, 'max': 1}
    },
    'Warehouse': {
        'parking_slots': {'min': 0, 'max': 200},
        'power_backup': {'min': 0, 'max': 1},
        'water_supply': {'min': 0, 'max': 1}
    },
    'Showroom': {
        'parking_slots': {'min': 0, 'max': 100},
        'power_backup': {'min': 0, 'max': 1},
        'water_supply': {'min': 0, 'max': 1}
    },
    'Industrial Space': {
        'parking_slots': {'min': 0, 'max': 500},
        'power_backup': {'min': 0, 'max': 1},
        'water_supply': {'min': 0, 'max': 1}
    },
    'Co-working Space': {
        'parking_slots': {'min': 0, 'max': 50},
        'power_backup': {'min': 0, 'max': 1},
        'water_supply': {'min': 0, 'max': 1}
    }
}

# --- Commercial Amenity Impact Percentages ---
COMMERCIAL_AMENITY_IMPACT = {
    'parking': 5.0, 'power_backup': 4.0, 'water_supply': 2.0, 'security': 3.5,
    'atm_near_me': 0.5, 'airport_near_me': 1.5, 'bus_stop_near_me': 1.0,
    'hospital_near_me': 0.75, 'mall_near_me': 2.0, 'market_near_me': 1.25,
    'metro_station_near_me': 2.5, 'park_near_me': 0.5, 'school_near_me': 0.75,
    'fire_safety': 2.5, 'loading_dock': 3.0, 'high_speed_internet': 3.0,
    'conference_room': 2.0, 'cafeteria': 1.5, 'reception_area': 2.0
}

# --- Load Model Resources ---
@st.cache_resource(show_spinner=True)
def load_resources() -> Tuple[Any, Any, List[str]]:
    """
    Loads model, scaler, and features list. Returns (model, scaler, features_list).
    If resources aren't found or fail to load, returns (None, None, None).
    """
    try:
        rf_model = joblib.load('mc.pkl')  # Update to your model filename
        scaler = joblib.load('sc.pkl')  # Update to your scaler filename
        features = joblib.load('fc.pkl')  # Update to your features filename
        # Normalize features to a list of column names
        if isinstance(features, (pd.Index, np.ndarray, list)):
            features_list = list(features)
        else:
            features_list = list(features)
        st.success("Commercial model and resources loaded successfully.")
        return rf_model, scaler, features_list
    except FileNotFoundError as e:
        st.error("Required file(s) not found. Please place 'commercial_model.pkl', 'commercial_scaler.pkl' and 'commercial_features.pkl' in the app directory.")
        st.info(str(e))
        return None, None, None
    except Exception as e:
        st.error("An error occurred while loading model resources.")
        st.info(str(e))
        return None, None, None

rf_model, scaler, features = load_resources()


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
            new_df[numerical_cols_for_current_model] = scaler.transform(new_df[numerical_cols_for_current_model])
        except Exception as e:
            st.error("Scaling failed. Ensure the scaler matches the model training features.")
            st.info(str(e))
            return None

    # Make prediction using the model
    try:
        log_pred = model.predict(new_df)[0]
        predicted_rent = np.expm1(log_pred)  # inverse of log1p
        if np.isnan(predicted_rent) or predicted_rent < 0:
            return None
        return float(predicted_rent)
    except Exception as e:
        st.error("Prediction failed. See details below.")
        st.info(str(e))
        return None


# --- Validation Functions ---
def validate_commercial_property(data_dict: Dict[str, Any]) -> List[str]:
    """Return warnings_list for commercial properties."""
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

    # Property type checks
    property_type = data_dict.get('Property_Type', data_dict.get('property_type', ''))
    parking_slots = data_dict.get('Number_Of_Parking_Slots', data_dict.get('parking_slots', 0))
    power_backup = data_dict.get('Power_Backup', data_dict.get('power_backup', 0))
    water_supply = data_dict.get('Water_Supply', data_dict.get('water_supply', 0))
    size = total_size

    if property_type in COMMERCIAL_PROPERTY_RULES:
        rules = COMMERCIAL_PROPERTY_RULES[property_type]
        if parking_slots < rules['parking_slots']['min'] or parking_slots > rules['parking_slots']['max']:
            warnings.append(f"For {property_type}, parking slots should be between {rules['parking_slots']['min']} and {rules['parking_slots']['max']}!")
        if power_backup < rules['power_backup']['min'] or power_backup > rules['power_backup']['max']:
            warnings.append(f"For {property_type}, power backup should be between {rules['power_backup']['min']} and {rules['power_backup']['max']}!")
        if water_supply < rules['water_supply']['min'] or water_supply > rules['water_supply']['max']:
            warnings.append(f"For {property_type}, water supply should be between {rules['water_supply']['min']} and {rules['water_supply']['max']}!")

    if property_type in COMMERCIAL_SIZE_GUIDELINES:
        guidelines = COMMERCIAL_SIZE_GUIDELINES[property_type]
        if size < guidelines['min'] or size > guidelines['max']:
            warnings.append(f"For {property_type}, size should be between {guidelines['min']} and {guidelines['max']} sq ft!")

    # Floor validations
    if data_dict.get('Floor_No', data_dict.get('floor_no', 0)) > data_dict.get('Total_floors_In_Building', data_dict.get('total_floors', 0)):
        warnings.append("Floor number cannot exceed total floors in building!")

    # Business type specific validations
    business_type = data_dict.get('Business_Type', '')
    if business_type == 'Restaurant' and water_supply == 0:
        warnings.append("Restaurants must have water supply!")
    if business_type in ['Data Center', 'IT Services'] and power_backup == 0:
        warnings.append(f"{business_type} businesses should have power backup!")

    # Abnormal large counts
    if parking_slots >= 100:
        if property_type not in ['Warehouse', 'Industrial Space']:
            warnings.append(f"Having {parking_slots} parking slots in a {property_type} is unusual!")
        if size < 5000:
            warnings.append(f"Having {parking_slots} parking slots in a {size} sq ft property is unusual!")

    return warnings


# --- Streamlit UI ---
st.title("Commercial Rental Price Prediction App")
st.markdown("Enter commercial property details and predict a fair rental price.")

if rf_model is None or scaler is None or features is None:
    st.warning("Cannot run prediction. Ensure 'commercial_model.pkl', 'commercial_scaler.pkl' and 'commercial_features.pkl' are available in the app directory.")
else:
    col1, col2 = st.columns(2)

    with col1:
        st.header("Property Details")
        size = st.number_input("Size In Sqft", min_value=0, max_value=200000, value=2000, key='size')
        with st.expander("Area Details"):
            area_type_options = ["Carpet Area", "Built-up Area", "Super Area"]
            area_type = st.selectbox("Select Area Type:", area_type_options, key='area_type')
            area_value = st.number_input("Enter Area Value (Sqft)", min_value=0, max_value=200000, value=1800, key='area_value')

        parking_slots = st.number_input("Number of Parking Slots", min_value=0, max_value=500, value=5, key='parking_slots')
        total_floors = st.number_input("Total Floors In Building", min_value=0, max_value=50, value=4, key='total_floors')
        floor_no = st.number_input("Floor No", min_value=0, max_value=total_floors if total_floors > 0 else 50, value=1, key='floor_no')
        property_age = st.number_input("Property Age (in years)", min_value=0, max_value=100, value=5, key='property_age')

        security_deposit = st.number_input("Security Deposit", min_value=0, value=50000, key='security_deposit')
        road_connectivity = st.slider("Road Connectivity (1-10)", min_value=1, max_value=10, value=5, key='road_connectivity')

    with col2:
        st.header("Categorical & Binary Features")
        area_options = sorted(list(AREA_TO_ZONE.keys()))
        area = st.selectbox("Select Area:", area_options, index=0, key='area')

        default_zone = AREA_TO_ZONE.get(area, 'West Zone')
        zone_options = ['East Zone', 'North Zone', 'South Zone', 'West Zone', 'Central Zone', 'Rural']
        try:
            zone_index = zone_options.index(default_zone)
        except ValueError:
            zone_index = 0
        zone = st.selectbox("Select Zone:", zone_options, index=zone_index, key='zone')

        property_type_options = ['Office Space', 'Retail Shop', 'Restaurant', 'Warehouse', 'Showroom', 'Industrial Space', 'Co-working Space']
        property_type = st.selectbox("Property Type:", property_type_options, key='property_type')

        business_type_options = ['General', 'IT Services', 'Retail', 'Restaurant', 'Manufacturing', 'Warehouse', 'Data Center', 'Healthcare', 'Education']
        business_type = st.selectbox("Business Type:", business_type_options, key='business_type')

        furnishing_status_options = ['Fully Furnished', 'Semi Furnished', 'Unfurnished']
        furnishing_status = st.selectbox("Select Furnishing Status:", furnishing_status_options, key='furnishing_status')

        zone_type_options = ['Commercial Zone', 'Industrial Zone', 'Mixed Use', 'Special Economic Zone']
        zone_type = st.selectbox("Zone Type:", zone_type_options, key='zone_type')

        facing_options = ['North', 'South', 'East', 'West', 'North-East', 'North-West', 'South-East', 'South-West']
        facing = st.selectbox("Facing:", facing_options, key='facing')

        brokerage_options = ['No Brokerage', 'With Brokerage']
        brokerage = st.selectbox("Brokerage:", brokerage_options, key='brokerage')

        maintenance_charge_options = ['Maintenance Not Included', 'Maintenance Included']
        maintenance_charge = st.selectbox("Maintenance Charge:", maintenance_charge_options, key='maintenance_charge')

        # Amenities state initialization
        if 'amenity_states' not in st.session_state:
            st.session_state['amenity_states'] = {k: False for k in COMMERCIAL_AMENITY_IMPACT.keys()}

        st.subheader("Amenities & Proximity (Check if available)")
        with st.expander("Property Amenities"):
            col_a, col_b = st.columns(2)
            with col_a:
                st.session_state['amenity_states']['parking'] = st.checkbox("Parking (+5.0%)", key='parking_cb', value=st.session_state['amenity_states'].get('parking', False))
                st.session_state['amenity_states']['power_backup'] = st.checkbox("Power Backup (+4.0%)", key='power_backup_cb', value=st.session_state['amenity_states'].get('power_backup', False))
                st.session_state['amenity_states']['water_supply'] = st.checkbox("Water Supply (+2.0%)", key='water_supply_cb', value=st.session_state['amenity_states'].get('water_supply', False))
                st.session_state['amenity_states']['security'] = st.checkbox("Security (+3.5%)", key='security_cb', value=st.session_state['amenity_states'].get('security', False))
                st.session_state['amenity_states']['fire_safety'] = st.checkbox("Fire Safety (+2.5%)", key='fire_safety_cb', value=st.session_state['amenity_states'].get('fire_safety', False))
                st.session_state['amenity_states']['loading_dock'] = st.checkbox("Loading Dock (+3.0%)", key='loading_dock_cb', value=st.session_state['amenity_states'].get('loading_dock', False))
                st.session_state['amenity_states']['high_speed_internet'] = st.checkbox("High Speed Internet (+3.0%)", key='high_speed_internet_cb', value=st.session_state['amenity_states'].get('high_speed_internet', False))
            with col_b:
                st.session_state['amenity_states']['conference_room'] = st.checkbox("Conference Room (+2.0%)", key='conference_room_cb', value=st.session_state['amenity_states'].get('conference_room', False))
                st.session_state['amenity_states']['cafeteria'] = st.checkbox("Cafeteria (+1.5%)", key='cafeteria_cb', value=st.session_state['amenity_states'].get('cafeteria', False))
                st.session_state['amenity_states']['reception_area'] = st.checkbox("Reception Area (+2.0%)", key='reception_area_cb', value=st.session_state['amenity_states'].get('reception_area', False))

        with st.expander("Proximity to Essential Services"):
            col_c, col_d = st.columns(2)
            with col_c:
                st.session_state['amenity_states']['atm_near_me'] = st.checkbox("ATM Near Me (+0.5%)", key='atm_near_me_cb', value=st.session_state['amenity_states'].get('atm_near_me', False))
                st.session_state['amenity_states']['bus_stop_near_me'] = st.checkbox("Bus Stop Near Me (+1.0%)", key='bus_stop_near_me_cb', value=st.session_state['amenity_states'].get('bus_stop_near_me', False))
                st.session_state['amenity_states']['mall_near_me'] = st.checkbox("Mall Near Me (+2.0%)", key='mall_near_me_cb', value=st.session_state['amenity_states'].get('mall_near_me', False))
                st.session_state['amenity_states']['metro_station_near_me'] = st.checkbox("Metro Station Near Me (+2.5%)", key='metro_station_near_me_cb', value=st.session_state['amenity_states'].get('metro_station_near_me', False))
                st.session_state['amenity_states']['school_near_me'] = st.checkbox("School Near Me (+0.75%)", key='school_near_me_cb', value=st.session_state['amenity_states'].get('school_near_me', False))
            with col_d:
                st.session_state['amenity_states']['airport_near_me'] = st.checkbox("Airport Near Me (+1.5%)", key='airport_near_me_cb', value=st.session_state['amenity_states'].get('airport_near_me', False))
                st.session_state['amenity_states']['hospital_near_me'] = st.checkbox("Hospital Near Me (+0.75%)", key='hospital_near_me_cb', value=st.session_state['amenity_states'].get('hospital_near_me', False))
                st.session_state['amenity_states']['market_near_me'] = st.checkbox("Market Near Me (+1.25%)", key='market_near_me_cb', value=st.session_state['amenity_states'].get('market_near_me', False))
                st.session_state['amenity_states']['park_near_me'] = st.checkbox("Park Near Me (+0.5%)", key='park_near_me_cb', value=st.session_state['amenity_states'].get('park_near_me', False))

    # Projection inputs and listed price
    st.markdown("---")
    st.subheader("Future Rental Rate Projection")
    projection_years = st.slider("Years from now to project:", min_value=1, max_value=20, value=5, key='projection_years')
    annual_growth_rate = st.slider("Expected Annual Growth Rate (%):", min_value=0.0, max_value=15.0, value=4.0, step=0.1, key='annual_growth_rate')
    listed_price = st.number_input("Enter the Listed Price of the property for comparison:", min_value=0, value=100000, key='listed_price_comp')

    # Predict button
    if st.button("Predict Rent"):
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
        amenities_count = sum(1 for k, v in st.session_state['amenity_states'].items() if v)

        user_input_data = {
            'Size_In_Sqft': size,
            'Carpet_Area_Sqft': converted_carpet_area,
            'Floor_No': floor_no, 'Total_floors_In_Building': total_floors, 'Road_Connectivity': road_connectivity,
            'Security_Deposit': security_deposit, 'Property_Age': property_age,
            'Number_Of_Parking_Slots': parking_slots,
            'Power_Backup': 1 if st.session_state['amenity_states'].get('power_backup', False) else 0,
            'Water_Supply': 1 if st.session_state['amenity_states'].get('water_supply', False) else 0,
            'ATM_Near_me': 1 if st.session_state['amenity_states'].get('atm_near_me', False) else 0,
            'Airport_Near_me': 1 if st.session_state['amenity_states'].get('airport_near_me', False) else 0,
            'Bus_Stop_Near_me': 1 if st.session_state['amenity_states'].get('bus_stop_near_me', False) else 0,
            'Hospital_Near_me': 1 if st.session_state['amenity_states'].get('hospital_near_me', False) else 0,
            'Mall_Near_me': 1 if st.session_state['amenity_states'].get('mall_near_me', False) else 0,
            'Market_Near_me': 1 if st.session_state['amenity_states'].get('market_near_me', False) else 0,
            'Metro_Station_Near_me': 1 if st.session_state['amenity_states'].get('metro_station_near_me', False) else 0,
            'Park_Near_me': 1 if st.session_state['amenity_states'].get('park_near_me', False) else 0,
            'School_Near_me': 1 if st.session_state['amenity_states'].get('school_near_me', False) else 0,

            # Categorical fields
            'City': 'Nagpur', 'Area': area, 'Zone': zone, 'Property_Type': property_type,
            'Business_Type': business_type, 'Furnishing_Status': furnishing_status,
            'Maintenance_Charge': maintenance_charge, 'Brokerage': brokerage,
            'Zone_Type': zone_type, 'Facing': facing,

            # Validation-only fields
            'area_type': area_type, 'area_value': area_value
        }

        # Validate
        validation_warnings = validate_commercial_property(user_input_data)
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
        for amenity_key, impact in COMMERCIAL_AMENITY_IMPACT.items():
            state_val = st.session_state['amenity_states'].get(amenity_key, False)
            if state_val:
                total_amenity_impact += impact
                amenity_impact_details[amenity_key] = impact

        adjusted_pred = None
        if base_pred is not None:
            adjusted_pred = base_pred * (1 + total_amenity_impact / 100.0)
            
            # Apply warning deductions - 20% per warning (less severe than residential)
            if num_warnings > 0:
                for _ in range(num_warnings):
                    adjusted_pred *= 0.8
                st.error(f"Applied {num_warnings} warning deduction(s): Each warning reduces the rent by 20% (total reduction: {100*(1-0.8**num_warnings):.1f}%)")

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
                FAIR_PRICE_TOLERANCE = 0.25  # Commercial properties have more variability
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

                # Future projection
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
