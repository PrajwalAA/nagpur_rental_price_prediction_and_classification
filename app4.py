# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime
import matplotlib.pyplot as plt
import re
import seaborn as sns
from typing import Tuple, List, Dict, Any

st.set_page_config(page_title="Property Price Prediction", layout="wide")

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
    '1 RK': {'min': 150, 'max': 1000},    # Based on data: 180-1000 sqft
    '1 BHK': {'min': 350, 'max': 1500},   # Based on data: 400-1500 sqft
    '2 BHK': {'min': 500, 'max': 2500},   # Based on data: 500-2500 sqft
    '3 BHK': {'min': 1000, 'max': 4000},  # Based on data: 1000-4000 sqft
    '4 BHK': {'min': 1500, 'max': 5000},  # Based on data: 1500-4000 sqft
    '5+ BHK': {'min': 1500, 'max': 10000} # Based on data: 1500-2600+ sqft
}

PROPERTY_ROOM_RULES = {
    'Studio Apartment': {
        'bedrooms': {'min': 0, 'max': 0},
        'bathrooms': {'min': 1, 'max': 1},
        'balconies': {'min': 0, 'max': 1}
    },
    'Flat': {
        'bedrooms': {'min': 0, 'max': 5},    # Data shows 0-5 bedrooms
        'bathrooms': {'min': 1, 'max': 6},   # Data shows 1-6 bathrooms
        'balconies': {'min': 0, 'max': 5}    # Data shows 0-5 balconies
    },
    'Independent House': {
        'bedrooms': {'min': 1, 'max': 10},   # Data shows 1-10 bedrooms
        'bathrooms': {'min': 1, 'max': 10},  # Data shows 1-10 bathrooms
        'balconies': {'min': 0, 'max': 10}   # Data shows 0-10 balconies
    },
    'Independent Builder Floor': {
        'bedrooms': {'min': 1, 'max': 6},    # Data shows 1-6 bedrooms
        'bathrooms': {'min': 1, 'max': 6},   # Data shows 1-6 bathrooms
        'balconies': {'min': 0, 'max': 5}    # Data shows 0-5 balconies
    },
    'Villa': {
        'bedrooms': {'min': 2, 'max': 10},   # Data shows 2-10 bedrooms
        'bathrooms': {'min': 2, 'max': 10},  # Data shows 2-10 bathrooms
        'balconies': {'min': 1, 'max': 10}   # Data shows 1-10 balconies
    },
    'Duplex': {
        'bedrooms': {'min': 2, 'max': 6},    # Data shows 2-6 bedrooms
        'bathrooms': {'min': 2, 'max': 6},   # Data shows 2-6 bathrooms
        'balconies': {'min': 1, 'max': 5}    # Data shows 1-5 balconies
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
        'bathrooms': {'min': 1, 'max': 2},   # Data shows 1-2 bathrooms
        'balconies': {'min': 0, 'max': 2}    # Data shows 0-2 balconies
    },
    '2 BHK': {
        'bedrooms': {'min': 2, 'max': 2},
        'bathrooms': {'min': 1, 'max': 3},   # Data shows 1-3 bathrooms
        'balconies': {'min': 0, 'max': 3}    # Data shows 0-3 balconies
    },
    '3 BHK': {
        'bedrooms': {'min': 3, 'max': 3},
        'bathrooms': {'min': 2, 'max': 4},   # Data shows 2-4 bathrooms
        'balconies': {'min': 1, 'max': 4}    # Data shows 1-4 balconies
    },
    '4 BHK': {
        'bedrooms': {'min': 4, 'max': 4},
        'bathrooms': {'min': 2, 'max': 5},   # Data shows 2-5 bathrooms
        'balconies': {'min': 1, 'max': 5}    # Data shows 1-5 balconies
    },
    '5+ BHK': {
        'bedrooms': {'min': 5, 'max': 10},   # Data shows 5-10 bedrooms
        'bathrooms': {'min': 3, 'max': 10},  # Data shows 3-10 bathrooms
        'balconies': {'min': 1, 'max': 10}   # Data shows 1-10 balconies
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

# --- Commercial Property Constants ---
FLOOR_WEIGHTAGE = {
    0: 0.0,    # Ground floor - base price
    1: 2.0,    # 1st floor - 2% increase
    2: 4.0,    # 2nd floor - 4% increase
    3: 6.0,    # 3rd floor - 6% increase
    4: 8.0,    # 4th floor - 8% increase
    5: 10.0,   # 5th floor - 10% increase
    6: 12.0,   # 6th floor - 12% increase
    7: 14.0,   # 7th floor - 14% increase
    8: 16.0,   # 8th floor - 16% increase
    9: 18.0,   # 9th floor - 18% increase
    10: 20.0   # 10th floor - 20% increase
}

# --- Load Model Resources ---
@st.cache_resource(show_spinner=True)
def load_rental_resources() -> Tuple[Any, Any, List[str]]:
    """
    Loads rental model, scaler, and features list. Returns (model, scaler, features_list).
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
        return rf_model, scaler, features_list
    except FileNotFoundError as e:
        st.error("Required file(s) not found. Please place 'm.pkl', 's.pkl' and 'f.pkl' in the app directory.")
        return None, None, None
    except Exception as e:
        st.error("An error occurred while loading rental model resources.")
        return None, None, None

@st.cache_resource(show_spinner=True)
def load_commercial_resources() -> Tuple[Any, Any, List[str]]:
    """
    Loads commercial model, scaler, and features list. Returns (model, scaler, features_list).
    If resources aren't found or fail to load, returns (None, None, None).
    """
    try:
        model = joblib.load('mc.pkl')
        scaler = joblib.load('sc.pkl')
        feature_names = joblib.load('fc.pkl')
        return model, scaler, feature_names
    except Exception as e:
        st.error(f"Error loading commercial model components: {e}")
        return None, None, None

# --- Prediction Functions ---
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
        return None

def preprocess_commercial_input(user_data, feature_names, scaler):
    df = pd.DataFrame([user_data])
    
    def floor_to_int_list(floor_str):
        floor_str = str(floor_str).lower()
        floor_str = floor_str.replace('floor', '').replace('floors', '').replace(' ', '')
        floor_str = floor_str.replace('ground', '0').replace('gf', '0')
        parts = re.split(r'[,]', floor_str)
        floor_numbers = []
        for p in parts:
            try:
                floor_numbers.append(int(p))
            except:
                continue
        if floor_numbers:
            return ','.join(map(str, sorted(floor_numbers)))
        return None
    
    df['floor_no'] = df['floor_no'].apply(floor_to_int_list)
    
    def total_floors_to_int(floor_str):
        try:
            return int(str(floor_str).lower().replace('floors', '').replace('floor', '').strip())
        except:
            return None
    
    df['total_floors'] = df['total_floors'].apply(total_floors_to_int)
    
    def size_to_int(size_str):
        try:
            return int(str(size_str).lower().replace('sqft','').replace('sq.ft','').strip())
        except:
            return None
    
    df['size_in_sqft'] = df['size_in_sqft'].apply(size_to_int)
    df['carpet_area_sqft'] = df['carpet_area_sqft'].apply(size_to_int)
    
    def extract_amenities_list(text):
        text = str(text).lower()
        text = re.sub(r'\(\d+\)', '', text)
        amenities = [x.strip() for x in text.split(',') if x.strip() != '']
        return amenities
    
    all_amenities = ['parking', 'vastu', 'lift', 'cabin', 'meeting room', 'dg and ups', 
                    'water storage', 'staircase', 'security', 'cctv', 'power backup', 
                    'reception area', 'pantry', 'fire extinguishers', 'fire safety', 
                    'oxygen duct', 'food court', 'furnishing', 'internet', 'fire sensors']
    
    for amenity in all_amenities:
        amenity_col = amenity.replace(' ', '_')
        df[amenity_col] = df['amenities_count'].apply(
            lambda x: 1 if amenity in extract_amenities_list(x) else 0
        )
    
    df['property_age'] = df['property_age'].astype(int)
    df['lock_in_period_in_months'] = df['lock in period'].str.replace('months', '', regex=False) \
                                     .str.replace('month', '', regex=False) \
                                     .str.strip() \
                                     .fillna(0) \
                                     .astype(int)
    
    categorical_features = ['listing litle', 'city', 'area', 'zone', 'location_hub',
                           'property_type', 'ownership', 'floor_no',
                           'electric_charge_included', 'water_charge_included',
                           'possession_status', 'posted_by', 'negotiable', 'brokerage']
    
    for feature in categorical_features:
        if feature in df.columns:
            df = pd.get_dummies(df, columns=[feature])
    
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    
    df = df[feature_names]
    
    numerical_cols = ['size_in_sqft', 'carpet_area_sqft', 'private_washroom',
                     'public_washroom', 'total_floors', 'property_age',
                     'expected rent increases yearly', 'fire_extinguishers',
                     'food_court', 'cabin', 'lift', '0', 'water_storage', 'dg',
                     'fire_safety', 'security', 'cctv', 'oxygen_duct', 'furnishing', 'vastu',
                     'reception_area', 'internet', 'water_supply', 'fire_sensors',
                     'power_backup', 'dg_and_ups', 'parking', 'pantry',
                     'lock_in_period_in_months']
    
    numerical_cols_present = [col for col in numerical_cols if col in df.columns]
    
    if numerical_cols_present:
        df[numerical_cols_present] = scaler.transform(df[numerical_cols_present])
    
    return df

def calculate_floor_adjusted_rent(base_rent, selected_floors):
    if not selected_floors:
        return base_rent, 0.0
    
    # Calculate average weightage for selected floors
    total_weightage = sum(FLOOR_WEIGHTAGE.get(int(floor), 0) for floor in selected_floors)
    avg_weightage = total_weightage / len(selected_floors)
    
    # Apply weightage to base rent
    adjusted_rent = base_rent * (1 + avg_weightage / 100)
    
    return adjusted_rent, avg_weightage

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

# --- Streamlit UI ---
st.title("Property Price Prediction App")

# Create tabs
tab1, tab2 = st.tabs(["Rental Price Prediction", "Commercial Price Prediction"])

# Tab 1: Rental Price Prediction
with tab1:
    st.markdown("Enter property details and predict a fair rental price.")
    
    # Load rental model resources
    rf_model, scaler, features = load_rental_resources()
    
    if rf_model is None or scaler is None or features is None:
        st.warning("Cannot run prediction. Ensure 'm.pkl', 's.pkl' and 'f.pkl' are available in the app directory.")
    else:
        col1, col2 = st.columns(2)

        with col1:
            st.header("Property Details")
            size = st.number_input("Size In Sqft", min_value=0, max_value=20000, value=1000, key='rental_size')
            with st.expander("Area Details"):
                area_type_options = ["Carpet Area", "Built-up Area", "Super Area"]
                area_type = st.selectbox("Select Area Type:", area_type_options, key='rental_area_type')
                area_value = st.number_input("Enter Area Value (Sqft)", min_value=0, max_value=50000, value=1500, key='rental_area_value')

            bedrooms = st.number_input("Number of Bedrooms", min_value=0, max_value=10, value=2, key='rental_bedrooms')
            bathrooms = st.number_input("Number of Bathrooms", min_value=0, max_value=10, value=2, key='rental_bathrooms')
            balcony = st.number_input("Number of Balconies", min_value=0, max_value=10, value=1, key='rental_balcony')
            total_floors = st.number_input("Total Floors In Building", min_value=0, max_value=50, value=4, key='rental_total_floors')
            floor_no = st.number_input("Floor No", min_value=0, max_value=total_floors if total_floors > 0 else 50, value=1, key='rental_floor_no')
            property_age = st.number_input("Property Age (in years)", min_value=0, max_value=100, value=5, key='rental_property_age')

            security_deposite = st.number_input("Security Deposite", min_value=0, value=20000, key='rental_security_deposite')
            road_connectivity = st.slider("Road Connectivity (1-10)", min_value=1, max_value=10, value=5, key='rental_road_connectivity')

        with col2:
            st.header("Categorical & Binary Features")
            area_options = sorted(list(AREA_TO_ZONE.keys()))
            area = st.selectbox("Select Area:", area_options, index=0, key='rental_area')

            default_zone = AREA_TO_ZONE.get(area, 'West Zone')
            zone_options = ['East Zone', 'North Zone', 'South Zone', 'West Zone', 'Central Zone', 'Rural']
            try:
                zone_index = zone_options.index(default_zone)
            except ValueError:
                zone_index = 0
            zone = st.selectbox("Select Zone:", zone_options, index=zone_index, key='rental_zone')

            furnishing_status_options = ['Fully Furnished', 'Semi Furnished', 'Unfurnished']
            furnishing_status = st.selectbox("Select Furnishing Status:", furnishing_status_options, key='rental_furnishing_status')

            recommended_for_options = ['Anyone', 'Bachelors', 'Family', 'Family and Bachelors', 'Family and Company']
            recommended_for = st.selectbox("Recommended For:", recommended_for_options, key='rental_recommended_for')

            water_supply_options_categorical = ['Borewell', 'Both', 'Municipal']
            municipal_bore_water = st.selectbox("Municipal Water Or Bore Water:", water_supply_options_categorical, key='rental_municipal_bore_water')

            type_of_society_options = ['Gated', 'Non-Gated', 'Township']
            type_of_society = st.selectbox("Type of Society:", type_of_society_options, key='rental_type_of_society')

            room_type_options = ['1 RK', '1 BHK', '2 BHK', '3 BHK', '4 BHK', '5+ BHK']
            room_type = st.selectbox("Room Type:", room_type_options, key='rental_room_type')

            # Auto-set bedrooms for 1 RK
            if room_type == "1 RK":
                st.info("1 RK selected: Number of bedrooms automatically set to 0")
                bedrooms = 0

            property_type_options = ['Flat', 'Studio Apartment', 'Independent House', 'Independent Builder Floor', 'Villa', 'Duplex']
            property_type = st.selectbox("Property Type:", property_type_options, key='rental_property_type')

            if property_type == "Duplex":
                st.info("Duplex selected: Total floors automatically set to 2")
                total_floors = 2

            brokerage_options = ['No Brokerage', 'With Brokerage']
            brokerage = st.selectbox("Brokerage:", brokerage_options, key='rental_brokerage')

            maintenance_charge_options = ['Maintenance Not Included', 'Maintenance Included']
            maintenance_charge = st.selectbox("Maintenance Charge:", maintenance_charge_options, key='rental_maintenance_charge')

            # Amenities state initialization
            if 'rental_amenity_states' not in st.session_state:
                st.session_state['rental_amenity_states'] = {k: False for k in AMENITY_IMPACT.keys()}

            st.subheader("Amenities & Proximity (Check if available)")
            with st.expander("Property Amenities"):
                col_a, col_b = st.columns(2)
                with col_a:
                    st.session_state['rental_amenity_states']['gym'] = st.checkbox("Gym (+2.5%)", key='rental_gym_cb', value=st.session_state['rental_amenity_states'].get('gym', False))
                    st.session_state['rental_amenity_states']['intercom'] = st.checkbox("Intercom (+1.0%)", key='rental_intercom_cb', value=st.session_state['rental_amenity_states'].get('intercom', False))
                    st.session_state['rental_amenity_states']['pet_allowed'] = st.checkbox("Pet Allowed (+2.0%)", key='rental_pet_allowed_cb', value=st.session_state['rental_amenity_states'].get('pet_allowed', False))
                    st.session_state['rental_amenity_states']['security'] = st.checkbox("Security (+3.0%)", key='rental_security_cb', value=st.session_state['rental_amenity_states'].get('security', False))
                    st.session_state['rental_amenity_states']['gas_pipeline'] = st.checkbox("Gas Pipeline (+1.0%)", key='rental_gas_pipeline_cb', value=st.session_state['rental_amenity_states'].get('gas_pipeline', False))
                    st.session_state['rental_amenity_states']['power_backup'] = st.checkbox("Power Backup (+2.5%)", key='rental_power_backup_cb', value=st.session_state['rental_amenity_states'].get('power_backup', False))
                    st.session_state['rental_amenity_states']['fire_support'] = st.checkbox("Fire Support (+1.0%)", key='rental_fire_support_cb', value=st.session_state['rental_amenity_states'].get('fire_support', False))
                    st.session_state['rental_amenity_states']['vastu'] = st.checkbox("Vastu Compliant (+3.0%)", key='rental_vastu_cb', value=st.session_state['rental_amenity_states'].get('vastu', False))
                with col_b:
                    st.session_state['rental_amenity_states']['gated_community'] = st.checkbox("Gated Community (+5.0%)", key='rental_gated_community_cb', value=st.session_state['rental_amenity_states'].get('gated_community', False))
                    st.session_state['rental_amenity_states']['lift'] = st.checkbox("Lift (+1.5%)", key='rental_lift_cb', value=st.session_state['rental_amenity_states'].get('lift', False))
                    st.session_state['rental_amenity_states']['pool'] = st.checkbox("Pool (+3.5%)", key='rental_pool_cb', value=st.session_state['rental_amenity_states'].get('pool', False))
                    st.session_state['rental_amenity_states']['water_supply_amenity'] = st.checkbox("Water Supply (amenity) (+1.25%)", help="Check if this specific water supply amenity is available", key='rental_water_supply_amenity_cb', value=st.session_state['rental_amenity_states'].get('water_supply_amenity', False))
                    st.session_state['rental_amenity_states']['wifi'] = st.checkbox("WiFi (+1.5%)", key='rental_wifi_cb', value=st.session_state['rental_amenity_states'].get('wifi', False))
                    st.session_state['rental_amenity_states']['sports_facility'] = st.checkbox("Sports Facility (+2.0%)", key='rental_sports_facility_cb', value=st.session_state['rental_amenity_states'].get('sports_facility', False))
                    st.session_state['rental_amenity_states']['kids_area'] = st.checkbox("Kids Area (+0.75%)", key='rental_kids_area_cb', value=st.session_state['rental_amenity_states'].get('kids_area', False))
                    st.session_state['rental_amenity_states']['garden'] = st.checkbox("Garden (+1.5%)", key='rental_garden_cb', value=st.session_state['rental_amenity_states'].get('garden', False))
                    st.session_state['rental_amenity_states']['parking'] = st.checkbox("Parking (+2.5%)", key='rental_parking_cb', value=st.session_state['rental_amenity_states'].get('parking', False))

            with st.expander("Proximity to Essential Services"):
                col_c, col_d = st.columns(2)
                with col_c:
                    st.session_state['rental_amenity_states']['atm_near_me'] = st.checkbox("ATM Near Me (+0.5%)", key='rental_atm_near_me_cb', value=st.session_state['rental_amenity_states'].get('atm_near_me', False))
                    st.session_state['rental_amenity_states']['bus_stop_near_me'] = st.checkbox("Bus Stop Near Me (+0.25%)", key='rental_bus_stop_near_me_cb', value=st.session_state['rental_amenity_states'].get('bus_stop_near_me', False))
                    st.session_state['rental_amenity_states']['mall_near_me'] = st.checkbox("Mall Near Me (+1.25%)", key='rental_mall_near_me_cb', value=st.session_state['rental_amenity_states'].get('mall_near_me', False))
                    st.session_state['rental_amenity_states']['metro_station_near_me'] = st.checkbox("Metro Station Near Me (+1.0%)", key='rental_metro_station_near_me_cb', value=st.session_state['rental_amenity_states'].get('metro_station_near_me', False))
                    st.session_state['rental_amenity_states']['school_near_me'] = st.checkbox("School Near Me (+0.75%)", key='rental_school_near_me_cb', value=st.session_state['rental_amenity_states'].get('school_near_me', False))
                with col_d:
                    st.session_state['rental_amenity_states']['airport_near_me'] = st.checkbox("Airport Near Me (+1.0%)", key='rental_airport_near_me_cb', value=st.session_state['rental_amenity_states'].get('airport_near_me', False))
                    st.session_state['rental_amenity_states']['hospital_near_me'] = st.checkbox("Hospital Near Me (+0.75%)", key='rental_hospital_near_me_cb', value=st.session_state['rental_amenity_states'].get('hospital_near_me', False))
                    st.session_state['rental_amenity_states']['market_near_me'] = st.checkbox("Market Near Me (+0.75%)", key='rental_market_near_me_cb', value=st.session_state['rental_amenity_states'].get('market_near_me', False))
                    st.session_state['rental_amenity_states']['park_near_me'] = st.checkbox("Park Near Me (+0.5%)", key='rental_park_near_me_cb', value=st.session_state['rental_amenity_states'].get('park_near_me', False))

        # Projection inputs and listed price
        st.markdown("---")
        st.subheader("Future Rental Rate Projection")
        projection_years = st.slider("Years from now to project:", min_value=1, max_value=20, value=5, key='rental_projection_years')
        annual_growth_rate = st.slider("Expected Annual Growth Rate (%):", min_value=0.0, max_value=15.0, value=3.5, step=0.1, key='rental_annual_growth_rate')
        listed_price = st.number_input("Enter the Listed Price of the property for comparison:", min_value=0, value=25000, key='rental_listed_price_comp')

        # Predict button
        if st.button("Predict Rent", key='rental_predict_button'):
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
            amenities_count = sum(1 for k, v in st.session_state['rental_amenity_states'].items() if v)

            user_input_data = {
                'Size_In_Sqft': size,
                'Carpet_Area_Sqft': converted_carpet_area,
                'Bedrooms': bedrooms, 'Bathrooms': bathrooms,
                'Balcony': balcony, 'Number_Of_Amenities': amenities_count,
                'Security_Deposite': security_deposite,
                'Floor_No': floor_no, 'Total_floors_In_Building': total_floors, 'Road_Connectivity': road_connectivity,
                # Model boolean numeric flags
                'gym': 1 if st.session_state['rental_amenity_states'].get('gym', False) else 0,
                'gated_community': 1 if st.session_state['rental_amenity_states'].get('gated_community', False) else 0,
                'intercom': 1 if st.session_state['rental_amenity_states'].get('intercom', False) else 0,
                'lift': 1 if st.session_state['rental_amenity_states'].get('lift', False) else 0,
                'pet_allowed': 1 if st.session_state['rental_amenity_states'].get('pet_allowed', False) else 0,
                'pool': 1 if st.session_state['rental_amenity_states'].get('pool', False) else 0,
                'security': 1 if st.session_state['rental_amenity_states'].get('security', False) else 0,
                'water_supply': 1 if st.session_state['rental_amenity_states'].get('water_supply_amenity', False) else 0,
                'wifi': 1 if st.session_state['rental_amenity_states'].get('wifi', False) else 0,
                'gas_pipeline': 1 if st.session_state['rental_amenity_states'].get('gas_pipeline', False) else 0,
                'sports_facility': 1 if st.session_state['rental_amenity_states'].get('sports_facility', False) else 0,
                'kids_area': 1 if st.session_state['rental_amenity_states'].get('kids_area', False) else 0,
                'power_backup': 1 if st.session_state['rental_amenity_states'].get('power_backup', False) else 0,
                'Garden': 1 if st.session_state['rental_amenity_states'].get('garden', False) else 0,
                'Fire_Support': 1 if st.session_state['rental_amenity_states'].get('fire_support', False) else 0,
                'Parking': 1 if st.session_state['rental_amenity_states'].get('parking', False) else 0,
                'ATM_Near_me': 1 if st.session_state['rental_amenity_states'].get('atm_near_me', False) else 0,
                'Airport_Near_me': 1 if st.session_state['rental_amenity_states'].get('airport_near_me', False) else 0,
                'Bus_Stop__Near_me': 1 if st.session_state['rental_amenity_states'].get('bus_stop_near_me', False) else 0,
                'Hospital_Near_me': 1 if st.session_state['rental_amenity_states'].get('hospital_near_me', False) else 0,
                'Mall_Near_me': 1 if st.session_state['rental_amenity_states'].get('mall_near_me', False) else 0,
                'Market_Near_me': 1 if st.session_state['rental_amenity_states'].get('market_near_me', False) else 0,
                'Metro_Station_Near_me': 1 if st.session_state['rental_amenity_states'].get('metro_station_near_me', False) else 0,
                'Park_Near_me': 1 if st.session_state['rental_amenity_states'].get('park_near_me', False) else 0,
                'School_Near_me': 1 if st.session_state['rental_amenity_states'].get('school_near_me', False) else 0,
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
                # keys in the sess state can differ slightly; attempt to map logically
                # we used some keys with different names (e.g., 'water_supply_amenity' vs 'water_supply'), check both
                state_val = st.session_state['rental_amenity_states'].get(amenity_key, st.session_state['rental_amenity_states'].get(amenity_key.replace('near_me', '_near_me'), False))
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

# Tab 2: Commercial Price Prediction
with tab2:
    # Custom CSS for white text, dark theme, and compact layout
    st.markdown("""
    <style>
        /* Set default text color to white and background to dark */
        .stApp {
            background-color: #0E1117;
            color: white;
        }
        html, body, .stMarkdown, h1, h2, h3, h4, h5, h6, p, span, div, label {
            color: white !important;
        }
        /* Make widgets more compact and ensure text is white */
        .stSelectbox > div > div > div { color: white; }
        .stNumberInput > div > div > input { color: white; }
        .stMultiSelect > div > div > div { color: white; }
        .element-container { margin-bottom: 0.5rem; }
        .stForm { border: 0px; padding: 0rem; }
        /* Style expander header */
        .streamlit-expanderHeader {
            background-color: #262730;
            border-radius: 5px;
        }
        /* Style multiselect dropdown */
        .stMultiSelect div[data-baseweb="select"] span {
            color: white;
        }
        /* Custom styling for price display */
        .price-container {
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            padding: 20px;
            border-radius: 10px;
            margin: 10px 0;
        }
        .price-label {
            font-size: 14px;
            opacity: 0.9;
        }
        .price-value {
            font-size: 28px;
            font-weight: bold;
        }
        .price-change {
            font-size: 16px;
            margin-top: 5px;
        }
        .positive-change {
            color: #4ade80;
        }
        .negative-change {
            color: #f87171;
        }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 style="text-align: center;">🏢 Commercial Property Rent Predictor</h1>', unsafe_allow_html=True)
    
    # Load commercial model resources
    model, scaler, feature_names = load_commercial_resources()
    
    if model is None or scaler is None or feature_names is None:
        st.error("Unable to load commercial model components. Please check your files.")
    else:
        # Display floor weightage information
        with st.expander("📊 Floor Premium Rates", expanded=False):
            st.write("Rent adjustments based on floor selection:")
            weightage_df = pd.DataFrame(list(FLOOR_WEIGHTAGE.items()), columns=['Floor', 'Premium (%)'])
            weightage_df['Floor'] = weightage_df['Floor'].apply(lambda x: f"Floor {x}")
            st.dataframe(weightage_df, hide_index=True, use_container_width=True)
        
        with st.form("commercial_prediction_form"):
            # --- Compact Input Grid ---
            col1, col2, col3 = st.columns(3)
            with col1:
                property_type = st.selectbox("Property Type", ['showroom', 'shop', 'bare shell office', 'ready to use office', 'commercial property', 'werehouse', 'godown'], index=0, key='commercial_property_type')
                size_sqft = st.number_input("Size (sqft)", min_value=100, max_value=100000, value=1000, step=50, key='commercial_size_sqft')
                area = st.selectbox("Area", ['manewada', 'jaitala', 'besa', 'omkar nagar', 'itwari', 'hingna', 'sitabuldi', 'mahal', 'kharbi', 'mihan', 'pratap nagar', 'ramdaspeth', 'dharampeth', 'gandhibag', 'chatrapati nagar', 'nandanwan', 'sadar', 'dighori', 'somalwada', 'ganeshpeth colony', 'mhalgi nagar', 'sakkardara', 'babulban', 'manish nagar', 'dhantoli', 'khamla', 'laxminagar', 'ajni', 'wathoda', 'hulkeshwar', 'pardi', 'new indora', 'civil lines', 'gadhibag', 'bagadganj', 'swawlambi nagar', 'manawada', 'trimurti nagar', 'lakadganj', 'shivaji nagar'], index=0, key='commercial_area')
            with col2:
                carpet_area = st.number_input("Carpet Area (sqft)", min_value=100, max_value=100000, value=800, step=50, key='commercial_carpet_area')
                zone = st.selectbox("Zone", ['south', 'west', 'east', 'north'], index=0, key='commercial_zone')
                location_hub = st.selectbox("Location Hub", ['commercial project', 'others', 'retail complex/building', 'market/high street', 'business park', 'it park', 'residential'], index=0, key='commercial_location_hub')
            with col3:
                ownership = st.selectbox("Ownership", ['freehold', 'leasehold', 'cooperative society', 'power_of_attorney'], index=0, key='commercial_ownership')
                total_floors = st.selectbox("Total Floors", ['3 floors', '1 floor', '2 floors', '4 floors', '5 floors', '8 floors', '7 floors', '6 floors', '15 floors', '9 floors', '10 floors'], index=0, key='commercial_total_floors')
                # --- Floor Selection with Multiselect Dropdown ---
                floor_options = [f"Floor {i}" for i in range(0, 11)]
                selected_floors = st.multiselect("Select Available Floors", floor_options, default=["Floor 0"], key='commercial_selected_floors')

            col_a, col_b, col_c = st.columns(3)
            with col_a:
                private_washroom = st.number_input("Private Washrooms", min_value=0, max_value=20, value=1, key='commercial_private_washroom')
            with col_b:
                public_washroom = st.number_input("Public Washrooms", min_value=0, max_value=20, value=1, key='commercial_public_washroom')
            with col_c:
                property_age = st.number_input("Property Age (years)", min_value=0, max_value=100, value=5, key='commercial_property_age')

            # --- Expanders for less critical details ---
            with st.expander("Amenities & Charges"):
                amenities_options = ['parking', 'vastu', 'lift', 'cabin', 'meeting room', 'dg and ups', 'water storage', 'staircase', 'security', 'cctv', 'power backup', 'reception area', 'pantry', 'fire extinguishers', 'fire safety', 'oxygen duct', 'food court', 'furnishing', 'internet', 'fire sensors']
                selected_amenities = st.multiselect("Select Amenities", amenities_options, key='commercial_selected_amenities')
                electric_charge = st.selectbox("Electric Charge Included", ['yes', 'no'], index=0, key='commercial_electric_charge')
                water_charge = st.selectbox("Water Charge Included", ['yes', 'no'], index=0, key='commercial_water_charge')

            with st.expander("Other Details"):
                possession_status = st.selectbox("Possession Status", ['ready to move', 'Under Construction'], index=0, key='commercial_possession_status')
                posted_by = st.selectbox("Posted By", ['owner', 'housing expert', 'broker'], index=0, key='commercial_posted_by')
                lock_in_period_str = st.selectbox("Lock-in Period", ['2 months', '6 months', '12 months', '3 months', '1 month', '11 months', '4 months', '10 months', '6  months', '8  months', '4  months', '36 months'], index=0, key='commercial_lock_in_period')
                expected_rent_increase_str = st.selectbox("Yearly Rent Increase", ['0.05', '0.10'], index=0, key='commercial_expected_rent_increase')
                negotiable = st.selectbox("Negotiable", ['yes', 'no'], index=0, key='commercial_negotiable')
                brokerage = st.selectbox("Brokerage", ['yes', 'no'], index=0, key='commercial_brokerage')
            
            # --- Submit Button ---
            predict_button = st.form_submit_button("Predict Rent Price", use_container_width=True)
            
            if predict_button:
                # Process the selected floors from multiselect
                floor_numbers = [floor.replace("Floor ", "") for floor in selected_floors]
                floor_no_str = ",".join(sorted(floor_numbers))
                
                lock_in_period = int(re.sub(r'\D', '', lock_in_period_str))
                expected_rent_increase = float(expected_rent_increase_str)
                
                user_data = {
                    'listing litle': property_type, 'city': 'nagpur', 'area': area, 'zone': zone,
                    'location_hub': location_hub, 'property_type': property_type, 'ownership': ownership,
                    'size_in_sqft': size_sqft, 'carpet_area_sqft': carpet_area,
                    'private_washroom': private_washroom, 'public_washroom': public_washroom,
                    'floor_no': floor_no_str, 'total_floors': total_floors,
                    'amenities_count': ', '.join(selected_amenities),
                    'electric_charge_included': electric_charge, 'water_charge_included': water_charge,
                    'property_age': property_age, 'possession_status': possession_status,
                    'posted_by': posted_by, 'lock in period': f"{lock_in_period} months",
                    'expected rent increases yearly': expected_rent_increase,
                    'negotiable': negotiable, 'brokerage': brokerage
                }
                
                processed_df = preprocess_commercial_input(user_data, feature_names, scaler)
                
                try:
                    prediction_log = model.predict(processed_df)[0]
                    base_prediction = np.expm1(prediction_log)
                    
                    # Calculate floor-adjusted rent
                    adjusted_rent, avg_weightage = calculate_floor_adjusted_rent(base_prediction, floor_numbers)
                    
                    st.session_state.commercial_base_prediction = base_prediction
                    st.session_state.commercial_adjusted_prediction = adjusted_rent
                    st.session_state.commercial_avg_weightage = avg_weightage
                    st.session_state.commercial_user_data = user_data
                    st.session_state.commercial_processed_df = processed_df
                    st.session_state.commercial_selected_floors = selected_floors
                    st.success("Prediction successful! See the results below.")

                except Exception as e:
                    st.error(f"Error making prediction: {e}")

        # --- Prediction Results Section ---
        if 'commercial_base_prediction' in st.session_state:
            st.markdown("---")
            st.markdown('<h2>Prediction Results</h2>', unsafe_allow_html=True)
            
            base_prediction = st.session_state.commercial_base_prediction
            adjusted_prediction = st.session_state.commercial_adjusted_prediction
            avg_weightage = st.session_state.commercial_avg_weightage
            user_data = st.session_state.commercial_user_data
            selected_floors = st.session_state.commercial_selected_floors
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Base Price Display
                st.markdown('<div class="price-container">', unsafe_allow_html=True)
                st.markdown('<div class="price-label">Base Rent Price (Ground Floor)</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="price-value">₹{base_prediction:.2f}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Adjusted Price Display
                st.markdown('<div class="price-container" style="background: linear-gradient(135deg, #f59e0b 0%, #ef4444 100%);">', unsafe_allow_html=True)
                st.markdown('<div class="price-label">Estimated Rent Price (Floor Adjusted)</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="price-value">₹{adjusted_prediction:.2f}</div>', unsafe_allow_html=True)
                
                # Show percentage change
                if avg_weightage > 0:
                    st.markdown(f'<div class="price-change positive-change">+{avg_weightage:.1f}% Floor Premium</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="price-change">No Floor Premium</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown('<h4>Property Summary</h4>', unsafe_allow_html=True)
                st.write(f"**Property Type:** {user_data['property_type'].title()}")
                st.write(f"**Size:** {user_data['size_in_sqft']} sqft")
                st.write(f"**Area:** {user_data['area'].title()}")
                # Display selected floors
                floors_list = user_data['floor_no'].split(',')
                if len(floors_list) == 1:
                    st.write(f"**Floor:** Floor {floors_list[0]}")
                else:
                    st.write(f"**Floors:** {', '.join([f'Floor {f}' for f in floors_list])}")
            
            with col2:
                st.markdown('<h4>Price Comparison</h4>', unsafe_allow_html=True)
                lower_bound = adjusted_prediction * 0.85
                upper_bound = adjusted_prediction * 1.15
                st.write(f"**Fair Range:** ₹{lower_bound:.2f} - ₹{upper_bound:.2f}")
                comparison_price = st.number_input("Enter Listed Price", min_value=0.0, value=float(adjusted_prediction), step=1000.0, key='commercial_comparison_price')
                if comparison_price < lower_bound:
                    st.warning("Listed price is **BELOW** fair range.")
                elif comparison_price > upper_bound:
                    st.warning("Listed price is **ABOVE** fair range.")
                else:
                    st.success("Listed price is **FAIR**.")
                
                st.markdown('<h4>Future Projection</h4>', unsafe_allow_html=True)
                years = st.slider("Years", min_value=1, max_value=10, value=5, key='commercial_years')
                growth_rate = st.slider("Growth (%)", min_value=0.0, max_value=15.0, value=5.0, step=0.5, key='commercial_growth_rate')
                projected_price = adjusted_prediction * ((1 + growth_rate/100) ** years)
                st.write(f"**Rent in {years} years:** ₹{projected_price:.2f}")
                
                # --- MODIFIED PLOTTING SECTION ---
                fig, ax = plt.subplots(figsize=(10, 5))
                years_range = np.arange(0, years + 1)
                prices = [adjusted_prediction * ((1 + growth_rate/100) ** y) for y in years_range]
                
                # Set plot background to white
                ax.set_facecolor('#FFFFFF')
                fig.patch.set_facecolor('#FFFFFF')
                
                # Plot line with a visible color
                ax.plot(years_range, prices, marker='o', linestyle='-', color='#1f77b4')
                
                # Set all text and grid elements to black
                ax.set_title(f'Rent Projection ({growth_rate}% Growth)', color='black')
                ax.set_xlabel('Years', color='black')
                ax.set_ylabel('Rent Price (₹)', color='black')
                ax.tick_params(colors='black')
                ax.grid(True, linestyle='--', color='black', alpha=0.3)
                
                st.pyplot(fig)
                
                # Floor Impact Analysis
                st.markdown('<h4>Floor Impact Analysis</h4>', unsafe_allow_html=True)
                floor_impact_data = []
                for floor in selected_floors:
                    floor_num = int(floor.replace("Floor ", ""))
                    premium = FLOOR_WEIGHTAGE.get(floor_num, 0)
                    floor_price = base_prediction * (1 + premium / 100)
                    floor_impact_data.append({
                        'Floor': floor,
                        'Premium (%)': premium,
                        'Price': f"₹{floor_price:.2f}"
                    })
                
                if floor_impact_data:
                    impact_df = pd.DataFrame(floor_impact_data)
                    st.dataframe(impact_df, hide_index=True, use_container_width=True)
