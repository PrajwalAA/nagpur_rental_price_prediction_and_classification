import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime
import matplotlib.pyplot as plt

# ---------------- Constants ---------------- #

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

AREA_TO_ZONE = {
    # (Add your mapping here)
}

ROOM_SIZE_GUIDELINES = {
    '1 RK': {'min': 200, 'max': 400},
    '1 BHK': {'min': 400, 'max': 700},
    '2 BHK': {'min': 700, 'max': 1100},
    '3 BHK': {'min': 1100, 'max': 1500},
    '4 BHK': {'min': 1500, 'max': 2200},
    '5+ BHK': {'min': 2200, 'max': 10000}
}

PROPERTY_ROOM_RULES = {
    'Studio Apartment': {'bedrooms': {'min':0,'max':0}, 'bathrooms':{'min':1,'max':1}, 'balconies':{'min':0,'max':1}},
    'Flat': {'bedrooms': {'min':0,'max':5}, 'bathrooms':{'min':1,'max':6}, 'balconies':{'min':0,'max':5}},
    'Independent House': {'bedrooms': {'min':1,'max':10}, 'bathrooms':{'min':1,'max':10}, 'balconies':{'min':0,'max':10}},
    'Independent Builder Floor': {'bedrooms': {'min':1,'max':6}, 'bathrooms':{'min':1,'max':6}, 'balconies':{'min':0,'max':5}},
    'Villa': {'bedrooms': {'min':2,'max':10}, 'bathrooms':{'min':2,'max':10}, 'balconies':{'min':1,'max':10}},
    'Duplex': {'bedrooms': {'min':2,'max':6}, 'bathrooms':{'min':2,'max':6}, 'balconies':{'min':1,'max':5}}
}

ROOM_TYPE_RULES = {
    '1 RK': {'bedrooms':{'min':0,'max':0}, 'bathrooms':{'min':1,'max':1}, 'balconies':{'min':0,'max':1}},
    '1 BHK': {'bedrooms':{'min':1,'max':1}, 'bathrooms':{'min':1,'max':2}, 'balconies':{'min':0,'max':2}},
    '2 BHK': {'bedrooms':{'min':2,'max':2}, 'bathrooms':{'min':1,'max':3}, 'balconies':{'min':0,'max':3}},
    '3 BHK': {'bedrooms':{'min':3,'max':3}, 'bathrooms':{'min':2,'max':4}, 'balconies':{'min':1,'max':4}},
    '4 BHK': {'bedrooms':{'min':4,'max':4}, 'bathrooms':{'min':2,'max':5}, 'balconies':{'min':1,'max':5}},
    '5+ BHK': {'bedrooms':{'min':5,'max':10}, 'bathrooms':{'min':3,'max':10}, 'balconies':{'min':1,'max':10}}
}

AMENITY_IMPACT = {
    'gym':2.5,'gated_community':5.0,'intercom':1.0,'lift':1.5,'pet_allowed':2.0,'pool':3.5,'security':3.0,
    'water_supply_amenity':1.25,'wifi':1.5,'gas_pipeline':1.0,'sports_facility':2.0,'kids_area':0.75,
    'power_backup':2.5,'garden':1.5,'fire_support':1.0,'parking':6.5,'atm_near_me':0.5,'airport_near_me':1.0,
    'bus_stop_near_me':0.25,'hospital_near_me':0.75,'mall_near_me':1.25,'market_near_me':0.75,'metro_station_near_me':1.0,
    'park_near_me':0.5,'school_near_me':0.75,'vastu':3.0
}

# ---------------- Load Model ---------------- #

@st.cache_resource
def load_model_resources():
    try:
        model = joblib.load('m.pkl')
        scaler = joblib.load('s.pkl')
        features = joblib.load('f.pkl')
        st.success("Model loaded successfully.")
        return model, scaler, features
    except FileNotFoundError as e:
        st.error("Model files not found. Ensure 'm.pkl', 's.pkl', 'f.pkl' exist.")
        return None, None, None

model, scaler, features = load_model_resources()

# ---------------- Prediction ---------------- #

def predict_rent(model, scaler, columns, data_dict):
    if model is None or scaler is None or columns is None:
        return None
    df = pd.DataFrame([data_dict])
    
    # One-hot encode categorical features
    for col in CATEGORICAL_FEATURES:
        if col in df:
            dummies = pd.get_dummies(df[col], prefix=col)
            df = df.drop(columns=[col]).join(dummies)
    
    # Align with training columns
    for col in columns:
        if col not in df:
            df[col] = 0
    df = df[columns]
    
    # Scale numerical features
    num_cols = [c for c in NUMERICAL_FEATURES if c in columns]
    if num_cols:
        df[num_cols] = scaler.transform(df[num_cols])
    
    try:
        pred_log = model.predict(df)[0]
        return np.expm1(pred_log)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return None

# ---------------- Validation ---------------- #

def validate_property(data):
    warnings = []
    # Area validation
    if data.get('area_value',0) > data.get('size',0):
        warnings.append("Area value cannot exceed total size.")
    
    # Room & property type validation
    room_type = data.get('room_type')
    prop_type = data.get('property_type')
    
    # Bedrooms/Bathrooms/Balconies
    if prop_type in PROPERTY_ROOM_RULES:
        rules = PROPERTY_ROOM_RULES[prop_type]
        for k in ['bedrooms','bathrooms','balconies']:
            val = data.get(k if k!='balconies' else 'balcony',0)
            if val < rules[k]['min'] or val > rules[k]['max']:
                warnings.append(f"{prop_type} {k} should be between {rules[k]['min']} and {rules[k]['max']}.")
    if room_type in ROOM_TYPE_RULES:
        rules = ROOM_TYPE_RULES[room_type]
        for k in ['bedrooms','bathrooms','balconies']:
            val = data.get(k if k!='balconies' else 'balcony',0)
            if val < rules[k]['min'] or val > rules[k]['max']:
                warnings.append(f"{room_type} {k} should be between {rules[k]['min']} and {rules[k]['max']}.")
    
    # Room size
    if room_type in ROOM_SIZE_GUIDELINES:
        sz = data.get('size',0)
        g = ROOM_SIZE_GUIDELINES[room_type]
        if sz < g['min'] or sz > g['max']:
            warnings.append(f"{room_type} size should be between {g['min']} and {g['max']} sqft.")
    
    return warnings

# ---------------- Streamlit UI ---------------- #

st.title("Rental Price Prediction App")

if model is not None:
    st.header("Property Details")
    col1, col2 = st.columns(2)
    
    with col1:
        size = st.number_input("Size (sqft)", value=1000)
        area_type = st.selectbox("Area Type", ["Carpet Area","Built-up Area","Super Area"])
        area_value = st.number_input("Area Value (sqft)", value=1500)
        bedrooms = st.number_input("Bedrooms", value=2)
        bathrooms = st.number_input("Bathrooms", value=2)
        balcony = st.number_input("Balconies", value=1)
        total_floors = st.number_input("Total Floors", value=4)
        floor_no = st.number_input("Floor No", min_value=0, max_value=total_floors, value=1)
        property_age = st.number_input("Property Age", value=5)
        
    with col2:
        area = st.selectbox("Area", list(AREA_TO_ZONE.keys()))
        zone = AREA_TO_ZONE.get(area,"West Zone")
        st.text(f"Auto Zone: {zone}")
        room_type = st.selectbox("Room Type", list(ROOM_SIZE_GUIDELINES.keys()))
        property_type = st.selectbox("Property Type", ['Flat','Studio Apartment','Independent House','Villa','Duplex'])
        brokerage = st.selectbox("Brokerage", ['No Brokerage','With Brokerage'])
        furnishing_status = st.selectbox("Furnishing Status", ['Fully Furnished','Semi Furnished','Unfurnished'])
        recommended_for = st.selectbox("Recommended For", ['Anyone','Bachelors','Family'])
        type_of_society = st.selectbox("Society Type", ['Gated','Non-Gated','Township'])
        municipal_bore_water = st.selectbox("Water Source", ['Borewell','Both','Municipal'])
    
    st.subheader("Amenities")
    if 'amenities' not in st.session_state:
        st.session_state['amenities'] = {k: False for k in AMENITY_IMPACT}
    
    for i, amenity in enumerate(AMENITY_IMPACT):
        st.session_state['amenities'][amenity] = st.checkbox(f"{amenity.replace('_',' ').title()} (+{AMENITY_IMPACT[amenity]}%)", value=st.session_state['amenities'][amenity])
    
    projection_years = st.slider("Project Years", 1, 20, 5)
    annual_growth_rate = st.slider("Annual Growth Rate (%)", 0.0,10.0,3.5)
    listed_price = st.number_input("Listed Price", value=25000)
    
    if st.button("Predict Rent"):
        # Calculate carpet area
        ratio = 1.0
        if area_type=="Built-up Area": ratio=0.85
        elif area_type=="Super Area": ratio=0.70
        carpet_area = area_value * ratio
        
        data_dict = {
            'Size_In_Sqft': size,
            'Carpet_Area_Sqft': carpet_area,
            'Bedrooms': bedrooms,
            'Bathrooms': bathrooms,
            'Balcony': balcony,
            'Number_Of_Amenities': sum(st.session_state['amenities'].values()),
            'Floor_No': floor_no,
            'Total_floors_In_Building': total_floors,
            'Property_Age': property_age,
            'Road_Connectivity': 5, # default
            'City':'Nagpur',
            'Area': area, 'Zone': zone, 'Frurnishing_Status': furnishing_status,
            'Recomened for': recommended_for, 'Muncipla Water Or Bore Water': municipal_bore_water,
            'Type of Society': type_of_society, 'Room': room_type, 'Type': property_type,
            'Brokerage': brokerage, 'Maintenance_Charge': 'Maintenance Included',
        }
        # Add amenities 0/1
        for k,v in st.session_state['amenities'].items():
            data_dict[k] = int(v)
        
        warnings = validate_property(data_dict)
        if warnings:
            st.warning("Validation Warnings:\n" + "\n".join(warnings))
        
        predicted_rent = predict_rent(model, scaler, features, data_dict)
        if predicted_rent is None: st.error("Prediction failed.")
        else:
            # Adjust with amenities
            total_amenity_pct = sum([AMENITY_IMPACT[k] for k,v in st.session_state['amenities'].items() if v])
            adjusted_rent = predicted_rent*(1 + total_amenity_pct/100)
            
            st.success(f"Predicted Rent: ₹{adjusted_rent:.0f}")
            
            # Future projection
            future_rents = [adjusted_rent*(1 + annual_growth_rate/100)**i for i in range(1, projection_years+1)]
            years = list(range(1, projection_years+1))
            fig, ax = plt.subplots()
            ax.plot(years, future_rents, marker='o')
            ax.set_title("Future Rent Projection")
            ax.set_xlabel("Years")
            ax.set_ylabel("Rent")
            st.pyplot(fig)
            
            # Comparison
            diff_pct = (listed_price - adjusted_rent)/adjusted_rent*100
            if abs(diff_pct) <= 30:
                st.info(f"Listed price ₹{listed_price} is reasonable ({diff_pct:+.1f}%)")
            elif diff_pct>30:
                st.warning(f"Listed price ₹{listed_price} seems overpriced ({diff_pct:+.1f}%)")
            else:
                st.warning(f"Listed price ₹{listed_price} seems underpriced ({diff_pct:+.1f}%)")
