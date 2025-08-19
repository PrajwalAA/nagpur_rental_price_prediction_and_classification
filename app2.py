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


        # --- Organized Amenities & Proximity using expanders ---
        st.subheader("Amenities & Proximity (Check if available)")

        # Define specific costs for each amenity (you can adjust these values)
        amenity_costs = {
            'gym': 500, 'gated_community': 1000, 'intercom': 200, 'lift': 300, 
            'pet_allowed': 400, 'pool': 700, 'security': 600, 'water_supply_amenity': 250,
            'wifi': 300, 'gas_pipeline': 200, 'sports_facility': 400, 'kids_area': 150,
            'power_backup': 500, 'garden': 300, 'fire_support': 200, 'parking': 500,
            'atm_near_me': 100, 'airport_near_me': 200, 'bus_stop_near_me': 50, 
            'hospital_near_me': 150, 'mall_near_me': 250, 'market_near_me': 150,
            'metro_station_near_me': 200, 'park_near_me': 100, 'school_near_me': 150
        }

        # Store checkbox states in session state to access them after button click
        # This is crucial for calculating total amenity cost
        if 'amenity_states' not in st.session_state:
            st.session_state['amenity_states'] = {}
            for amenity_key in amenity_costs.keys():
                st.session_state['amenity_states'][amenity_key] = False

        with st.expander("Property Amenities"):
            col_a, col_b = st.columns(2)
            with col_a:
                st.session_state['amenity_states']['gym'] = st.checkbox("Gym (+Rs 500)", key='gym_cb')
                st.session_state['amenity_states']['intercom'] = st.checkbox("Intercom (+Rs 200)", key='intercom_cb')
                st.session_state['amenity_states']['pet_allowed'] = st.checkbox("Pet Allowed (+Rs 400)", key='pet_allowed_cb')
                st.session_state['amenity_states']['security'] = st.checkbox("Security (+Rs 600)", key='security_cb')
                st.session_state['amenity_states']['gas_pipeline'] = st.checkbox("Gas Pipeline (+Rs 200)", key='gas_pipeline_cb')
                st.session_state['amenity_states']['power_backup'] = st.checkbox("Power Backup (+Rs 500)", key='power_backup_cb')
                st.session_state['amenity_states']['fire_support'] = st.checkbox("Fire Support (+Rs 200)", key='fire_support_cb')
            with col_b:
                st.session_state['amenity_states']['gated_community'] = st.checkbox("Gated Community (+Rs 1000)", key='gated_community_cb')
                st.session_state['amenity_states']['lift'] = st.checkbox("Lift (+Rs 300)", key='lift_cb')
                st.session_state['amenity_states']['pool'] = st.checkbox("Pool (+Rs 700)", key='pool_cb')
                st.session_state['amenity_states']['water_supply_amenity'] = st.checkbox("Water Supply (as amenity) (+Rs 250)", help="Check if this specific water supply amenity is available", key='water_supply_amenity_cb')
                st.session_state['amenity_states']['wifi'] = st.checkbox("WiFi (+Rs 300)", key='wifi_cb')
                st.session_state['amenity_states']['sports_facility'] = st.checkbox("Sports Facility (+Rs 400)", key='sports_facility_cb')
                st.session_state['amenity_states']['kids_area'] = st.checkbox("Kids Area (+Rs 150)", key='kids_area_cb')
                st.session_state['amenity_states']['garden'] = st.checkbox("Garden (+Rs 300)", key='garden_cb')
                st.session_state['amenity_states']['parking'] = st.checkbox("Parking (+Rs 500)", key='parking_cb')

        with st.expander("Proximity to Essential Services"):
            col_c, col_d = st.columns(2)
            with col_c:
                st.session_state['amenity_states']['atm_near_me'] = st.checkbox("ATM Near Me (+Rs 100)", key='atm_near_me_cb')
                st.session_state['amenity_states']['bus_stop_near_me'] = st.checkbox("Bus Stop Near Me (+Rs 50)", key='bus_stop_near_me_cb')
                st.session_state['amenity_states']['mall_near_me'] = st.checkbox("Mall Near Me (+Rs 250)", key='mall_near_me_cb')
                st.session_state['amenity_states']['metro_station_near_me'] = st.checkbox("Metro Station Near Me (+Rs 200)", key='metro_station_near_me_cb')
                st.session_state['amenity_states']['school_near_me'] = st.checkbox("School Near Me (+Rs 150)", key='school_near_me_cb')
            with col_d:
                st.session_state['amenity_states']['airport_near_me'] = st.checkbox("Airport Near Me (+Rs 200)", key='airport_near_me_cb')
                st.session_state['amenity_states']['hospital_near_me'] = st.checkbox("Hospital Near Me (+Rs 150)", key='hospital_near_me_cb')
                st.session_state['amenity_states']['market_near_me'] = st.checkbox("Market Near Me (+Rs 150)", key='market_near_me_cb')
                st.session_state['amenity_states']['park_near_me'] = st.checkbox("Park Near Me (+Rs 100)", key='park_near_me_cb')


    # --- New User Inputs for Future Rate Prediction ---
    st.markdown("---")
    st.subheader("Future Rental Rate Projection")
    projection_years = st.slider("Years from now to project:", min_value=1, max_value=20, value=5, key='projection_years')
    annual_growth_rate = st.slider("Expected Annual Growth Rate (%):", min_value=0.0, max_value=10.0, value=3.5, step=0.1, key='annual_growth_rate')
    
    # This remains the user's input for the "actual" listed price for comparison.
    listed_price = st.number_input("Enter the Listed Price of the property for comparison:", min_value=0, value=25000, key='listed_price_comp')


    # When the user clicks the predict button
    if st.button("Predict Rent"):
        # Calculate amenity additions based on checked boxes
        amenity_additions = 0
        for amenity_key, cost in amenity_costs.items():
            if st.session_state['amenity_states'].get(amenity_key, False):
                amenity_additions += cost
        
        user_input_data = {
            'Size_In_Sqft': size, 'Carpet_Area_Sqft': carpet_area, 'Bedrooms': bedrooms, 'Bathrooms': bathrooms,
            'Balcony': balcony, 'Number_Of_Amenities': amenities_count, 'Security_Deposite': security_deposite,
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
            'Brokerage': brokerage, 'Maintenance_Charge': maintenance_charge
        }

        st.markdown("---")
        st.subheader("Prediction Results")

        # Get and display the current date
        today = datetime.date.today()
        st.info(f"Prediction based on today's market conditions: **{today.strftime('%B %d, %Y')}**")

        # Predict with the single Model (base predicted rent)
        base_predicted_rent = predict_rent_with_model(rf_model, scaler, features, user_input_data)
        
        # Calculate Adjusted Predicted Rent by adding amenity costs
        adjusted_predicted_rent = base_predicted_rent + amenity_additions if base_predicted_rent is not None else None


        if base_predicted_rent is not None:
            st.success(f"Base Predicted Rent (without amenities): **Rs {base_predicted_rent:,.2f}**")
            st.info(f"**Additional Value from Selected Amenities:** Rs {amenity_additions:,.2f}")
            if adjusted_predicted_rent is not None:
                # Display Adjusted Predicted Rent in a different color (e.g., blue)
                st.markdown(f"<span style='color:white; font-weight:bold; font-size: 2em;'>Predicted Rent Rs {adjusted_predicted_rent:,.2f}</span>", unsafe_allow_html=True)

                # --- Future Rent Calculation (using adjusted_predicted_rent) ---
                future_predicted_rent_adjusted = adjusted_predicted_rent * (1 + annual_growth_rate / 100)**projection_years
                
                st.info(f"**Projected Adjusted Rent in {projection_years} years:**")
                st.success(f"Rs {future_predicted_rent_adjusted:,.2f} (assuming a {annual_growth_rate:.1f}% annual growth rate)")

                # --- Price Comparison (comparing Listed Price to Adjusted Predicted Rent) ---
                FAIR_PRICE_TOLERANCE = 0.5
                
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
                    st.success(f"Listed price {listed_price:,.2f} appears to be **Fairly Priced** compared to Adjusted Predicted Rent.")

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
