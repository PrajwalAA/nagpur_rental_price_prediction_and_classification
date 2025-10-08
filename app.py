import streamlit as st
import joblib
import pandas as pd
import numpy as np

# --- 1. Load the saved objects ---
# Use st.cache_data to load the model/scaler only once
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('mc.pkl') # 'mc.pkl' for model
        scaler = joblib.load('sc.pkl') # 'sc.pkl' for scaler
        feature_names = joblib.load('fc.pkl') # 'fc.pkl' for feature names
        return model, scaler, feature_names
    except Exception as e:
        st.error(f"Error loading assets: {e}. Make sure mc.pkl, sc.pkl, and fc.pkl are in the same directory.")
        return None, None, None

rf_model, scaler, feature_names = load_assets()

if rf_model is not None:
    # --- 2. Set up the Streamlit app interface ---
    st.title('Machine Learning Model Predictor')
    st.write('Enter the feature values below to get a prediction.')

    # Use a sidebar for input features
    st.sidebar.header('Input Features')

    # Create a dictionary to hold user inputs
    user_input = {}
    
    # Define a function to collect user input for each feature
    def get_user_input(features):
        for feature in features:
            # You should customize the widget based on the feature type (e.g., slider, selectbox)
            # This example uses a number_input for all features
            # You may need min_value and max_value based on your data
            default_value = 0.0 # Placeholder default
            user_input[feature] = st.sidebar.number_input(f'{feature}', value=default_value)
        return pd.DataFrame([user_input])

    # Get the input features from the user
    input_df = get_user_input(feature_names)

    # Display the collected input
    st.subheader('User Input')
    st.write(input_df)

    # --- 3. Make Prediction ---
    if st.button('Predict'):
        try:
            # 1. Scaling the input data
            # Ensure the order of columns matches the training data
            scaled_features = scaler.transform(input_df)
            
            # 2. Make the prediction
            prediction = rf_model.predict(scaled_features)
            
            # 3. Display the result
            st.subheader('Prediction Result')
            # Customise the output format based on your model's task (e.g., classification label, regression value)
            st.success(f'The predicted value is: {prediction[0]:.2f}') 

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

# --- Instructions for running the app ---
st.markdown("""
---
**To run this app locally:**
1.  Make sure you have Streamlit installed: `pip install streamlit scikit-learn pandas joblib`
2.  Save the code above as `app.py`.
3.  Ensure your saved files (`mc.pkl`, `sc.pkl`, `fc.pkl`) are in the same directory.
4.  Run from your terminal: `streamlit run app.py`
""")
