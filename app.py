import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Page config
st.set_page_config(page_title="House Price Predictor", page_icon="🏠")

# Title
st.title("🏠 Lahore House Price Predictor")
st.write("Enter property details to get price prediction")


# Load model
@st.cache_resource
def load_model():
    # List of possible model files to try (in order of preference)
    model_files = [
        'lahore_house_model.joblib',
        'model.joblib',
        'pipe.pkl'
    ]

    for model_file in model_files:
        if os.path.exists(model_file):
            try:
                if model_file.endswith('.joblib'):
                    model = joblib.load(model_file)
                    st.success(f"✅ Model loaded successfully from {model_file}!")
                else:  # .pkl file
                    import pickle
                    with open(model_file, 'rb') as file:
                        model = pickle.load(file)
                    st.success(f"✅ Model loaded successfully from {model_file}!")

                # Verify model has predict method
                if hasattr(model, 'predict'):
                    return model
                else:
                    st.error(f"❌ {model_file} doesn't contain a valid model")
                    continue

            except Exception as e:
                st.warning(f"⚠️ Failed to load {model_file}: {str(e)}")
                continue

    # If no model loaded successfully
    st.error("❌ Could not load any model file!")
    import sklearn
    st.warning(f"🔍 Current scikit-learn version: {sklearn.__version__}")
    st.info("💡 **Required:** scikit-learn version 1.6.1 (same as Colab)")
    st.write("**Steps to fix:**")
    st.write("1. Upgrade scikit-learn: `pip install scikit-learn==1.6.1`")
    st.write("2. In Colab, re-save with joblib:")
    st.code("""
import joblib
joblib.dump(pipe, 'lahore_house_model.joblib')
from google.colab import files
files.download('lahore_house_model.joblib')
    """, language="python")
    return None


model = load_model()

# Show valid categories if model is loaded
if model:
    try:
        # Extract valid categories from the trained model
        preprocessor = model.named_steps['columntransformer']
        onehot_encoder = preprocessor.named_transformers_['passthrough']

    except:
        valid_locations = ['others']  # fallback
        valid_types = ['House', 'Flat', 'Upper Portion', 'Lower Portion', 'Farm House']

if model:
    # Input fields
    st.subheader("Property Details")

    # Location - Exact locations from your training data
    training_locations = [
        'DHA Defence, Lahore, Punjab', 'Bahria Town, Lahore, Punjab', 'Askari, Lahore, Punjab',
        'Raiwind Road, Lahore, Punjab', 'Johar Town, Lahore, Punjab', 'Bahria Orchard, Lahore, Punjab',
        'State Life Housing Society, Lahore, Punjab', 'Park View City, Lahore, Punjab',
        'Central Park Housing Scheme, Lahore, Punjab', 'GT Road, Lahore, Punjab',
        'DHA 11 Rahbar, Lahore, Punjab', 'Paragon City, Lahore, Punjab', 'Wapda Town, Lahore, Punjab',
        'Al Rehman Garden, Lahore, Punjab', 'Allama Iqbal Town, Lahore, Punjab',
        'Valencia Housing Society, Lahore, Punjab', 'Khayaban-e-Amin, Lahore, Punjab',
        'Marghzar Officers Colony, Lahore, Punjab', 'Bedian Road, Lahore, Punjab',
        'Gulberg, Lahore, Punjab', 'Punjab Coop Housing Society, Lahore, Punjab',
        'Canal Garden, Lahore, Punjab', 'Formanites Housing Scheme, Lahore, Punjab',
        'Lahore Medical Housing Society, Lahore, Punjab', 'Nasheman-e-Iqbal, Lahore, Punjab',
        'Cantt, Lahore, Punjab', 'Eden, Lahore, Punjab', 'Model Town, Lahore, Punjab',
        'New Lahore City, Lahore, Punjab', 'Samanabad, Lahore, Punjab', 'Jubilee Town, Lahore, Punjab',
        'Ferozepur Road, Lahore, Punjab', 'Multan Road, Lahore, Punjab',
        'Pak Arab Housing Society, Lahore, Punjab', 'EME Society, Lahore, Punjab',
        'Lahore Motorway City, Lahore, Punjab', 'Architects Engineers Housing Society, Lahore, Punjab',
        'Bankers Avenue Cooperative Housing Society, Lahore, Punjab', 'Green City, Lahore, Punjab',
        'Bahria Nasheman, Lahore, Punjab', 'Divine Gardens, Lahore, Punjab', 'LDA Avenue, Lahore, Punjab',
        'Sabzazar Scheme, Lahore, Punjab', 'Kahna, Lahore, Punjab', 'Defence Road, Lahore, Punjab',
        'Punjab University Employees Society, Lahore, Punjab', 'Garden Town, Lahore, Punjab',
        'Shadab Garden, Lahore, Punjab', 'Punjab Govt Employees Society, Lahore, Punjab',
        'Military Accounts Housing Society, Lahore, Punjab', 'Thokar Niaz Baig, Lahore, Punjab',
        'Main Canal Bank Road, Lahore, Punjab', 'Ghous Garden, Lahore, Punjab',
        'Bankers Co-operative Housing Society, Lahore, Punjab', 'Township, Lahore, Punjab',
        'Lalazaar Garden, Lahore, Punjab', 'Cavalry Ground, Lahore, Punjab', 'College Road, Lahore, Punjab',
        'Vital Homes Housing Scheme, Lahore, Punjab', 'Faisal Town, Lahore, Punjab',
        'OPF Housing Scheme, Lahore, Punjab', 'Tariq Gardens, Lahore, Punjab', 'Izmir Town, Lahore, Punjab',
        'Harbanspura, Lahore, Punjab', 'Audit & Accounts Housing Society, Lahore, Punjab',
        'IEP Engineers Town, Lahore, Punjab', 'NFC 1, Lahore, Punjab', 'PIA Housing Scheme, Lahore, Punjab',
        'PCSIR Housing Scheme, Lahore, Punjab', 'Gulshan-e-Ravi, Lahore, Punjab',
        'Sui Gas Housing Society, Lahore, Punjab', 'Al Noor Park Housing Society, Lahore, Punjab',
        'Cavalry Extension, Lahore, Punjab', 'Green Cap Housing Society, Lahore, Punjab',
        'Chinar Bagh, Lahore, Punjab', 'Nawab Town, Lahore, Punjab', 'Al-Hamd Park, Lahore, Punjab',
        'HBFC Housing Society, Lahore, Punjab', 'Gulshan-e-Lahore, Lahore, Punjab',
        'Revenue Society, Lahore, Punjab', 'Al-Hamad Colony (AIT), Lahore, Punjab',
        'Hamza Town, Lahore, Punjab', 'Canal Bank Housing Scheme, Lahore, Punjab',
        'UET Housing Society, Lahore, Punjab', 'Ichhra, Lahore, Punjab', 'Fazaia Housing Scheme, Lahore, Punjab',
        'Walton Road, Lahore, Punjab', 'Al-Hafiz Town, Lahore, Punjab', 'Chungi Amar Sadhu, Lahore, Punjab',
        'Punjab Small Industries Colony, Lahore, Punjab', 'Shahtaj Colony, Lahore, Punjab',
        'PCSIR Staff Colony, Lahore, Punjab', 'Jail Road, Lahore, Punjab', 'Taj Bagh Scheme, Lahore, Punjab',
        'Awan Town, Lahore, Punjab', 'Shershah Colony - Raiwind Road, Lahore, Punjab',
        'Punjab Government Servant Housing Foundation, Lahore, Punjab', 'Canal Fort II, Lahore, Punjab',
        'Elite Town, Lahore, Punjab', 'Park Avenue Housing Scheme, Lahore, Punjab',
        'Super Town, Lahore, Punjab', 'Rail Town (Canal City), Lahore, Punjab',
        'Sunfort Gardens, Lahore, Punjab', 'Rehan Garden, Lahore, Punjab', 'Shah Jamal, Lahore, Punjab',
        'Zaitoon City, Lahore, Punjab', 'others'
    ]

    location = st.selectbox(
        "Location",
        training_locations,
        index=0,  # Default to first location
        help="Select the exact area where your property is located"
    )

    # Property Type - Exact types from your training data
    property_type = st.selectbox(
        "Property Type",
        ['House', 'Flat']  # Only these two types in your training data
    )

    # Area - Convert to square feet like in training
    col1, col2 = st.columns(2)
    with col1:
        area_size = st.number_input("Area Size", min_value=1.0, value=5.0, step=0.5)
    with col2:
        area_unit = st.selectbox("Unit", ["Marla", "Kanal"])


    # Convert area to square feet (same as your training data conversion)
    def convert_area_to_sqft(size, unit):
        if unit.lower() == "marla":
            return size * 272.25
        elif unit.lower() == "kanal":
            return size * 5445
        else:
            return size


    # Convert area to square feet for the model
    area_sqft = convert_area_to_sqft(area_size, area_unit)

    # Display the conversion for user reference
    st.info(f"📏 {area_size} {area_unit} = {area_sqft:,.0f} sq ft")

    # Bedrooms and Bathrooms
    col3, col4 = st.columns(2)
    with col3:
        bedrooms = st.selectbox("Bedrooms", [1, 2, 3, 4, 5, 6, 7, 8], index=2)
    with col4:
        bathrooms = st.selectbox("Bathrooms", [1, 2, 3, 4, 5, 6, 7, 8], index=1)

    # Predict button
    if st.button("Predict Price", type="primary"):
        # Create input dataframe with correct column names and converted area
        input_data = pd.DataFrame({
            'Type': [property_type],  # Model expects 'Type' first
            'Location': [location],  # Then 'Location'
            'Area': [area_sqft],  # Converted area in square feet (not string)
            'Bath(s)': [bathrooms],  # Model expects 'Bath(s)'
            'Bedroom(s)': [bedrooms]  # Model expects 'Bedroom(s)'
        })

        # Make prediction
        try:
            prediction = model.predict(input_data)[0]

            # Convert from log scale (since you used np.log1p in training)
            predicted_price = np.expm1(prediction)

            st.success(f"💰 Predicted Price: PKR {predicted_price:,.0f}")

            # Show input summary
            st.info(
                f"📋 Property: {bedrooms} bed, {bathrooms} bath {property_type.lower()} in {location.split(',')[0]} ({area_size} {area_unit})")

        except Exception as e:
            st.error(f"❌ Prediction error: {str(e)}")

            if "unknown categories" in str(e).lower():
                st.warning("🔍 This means the location/property type wasn't in the training data")

                # Try to extract valid categories from the model
                try:
                    preprocessor = model.named_steps['columntransformer']
                    onehot_encoder = preprocessor.named_transformers_['passthrough']
                    if hasattr(onehot_encoder, 'categories_'):
                        st.write("**Valid Locations:**", list(onehot_encoder.categories_[0]))
                        st.write("**Valid Property Types:**", list(onehot_encoder.categories_[1]))
                except:
                    st.info("💡 Try selecting 'others' for location if your area isn't listed")

            st.write("🔍 Debug info:")
            st.write("Input data:")
            st.dataframe(input_data)

    # Show sample data info
    st.subheader("📖 How to use:")
    st.write("1. Select your property location from the dropdown")
    st.write("2. Choose the property type")
    st.write("3. Enter area size and select unit (Marla/Kanal)")
    st.write("4. Select number of bedrooms and bathrooms")
    st.write("5. Click 'Predict Price' to get the estimated price")

else:
    st.warning("⚠️ Model not loaded. Please check your pipe.pkl file.")
    st.info("💡 Make sure you have:")
    st.write("- Downloaded pipe.pkl from Google Colab")
    st.write("- Placed it in the same folder as this app.py file")
    st.write("- Installed all required packages: `pip install streamlit pandas numpy scikit-learn`")