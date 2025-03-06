import streamlit as st
import numpy as np
import pickle
import os

# Custom CSS for styling
st.markdown(
    """
    <style>
        /* Set a light background */
        body {
            background-color: #ffffff;
            color: black;
        }

        /* Style the image */
        .stImage img {
            border-radius: 15px;  /* Rounded corners */
            transition: transform 0.3s ease, box-shadow 0.3s ease; /* Smooth transition */
        }

        .stImage img:hover {
            transform: scale(1.05);  /* Zoom effect */
            box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.2); /* Shadow */
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit UI
st.title("🌾 Crop Recommendation System")

# Display main image with hover effect
st.image("134.jpg", use_container_width=True, caption="Indian Agriculture and Farming")

st.write("""
### 🚜 About Farming in India
India has a diverse agricultural landscape, with crops like **Rice, Wheat, Maize, and Pulses** being major contributors to the economy. 
Factors such as **soil nutrients, temperature, rainfall, and humidity** play a crucial role in determining the best crop for cultivation.

Using **Machine Learning**, we can analyze these parameters and suggest the most suitable crop for optimal yield. 🌱
""")

# Load the model and scalers
model_path = "C:/Users/Tejas Divekar/ML Projects/Crop Recommend System/model.pkl"
scaler_path = "C:/Users/Tejas Divekar/ML Projects/Crop Recommend System/standardscaler.pkl"
minmax_path = "C:/Users/Tejas Divekar/ML Projects/Crop Recommend System/minmaxscaler.pkl"

# Check if all required files exist
if not all(os.path.exists(path) for path in [model_path, scaler_path, minmax_path]):
    st.error("❌ One or more required files (model.pkl, standardscaler.pkl, minmaxscaler.pkl) are missing!")
    st.stop()

# Load models and scalers
model = pickle.load(open(model_path, 'rb'))
sc = pickle.load(open(scaler_path, 'rb'))
mx = pickle.load(open(minmax_path, 'rb'))

# Crop dictionary for mapping prediction to crop name
crop_dict = {
    1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 
    6: "Papaya", 7: "Orange", 8: "Apple", 9: "Muskmelon", 
    10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana", 
    14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 
    17: "Mungbean", 18: "Mothbeans", 19: "Pigeonpeas", 
    20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
}

# Example default values
example_values = {
    "N": 50.0, "P": 40.0, "K": 30.0, "temp": 25.5, "humidity": 70.0, "ph": 6.5, "rainfall": 120.0
}

# Input fields for user data
st.write("### 📊 Enter Soil and Climate Parameters:")
N = st.number_input("Nitrogen", min_value=0.0, max_value=500.0, step=0.1, value=example_values["N"])
P = st.number_input("Phosphorus", min_value=0.0, max_value=500.0, step=0.1, value=example_values["P"])
K = st.number_input("Potassium", min_value=0.0, max_value=500.0, step=0.1, value=example_values["K"])
temp = st.number_input("Temperature (°C)", min_value=-10.0, max_value=50.0, step=0.1, value=example_values["temp"])
humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, step=0.1, value=example_values["humidity"])
ph = st.number_input("pH Level", min_value=0.0, max_value=14.0, step=0.1, value=example_values["ph"])
rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, step=0.1, value=example_values["rainfall"])

# Predict best crop
if st.button("Predict Best Crop"):
    try:
        # Prepare input array
        feature_list = np.array([N, P, K, temp, humidity, ph, rainfall]).reshape(1, -1)

        # Scale the features
        mx_features = mx.transform(feature_list)
        sc_mx_features = sc.transform(mx_features)

        # Make prediction
        prediction = model.predict(sc_mx_features)
        crop = crop_dict.get(prediction[0], "Unknown Crop")

        # Display result
        st.success(f"✅ Recommended Crop: **{crop}**")
    
    except Exception as e:
        st.error(f"⚠️ Error: {str(e)}")
