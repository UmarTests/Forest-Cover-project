import streamlit as st
import numpy as np
import joblib
import pandas as pd

# Load trained model and scaler with absolute paths
model = joblib.load(r"C:\Users\mohdq\OneDrive\Desktop\internship projects\forest_cover.pro\forest_cover_prediction\best_rf_forest_cover_model.joblib")
scaler = joblib.load(r"C:\Users\mohdq\OneDrive\Desktop\internship projects\forest_cover.pro\forest_cover_prediction\scaler.joblib")

# Load feature names
feature_names = joblib.load(r"C:\Users\mohdq\OneDrive\Desktop\internship projects\forest_cover.pro\forest_cover_prediction\feature_names.joblib")
st.write("Original Feature Names Count:", len(feature_names))

# If feature_names doesn't have 54 columns, correct it:
if len(feature_names) != 54:
    feature_names = (
        ["Elevation", "Aspect", "Slope", "Horizontal_Distance_To_Hydrology", "Vertical_Distance_To_Hydrology",
         "Horizontal_Distance_To_Roadways", "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm", "Horizontal_Distance_To_Fire_Points"]
        + [f"Wilderness_Area{i}" for i in range(1, 5)]
        + [f"Soil_Type{i}" for i in range(1, 41)]
    )
    st.write("Corrected Feature Names Count:", len(feature_names))
    # Optionally, overwrite the file:
    joblib.dump(feature_names, r"C:\Users\mohdq\OneDrive\Desktop\internship projects\forest_cover.pro\feature_names.joblib")

# UI Title
st.title("🌲 Forest Cover Type Prediction")
st.markdown("### Enter land characteristics to predict the forest cover type.")

# User Input Sections
with st.expander("📏 Terrain Features"):
    elevation = st.slider("Elevation (meters)", 0, 5000, 2500)
    aspect = st.slider("Aspect (compass direction in degrees)", 0, 360, 180)
    slope = st.slider("Slope (degrees)", 0, 90, 10)

with st.expander("💧 Hydrology & Fire Points"):
    horizontal_dist_hydrology = st.slider("Distance to Nearest Water (meters)", 0, 1000, 250)
    vertical_dist_hydrology = st.slider("Vertical Distance to Water (meters)", -500, 500, 0)
    horizontal_dist_roadways = st.slider("Distance to Roadways (meters)", 0, 7000, 3500)
    horizontal_dist_firepoints = st.slider("Distance to Fire Points (meters)", 0, 7000, 3500)

with st.expander("🌞 Hillshade Factors"):
    hillshade_9am = st.slider("Hillshade at 9 AM", 0, 255, 200)
    hillshade_noon = st.slider("Hillshade at Noon", 0, 255, 220)
    hillshade_3pm = st.slider("Hillshade at 3 PM", 0, 255, 150)

with st.expander("🌲 Wilderness Area & Soil Type"):
    wilderness_area = st.selectbox("Wilderness Area", ["Rawah", "Neota", "Comanche Peak", "Cache la Poudre"])
    soil_type = st.selectbox("Soil Type", [f"Type {i}" for i in range(1, 41)])

# Encode Wilderness Area (One-Hot)
# Here, we assume that during training, the wilderness areas were one-hot encoded in the order: 
# Wilderness_Area1, Wilderness_Area2, Wilderness_Area3, Wilderness_Area4.
# We'll map the selected wilderness to the appropriate one-hot encoding.
wilderness_mapping = {
    "Rawah": 1,
    "Neota": 2,
    "Comanche Peak": 3,
    "Cache la Poudre": 4
}
selected_wilderness = wilderness_mapping[wilderness_area]
wilderness_encoded = [1 if i == selected_wilderness else 0 for i in range(1, 5)]

# Encode Soil Type (One-Hot)
# Convert "Type X" to integer X, then one-hot encode it.
soil_num = int(soil_type.split()[-1])
soil_type_encoded = [1 if i == soil_num else 0 for i in range(1, 41)]

# Prepare input data with correct feature names (54 features total)
input_data = pd.DataFrame([[
    elevation, aspect, slope, horizontal_dist_hydrology, vertical_dist_hydrology,
    horizontal_dist_roadways, hillshade_9am, hillshade_noon, hillshade_3pm,
    horizontal_dist_firepoints
] + wilderness_encoded + soil_type_encoded], columns=feature_names)

# Scale input data
input_scaled = scaler.transform(input_data)

# Prediction
if st.button("🌳 Predict Forest Cover Type"):
    prediction = model.predict(input_scaled)[0]
    
    cover_types = {
        1: "Spruce/Fir",
        2: "Lodgepole Pine",
        3: "Ponderosa Pine",
        4: "Cottonwood/Willow",
        5: "Aspen",
        6: "Douglas-fir",
        7: "Krummholz"
    }
    result = f"🌲 **Predicted Forest Cover Type:** {cover_types[prediction]}"
    st.markdown(f"## {result}")
