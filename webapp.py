import streamlit as st
import numpy as np
import pickle

# Load the model and scaler
with open("gdp_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

st.title("🌍 GDP Predictor App")
st.markdown("Predict the GDP of a nation based on socioeconomic indicators.")

# User inputs
country_encoded = st.number_input("Country (encoded)", min_value=0)
year = st.number_input("Year", min_value=1960, max_value=2030, value=2022)
population = st.number_input("Population", min_value=0.0)
life_expectancy = st.number_input("Life Expectancy", min_value=0.0)
unemployment_rate = st.number_input("Unemployment Rate (%)", min_value=0.0)
co2_emissions = st.number_input("CO2 Emissions (kt)", min_value=0.0)
electricity_access = st.number_input("Access to Electricity (%)", min_value=0.0, max_value=100.0)

# Predict button
if st.button("Predict GDP"):
    input_data = np.array([[country_encoded, year, population, life_expectancy, unemployment_rate, co2_emissions, electricity_access]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    st.success(f"Predicted GDP: ${prediction:,.2f}")