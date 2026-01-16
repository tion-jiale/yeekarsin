import streamlit as st
import pandas as pd
import pickle

@st.cache_resource
def load_model():
    with open("house_price_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

st.title("🏠 House Price Predictor")

rooms = st.number_input("Rooms", 1.0, 15.0, 5.0)
distance = st.number_input("Distance (km)", 0.1, 20.0, 2.0)

if st.button("Predict"):
    X = pd.DataFrame([[rooms, distance]], columns=["Rooms", "Distance"])
    price = model.predict(X)[0]
    st.success(f"Estimated price: ${price:.2f} thousand")


