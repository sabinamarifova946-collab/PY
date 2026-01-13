import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ===============================
# MODELNI YUKLASH
# ===============================
@st.cache_resource
def load_model():
    data = joblib.load("knn_car_price_model.pkl")
    return data

data = load_model()

model = data["model"]
columns = data["columns"]
imputer = data["imputer"]
scaler = data["scaler"]
encoders = data["label_encoders"]

# ===============================
# SAHIFA SOZLAMASI
# ===============================
st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="🚗",
    layout="centered"
)

st.title("🚗 Car Price Prediction App")
st.write("KNN Regressor yordamida mashina narxini bashorat qilish")

# ===============================
# INPUT FORM
# ===============================
st.header("Mashina ma’lumotlarini kiriting")

with st.form("car_form"):
    brand = st.text_input("Brand", "Toyota")
    model_name = st.text_input("Model", "Camry")
    year = st.number_input("Year", 1990, 2025, 2023)
    engine = st.number_input("Engine Size (L)", 0.5, 8.0, 2.5)
    fuel = st.selectbox("Fuel Type", ["Petrol", "Diesel", "Hybrid", "Electric"])
    transmission = st.selectbox("Transmission", ["Automatic", "Manual"])
    mileage = st.number_input("Mileage (km)", 0, 500000, 15000)
    condition = st.selectbox("Condition", ["New", "Used"])

    submit = st.form_submit_button("🔮 Bashorat qilish")

# ===============================
# PREDICT FUNKSIYA
# ===============================
def predict_price(input_dict):
    df_new = pd.DataFrame([input_dict])
    df_new = df_new.reindex(columns=columns, fill_value=np.nan)

    for col, le in encoders.items():
        if col in df_new:
            df_new[col] = df_new[col].apply(
                lambda x: le.transform([x])[0]
                if x in le.classes_
                else le.transform([le.classes_[0]])[0]
            )

    df_new = imputer.transform(df_new.to_numpy())
    df_new = scaler.transform(df_new)

    return model.predict(df_new)[0]

# ===============================
# NATIJA
# ===============================
if submit:
    input_data = {
        "Brand": brand,
        "Model": model_name,
        "Year": year,
        "Engine Size": engine,
        "Fuel Type": fuel,
        "Transmission": transmission,
        "Mileage": mileage,
        "Condition": condition
    }

    price = predict_price(input_data)

    st.success(f"💰 Bashorat qilingan narx: **{price:,.0f} $**")

    # ===============================
    # GRAFIK — Feature Summary
    # ===============================
    st.subheader("📊 Kiritilgan ma’lumotlar")
    df_show = pd.DataFrame(input_data.items(), columns=["Feature", "Value"])
    st.table(df_show)

    # ===============================
    # VIZUAL BAR CHART
    # ===============================
    numeric_data = {
        "Year": year,
        "Engine": engine,
        "Mileage": mileage
    }

    fig, ax = plt.subplots()
    ax.bar(numeric_data.keys(), numeric_data.values())
    ax.set_title("Numeric Features Overview")

    st.pyplot(fig)
