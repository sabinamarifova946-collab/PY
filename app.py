import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import warnings

# ===============================
# SKLEARN WARNING LARINI O‘CHIRISH
# ===============================
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=Warning)

# ===============================
# MODELNI YUKLASH
# ===============================
@st.cache_resource
def load_model():
    """
    KNN model, imputer, scaler va label encoderlarni yuklaydi
    """
    data = joblib.load("knn_car_price_model.pkl")
    return data

data = load_model()

# Modelni, ustunlar ro'yxatini va preprocesslarni olish
model = data.get("model")
columns = data.get("columns")
imputer = data.get("imputer")
scaler = data.get("scaler")
encoders = data.get("label_encoders", {})

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
    """
    Input dictionary asosida bashorat chiqaradi (sklearn 1.8+ va column mismatch bilan mos)
    """
    # 1️⃣ Foydalanuvchi inputi DataFrame ga
    df_new = pd.DataFrame([input_dict])

    # 2️⃣ Barcha categorical ustunlarni LabelEncoder bilan encode qilish
    for col, le in encoders.items():
        if col in df_new.columns:
            df_new[col] = df_new[col].apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else le.transform([le.classes_[0]])[0]
            )

    # 3️⃣ Reindex: Model trenining barcha ustunlari bilan moslash
    df_new = df_new.reindex(columns=columns, fill_value=0)  # NaN o‘rniga 0

    # 4️⃣ Imputer va scaler bilan transform
    try:
        df_new = imputer.transform(df_new)
    except AttributeError:
        df_new = imputer.fit_transform(df_new)

    df_new = scaler.transform(df_new)

    # 5️⃣ Bashorat
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

    try:
        price = predict_price(input_data)
        st.success(f"💰 Bashorat qilingan narx: {price:,.0f} $")
    except Exception as e:
        st.error(f"Xatolik yuz berdi: {e}")

    # ===============================
    # GRAFIK — Feature Summary
    # ===============================
    st.subheader("📊 Kiritilgan ma’lumotlar")
    # PyArrow mosligi uchun Value ustunini stringga aylantirish
    df_show = pd.DataFrame(
        [(k, str(v)) for k, v in input_data.items()],
        columns=["Feature", "Value"]
    )
    st.table(df_show)

    # ===============================
    # VIZUAL BAR CHART
    # ===============================
    numeric_data = {
        "Year": year,
        "Engine Size": engine,
        "Mileage": mileage
    }
    fig, ax = plt.subplots()
    ax.bar(numeric_data.keys(), numeric_data.values(), color=['skyblue', 'orange', 'green'])
    ax.set_title("Numeric Features Overview")
    ax.set_ylabel("Value")
    st.pyplot(fig)
