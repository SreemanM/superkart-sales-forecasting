
import os
import requests
import pandas as pd
import streamlit as st

API_ROOT = os.getenv("API_ROOT", "http://backend:7860")

st.set_page_config(page_title="SuperKart Sales Forecast", layout="centered")
st.title("SuperKart Sales Forecast")
st.caption("Enter product/store attributes to obtain the model's predicted sales.")

with st.form("single_prediction"):
    Product_Weight = st.number_input("Product Weight", min_value=0.0, value=12.66)
    Product_Sugar_Content = st.selectbox(
        "Product Sugar Content", ["Low Sugar", "Regular", "No Sugar"]
    )
    Product_Allocated_Area = st.number_input(
        "Product Allocated Area", min_value=0.0, value=0.027, format="%.3f"
    )
    Product_MRP = st.number_input("Product MRP", min_value=0.0, value=117.08)
    Store_Size = st.selectbox("Store Size", ["Small", "Medium", "High"])
    Store_Location_City_Type = st.selectbox(
        "Store Location City Type", ["Tier 1", "Tier 2", "Tier 3"]
    )
    Store_Type = st.selectbox(
        "Store Type",
        ["Food Mart", "Supermarket Type1", "Supermarket Type2", "Supermarket Type3", "Departmental Store"],
    )
    Product_Id_char = st.selectbox("Product ID Prefix", ["FD", "DR", "NC"])
    Store_Age_Years = st.number_input("Store Age (Years)", min_value=0, value=16)
    Product_Type_Category = st.selectbox(
        "Product Type Category", ["Perishables", "Non Perishables"]
    )

    submitted = st.form_submit_button("Predict Sales")

if submitted:
    payload = {
        "Product_Weight": Product_Weight,
        "Product_Sugar_Content": Product_Sugar_Content,
        "Product_Allocated_Area": Product_Allocated_Area,
        "Product_MRP": Product_MRP,
        "Store_Size": Store_Size,
        "Store_Location_City_Type": Store_Location_City_Type,
        "Store_Type": Store_Type,
        "Product_Id_char": Product_Id_char,
        "Store_Age_Years": Store_Age_Years,
        "Product_Type_Category": Product_Type_Category,
    }

    response = requests.post(f"{API_ROOT}/v1/predict", json=payload, timeout=60)
    if response.ok:
        st.success(f"Predicted sales: {response.json()['predicted_sales']}")
    else:
        st.error(response.text)

st.divider()
st.subheader("Batch Prediction")
uploaded_file = st.file_uploader("Upload batch CSV", type=["csv"])

if uploaded_file is not None and st.button("Run Batch Prediction"):
    files = {
        "file": (
            uploaded_file.name,
            uploaded_file.getvalue(),
            "text/csv",
        )
    }
    response = requests.post(
        f"{API_ROOT}/v1/predictbatch",
        files=files,
        timeout=120,
    )

    if response.ok:
        predictions = response.json()
        output = pd.DataFrame(
            {
                "row_index": list(predictions.keys()),
                "predicted_sales": list(predictions.values()),
            }
        )
        st.dataframe(output, use_container_width=True)
    else:
        st.error(response.text)
