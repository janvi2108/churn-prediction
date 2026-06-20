import streamlit as st
import pandas as pd
import pickle
from tensorflow.keras.models import load_model

# Load model
model = load_model('model.h5')

# Load encoders and scaler
with open('onehot_encoder_geography.pkl', 'rb') as f:
    onehot_encoder = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('label_encoder_gender.pkl', 'rb') as f:
    label_encoder_gender = pickle.load(f)

# Streamlit App
st.title("Customer Churn Prediction")

# User Input
geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
gender = st.selectbox("Gender", ["Male", "Female"])

age = st.number_input("Age", min_value=18, max_value=100, value=30)
balance = st.number_input("Balance", min_value=0.0, value=1000.0)
credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=600)
estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=50000.0)

tenure = st.number_input("Tenure", min_value=0, max_value=10, value=3)
num_of_products = st.number_input("Number of Products", min_value=1, max_value=4, value=1)

has_cr_card = st.selectbox("Has Credit Card", ["Yes", "No"])
is_active_member = st.selectbox("Is Active Member", ["Yes", "No"])

if st.button("Predict"):

    # Create input dataframe
    input_data_df = pd.DataFrame({
        'CreditScore': [credit_score],
        'Gender': [gender],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [1 if has_cr_card == "Yes" else 0],
        'IsActiveMember': [1 if is_active_member == "Yes" else 0],
        'EstimatedSalary': [estimated_salary]
    })

    # Encode Gender
    input_data_df['Gender'] = input_data_df['Gender'].map({
    'Female': 0,
    'Male': 1
})

    # Encode Geography
    geography_encoded = onehot_encoder.transform(
        [[geography]]
    ).toarray()

    geography_encoded_df = pd.DataFrame(
        geography_encoded,
        columns=onehot_encoder.get_feature_names_out(['Geography'])
    )

    # Combine data
    input_data_df = pd.concat(
        [input_data_df.reset_index(drop=True),
         geography_encoded_df.reset_index(drop=True)],
        axis=1
    )

    # Ensure correct column order
    input_data_df = input_data_df[scaler.feature_names_in_]

    # Scale
    input_data_scaled = scaler.transform(input_data_df)

    # Predict
    prediction = model.predict(input_data_scaled)
    prediction_probability = prediction[0][0]

    st.write(f"Churn Probability: {prediction_probability:.2%}")

    if prediction_probability > 0.5:
        st.error("Customer is likely to churn.")
    else:
        st.success("Customer is not likely to churn.")