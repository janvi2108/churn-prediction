# Customer Churn Prediction System

## Overview

Customer Churn Prediction System is a machine learning web application that predicts the likelihood of a customer leaving a bank based on customer demographics, account information, and engagement metrics. The application uses an Artificial Neural Network (ANN) built with TensorFlow and provides real-time predictions through an interactive Streamlit interface.

## Features

* Real-time customer churn prediction
* Interactive web interface using Streamlit
* Data preprocessing with feature scaling and encoding
* ANN-based deep learning model
* Probability-based churn analysis
* User-friendly input forms

## Tech Stack

* Python
* TensorFlow / Keras
* Scikit-learn
* Pandas
* NumPy
* Streamlit
* Matplotlib
* Seaborn

## Dataset Features

The model utilizes the following customer attributes:

* Credit Score
* Geography
* Gender
* Age
* Tenure
* Account Balance
* Number of Products
* Has Credit Card
* Is Active Member
* Estimated Salary

## Project Structure

```bash
├── app.py
├── model.h5
├── scaler.pkl
├── onehot_encoder_geography.pkl
├── label_encoder_gender.pkl
├── requirements.txt
├── runtime.txt
└── Churn_Modelling.csv
```

## Installation

1. Clone the repository:

```bash
git clone https://github.com/janvi2108/churn-prediction.git
cd churn-prediction
```

2. Create a virtual environment:

```bash
python -m venv venv
```

3. Activate the environment:

```bash
# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

4. Install dependencies:

```bash
pip install -r requirements.txt
```

5. Run the application:

```bash
streamlit run app.py
```

## Model Development

The churn prediction model was developed using:

* Data preprocessing and feature engineering
* One-Hot Encoding for Geography
* Label Encoding for Gender
* Standard Scaling for numerical features
* Artificial Neural Network (ANN) using TensorFlow/Keras

## Prediction Output

The application provides:

* Churn Probability Score
* Customer Retention Risk Assessment
* Churn / No Churn Classification

## Future Improvements

* Model explainability using SHAP
* Advanced ensemble learning models
* Customer segmentation analysis
* Deployment with CI/CD pipelines
* Real-time monitoring dashboard

## Author

Janvi Gaba
