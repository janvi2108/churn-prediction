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

## Deployment Note

Initially, the application was deployed on Streamlit Community Cloud. However, deployment issues were encountered due to TensorFlow compatibility with the default Python environment provided by Streamlit Cloud. The platform was provisioning Python 3.14, while the TensorFlow version required for this project did not have compatible wheels available for that Python version, resulting in dependency installation failures.

To ensure stable deployment and compatibility with TensorFlow, the application was migrated to Render, where the Python runtime version could be explicitly configured. Render provided greater flexibility for managing dependencies and successfully supported the deployment of the trained ANN model along with the required preprocessing artifacts (encoders and scaler files).

### Challenges Faced

* TensorFlow installation failures due to Python version incompatibility.
* Dependency resolution errors during Streamlit Cloud deployment.
* Runtime configuration issues despite specifying Python version files.
* Model artifact management and deployment configuration.

### Solution

* Migrated deployment from Streamlit Community Cloud to Render.
* Configured Python 3.11 environment for TensorFlow compatibility.
* Included model artifacts (`.h5`, encoder, and scaler files) in deployment.
* Successfully deployed the Streamlit application using Render Web Services.


## Author

Janvi Gaba
