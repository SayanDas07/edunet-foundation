import streamlit as st
import numpy as np
import joblib
import pandas as pd

st.set_page_config(page_title="Diabetes Prediction", layout="wide")

@st.cache_resource
def load_model():
    model = joblib.load("diabetes_trained_model.pkl")  
    scaler = joblib.load("scaler.pkl")        
    return model, scaler

model, scaler = load_model()


data = pd.read_csv("diabetes.csv")
feature_means = data.drop(columns='Outcome').mean()

st.markdown("""
    <style>
    body {
        font-family: 'Roboto', 'Segoe UI', sans-serif;
        background-color: #f9fbfd;
        margin: 0;
        padding: 0;
        color: #2c3e50;
    }
    
    /* Main title styles */
    .title {
        text-align: center;
        color: #1e88e5;
        font-size: 2.8em;
        margin-top: 40px;
        margin-bottom: 10px;
        font-weight: 700;
        letter-spacing: -0.5px;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.05);
    }
    
    /* Subtitle/description styles */
    .description {
        text-align: center;
        color: #546e7a;
        font-size: 1.2em;
        margin: 0 auto 50px;
        max-width: 800px;
        line-height: 1.6;
        font-weight: 300;
    }
    
    /* Main container styling */
    .input-container {
        background-color: #ffffff;
        padding: 40px;
        border-radius: 16px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.08);
        margin: 0 auto 50px;
        max-width: 1000px;
        border: 1px solid rgba(0, 0, 0, 0.05);
    }
    
    /* Form section headers */
    .section-header {
        font-size: 1.3em;
        font-weight: 600;
        color: #1e88e5;
        margin-bottom: 25px;
        padding-bottom: 10px;
        border-bottom: 2px solid #e3f2fd;
    }
    
    /* Input field labels */
    .input-label {
        font-weight: 500;
        font-size: 1.05em;
        color: #455a64;
        margin-bottom: 8px;
        display: block;
    }
    
    /* Submit button styling */
    .btn {
        background: linear-gradient(135deg, #1e88e5 0%, #1565c0 100%);
        color: white;
        font-weight: 600;
        padding: 14px 32px;
        border-radius: 8px;
        cursor: pointer;
        font-size: 1.2em;
        width: 100%;
        transition: all 0.3s ease;
        border: none;
        margin-top: 20px;
        box-shadow: 0 4px 12px rgba(30, 136, 229, 0.3);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .btn:hover {
        background: linear-gradient(135deg, #1976d2 0%, #0d47a1 100%);
        box-shadow: 0 6px 16px rgba(30, 136, 229, 0.4);
        transform: translateY(-2px);
    }
    
    .btn:active {
        transform: translateY(1px);
        box-shadow: 0 2px 8px rgba(30, 136, 229, 0.4);
    }
    
    /* Result message styling */
    .result {
        font-size: 1.6em;
        font-weight: 700;
        margin-top: 30px;
        padding: 20px;
        text-align: center;
        border-radius: 12px;
        transition: all 0.3s ease;
        animation: fadeIn 0.5s ease-in-out;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .result-positive {
        background-color: #ffebee;
        color: #d32f2f;
        border-left: 5px solid #d32f2f;
    }
    
    .result-negative {
        background-color: #e8f5e9;
        color: #2e7d32;
        border-left: 5px solid #2e7d32;
    }
    
    /* Required field marker */
    .required {
        color: #f44336;
        font-weight: bold;
        margin-left: 4px;
    }
    
    /* Input field styling */
    .input-container input, .form-group input {
        border-radius: 8px;
        padding: 14px;
        margin-bottom: 20px;
        border: 2px solid #e0e0e0;
        font-size: 1.1em;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: inset 0 1px 3px rgba(0,0,0,0.05);
    }
    
    .input-container input:focus, .form-group input:focus {
        outline: none;
        border-color: #1e88e5;
        box-shadow: 0 0 0 3px rgba(30, 136, 229, 0.2);
    }
    
    .input-container input::placeholder, .form-group input::placeholder {
        color: #bdbdbd;
        font-weight: 300;
    }
    
    /* Column layout */
    .columns {
        display: flex;
        justify-content: space-between;
        gap: 30px;
    }
    
    .col {
        flex: 1;
    }
    
    /* Error message styling */
    .error-msg {
        color: #f44336;
        font-weight: 500;
        margin: 15px 0;
        padding: 12px 20px;
        background-color: #ffebee;
        border-radius: 6px;
        border-left: 4px solid #f44336;
        display: flex;
        align-items: center;
    }
    
    .error-icon {
        margin-right: 10px;
        font-size: 1.2em;
    }
    
    .error-msg-text {
        font-size: 1em;
    }
    
    /* Form layout */
    .form-container {
        display: flex;
        flex-wrap: wrap;
        gap: 20px;
        margin-bottom: 30px;
    }
    
    .form-group {
        flex: 1;
        min-width: 250px;
        padding: 10px;
    }
    
    .form-group label {
        display: block;
        margin-bottom: 10px;
        font-weight: 500;
        color: #455a64;
    }
    
    /* Helper text under fields */
    .helper-text {
        font-size: 0.85em;
        color: #78909c;
        margin-top: -15px;
        margin-bottom: 15px;
    }
    
    /* Tooltip styling */
    .tooltip {
        position: relative;
        display: inline-block;
        margin-left: 5px;
        cursor: help;
    }
    
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 200px;
        background-color: #455a64;
        color: #fff;
        text-align: center;
        border-radius: 6px;
        padding: 10px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
        font-size: 0.9em;
        font-weight: normal;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    
    /* Card styling for sections */
    .card {
        background-color: white;
        border-radius: 12px;
        padding: 25px;
        margin-bottom: 30px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        border: 1px solid #e0e0e0;
    }
    
    /* Style Streamlit elements */
    .stButton button {
        background: linear-gradient(135deg, #1e88e5 0%, #1565c0 100%);
        color: white;
        font-weight: 600;
        padding: 14px 32px;
        border-radius: 8px;
        cursor: pointer;
        font-size: 1.2em;
        width: 100%;
        transition: all 0.3s ease;
        border: none;
        margin-top: 20px;
        box-shadow: 0 4px 12px rgba(30, 136, 229, 0.3);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton button:hover {
        background: linear-gradient(135deg, #1976d2 0%, #0d47a1 100%);
        box-shadow: 0 6px 16px rgba(30, 136, 229, 0.4);
        transform: translateY(-2px);
    }
    
    .stSelectbox, .stNumberInput, .stTextInput {
        margin-bottom: 20px;
    }
    
    .stTextInput input, .stNumberInput input {
        padding: 14px !important;
        font-size: 1.1em !important;
        border: 2px solid #e0e0e0 !important;
        border-radius: 8px !important;
    }
    
    .stTextInput input:focus, .stNumberInput input:focus {
        border-color: #1e88e5 !important;
        box-shadow: 0 0 0 3px rgba(30, 136, 229, 0.2) !important;
    }
    
    /* Responsive design adjustments */
    @media (max-width: 992px) {
        .title {
            font-size: 2.4em;
        }
        
        .description {
            font-size: 1.1em;
            padding: 0 20px;
        }
        
        .input-container {
            padding: 30px 20px;
            margin: 0 15px 40px;
        }
    }
    
    @media (max-width: 768px) {
        .title {
            font-size: 2em;
        }
        
        .columns {
            flex-direction: column;
            gap: 0;
        }
        
        .btn {
            padding: 12px 24px;
            font-size: 1.1em;
        }
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='title'>Diabetes Prediction App</h1>", unsafe_allow_html=True)

st.markdown("<p class='description'>This app predicts the likelihood of diabetes based on patient data such as age, BMI, glucose levels, and more.</p>", unsafe_allow_html=True)

col1, col2 = st.columns([1, 1])


with col1:
    pregnancies = st.text_input("Pregnancies (count) *", "", placeholder="e.g., 0 - 20")
    glucose = st.text_input("Glucose Level (mg/dL) *", "", placeholder="e.g., 70 - 200 mg/dL")
    blood_pressure = st.text_input("Blood Pressure (Systolic in mmHg) *", "", placeholder="e.g., 60 - 180 mmHg")
    skin_thickness = st.text_input("Skin Thickness (mm)", "", placeholder="e.g., 10 - 99 mm")

with col2:
    insulin = st.text_input("Insulin (µU/mL)", "", placeholder="e.g., 0 - 500 µU/mL")
    bmi = st.text_input("BMI (kg/m²)", "", placeholder="e.g., 15.0 - 45.0 kg/m²")
    diabetes_pedigree = st.text_input("Diabetes Pedigree Function", "", placeholder="e.g., 0.1 - 2.5")
    age = st.text_input("Age (years) *", "", placeholder="e.g., 10 - 120 years")

# Validate input to ensure required fields are not empty
if st.button("Predict Diabetes", key="predict", help="Click to predict the likelihood of diabetes based on the entered data", use_container_width=True):
    # Check if the necessary fields are filled
    missing_fields = []
    if not pregnancies:
        missing_fields.append('Pregnancies')
    if not glucose:
        missing_fields.append('Glucose Level')
    if not blood_pressure:
        missing_fields.append('Blood Pressure')
    if not age:
        missing_fields.append('Age')

    if missing_fields:
        st.markdown(f"<div class='error-msg'><span class='error-icon'>⚠️</span><span class='error-msg-text'>Please fill all the required fields marked with a star (*): {', '.join(missing_fields)}</span></div>", unsafe_allow_html=True)
    else:
       
        user_data = np.array([
            int(pregnancies) if pregnancies else feature_means['Pregnancies'],  
            float(glucose) if glucose else feature_means['Glucose'], 
            float(blood_pressure) if blood_pressure else feature_means['BloodPressure'],  
            float(skin_thickness) if skin_thickness else feature_means['SkinThickness'],  
            float(insulin) if insulin else feature_means['Insulin'],  
            float(bmi) if bmi else feature_means['BMI'],  
            float(diabetes_pedigree) if diabetes_pedigree else feature_means['DiabetesPedigreeFunction'],  
            int(age) if age else feature_means['Age']  
        ]).reshape(1, -1)

        # Scale the input data
        user_data_scaled = scaler.transform(user_data)

        prediction = model.predict(user_data_scaled)
        prediction_proba = model.predict_proba(user_data_scaled)[0]

        # Display result 
        if prediction[0] == 1:
            st.markdown(f"<div class='result result-positive'>The model predicts you <strong>may have diabetes</strong> with a confidence of {prediction_proba[1]:.2f}.</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='result result-negative'>The model predicts you <strong>do not have diabetes</strong> with a confidence of {prediction_proba[0]:.2f}.</div>", unsafe_allow_html=True)