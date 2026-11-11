import joblib
import keras
import numpy as np
import pandas as pd
import streamlit as st

st.title('Credit Card Fraud Detection - Deep Learning Deployment')
st.write('Interactive web app using the best trained ANN model.')

@st.cache_resource
def load_dl_artifacts():
    model = keras.models.load_model('best_dl_model.keras')
    scaler = joblib.load('scaler_dl.pkl')
    return model, scaler

model, scaler = load_dl_artifacts()

st.sidebar.header('Transaction Features')
amount = st.sidebar.number_input('Transaction Amount', value=100.0)
time_val = st.sidebar.number_input('Transaction Time (Sec)', value=0.0)

# Collect PCA features (V1 - V28)
features = {}
for i in range(1, 29):
    features[f'V{i}'] = st.sidebar.slider(f'V{i}', -5.0, 5.0, 0.0)

if st.button('Predict Fraud (DL)'):
    input_df = pd.DataFrame([features])
    input_df['NormalizedAmount'] = scaler.transform([[amount]])[0][0]
    input_df['NormalizedTime'] = scaler.transform([[time_val]])[0][0]
    
    arr = input_df.values
    probability = model.predict(arr, verbose=0).ravel()[0]
    prediction = 1 if probability > 0.5 else 0
    
    if prediction == 1:
        st.error(f'Fraudulent Transaction Detected! (Probability: {probability:.4f})')
    else:
        st.success(f'Legitimate Transaction. (Fraud Probability: {probability:.4f})')