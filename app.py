import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model  # TensorFlow's Keras
import pickle


model = load_model("./pickle/battery_life_model.h5") 

with open('./pickle/scaler.pkl', 'rb') as scaler_file:
   scaler= pickle.load(scaler_file)

with open('./pickle/onehot_encoder.pkl', 'rb') as encode_file:
    onehot_encoder = pickle.load(encode_file)


def prepare_input(type, Capacity, Re, Rct):
    #encode "type"
    type_encoded = onehot_encoder.transform([[type]])  #mảng 2D

    # vector đặc trưng đầu vào 
    F_input = np.concatenate((np.array([[Capacity, Re, Rct]]), type_encoded),axis=1)
    print("Input shape:", F_input.shape)

    # chuẩn hóa đặc trưng đầu vào
    F_input_scaled = scaler.transform(F_input) #mảng 2D

    return F_input_scaled

def predict_battery_life(**predict_input):
    F_input_scaled = prepare_input(**predict_input)

    # đầu ra dự đoán (ambient_temperature)
    y_predicted = model.predict(F_input_scaled)

    return y_predicted[0]



st.title("Battery Life Prediction using ANN")

# Input fields
type = st.selectbox("Select Discharge Type", ['charge', 'discharge', 'impedance'])
Capacity = st.number_input("Enter Capacity", min_value=0.0)
Re = st.number_input("Enter Re", min_value=-1e12, max_value=1e12)
Rct = st.number_input("Enter Rct", min_value=-1e12, max_value=1e12)

# Button
if st.button('Predict Battery Life'):
    predicted_battery_life = predict_battery_life(type=type, Capacity=Capacity, Re=Re, Rct=Rct)
    st.write(f"The predicted battery life is: {predicted_battery_life} units")
