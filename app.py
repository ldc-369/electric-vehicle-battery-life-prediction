import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model  # TensorFlow's Keras
import pickle




model = load_model("./pickle/model.h5") 

with open('./pickle/scaler_F.pkl', 'rb') as f1:
   scaler_F= pickle.load(f1)
   
with open('./pickle/scaler_y.pkl', 'rb') as f2:
   scaler_y= pickle.load(f2)

with open('./pickle/onehot_encoder.pkl', 'rb') as f3:
    onehot_encoder = pickle.load(f3)

def prepare_input(type, Capacity, Re, Rct):
    # encode "type"
    type_encoded = onehot_encoder.transform(np.array([[type]]))  # array 2D (1, 3)

    F_predict = np.concatenate((np.array([[Capacity, Re, Rct]]), type_encoded),axis=1)
    print("Input shape:", F_predict.shape) # 2D (1, 6)

    # scaling
    F_predict = scaler_F.transform(F_predict) # 2D (1, 6)
    
    return F_predict

def predict_battery_life(**predict_input):
    F_predict = prepare_input(**predict_input)

    # đầu ra dự đoán (ambient_temperature)
    y_predict = model.predict(F_predict)

    return scaler_y.inverse_transform(y_predict)[0][0]

st.title("Battery Life Prediction using ANN")

# Input fields
type = st.selectbox("Select Discharge Type", ['charge', 'discharge', 'impedance'])
Capacity = st.number_input("Enter Capacity", min_value=0.0, format="%.8f")
Re = st.number_input("Enter Re", min_value=-1e12, max_value=1e12, format="%.8f")
Rct = st.number_input("Enter Rct", min_value=-1e12, max_value=1e12, format="%.8f")

# Button
if st.button('Predict Battery Life'):
    predicted_battery_life = predict_battery_life(type=type, Capacity=Capacity, Re=Re, Rct=Rct)
    st.write(f"The predicted battery life is: {predicted_battery_life:.2f} units")
