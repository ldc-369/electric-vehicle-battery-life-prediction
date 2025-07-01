import numpy as np
from tensorflow.keras.models import load_model  # TensorFlow's Keras
import pickle

# load model
model = load_model("./model/model.keras")

with open('./model/X_scaler.pkl', 'rb') as f1:
   X_scaler = pickle.load(f1)
   
with open('./model/Y_scaler.pkl', 'rb') as f2:
   Y_scaler = pickle.load(f2)

with open('./model/type_encoder.pkl', 'rb') as f3:
    type_encoder = pickle.load(f3)

def prepare_input(type_battery, capacity, re, rct):
    # encode
    type_encoded = type_encoder.transform(np.array([[type_battery.lower()]])) # array(1, 3)
    
    X_remain = np.array([[capacity, re, rct]]) # array(1, 3)
    X_predict = np.concatenate((type_encoded, X_remain), axis=1)  # array(1, 6)

    # scale
    X_predict = X_scaler.transform(X_predict)
    
    return X_predict

def predict_battery_life(**predict_input):
    X_predict = prepare_input(**predict_input)

    # ambient_temperature
    Y_predict = model.predict(X_predict)

    return Y_scaler.inverse_transform(Y_predict)

Y_predict = predict_battery_life(type_battery = 'discharge', capacity = 1.674305, re = -4.976500e+11, rct = 1.055903e+12)

print(f"Predicted Battery Life: {Y_predict}")