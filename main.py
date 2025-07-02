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

def prepare_input(input):
    # encode
    type_encoded = type_encoder.transform(np.array([[input['Type'].lower()]])) # # array(1, 3)
    
    # scale
    X_remain = np.array([[input['Capacity'], input['Re'], input['Rct']]]) # array(1, 3)
    X_remain = X_scaler.transform(X_remain)

    #concatenate
    X_predict = np.concatenate((type_encoded, X_remain), axis=1)  # array(1, 6)

    return X_predict

def predict_battery_life(input):
    X_predict = prepare_input(input)
    
    Y_predict = model.predict(X_predict)

    return Y_scaler.inverse_transform(Y_predict)

input = {
    "Type": 'discharge', 
    "Capacity": 0.990759, 
    "Re": 0.072553, 
    "Rct": 0.101419
}

Y_predict = predict_battery_life(input)

print(f"Predicted ambient_temperature: {Y_predict}")