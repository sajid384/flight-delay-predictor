import joblib
import pandas as pd

# Load model and feature list
model = joblib.load("flight_delay_model.pkl")
model_features = joblib.load("model_features.pkl")

# Create dictionary with default 0 values
input_dict = {feature: 0 for feature in model_features}

# Fill actual values
input_dict["MONTH"] = 3
input_dict["DAY_OF_WEEK"] = 2
input_dict["DISTANCE"] = 1500
input_dict["TAXI_OUT"] = 15

# Example airline
airline_col = "OP_UNIQUE_CARRIER_DL"
if airline_col in input_dict:
    input_dict[airline_col] = 1

# Convert to DataFrame with correct column order
new_data = pd.DataFrame([input_dict], columns=model_features)

# Predict
prediction = model.predict(new_data)[0]
probability = model.predict_proba(new_data)[0][1]

if prediction == 1:
    print(f"Flight will be DELAYED (Probability: {probability:.2f})")
else:
    print(f"Flight will be ON TIME (Probability: {probability:.2f})")