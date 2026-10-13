import joblib
import pandas as pd

# Load the trained model
model = joblib.load("water_model.pkl")

print("================================")
print(" WATER CONSUMPTION PREDICTOR")
print("================================")

# Get input from user
temperature = float(input("Temperature (°C): "))
rainfall = float(input("Rainfall: "))
population = int(input("Population: "))
previous_usage = float(input("Previous water usage (liters): "))
day_of_week = int(input("Day of week (0=Monday, 6=Sunday): "))
month = int(input("Month (1-12): "))

# Create input DataFrame
input_data = pd.DataFrame({
    "Temperature": [temperature],
    "Rainfall": [rainfall],
    "Population": [population],
    "Previous_Usage": [previous_usage],
    "Day_of_Week": [day_of_week],
    "Month": [month]
})

# Make prediction
prediction = model.predict(input_data)

print("\nPredicted Water Consumption:")
print(round(prediction[0], 2), "liters")