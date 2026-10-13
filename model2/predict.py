import joblib
import pandas as pd

# Load trained model
model = joblib.load("house_price_model.pkl")

# Get input from user
area = float(input("Enter area (sq ft): "))
bedrooms = int(input("Enter number of bedrooms: "))
bathrooms = int(input("Enter number of bathrooms: "))
house_age = int(input("Enter house age (years): "))
distance_city = float(input("Enter distance from city center (km): "))

# Create input data
input_data = pd.DataFrame([{
    "Area": area,
    "Bedrooms": bedrooms,
    "Bathrooms": bathrooms,
    "House_Age": house_age,
    "Distance_City": distance_city
}])

# Make prediction
prediction = model.predict(input_data)

print("Predicted House Price:", prediction[0])