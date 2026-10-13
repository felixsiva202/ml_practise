import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
data = pd.read_csv("data/water_data.csv")

print("Dataset loaded")
print("Shape:", data.shape)

# Input features
X = data[
    [
        "Temperature",
        "Rainfall",
        "Population",
        "Previous_Usage",
        "Day_of_Week",
        "Month"
    ]
]

# Target
y = data["Water_Usage"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

print("Model training completed")

# Predict test data
predictions = model.predict(X_test)

# Evaluate model
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("MAE:", mae)
print("R2 Score:", r2)

# Save model
joblib.dump(model, "water_model.pkl")

print("Model saved successfully")