import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

import joblib


# 1. Create sample house price data
np.random.seed(42)

data = {
    "Area": np.random.randint(500, 5000, 1000),
    "Bedrooms": np.random.randint(1, 6, 1000),
    "Bathrooms": np.random.randint(1, 4, 1000),
    "House_Age": np.random.randint(1, 30, 1000),
    "Distance_City": np.random.randint(1, 30, 1000)
}

df = pd.DataFrame(data)


# 2. Generate house price
df["Price"] = (
    df["Area"] * 5000
    + df["Bedrooms"] * 500000
    + df["Bathrooms"] * 300000
    - df["House_Age"] * 50000
    - df["Distance_City"] * 100000
)


# 3. Display dataset
print(df.head())
print("Dataset shape:", df.shape)


# 4. Separate features and target
X = df[[
    "Area",
    "Bedrooms",
    "Bathrooms",
    "House_Age",
    "Distance_City"
]]

y = df["Price"]

print("Features:", X.shape)
print("Target:", y.shape)


# 5. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)


# 6. Create and train model
model = LinearRegression()

model.fit(X_train, y_train)

print("Model training completed")

# 7. Make predictions
y_pred = model.predict(X_test)

# 8. Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("R2 Score:", r2)

# 9. Save the trained model
joblib.dump(model, "house_price_model.pkl")

print("Model saved successfully")