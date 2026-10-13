import pandas as pd
import numpy as np

np.random.seed(42)

# Number of records
n = 2000

# Generate dates
dates = pd.date_range(
    start="2020-01-01",
    periods=n,
    freq="D"
)

# Generate input data
temperature = np.random.uniform(20, 40, n)
rainfall = np.random.uniform(0, 50, n)
population = np.random.randint(500, 5000, n)
previous_usage = np.random.uniform(20000, 150000, n)

# Date features
day_of_week = dates.dayofweek
month = dates.month

# Generate water consumption
water_usage = (
    0.70 * previous_usage
    + 1200 * temperature
    - 500 * rainfall
    + 8 * population
    + np.random.normal(0, 3000, n)
)

# Make sure water usage is positive
water_usage = np.maximum(water_usage, 1000)

# Create DataFrame
data = pd.DataFrame({
    "Date": dates,
    "Temperature": temperature.round(2),
    "Rainfall": rainfall.round(2),
    "Population": population,
    "Previous_Usage": previous_usage.round(2),
    "Day_of_Week": day_of_week,
    "Month": month,
    "Water_Usage": water_usage.round(2)
})

# Save dataset
data.to_csv("data/water_data.csv", index=False)

print("Dataset created successfully!")
print("Number of records:", len(data))
print(data.head())