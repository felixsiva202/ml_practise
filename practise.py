import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,root_mean_squared_error,r2_score

data = {
    "hours_studied":[1,2,3,4,5,6,7,8,9,10],
    "exam_score":[32,42,55,59,64,70,77,82,89,95]
}
df = pd.DataFrame(data)

x=df[["hours_studied"]]
y=df[["exam_score"]]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=42)

model=LinearRegression()
model.fit(x_train,y_train)

y_pred=model.predict(x_test)

print("Actual scores:", y_test.values.tolist())
print("Predicted scores:", y_pred.round(2))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", root_mean_squared_error(y_test, y_pred))
print("R2:", r2_score(y_test, y_pred))














