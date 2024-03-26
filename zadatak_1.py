from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import sklearn.linear_model as lm
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

data = pd.read_csv("data_C02_emission (1).csv")
data = data.drop(columns=["Make", "Model", "Vehicle Class"], axis=1)

input_var = [
    "Fuel Consumption City (L/100km)",
    "Fuel Consumption Hwy (L/100km)",
    "Fuel Consumption Comb (L/100km)",
    "Fuel Consumption Comb (mpg)",
    "Engine Size (L)",
    "Cylinders",
]
output = ["CO2 Emissions (g/km)"]
X = data[input_var].to_numpy()
y = data[output].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.01, random_state=True
)

plt.figure()
plt.scatter(x=X_train[:, 4], y=y_train, color="red", label="train data", s=20)
plt.scatter(x=X_test[:, 4], y=y_test, color="blue", label="test data", s=20)
plt.xlabel("Engine Size(L)")
plt.ylabel("CO2 Emissions (g/km)")
plt.title("train podatci")
plt.legend()
plt.show()

plt.subplot(2, 1, 1)
plt.hist(x=X_train[:, 0], bins=5)
plt.xlabel("Fuel Consumption City (L/100km)")

# skaliraj
sc = MinMaxScaler()
X_train_n = sc.fit_transform(X_train)
X_test_n = sc.transform(X_test)

plt.subplot(2, 1, 2)
plt.hist(x=X_train_n[:, 0], bins=5)
plt.xlabel("Fuel Consumption City (L/100km)")
plt.subplots_adjust(hspace=0.4)
plt.show()

# d
linearModel = lm.LinearRegression()
linearModel.fit(X_train_n, y_train)
print(linearModel.coef_)

# e
y_test_p = linearModel.predict(X_test_n)

plt.figure()
plt.scatter(x=X_test_n[:, 4], y=y_test, color="red", label="real values")
plt.scatter(x=X_test_n[:, 4], y=y_test_p, color="blue", label="predicted values")
plt.xlabel("engine size")
plt.ylabel("emission")
plt.title("Izlaz naucenog modela")
plt.legend()
plt.show()

# f
MSE = mean_squared_error(y_test, y_test_p)
RMSE = math.sqrt(MSE)
MAE = mean_absolute_error(y_test, y_test_p)
MAPE = mean_absolute_percentage_error(y_test, y_test_p)
R2 = r2_score(y_test, y_test_p)

print(MSE)
print(RMSE)
print(MAE)
print(MAPE)
print(R2)

# g
# povecanjem skupa za testiranje se povecava pogreska, ne znacajno, ali ipak malo
