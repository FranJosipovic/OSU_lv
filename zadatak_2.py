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
from sklearn.preprocessing import OneHotEncoder

data = pd.read_csv("data_C02_emission (1).csv")
data = data.drop(columns=["Make", "Model", "Vehicle Class", "Transmission"], axis=1)


ohe = OneHotEncoder()
X_encoded = ohe.fit_transform(data[["Fuel Type"]]).toarray()
ohe_columns = ohe.get_feature_names_out(["Fuel Type"])

ohe_df = pd.DataFrame(columns=ohe_columns, data=X_encoded)
new_data = pd.concat([data, ohe_df], axis=1)
new_data = new_data.drop(columns=["Fuel Type"], axis=1)

print(new_data)

input_var = [
    "Engine Size (L)",
    "Cylinders",
    "Fuel Consumption City (L/100km)",
    "Fuel Consumption Hwy (L/100km)",
    "Fuel Consumption Comb (L/100km)",
    "Fuel Consumption Comb (mpg)",
    "Fuel Type_D",
    "Fuel Type_E",
    "Fuel Type_X",
    "Fuel Type_Z",
]
output = ["CO2 Emissions (g/km)"]

X = new_data[input_var].to_numpy()
y = new_data[output].to_numpy()
