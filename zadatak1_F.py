import pandas as pd

data = pd.read_csv("data_C02_emission.csv")

#pretvaranje iz object u category
object_columns = data.select_dtypes(include=['object'])
data[object_columns.columns] = object_columns.astype('category')

dizelasi = data[data["Fuel Type"] == "D"]
print(dizelasi["Fuel Consumption City (L/100km)"].mean())

benzinci = data[data["Fuel Type"] == "X"]
print(benzinci["Fuel Consumption City (L/100km)"].mean())

print(dizelasi["Fuel Consumption City (L/100km)"].median())
print(benzinci["Fuel Consumption City (L/100km)"].median())