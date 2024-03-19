import pandas as pd

data = pd.read_csv("data_C02_emission.csv")

#pretvaranje iz object u category
object_columns = data.select_dtypes(include=['object'])
data[object_columns.columns] = object_columns.astype('category')

data = data.sort_values(by="Fuel Consumption City (L/100km)",ascending=True)
print("Najemanje")
print(data[['Make','Model','Fuel Consumption City (L/100km)']].head(3))
print("Najvise")
print(data[['Make','Model','Fuel Consumption City (L/100km)']].tail(3))
