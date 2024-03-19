import pandas as pd

data = pd.read_csv("data_C02_emission.csv")

#pretvaranje iz object u category
object_columns = data.select_dtypes(include=['object'])
data[object_columns.columns] = object_columns.astype('category')

new_data = data[(data["Engine Size (L)"] >= 2.5) & (data["Engine Size (L)"] <= 3.5)]
print(len(new_data))
print(new_data['CO2 Emissions (g/km)'].mean())