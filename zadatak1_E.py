import pandas as pd

data = pd.read_csv("data_C02_emission.csv")

# pretvaranje iz object u category
object_columns = data.select_dtypes(include=["object"])
data[object_columns.columns] = object_columns.astype("category")

cilindrasi_4_6_8 = data[data["Cylinders"].isin([4, 6, 8, 10, 12, 14, 16])]
print(len(cilindrasi_4_6_8))

print(cilindrasi_4_6_8.groupby("Cylinders")["CO2 Emissions (g/km)"].mean())
