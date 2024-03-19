import pandas as pd

data = pd.read_csv("data_C02_emission.csv")

#pretvaranje iz object u category
object_columns = data.select_dtypes(include=['object'])
data[object_columns.columns] = object_columns.astype('category')
manualci = data[data["Transmission"].str.startswith("M")]
print(len(manualci))