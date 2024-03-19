import pandas as pd

data = pd.read_csv("data_C02_emission.csv")

#pretvaranje iz object u category
object_columns = data.select_dtypes(include=['object'])
data[object_columns.columns] = object_columns.astype('category')

dizelasi_4_cilindarski = data[(data["Cylinders"] == 4) & (data["Fuel Type"] == 'D')]
index_najveceg = dizelasi_4_cilindarski["Fuel Consumption City (L/100km)"].argmax()
print(dizelasi_4_cilindarski.iloc[index_najveceg,:])