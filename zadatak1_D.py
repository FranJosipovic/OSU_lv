import pandas as pd

data = pd.read_csv("data_C02_emission.csv")

#pretvaranje iz object u category
object_columns = data.select_dtypes(include=['object'])
data[object_columns.columns] = object_columns.astype('category')

audii = data[data['Make'] == 'Audi']

audii_sa_4_cilindra = audii[audii['Cylinders'] == 4]
#mjerenja za audi
print(len(audii))

#CO2 za 4cilindraske audije
print(audii_sa_4_cilindra["CO2 Emissions (g/km)"].mean())