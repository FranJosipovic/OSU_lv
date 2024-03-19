import pandas as pd

data = pd.read_csv("data_C02_emission.csv")

print(len(data))
print(data.info())

print(data.isnull()) #nema

if(data.duplicated().sum() > 0):
    data.drop_duplicates()
    data.reset_index(drop=True)
    
print(data.duplicated()) #nema

#pretvaranje iz object u category
print(data.info())
object_columns = data.select_dtypes(include=['object'])
data[object_columns.columns] = object_columns.astype('category')
print(data.info())