import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("data_C02_emission.csv")

fuel_type_map_to_values = {"X":'red',"Z":'blue',"D":'yellow',"E":'orange',"N":'green'}

data.plot.scatter(
    x="Fuel Consumption City (L/100km)",
    y="CO2 Emissions (g/km)",
    c=data["Fuel Type"].map(fuel_type_map_to_values),
    s=50
)
plt.show()
