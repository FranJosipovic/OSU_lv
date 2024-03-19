import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("data_C02_emission.csv")

by_fuel = data.groupby("Cylinders")["CO2 Emissions (g/km)"].mean()
print(by_fuel)
by_fuel.plot(kind='bar',title="Prosjecna emisija po broju cilindara",ylabel="CO2 emisija",xlabel="Broj cilindara")
plt.show()