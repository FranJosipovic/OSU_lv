import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("data_C02_emission.csv")

plt.figure()
data["CO2 Emissions (g/km)"].plot(kind='hist',bins=20)
plt.show()