import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("data_C02_emission.csv")

by_fuel = data.groupby("Fuel Type").size()
by_fuel.plot(kind='bar',title="Broj vozila po tipu goriva",ylabel="Broj vozila",xlabel="Tip Goriva")
plt.show()