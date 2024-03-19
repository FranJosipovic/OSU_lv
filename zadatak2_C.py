import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("data_C02_emission.csv")

data.groupby("Fuel Type").boxplot(column=["Fuel Consumption Hwy (L/100km)"])
plt.show()