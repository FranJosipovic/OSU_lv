import numpy as np
import matplotlib.pyplot as plt

x = np.array([1, 2, 3, 3, 1])
y = np.array([1, 2, 2, 1, 1])

plt.figure()
plt.plot(
    x, y, color="blue", marker=".", linestyle="solid", linewidth="4", markersize="12"
)
plt.axis([0, 4, 0, 4])
plt.show()
