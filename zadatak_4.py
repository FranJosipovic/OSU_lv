import numpy as np
import matplotlib.pyplot as plt

jedinice = np.ones((50,50))
nulice = np.zeros((50,50))

gornji_dio = np.hstack((nulice,jedinice))
donji_dio = np.hstack((jedinice,nulice))

ukupno = np.vstack((gornji_dio,donji_dio))

plt.figure()
plt.imshow(ukupno,cmap='gray')
plt.show()