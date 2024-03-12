import numpy as np
import matplotlib.pyplot as plt

file = np.loadtxt("data.csv", delimiter=",", dtype=str)
data = file[1:, :].astype(np.float16)
print(f"Mjerenje izvršeno nad {len(data)} osoba")

plt.figure()

###svi ljudi###
visine = data[:, 1]
tezine = data[:, 2]
plt.subplot(1, 2, 1)
plt.scatter(visine, tezine, marker=".")
plt.title("Odnos svih visina i tezina")
plt.xlabel("Visina(cm)")
plt.ylabel("Tezina(kg)")
# plt.show()

###svakih50###
visine_50 = data[::50, 1]
tezine_50 = data[::50, 2]
plt.subplot(1, 2, 2)
plt.scatter(visine_50, tezine_50, marker=".")
plt.title("Odnos svakih 50 visina i tezina")
plt.xlabel("Visina(cm)")
plt.ylabel("Tezina(kg)")

plt.subplots_adjust(wspace=0.5)

print(visine.max())
print(visine.min())
print(visine.mean())

muskarci = data[data[:, 0] == 1]
zene = data[data[:, 0] == 0]

muskarci_visine = muskarci[:, 1]
print("Muskarci:")
print(muskarci_visine.max())
print(muskarci_visine.min())
print(muskarci_visine.mean())

zene_visine = zene[:, 1]
print("Zene:")
print(zene_visine.max())
print(zene_visine.min())
print(zene_visine.mean())

plt.show()
