import numpy as np
import matplotlib.pyplot as plt

img = plt.imread("road.jpg")
img = img[:, :, 0].copy()
print(img.shape)
print(img.dtype)
plt.figure()

###posvijetljena
plt.subplot(2,2,1)
plt.title("Posvijetljena")
img_gray = img + 150
img_gray[img_gray > 255] = 255
img_gray[img_gray < 150] = 255
plt.imshow(img_gray, cmap="gray")

###druga cetvrtina
plt.subplot(2,2,2)
plt.title("Druga cetvrtina")
druga_cetvrtina = img[:,160:321]
plt.imshow(druga_cetvrtina, cmap="gray")

###zarotirana
plt.subplot(2,2,3)
plt.title("zarotirana")
rotirana = np.rot90(img,-1)
plt.imshow(rotirana, cmap="gray")

###zrcaljena
plt.subplot(2,2,4)
plt.title("zrcaljena")
mirrored = np.fliplr(img)
plt.imshow(mirrored, cmap="gray")

plt.subplots_adjust(hspace=0.5)
plt.show()