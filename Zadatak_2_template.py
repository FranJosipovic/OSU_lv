import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as Image
from sklearn.cluster import KMeans

##nadi najbolje clustere
for i in range(1,7):
    img = Image.imread(f"imgs\\test_{i}.jpg")

    # pretvori vrijednosti elemenata slike u raspon 0 do 1
    img = img.astype(np.float64) / 255

    # transfromiraj sliku u 2D numpy polje (jedan red su RGB komponente elementa slike)
    w,h,d = img.shape
    img_array = np.reshape(img, (w*h, d))

    wcss=[]

    for i in range(1, 11):
        km = KMeans(n_clusters=i, n_init=5)
        km.fit(img_array)
        wcss.append(km.inertia_)

    plt.figure()
    plt.plot(range(1, 11), wcss, marker='o')
    plt.title('Elbow Method')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS')
    plt.show()

num_clusters = [2, 2, 2, 2, 4, 2]

for i, clusters in enumerate(num_clusters, start=1):
    # ucitaj sliku
    img = Image.imread(f"imgs\\test_{i}.jpg")

    # prikazi originalnu sliku
    plt.figure()
    plt.title("Originalna slika")
    plt.imshow(img)
    plt.tight_layout()
    plt.show()

    # pretvori vrijednosti elemenata slike u raspon 0 do 1
    img = img.astype(np.float64) / 255

    # transfromiraj sliku u 2D numpy polje (jedan red su RGB komponente elementa slike)
    w,h,d = img.shape
    img_array = np.reshape(img, (w*h, d))

    unique_colors = np.unique(img_array,axis=0)
    print(len(unique_colors))

    img_array_aprox = img_array.copy()
    kmeans = KMeans(n_clusters=clusters, random_state=42)
    kmeans.fit(img_array)
    new_colors = kmeans.cluster_centers_
    labels = kmeans.labels_
    # Replace each pixel with its nearest centroid
    img_array_aprox = new_colors[labels].reshape(w, h, d)

    # Display the image with reduced colors
    plt.figure()
    plt.title("Image with 5 colors")
    plt.imshow(img_array_aprox)
    plt.tight_layout()
    plt.show()

    for cluster_label in range(clusters):
        # Create a binary mask for pixels belonging to the current cluster
        binary_mask = labels.reshape(w, h) == cluster_label

        # Display the binary mask
        plt.figure()
        plt.title(f"Binary Image for Cluster {cluster_label + 1}")
        plt.imshow(binary_mask, cmap='gray')
        plt.tight_layout()
        plt.show()
