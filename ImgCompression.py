import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

img = plt.imread('tree.jpg')

height = img.shape[0]
width = img.shape[1]

img = img.reshape(height * width, 3)

kmeans = KMeans(n_clusters=6).fit(img)
labels = kmeans.predict(img)
clusters = kmeans.cluster_centers_

clone_img = np.zeros_like(img)

for i in range(len(clone_img)):
	clone_img[i] = clusters[labels[i]]

clone_img = clone_img.reshape(height, width, 3)

plt.imshow(clone_img)
plt.show()