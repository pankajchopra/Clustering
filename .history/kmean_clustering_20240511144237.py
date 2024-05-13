import numpy as np
import pandas as pd 
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


# Generate random data points
data = np.random.rand(100, 2)

# Initialize KMeans with 3 clusters
kmeans = KMeans(n_clusters=5)

# Fit the data to the KMeans model
kmeans.fit(data)

# Get cluster centroids and labels
centroids = kmeans.cluster_centers_
labels = kmeans.labels_



# Plot data points
plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis')

# Plot cluster centroids
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=200)
plt.title('KMeans Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
