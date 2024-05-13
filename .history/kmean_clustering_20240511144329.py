import numpy as np
import pandas as pd 
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


# Generate random data points
data = np.random.rand(100, 2)

# Step 2: Initialize KMeans with 3 clusters
kmeans = KMeans(n_clusters=5)

# Step 3: Fit the data to the KMeans model
kmeans.fit(data)

# Step 4: Get cluster centroids and labels
centroids = kmeans.cluster_centers_
labels = kmeans.labels_

# Step 5: Refine clusters by finding the mean of each cluster
refined_centroids = []
for cluster_label in range(kmeans.n_clusters):
    # Find data points belonging to the current cluster
    cluster_data = data[labels == cluster_label]
    # Calculate the mean of the data points in the cluster
    cluster_mean = np.mean(cluster_data, axis=0)
    # Append the mean to the refined centroids list
    refined_centroids.append(cluster_mean)

# Print the refined centroids
print("Refined Centroids:")
for idx, centroid in enumerate(refined_centroids):
    print(f"Cluster {idx + 1}: {centroid}")

# Plot data points
plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis')

# Plot cluster centroids
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=200)
plt.title('KMeans Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
