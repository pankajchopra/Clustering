import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN

# Generate sample data
X, y = make_blobs(n_samples=1000, centers=4, cluster_std=0.6, random_state=0)

# Fit the DBSCAN model
dbscan = DBSCAN(eps=0.5, min_samples=20)
clusters = dbscan.fit_predict(X)

# Plot the clustering results
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap='viridis')
plt.title('DBSCAN Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar()
plt.show()

# Print the number of clusters and noise points
unique_labels = set(clusters)
print(f"Number of clusters: {len(unique_labels) - (1 if -1 in unique_labels else 0)}")
print(f"Number of noise points: {list(clusters).count(-1)}")