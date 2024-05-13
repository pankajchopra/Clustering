import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering

# Generate sample data
X, y = make_blobs(n_samples=200, centers=4, cluster_std=0.6, random_state=0)

# Calculate the linkage matrix
Z = linkage(X, method='ward')

# Plot the dendrogram
plt.figure(figsize=(10, 6))
dendrogram(Z, truncate_mode='level', p=3)
plt.xlabel("Sample Index")
plt.ylabel("Distance")
plt.title("Dendrogram")
plt.show()

# Fit the AgglomerativeClustering model
clustering = AgglomerativeClustering(n_clusters=4, linkage='ward')
cluster_labels = clustering.fit_predict(X)

# Plot the clustering results
plt.figure(figsize=(8, 6), facecolor="black", edgecolor=white)
plt.scatter(X[:, 0], X[:, 1], c=cluster_labels)
plt.title("Hierarchical Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()