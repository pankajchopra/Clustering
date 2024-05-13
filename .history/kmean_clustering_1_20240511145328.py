import numpy as np

# Generate random data points
data = np.random.rand(100, 2)

# Initialize KMeans with 3 clusters
k = 3
centroids = data[np.random.choice(len(data), k, replace=False)]
print("Initial centroids:")
print(centroids)

# Number of iterations
max_iter = 10

# Iterative optimization
for i in range(max_iter):
    # Assign each data point to the nearest centroid
    distances = np.sqrt(((data - centroids[:, np.newaxis])**2).sum(axis=2))
    labels = np.argmin(distances, axis=0)

    # Update centroids
    new_centroids = np.array([data[labels == j].mean(axis=0) for j in range(k)])

    print(f"Iteration {i + 1}:")
    print("Updated centroids:")
    print(new_centroids)

    # Check for convergence
    if np.allclose(centroids, new_centroids):
        print("Convergence reached. Exiting loop.")
        break

    centroids = new_centroids
