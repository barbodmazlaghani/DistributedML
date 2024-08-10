import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time

start_time = time.time()

dataset = pd.read_csv('data.csv', header=None)
all_points = dataset.to_numpy()

num_clusters = 5
iterations = 100

initial_indices = np.random.choice(all_points.shape[0], num_clusters, replace=False)
cluster_centers = all_points[initial_indices]

for iteration in range(iterations):
    distance = np.linalg.norm(all_points[:, np.newaxis] - cluster_centers, axis=2)
    closest = np.argmin(distance, axis=1)
    cluster_centers = [all_points[closest == cluster_idx].mean(axis=0) for cluster_idx in range(num_clusters)]

fig, ax = plt.subplots(figsize=(10, 7))
palette = ['r', 'g', 'b', 'y', 'c', 'm']

for idx in range(num_clusters):
    ax.scatter(all_points[closest == idx].T[0], all_points[closest == idx].T[1], c=palette[idx], alpha=0.5)
    ax.scatter(cluster_centers[idx][0], cluster_centers[idx][1], color=palette[idx], marker='x')

fig.savefig('kmeans_result_serial.png')

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Time taken: {elapsed_time:.2f} seconds")

