from mpi4py import MPI
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

data_loading_start = time.time()

if rank == 0:
    dataset = pd.read_csv('data.csv', header=None)
    all_points = dataset.to_numpy()
    partition_size = len(all_points) // size
else:
    all_points = None
    partition_size = None

partition_size = comm.bcast(partition_size, root=0)
local_points = np.empty((partition_size, 2))
comm.Scatter(all_points, local_points, root=0)

data_loading_end = time.time()

if rank == 0:
    print(f"Time taken for data scattering: {data_loading_end - data_loading_start:.2f} seconds")

num_clusters = 5
iterations = 100

if rank == 0:
    initial_indices = np.random.choice(all_points.shape[0], num_clusters, replace=False)
    cluster_centers = all_points[initial_indices]
else:
    cluster_centers = np.empty((num_clusters, 2))

kmeans_start = time.time()

for _ in range(iterations):

    cluster_centers = comm.bcast(cluster_centers, root=0)
    local_sum = np.zeros((num_clusters, 2))
    local_count = np.zeros(num_clusters)
    
    for point in local_points:
        idx = np.argmin(np.linalg.norm(point - cluster_centers, axis=1))
        local_sum[idx] += point
        local_count[idx] += 1

    global_sums = comm.reduce(local_sum, op=MPI.SUM, root=0)
    global_counts = comm.reduce(local_count, op=MPI.SUM, root=0)

    if rank == 0:
        cluster_centers = [global_sums[i] / global_counts[i] if global_counts[i] > 0 else cluster_centers[i] for i in range(num_clusters)]

kmeans_end = time.time()

if rank == 0:
    closest = np.array([np.argmin(np.linalg.norm(point - cluster_centers, axis=1)) for point in all_points])

    fig, ax = plt.subplots(figsize=(10, 7))
    palette = ['r', 'g', 'b', 'y', 'c', 'm']

    for idx in range(num_clusters):
        ax.scatter(all_points[closest == idx].T[0], all_points[closest == idx].T[1], c=palette[idx], alpha=0.5)
        ax.scatter(cluster_centers[idx][0], cluster_centers[idx][1], color=palette[idx], marker='x')

    fig.savefig('kmeans_parallel_2n2c.png')

    print(f"Time taken for k-means clustering: {kmeans_end - kmeans_start:.2f} seconds")
    print(f"Total Time taken: {kmeans_end - data_loading_start:.2f} seconds")

