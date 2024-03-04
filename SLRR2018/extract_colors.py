# code=utf-8
# Final funciton extract colors
import numpy as np
import cupy as cp
from scipy.cluster.vq import kmeans

def kmeans_cupy(obs, k_or_guess, iter=20, thresh=1e-5):
    # Determine the data type of observation array
    dtype = obs.dtype
    
    # Initialize centroids
    if isinstance(k_or_guess, int):
        # Randomly choose k observations as initial centroids
        centroids = obs[cp.random.choice(obs.shape[0], k_or_guess, replace=False)]
    else:
        # Use the given initial centroids
        centroids = k_or_guess

    for _ in range(iter):
        # Assign each observation to the closest centroid
        distances = cp.linalg.norm(obs[:, cp.newaxis] - centroids, axis=2)
        labels = cp.argmin(distances, axis=1)
        
        # Compute new centroids as the mean of all observations assigned to each centroid
        new_centroids = cp.zeros_like(centroids)
        for i in range(centroids.shape[0]):
            cluster_points = obs[labels == i]
            # If a cluster is empty, reinitialize its centroid randomly
            if cluster_points.size == 0:
                new_centroids[i] = obs[cp.random.choice(obs.shape[0], 1)]
            else:
                new_centroids[i] = cluster_points.mean(axis=0)
        
        # Check for convergence (if centroids do not change)
        if cp.linalg.norm(new_centroids - centroids) < thresh:
            break
        
        centroids = new_centroids
    
    # Return the centroids and the distortion (sum of squared distances to closest centroid)
    distortion = cp.sum((obs - centroids[labels])**2)
    
    return centroids, distortion

def extract_colors(original_img, K=50, iter=200):
    """extract the color dictionary by using Kmeans
    Params:
        original_img: icput image(H,W,C)
        K: size of color dictionary. default:50
        iter: Kmeans iteration
    Return:
        colors: shape(3, K) range(0, 1)

    """
    # BGR
    img = original_img.copy().astype(cp.float64).reshape(-1, 3)  # N, 3
    h, w, c = original_img.shape
    B = R = 0
    G = T = 1
    R = P = 2
    eps = 1e-8
    # convert to spherical coordinates
    img_rtp = cp.empty_like(img)  # N, 3
    x = img[:, B]
    y = img[:, G]
    z = img[:, R]
    img_rtp[:, R] = cp.sqrt(cp.square(x) + cp.square(y) + cp.square(z))
    img_rtp[:, T] = cp.arccos(z / (img_rtp[:, R] + eps))
    img_rtp[:, P] = cp.arctan(y / (x + eps))
    # cluster on T, P
    # K = 10
    # cluster_tp, _ = kmeans(img_rtp[:, (T, P)].get(), K, iter=iter, thresh=1e-05, check_finite=True)
    try:
        cluster_tp, _ = kmeans_cupy(img_rtp[:, (T, P)], K, iter)
        img_tp = img_rtp[:, (T, P)][:, cp.newaxis, :]  # N, 1, 2
        img_r = img_rtp[:, R]  # N,
        distance = cp.linalg.norm(cluster_tp - img_tp, axis=2)  # , N,K
        min_index = cp.argmin(distance, axis=1).astype(cp.uint32)  # ,N
        K = min(K, cluster_tp.shape[0])
        cluster_r = cp.ones((K)).astype(cp.float64)
        for i in range(K):
            cluster_r[i] = (img_r[min_index == i].min())
        cluster_r = cluster_r[..., cp.newaxis]
        cluster_rtp = cp.concatenate([cluster_r, cluster_tp], axis=1)
    except:
        cluster_tp, _ = kmeans(img_rtp[:, (T, P)].get(), K, iter=iter, thresh=1e-05, check_finite=True)
        cluster_tp = cp.asarray(cluster_tp)
        img_tp = img_rtp[:, (T, P)][:, cp.newaxis, :]  # N, 1, 2
        img_r = img_rtp[:, R]  # N,
        distance = cp.linalg.norm(cluster_tp - img_tp, axis=2)  # , N,K
        min_index = cp.argmin(distance, axis=1).astype(cp.uint32)  # ,N
        K = min(K, cluster_tp.shape[0])
        cluster_r = cp.ones((K)).astype(cp.float64)
        for i in range(K):
            cluster_r[i] = (img_r[min_index == i].min())
        cluster_r = cluster_r[..., cp.newaxis]
        cluster_rtp = cp.concatenate([cluster_r, cluster_tp], axis=1)

    # convert to RGB
    colors = cp.empty_like(cluster_rtp)
    colors[:, B] = cluster_rtp[:, R] * cp.sin(cluster_rtp[:, T]) * cp.cos(cluster_rtp[:, P])
    colors[:, G] = cluster_rtp[:, R] * cp.sin(cluster_rtp[:, T]) * cp.sin(cluster_rtp[:, P])
    colors[:, R] = cluster_rtp[:, R] * cp.cos(cluster_rtp[:, T])
    colors = cp.clip(colors, a_min=0, a_max=1)
    return colors.transpose([1, 0])

