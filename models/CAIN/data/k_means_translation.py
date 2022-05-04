import numpy as np
import scipy as sp

def kmeans_centroids(segments, translations, color, k, return_seg_centroids=True):
    """
    K-means clustering of the centroids of the segments
    
    @param segments: list of numpy arrays of shape (n_i, 8)
    @param translations: numpy array (n, 2). Holds the translations of the segments
    @param color: numpy array (n, 3). Holds the colors of the segments
    @param k: number of clusters
    @return: list of length k containing the indices of components in each cluster
    """
    segment_centroids = [weighted_centroid(segments[i], translations[i]) for i in range(len(segments))]
    segment_centroids = np.array(segment_centroids) # (#segments, 2)
    print(segment_centroids.shape)
    # print(segment_centroids)

    # append color to segment_centroids
    segment_centroids = np.concatenate((segment_centroids, color), axis=1)
    mins = np.min(segment_centroids, axis=0)
    maxes = np.max(segment_centroids, axis=0)

    x_max = np.max(segment_centroids[:, 0])
    x_min = np.min(segment_centroids[:, 0])
    y_max = np.max(segment_centroids[:, 1])
    y_min = np.min(segment_centroids[:, 1])

    centroids = np.random.rand(k, 5)
    # centroids[:, 0] = centroids[:, 0] * (x_max - x_min) + x_min
    # centroids[:, 1] = centroids[:, 1] * (y_max - y_min) + y_min
    centroids = centroids * (maxes - mins) + mins

    clusters = np.zeros(segment_centroids.shape[0], dtype=np.int)
    for i in range(100):
        # assign each point to the nearest centroid
        for j in range(segment_centroids.shape[0]):
            clusters[j] = np.argmin(np.linalg.norm(segment_centroids[j] - centroids, axis=1))
        # print(clusters)
        # update the centroids
        for j in range(k):
            if np.sum(clusters == j) == 0:
                centroids[j] = np.random.rand(2)
                centroids[j, 0] = centroids[j, 0] * (x_max - x_min) + x_min
                centroids[j, 1] = centroids[j, 1] * (y_max - y_min) + y_min
                continue
            centroids[j] = segment_centroids[clusters == j].mean(axis=0)

    sorted_clusters = [np.where(clusters == i)[0] for i in range(k)]
    if return_seg_centroids:
        return sorted_clusters, segment_centroids
    return sorted_clusters
    
def weighted_centroid(segment, translation):
    """
    Calculates a weighted centroid of a segment.
    Weighting is done by looking at length of line segments for each point.

    @params segment: numpy array of shape (n, 8)
    @param translation: translation of the segment
    @return: numpy array of shape (2)
    """
    centroid = np.zeros(2)
    sum_weights = 0
    for i in range(segment.shape[0]):
        # print(segment[i, 0:2], segment[i, 6:8])
        # weight = np.linalg.norm(segment[i][0] - segment[i][-1])
        weight = np.linalg.norm(segment[i, 0:2] - segment[i, 6:8])
        # weight = 1
        centroid += (segment[i, 0:2] + segment[i, 6:8]) * weight
        # print(centroid)
        # print(weight)
        sum_weights += weight * 2
    centroid /= sum_weights
    return centroid + translation
