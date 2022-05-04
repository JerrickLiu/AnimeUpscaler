import cairosvg
import sys
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import torch
import torch.nn as nn
import cv2
from functools import reduce

def load_segments(vector_file:str):
    """
    Loads a vector file and returns a list of segments, colors, and transforms. A curve is numpy array of 
    8 floats, representing the 4 coordinates for a cubic bezier curve.

    @param vector_file: The path to the vector file.
    @return Python List of numpy arrays, np.array of colors, np.array of transforms.
    """
    if not vector_file.endswith('.svg'):
        print('Not a svg file')
        return

    df = pd.read_xml(vector_file)
    segments = []
    colors = []
    transforms = []

    for i in range(len(df)):
        color = df.iloc[i]['fill']

        # split color into r, g, b from hex code
        color_processed = color[1:]
        color_processed = np.array([int(color_processed[i:i+2], 16) for i in range(0, len(color_processed), 2)]).astype(np.float)
        colors.append(color_processed)

        path = df.iloc[i]['d']
        curves = path.split('C')
        curves = ([(curves[i].strip().split(' ')) for i in range(len(curves))])
        curves = curves[1:] # remove the first element, which is the M0
        curves[-1] = curves[-1][:-1] # remove the last element, which is the Z

        # pruning out M and Z in the middle
        for j in range(len(curves)):
            p = curves[j]
            if len(p) > 6:
                curves[j] = p[:6]
        # convert all string to float()
        for k in range(len(curves)):
            for j in range(len(curves[k])):
                curves[k][j] = float(curves[k][j])

        # prepend previous curve's last point to the curve
        for j in range(len(curves)):
            if j == 0:
                curves[j] = [0, 0] + curves[j]
            else:
                curves[j] = curves[j-1][-2:] + curves[j]

        curves_np = np.array(curves)
        segments.append(curves_np)

        # get transform from form 'translate(x, y)' to [x, y]
        transform = df.iloc[i]['transform']
        transform = transform.split('(')[1].split(')')[0].split(',')
        transform = [float(transform[0]), float(transform[1])]
        transform = np.array(transform).astype(np.float)
        transforms.append(transform)

    return segments, colors, transforms


def post_process_svg_info(svg_info):
    """
    Pads the svg_info to be all of same length. Also converts each element to a tensor.

    Args:
        svg_info: a tuple of (segments, colors, transforms).
    Returns:
        A tuple of (segments, colors, transforms) that are padded to be all of same length and are tensors.
    """

    # Find max length of segments
    segments, colors, transforms = svg_info

    max_len = 0
    for segment in segments:
        if segment.shape[0] > max_len:
            max_len = segment.shape[0]

    # Pad segments to max length with -1    
    for i in range(len(segments)):
        segments[i] = np.pad(segments[i], ((0, max_len - segments[i].shape[0]), (0, 0)), constant_values=-1)
    
    segments = torch.Tensor(segments)
    colors = torch.Tensor(colors).unsqueeze(1).expand(-1, max_len, -1)
    transforms = torch.Tensor(transforms).unsqueeze(1).expand(-1, max_len, -1)

    return segments, colors, transforms


def get_factors(n):    
    return set(reduce(list.__add__, 
                ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))

def k_box_clustering(segments, translations, k, return_seg_centroids=True):
    """
    Clusters the segments into k-boxes.

    Args:
        k: Number of clusters.
        svg_info: a tuple of (segments, colors, transforms)
        image: Pillow image of the original image.

    Returns:
        A list of k-boxes (each is a list of indices of segments).

    """

    # TODO: Don't hardcode this
    width, height = 426, 240

    aspect_ratio = width / height
    factors = list(get_factors(k))

    # Find two factors of k that are closest to the aspect ratio
    min_factors = []
    min_aspect_ratio = 9999
    for i in range(len(factors)):
        for j in range(i, len(factors)):
            if abs(factors[i] / factors[j] - aspect_ratio) < min_aspect_ratio and factors[i] * factors[j] == k:
                min_aspect_ratio = abs(factors[i] / factors[j] - aspect_ratio)
                min_factors = [factors[i], factors[j]]


    # Divide the image into k-boxes
    k_boxes = []

    if width > height:
        min_factor_x = min_factors[1]
        min_factor_y = min_factors[0]
        x_ratio = width / min_factors[1]
        y_ratio = height / min_factors[0]

    else:
        min_factor_x = min_factors[0]
        min_factor_y = min_factors[1]
        x_ratio = width / min_factors[0]
        y_ratio = height / min_factors[1]

    for i in range(min_factor_x):
        for j in range(min_factor_y):
            # Get the bounding box of the k-box
            x_start = x_ratio * i
            x_end = x_ratio * (i + 1)
            y_start = y_ratio * j
            y_end = y_ratio * (j + 1)

            box = (x_start, y_start, x_end, y_end)
            k_boxes.append(box)

    segment_centroids = [weighted_centroid(segments[i], translations[i]) for i in range(len(segments))]
    segment_centroids = np.array(segment_centroids)

    print(segment_centroids.shape)

    # Assign segments to k-boxes
    # If segment centroid is within the bounding box, assign it to the k-box

    clusters = [[] for i in range(k)]

    for i in range(len(segment_centroids)):
        segment_centroid = segment_centroids[i]
        for j in range(len(k_boxes)):
            box = k_boxes[j]

            # Check if segment centroid is within the bounding box
            if segment_centroid[0] >= box[0] and segment_centroid[0] <= box[2] and segment_centroid[1] >= box[1] and segment_centroid[1] <= box[3]:
                clusters[j].append(i)
                break

    if return_seg_centroids:
        return clusters, segment_centroids
    return clusters

def kmeans_centroids(segments, translations, color, k, return_seg_centroids=True):
    """
    K-means clustering of the centroids of the segments
    
    @param segments: list of numpy arrays of shape (n_i, 8)
    @param translations: numpy array (n, 2). Holds the translations of the segments
    @param color: color of the segments (n, 3)
    @param k: number of clusters
    @return: list of length k containing the indices of components in each cluster
    """
    s_centroids = [weighted_centroid(segments[i], translations[i]) for i in range(len(segments))]
    s_centroids = np.array(s_centroids) # (#segments, 2)
    # print(s_centroids.shape)
    # print(segment_centroids)

    # append color to segment_centroids
    segment_centroids = np.concatenate((s_centroids, np.array(color)/20), axis=1)
    mins = np.min(segment_centroids, axis=0)
    maxes = np.max(segment_centroids, axis=0)
    centroids = np.random.rand(k, 5)
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
                centroids[j] = np.random.rand(5)
                # centroids[j, 0] = centroids[j, 0] * (x_max - x_min) + x_min
                # centroids[j, 1] = centroids[j, 1] * (y_max - y_min) + y_min
                centroids[j] = centroids[j] * (maxes - mins) + mins
                continue
            centroids[j] = segment_centroids[clusters == j].mean(axis=0)

    sorted_clusters = [np.where(clusters == i)[0] for i in range(k)]
    if return_seg_centroids:
        return sorted_clusters, s_centroids
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
