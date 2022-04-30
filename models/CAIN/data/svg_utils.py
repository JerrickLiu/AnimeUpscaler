import cairosvg
import sys
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import torch
import torch.nn as nn

# file = sys.argv[1]

# print('file', file)
# dir = sys.argv[1]

# # with open(file, 'rb') as f:
    # # data = f.read()

# # print(data)

# for file in tqdm(os.listdir(dir)):
    # if file.endswith(".svg"):
        # # print(dir + file)
        # df = pd.read_xml(dir + file)
        # # print(df.head())
        # z_all = []

        # for i in range(len(df)):
            # print(df.iloc[i]['transform'])
            # path = df.iloc[i]['d']
            # curves = path.split('C')
            # per_curve = ([(curves[i].strip().split(' ')) for i in range(len(curves))])
            # # if per_curve[0][0] != 'M0':
                # # print("AHHHH")
            # per_curve = per_curve[1:]
            # per_curve[-1] = per_curve[-1][:-1]
            # zs = 0
            # for i in range(len(per_curve)):
                # p = per_curve[i]
                # if len(p) > 6:
                    # zs += 1
                    # per_curve[i] = p[:6]
                # # for n in p:
                    # # if n.isalpha():
                        # # print(per_curve)
                        # # print(n)
                        # # zs += 1
                        # # pass
            # # z_all.append(zs)
            # per_curve = np.array(per_curve)
            # # print(per_curve.shape)
            # # print(per_curve / len(curves))
#         # print(np.unique(np.array(z_all), return_counts=True))


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
        for i in range(len(curves)):
            p = curves[i]
            if len(p) > 6:
                curves[i] = p[:6]
        # convert all string to float()
        for i in range(len(curves)):
            for j in range(len(curves[i])):
                curves[i][j] = float(curves[i][j])

        # prepend previous curve's last point to the curve
        for i in range(len(curves)):
            if i == 0:
                curves[i] = [0, 0] + curves[i]
            else:
                curves[i] = curves[i-1][-2:] + curves[i]

        curves_np = np.array(curves)
        segments.append(curves_np)

        # get transform from form 'translate(x, y)' to [x, y]
        transform = df.iloc[i]['transform']
        transform = transform.split('(')[1].split(')')[0].split(',')
        transform = [float(transform[0]), float(transform[1])]
        transform = np.array(transform).astype(np.float)
        transforms.append(transform)

    return segments, colors, transforms


def pad_segments(segments):
    """
    Pads the segments to be of the same length.

    Args:
        segments: List of numpy arrays of curves.
    Returns:
        List of numpy arrays of padded curves.
    """

    # Find max length of segments
    max_len = 0
    for segment in segments:
        if segment.shape[0] > max_len:
            max_len = segment.shape[0]

    # Pad segments to max length with -1    
    for i in range(len(segments)):
        segments[i] = np.pad(segments[i], ((0, max_len - segments[i].shape[0]), (0, 0)), 'constant', constant_values=-1)
    return segments