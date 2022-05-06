"""
Dataset for Anime Data with corresponding vectorized frames
"""
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import torchvision.transforms as TF
import random
import sys
from .svg_utils import load_segments, post_process_svg_info
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn

def anime_augs(triplet):
    i, j, h, w = transforms.RandomResizedCrop.get_params(triplet[0], scale=(0.5, 1.0), ratio=(.75, 1.33))
    size = (triplet[0].shape[1], triplet[0].shape[2])
    if random.random() > .5:
        triplet = TF.functional.hflip(triplet)

    MAX_ANGLE = 15
    angle = (random.random() - .5) * 2 * MAX_ANGLE
    triplet = TF.functional.rotate(triplet, angle)
    triplet = TF.functional.resized_crop(triplet, i, j, h, w, size)
    return triplet


class AnimationVectorizedDataset(Dataset):
    """ Animation Vectorized dataset """

    def __init__(self, csv_file, root_dir, vectorized_dir, transform=None, triplet_fps=4):
        """
        Args:
            csv_file (string): Path to the csv file with annotations of scenes.
            root_dir (string): Directory with all the images.
            vectorized_dir (string): Directory where the vectorized frames are stored. Assumed to have the same structure as root_dir.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.animation_csv = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.vectorized_dir = vectorized_dir
        self.transform = transform
        self.frame_paths, self.vectorized_paths = self.make_dataset()
        self.triplet_fps = triplet_fps

    def make_dataset(self):
        """
        Returns: a list of lists, where each list is a list of paths to frames for a clip for orignal images and vectorized images.
        
        Each SVG triplet entry will contain the following:
            - Segments of the SVG
            - Color of the SVG
            - Transforms of the SVG
        """

        frame_paths = {}
        vectorized_paths = {}

        for index, folder in enumerate(os.listdir(self.root_dir)):
            # print(folder)
            video_path = os.path.join(self.root_dir, folder)
            vector_path = os.path.join(self.vectorized_dir, folder)

            if not os.path.isdir(video_path):
                continue

            # Create a key as the folder name + "-extracted_frames"
            frame_paths[folder] = []
            vectorized_paths[folder] = []

            for frame in sorted(os.listdir(video_path)):
                frame_path = os.path.join(video_path, frame)
                frame_paths[folder].append(frame_path)
            
            for frame in sorted(os.listdir(vector_path)):
                frame_vector_path = os.path.join(vector_path, frame)
                vectorized_paths[folder].append(frame_vector_path)
            # print(frame_paths[folder][:2],'\n', vectorized_paths[folder][:2], '\n\n\n\n')

        return frame_paths, vectorized_paths

    def __len__(self):
        """ 
        Returns the length of the dataset.

        TODO: This is not the correct way to do this. A better way is to use the actual number of trips in the dataset in a given scene.
        """
        return len(self.animation_csv)

    def load_image(self, image_path):
        """
        Loads an image from a given path
        """
        img = Image.open(image_path)
        img = img.convert('RGB')

        return img
    
    def __getitem__(self, idx):

        # Get the FPS of the scene
        scene_frame_length = self.animation_csv.iloc[idx]['Length (frames)']
        # print("SCENE LENGTH: ", scene_frame_length)
        scene_frame_seconds = self.animation_csv.iloc[idx]['Length (seconds)']
        # print(scene_frame_seconds)
        # print(self.animation_csv.iloc[idx])
        scene_fps = scene_frame_length // scene_frame_seconds

        # Get the start and end frame of the scene
        start_frame = self.animation_csv.iloc[idx]['Start Frame']
        end_frame = self.animation_csv.iloc[idx]['End Frame'] - 1

        interpolation_factor = scene_fps // self.triplet_fps

        if interpolation_factor * 2 > scene_frame_length:
            middle_frame = (start_frame + end_frame) // 2
        
        else:
            upper_bound_start_frame = scene_frame_length - interpolation_factor * 2

            start_frame = random.randint(start_frame, start_frame + upper_bound_start_frame)

            middle_frame = start_frame + interpolation_factor
            end_frame = min(end_frame, middle_frame + interpolation_factor)

        triplet_range = [int(start_frame), int(middle_frame), int(end_frame)]

        triplet = []
        svg_triplet = []
        svg_files = []
        svg_triplet_num_segments = []
        svg_prepadded_info = []

        folder_name = self.animation_csv.iloc[idx]['file_name'] + "-extracted_frames"

        # Get the time delta between the start and end frame
        time_delta = self.animation_csv.iloc[idx]['Length (seconds)']
        
        frame_size = (1, 3, 240, 424)
        # print(triplet_range)
        for frame_index in triplet_range:
            try: 
                # Rasterized image
                image = self.load_image(self.frame_paths[folder_name][frame_index])

                # SVG path
                svg_file = self.vectorized_paths[folder_name][frame_index]
                # render_svg(svg_file, show=True)

                # SVG info is a list of (segments, color, transforms)
                prepad_svg_info = list(load_segments(svg_file))

                # Get number of segments of SVG to be added
                num_segments = len(prepad_svg_info[0])

                # Pad the svg info and convert it to tensors
                svg_info = post_process_svg_info(prepad_svg_info)

                segments, colors, transforms = svg_info

                # Concatenate segments, colors, and transforms into a single tensor
                svg_tensor = torch.cat((segments, colors, transforms), dim=2)

                # Swap the first and last channels for padding
                svg_tensor = svg_tensor.permute(2, 1, 0)

                
            except IndexError:
                print("INDEX ERROR")
                print(folder_name, frame_index)
                print(start_frame, middle_frame, end_frame)
                exit()
            image = TF.ToTensor()(image)
            image = TF.functional.crop(image, 0, 0, frame_size[2], frame_size[3]) # hard coded an arbitrary 240p size
       
            # print(image.shape)
            if (image.shape[-1] != 424):
                print("Goddamn")
                image = np.zeros((image.shape[0], 3, 240, 424))

            triplet.append(image.reshape(frame_size))
            svg_triplet.append(svg_tensor)
            svg_triplet_num_segments.append(num_segments)
            svg_files.append(svg_file)
            svg_prepadded_info.append(prepad_svg_info)

        triplet = torch.cat(triplet, dim=0)
        svg_triplet = pad(svg_triplet, -1, [1, 2])
        svg_triplet = torch.stack(svg_triplet)

        if self.transform:
            triplet = self.transform(triplet)
            # for i in range(3):
                # plt.imshow(triplet[i].cpu().numpy().transpose(1,2,0))
                # plt.show()

        
        # print(svg_files)
#         print(triplet.shape)
        # print(svg_triplet.shape)
        # print(len(svg_triplet_num_segments))
        # print(len(svg_files))
#         print(time_delta)
        return triplet, svg_triplet, svg_triplet_num_segments, svg_files, time_delta, svg_prepadded_info

def pad(tensors, pad_value, dimensions):
    """
    Pads a list of tensors with a given value in the given dimensions. Currently only supports 2D padding
    """

    d1, d2 = dimensions

    # print("BEFORE: ")
    # for i in range(len(tensors)):
    #     print(tensors[i].shape)

    max_d1_length = tensors[0].shape[d1]
    for i in range(1, len(tensors)):
        if tensors[i].shape[d1] > max_d1_length:
            max_d1_length = tensors[i].shape[d1]

    max_d2_length = tensors[0].shape[d2]
    for i in range(1, len(tensors)):
        if tensors[i].shape[d2] > max_d2_length:
            max_d2_length = tensors[i].shape[d2]
    

    # Pad each tensor of the triplet to [max_length, max_segment_length, 13]
    for i in range(len(tensors)):
        pad_2d = (0, max_d2_length - tensors[i].shape[d2], 0, max_d1_length - tensors[i].shape[d1])
        tensors[i] = nn.functional.pad(tensors[i], pad_2d, "constant", pad_value)
    
    # print("AFTER: ")
    # for i in range(len(tensors)):
    #     print(tensors[i].shape)
    return tensors


def custom_collate(batch):
    # Get all svg tensors from first dim of the batch
    triplets = [item[0] for item in batch]
    svg_tensors = [item[1] for item in batch]
    num_segments = [item[2] for item in batch]
    svg_files = [item[3] for item in batch]
    time_deltas = [item[4] for item in batch]
    svg_prepadded_info = [item[5] for item in batch]

    svg_tensors = pad(svg_tensors, -1, [2, 3])

    triplets = torch.stack(triplets, dim=0)
    svg_tensors = torch.stack(svg_tensors, dim=0)
    # print("BEFORE: ", svg_tensors.shape)
    svg_tensors = svg_tensors.permute(0, 1, 4, 3, 2)

    return [triplets, svg_tensors, num_segments, svg_files, time_deltas, svg_prepadded_info]


def get_loader(mode, csv, root, vector_dir, batch_size, shuffle, num_workers, test_mode=None):
    dataset = AnimationVectorizedDataset(csv, root, vector_dir)
    return DataLoader(dataset, batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=custom_collate, pin_memory=True)

# data_root = "/media/sahil/DL_5TB/MachineLearning/anime_playlist_downloads/fight_scenes_extracted_frames/extracted_frames"

# csv = "/media/sahil/DL_5TB/MachineLearning/anime_playlist_downloads/fight_scenes_extracted_frames/metadata/all_scenes.csv"

# vectorized_dir = "/media/sahil/DL_5TB/MachineLearning/anime_playlist_downloads/fight_scenes_extracted_frames/extracted_frames_vectorized"

# loader = get_loader("train", csv, data_root, vectorized_dir, 2, False, 0)

# for i, (images, svg_info, num_segments, svg_files, time_delta, svg_prepadded_info) in enumerate(loader):
#     print(num_segments)
#     print(len(svg_prepadded_info[0][2]))
#     segments, colors, transforms = svg_prepadded_info[0]
#     print(len(segments))
#     print(svg_info.shape)
#     break

# anime_trips = AnimationVectorizedDataset(csv, data_root, vectorized_dir)
# stuff = anime_trips[0]
# print(svg_triplet[0].size())
# print(svg_triplet[1].size())
# print(svg_triplet[2].size())

