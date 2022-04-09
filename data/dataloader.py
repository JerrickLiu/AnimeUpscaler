import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import torchvision.transforms as TF

class AnimationDataset(Dataset):
    """ Animation dataset """

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations of scenes.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.animation_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.frame_paths = self.make_dataset()
    
    def make_dataset(self) -> list:
        """
        Returns: a list of lists, where each list is a list of paths to frames for a clip
        """

        frame_paths = []

        for index, folder in enumerate(os.listdir (self.root_dir)):
            video_path = os.path.join(self.root_dir, folder)

            if not os.path.isdir(video_path):
                continue

            frame_paths.append([])

            for frame in os.listdir(video_path):
                frame_path = os.path.join(video_path, frame)
                frame_paths[index].append(frame_path)
        
        return frame_paths


    def load_image(self, image_path):
        """
        Loads an image from a given path
        """
        img = Image.open(image_path)
        img = img.convert('RGB')
        return img

    def __len__(self):
        return sum([len(files) for r, d, files in os.walk(self.root_dir)])
    
    def __getitem__(self, idx):
        triplet_range = [0, 1, 2]

        triplet = []

        for frame_index in triplet_range:
            image = self.load_image(self.frame_paths[idx][frame_index])

            if self.transform:
                image = self.transform(image)
            
            triplet.append(image)
        
        return triplet