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


class AnimationDataset(Dataset):
    """ Animation dataset """

    def __init__(self, csv_file, root_dir, transform=anime_augs, triplet_fps=12):
        """
        Args:
            csv_file (string): Path to the csv file with annotations of scenes.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.animation_csv = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.frame_paths = self.make_dataset()
        self.triplet_fps = triplet_fps

    def make_dataset(self) -> list:
        """
        Returns: a list of lists, where each list is a list of paths to frames for a clip
        """

        frame_paths = {}

        for index, folder in enumerate(os.listdir (self.root_dir)):
            # print(folder)
            video_path = os.path.join(self.root_dir, folder)

            if not os.path.isdir(video_path):
                continue

            # Create a key as the foolder name + "-extracted_frames"
            frame_paths[folder] = []

            for frame in os.listdir(video_path):
                # print(frame)
                # print(frame[6:-4])
                frame_path = os.path.join(video_path, frame)
                frame_paths[folder].append(frame_path)

        return frame_paths


    def load_image(self, image_path):
        """
        Loads an image from a given path
        """
        img = Image.open(image_path)
        img = img.convert('RGB')
        return img

    def __len__(self):
        """ 
        Returns the length of the dataset.

        TODO: This is not the correct way to do this. A better way is to use the actual number of trips in the dataset in a given scene.
        """
        return len(self.animation_csv)
    
    def __getitem__(self, idx):

        # Get the FPS of the scene
        scene_frame_length = self.animation_csv.iloc[idx]['Length (frames)']
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



        # start_frame = self.animation_csv.iloc[idx]['Start Frame']
        # end_frame = self.animation_csv.iloc[idx]['End Frame'] - 1
        # middle_frame = (start_frame + end_frame) // 2

        triplet_range = [start_frame, middle_frame, end_frame]

        triplet = []

        folder_name = self.animation_csv.iloc[idx]['file_name'] + "-extracted_frames"

        # Get the time delta between the start and end frame
        time_delta = self.animation_csv.iloc[idx]['Length (seconds)']
        

        frame_size = (1, 3, 240, 424)
        for frame_index in triplet_range:
            try: 
                image = self.load_image(self.frame_paths[folder_name][int(frame_index)])
            except IndexError:
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
        triplet = torch.cat(triplet, dim=0)
 
        if self.transform:
            triplet = self.transform(triplet)
            # for i in range(3):
                # plt.imshow(triplet[i].cpu().numpy().transpose(1,2,0))
                # plt.show()
        
        return triplet, time_delta


# data_root = "/home/jerrick/disk_mount/anime_data/small_extracted_frames/extracted_frames"
# csv = "/home/jerrick/disk_mount/anime_data/small_extracted_frames/metadata/all_scenes.csv"
# anime_trips = AnimationDataset(csv, data_root)

# for i in range(5):
    # print(anime_trips[i])

def get_loader(mode, csv, root, batch_size, shuffle, num_workers, test_mode=None):
    dataset = AnimationDataset(csv, root)
    return DataLoader(dataset, batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
