# dataloader for multi frames (acceleration), modified from superslomo

import torch.utils.data as data
from PIL import Image
import os
import os.path
import random
import sys
import numpy as np
import torch
import cv2
import torch.nn.functional as F
import torchvision.transforms as TF
from .svg_utils import load_segments, post_process_svg_info
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


def _make_dataset(dir, vector_dir):
    framesPath = []
    svg_frames_path = []

    # Find and loop over all the clips in root `dir`.
    for index, folder in enumerate(sorted(os.listdir(dir))):
        clipsFolderPath = os.path.join(dir, folder)

        # Skip items which are not folders.
        if not (os.path.isdir(clipsFolderPath)):
            continue
        framesPath.append([])

        for image in sorted(os.listdir(clipsFolderPath)):
            # Add path to list.
            framesPath[index].append(os.path.join(clipsFolderPath, image))

        # Find and loop over all the clips in root `dir`.
    
    for index, folder in enumerate(sorted(os.listdir(vector_dir))):
        clipsFolderPath = os.path.join(vector_dir, folder)

        # Skip items which are not folders.
        if not (os.path.isdir(clipsFolderPath)):
            continue
        svg_frames_path.append([])

        for image in sorted(os.listdir(clipsFolderPath)):
            # Add path to list.
            svg_frames_path[index].append(os.path.join(clipsFolderPath, image))

    return framesPath, svg_frames_path



def _pil_loader(path, cropArea=None, resizeDim=None, frameFlip=0):
    with open(path, 'rb') as f:
        img = Image.open(f)
        # Resize image if specified.
        resized_img = img.resize(resizeDim, Image.ANTIALIAS) if (resizeDim != None) else img
        # cv2.imwrite(resize)
        # Crop image if crop area specified.
        if cropArea != None:
            cropped_img = resized_img.crop(cropArea)
        else:
            cropped_img = resized_img
        # Flip image horizontally if specified.
        flipped_img = cropped_img.transpose(Image.FLIP_LEFT_RIGHT) if frameFlip else cropped_img


        return flipped_img.convert('RGB')


    
    
class AniTriplet(data.Dataset):
    def __init__(self, root, svg_root, transform=None, resizeSize=(424, 240), randomCropSize=(352, 352), train=True, shift=0):
        # Populate the list with image paths for all the
        # frame in `root`.
        framesPath, vector_frames_path = _make_dataset(root, svg_root)
        # Raise error if no images found in root.
        if len(framesPath) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"))
                
        self.randomCropSize     = randomCropSize
        self.cropX0             = resizeSize[0] - randomCropSize[0]
        self.cropY0             = resizeSize[1] - randomCropSize[1]
        self.root               = root
        self.transform          = transform
        self.train              = train
        self.resizeSize         = resizeSize
        self.framesPath         = framesPath
        self.vectorized_paths   = vector_frames_path
        self.shift              = shift

    def __getitem__(self, index):
        sample = []
        svg_triplet = []
        svg_files = []
        svg_triplet_num_segments = []
        svg_prepadded_info = []
        inter = None
        cropArea = []
        shifts = []
        
        # if (self.train):
        #     ### Data Augmentation ###
        #     # To select random 9 frames from 12 frames in a clip
        #     firstFrame = 0
        #     # Apply random crop on the 9 input frames

            
        #     shiftX = random.randint(0, self.shift)//2 * 2
        #     shiftY = random.randint(0, self.shift)//2 * 2
        #     shiftX = shiftX * -1 if random.randint(0, 1) > 0 else shiftX
        #     shiftY = shiftY * -1 if random.randint(0, 1) > 0 else shiftY
    

        #     cropX0 = random.randint(max(0, -shiftX), min(self.cropX0 - shiftX, self.cropX0))
        #     cropY0 = random.randint(max(0, -shiftY), min(self.cropY0, self.cropY0 - shiftY))
            

        #     cropArea.append((cropX0, cropY0, cropX0 + self.randomCropSize[0], cropY0 + self.randomCropSize[1]))
        #     cropArea.append((cropX0 + shiftX//2, cropY0 + shiftY//2, cropX0 + shiftX//2 + self.randomCropSize[0], cropY0 + shiftY//2 + self.randomCropSize[1]))
        #     cropArea.append((cropX0 + shiftX, cropY0 + shiftY, cropX0 + shiftX + self.randomCropSize[0], cropY0 + shiftY + self.randomCropSize[1]))
            
        #     shifts.append((shiftX, shiftY))
        #     shifts.append((-shiftX, -shiftY))

        #     inter = 1
        #     reverse = random.randint(0, 1)
        #     if reverse:
        #         frameRange = [2, 1, 0]
        #         inter = 1

        #     else:
        #         frameRange = [0, 1, 2]
        #     randomFrameFlip = random.randint(0, 1)

        # else:
        #     cropArea.append((0, 0, self.randomCropSize[0], self.randomCropSize[1]))
        #     cropArea.append((0, 0, self.randomCropSize[0], self.randomCropSize[1]))
        #     cropArea.append((0, 0, self.randomCropSize[0], self.randomCropSize[1]))
        #     # IFrameIndex = ((index) % 7  + 1)
        #     # returnIndex = IFrameIndex - 1
        #     frameRange = [0, 1, 2]
        #     randomFrameFlip = 0
        #     inter = 1
        #     shifts = [(0, 0), (0, 0)]

        frame_size = (1, 3, 240, 424)
        
        frameRange = [0, 1, 2]
        # Loop over for all frames corresponding to the `index`.
        for frameIndex in frameRange:
            # Open image using pil and augment the image.

            # image = _pil_loader(self.framesPath[index][frameIndex], cropArea=cropArea[frameIndex],  resizeDim=self.resizeSize, frameFlip=randomFrameFlip)
            image_path = self.framesPath[index][frameIndex]
            svg_file = self.vectorized_paths[index][frameIndex]

            image_file_parent_dir = os.path.dirname(image_path)
            image_folder_name = os.path.basename(image_file_parent_dir)

            # Get the parent folder name of svg_file
            svg_file_parent_dir = os.path.dirname(svg_file)
            svg_folder_name = os.path.basename(svg_file_parent_dir)

            assert image_folder_name == svg_folder_name, "image_folder_name and svg_folder_name should be same"

            image = _pil_loader(self.framesPath[index][frameIndex], resizeDim=self.resizeSize)
            # image.save(str(frameIndex) + '.jpg')



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

            # Apply transformation if specified.
            if self.transform is not None:
                image = self.transform(image)
            
            image = TF.ToTensor()(image)
            sample.append(image.reshape(frame_size))
            svg_triplet.append(svg_tensor)
            svg_triplet_num_segments.append(num_segments)
            svg_files.append(svg_file)
            svg_prepadded_info.append(prepad_svg_info)

        t =  0.5

        sample = torch.cat(sample, dim=0)
        svg_triplet = pad(svg_triplet, -1, [1, 2])
        svg_triplet = torch.stack(svg_triplet)

        return sample, svg_triplet, svg_triplet_num_segments, svg_files, None, svg_prepadded_info

    def __len__(self):
        return len(self.framesPath)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
   
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

def get_loader(mode, root, vector_dir, batch_size, shuffle, num_workers, test_mode=None):
    dataset = AniTriplet(root, vector_dir)
    return DataLoader(dataset, batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=custom_collate, pin_memory=True)


# root = "/media/sahil/DL_5TB/MachineLearning/anime-ds/train_10k"
# vector_root = "/media/sahil/DL_5TB/MachineLearning/anime-ds_vectorized/train_10k"

# loader = get_loader("train", root, vector_root, 2, False, 0)

# for i, (images, svg_info, num_segments, svg_files, time_delta, svg_prepadded_info) in enumerate(loader):
#     im1 = images[:, 0, ...]
#     im2 = images[:, 2, ...]
#     gt = images[:, 1, ...]

#     print(im1.shape, im2.shape, gt.shape)
#     break

# data = AniTriplet(root, vector_root)

# triplets, svg_tensors, num_segments, svg_files, time_deltas, svg_prepadded_inf = data[0]
