import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from .svg_utils import load_segments, post_process_svg_info
import torch.nn as nn
import torchvision.transforms as TF

class Video(Dataset):
    def __init__(self, data_root, fmt='png'):
        images = sorted(glob.glob(os.path.join(data_root, '*.%s' % fmt)))
        for im in images:
            try:
                float_ind = float(im.split('_')[-1][:-4])
            except ValueError:
                os.rename(im, '%s_%.06f.%s' % (im[:-4], 0.0, fmt))
        # re
        images = sorted(glob.glob(os.path.join(data_root, '*.%s' % fmt)))
        self.imglist = [[images[i], images[i+1]] for i in range(len(images)-1)]
        print('[%d] images ready to be loaded' % len(self.imglist))


    def __getitem__(self, index):
        imgpaths = self.imglist[index]

        # Load images
        img1 = Image.open(imgpaths[0])
        img2 = Image.open(imgpaths[1])

        T = transforms.ToTensor()
        img1 = T(img1)
        img2 = T(img2)

        imgs = [img1, img2] 
        meta = {'imgpath': imgpaths}
        return imgs, meta

    def __len__(self):
        return len(self.imglist)


class VideoVectorized(Dataset):
    def __init__(self, data_root, svg_root, fmt='png'):
        images = sorted(glob.glob(os.path.join(data_root, '*.%s' % fmt)))
        for im in images:
            try:
                float_ind = float(im.split('_')[-1][:-4])
            except ValueError:
                os.rename(im, '%s_%.06f.%s' % (im[:-4], 0.0, fmt))

        images = sorted(glob.glob(os.path.join(data_root, '*.%s' % fmt)))
        self.imglist = [[images[i], images[i+1]] for i in range(len(images)-1)]
        print('[%d] images ready to be loaded' % len(self.imglist))

        svgs = sorted(glob.glob(os.path.join(svg_root, '*.svg')))

        for svg in svgs:
            try:
                float_ind = float(svg.split('_')[-1][:-4])
            except ValueError:
                os.rename(svg, '%s_%.06f.%s' % (svg[:-4], 0.0, 'svg'))

        svgs = sorted(glob.glob(os.path.join(svg_root, '*.svg')))
        self.svg_list = [[svgs[i], svgs[i+1]] for i in range(len(svgs)-1)]
        print('[%d] svg images ready to be loaded' % len(self.imglist))

        assert len(self.imglist) == len(self.svg_list), 'Number of images and svg images must be equal'


    def __getitem__(self, index):
        imgpaths = self.imglist[index]
        svgpaths = self.svg_list[index]

        # Load images
        img1 = Image.open(imgpaths[0])
        img2 = Image.open(imgpaths[1])

        svgs = []
        svgs_num_segments = []
        svg_files = []
        svg_prepadded_info = []

        frame_size = (1, 3, 240, 424)

        for svg_path in svgpaths:
            # SVG info is a list of (segments, color, transforms)
            prepad_svg_info = list(load_segments(svg_path))

            # Get number of segments of SVG to be added
            num_segments = len(prepad_svg_info[0])

            # Pad the svg info and convert it to tensors
            svg_info = post_process_svg_info(prepad_svg_info)

            segments, colors, transform = svg_info

            # Concatenate segments, colors, and transforms into a single tensor
            svg_tensor = torch.cat((segments, colors, transform), dim=2)

            # Swap the first and last channels for padding
            svg_tensor = svg_tensor.permute(2, 1, 0)

            svgs.append(svg_tensor)
            svgs_num_segments.append(num_segments)
            svg_files.append(svg_path)
            svg_prepadded_info.append(prepad_svg_info)

        svgs = pad(svgs, -1, [1, 2])
        svgs = torch.stack(svgs)

        T = transforms.ToTensor()
        img1 = T(img1)
        img1 = TF.functional.crop(img1, 0, 0, frame_size[2], frame_size[3])
        img1 = img1.reshape(frame_size)
        
        img2 = T(img2)
        img2 = TF.functional.crop(img2, 0, 0, frame_size[2], frame_size[3])
        img2 = img2.reshape(frame_size)


        imgs = [img1, img2]
        imgs = torch.cat(imgs, dim=0) 
        meta = {'imgpath': imgpaths}
        return imgs, meta, svgs, svgs_num_segments, svg_files, svg_prepadded_info

    def __len__(self):
        return len(self.imglist)

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
    imgs = [item[0] for item in batch]

    # Keep the default collate function for the rest of the batch
    meta = {'imgpath': [[], []]}

    for i in range(len(batch)):
        meta['imgpath'][0].append(batch[i][1]['imgpath'][0])
        meta['imgpath'][1].append(batch[i][1]['imgpath'][1])

    svg_tensors = [item[2] for item in batch]
    num_segments = [item[3] for item in batch]
    svg_files = [item[4] for item in batch]
    svg_prepadded_info = [item[5] for item in batch]

    svg_tensors = pad(svg_tensors, -1, [2, 3])

    imgs = torch.stack(imgs, dim=0)
    svg_tensors = torch.stack(svg_tensors, dim=0)
    # print("BEFORE: ", svg_tensors.shape)
    svg_tensors = svg_tensors.permute(0, 1, 4, 3, 2)

    return [imgs, meta, svg_tensors, num_segments, svg_files, svg_prepadded_info]

def get_loader(mode, data_root, batch_size, img_fmt='png', shuffle=False, num_workers=0, n_frames=1):
    if mode == 'train':
        is_training = True
    else:
        is_training = False
    dataset = Video(data_root, fmt=img_fmt)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)

def get_vectorized_loader(mode, data_root, svg_root, batch_size, img_fmt='png', shuffle=False, num_workers=0, n_frames=1):
    if mode == 'train':
        is_training = True
    else:
        is_training = False

    dataset = VideoVectorized(data_root, svg_root, fmt=img_fmt)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=custom_collate, pin_memory=True)
