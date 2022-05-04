import math
import numpy as np

import torch
import torch.nn as nn

from .common import *


class Encoder(nn.Module):
    def __init__(self, n_resgroups=5, n_resblocks=12, in_channels=3, depth=3, vector_intermediate=False):
        super(Encoder, self).__init__()

        # Shuffle pixels to expand in channel dimension
        # shuffler_list = [PixelShuffle(0.5) for i in range(depth)]
        # self.shuffler = nn.Sequential(*shuffler_list)
        self.shuffler = PixelShuffle(1 / 2**depth)

        relu = nn.LeakyReLU(0.2, True)

        self.vector_intermediate = vector_intermediate
        
        # FF_RCAN or FF_Resblocks
        if self.vector_intermediate:
            self.interpolate = VectorIntermediateInterpolation(n_resgroups, n_resblocks, in_channels * (4**depth), act=relu)

        else:
            self.interpolate = Interpolation(n_resgroups, n_resblocks, in_channels * (4**depth), act=relu)
        
    def forward(self, x1, x2, intermediate=None):
        """
        Encoder: Shuffle-spread --> Feature Fusion --> Return fused features
        """
        feats1 = self.shuffler(x1)
        feats2 = self.shuffler(x2)

        if self.vector_intermediate:
            assert intermediate is not None, "Intermediate tensor is required for vector_intermediate=True"
            feats_intermediate = self.shuffler(intermediate)
            feats = self.interpolate(feats_intermediate)

        else:
            feats = self.interpolate(feats1, feats2)

        return feats


class Decoder(nn.Module):
    def __init__(self, depth=3):
        super(Decoder, self).__init__()

        # shuffler_list = [PixelShuffle(2) for i in range(depth)]
        # self.shuffler = nn.Sequential(*shuffler_list)
        self.shuffler = PixelShuffle(2**depth)

    def forward(self, feats):
        out = self.shuffler(feats)
        return out


class CAIN(nn.Module):
    def __init__(self, n_resgroups=5, n_resblocks=12, depth=3, in_channels=3, vector_intermediate=False):
        super(CAIN, self).__init__()
        
        self.encoder = Encoder(n_resgroups, n_resblocks, in_channels=in_channels, depth=depth, vector_intermediate=vector_intermediate)
        self.decoder = Decoder(depth=depth)
        self.vector_intermediate = vector_intermediate

    def forward(self, x1, x2, intermediate=None):
        x1, m1 = sub_mean(x1)
        x2, m2 = sub_mean(x2)

        if not self.training:
            paddingInput, paddingOutput = InOutPaddings(x1)
            x1 = paddingInput(x1)
            x2 = paddingInput(x2)

        if self.vector_intermediate:
            assert intermediate is not None, 'Intermediate is None'
            feats = self.encoder(x1, x2, intermediate)
            out = self.decoder(feats)

        else:
            feats = self.encoder(x1, x2)
            out = self.decoder(feats)

        if not self.training:
            out = paddingOutput(out)

        mi = (m1 + m2) / 2
        out = out + mi

        return out, feats
