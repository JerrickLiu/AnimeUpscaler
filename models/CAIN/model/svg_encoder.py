import torch
import torch.nn as nn
from CAIN.data.svg_utils import load_segments, pad_segments
import numpy as np
import os
import sys

class SVGEncoder(nn.Module):
    """
    SVG Encoder that encodes a parsed SVG file into a latent vector. Takes in a concatenated vector of SVG segments, colors, and transforms as the input vector.
    """

    def __init__(self, input_size, embed_size, hidden_size, output_size, bidirectional=False, num_layers=2):
        super(SVGEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.bidirectional = bidirectional
        self.directions = 2 if bidirectional else 1

        self.embed = torch.nn.Embedding(input_size, embed_size)

        self.gru = nn.GRU(input_size, hidden_size, bidirectional=bidirectional, num_layers=num_layers, dropout=0.5, batch_first=True)
    
    def forward(self, x):
        """
        Takes in a concatenated vector of SVG segments, colors, and transforms as the input vector.

        @param x: The input vector. Should be a concatenated vector of SVG segments, colors, and transforms. Dim of x is (batch_size, max length of segment, 13).

        """

        # x = self.embed(x)
        # print(x.shape)
        x, hidden = self.gru(x)

        if self.bidirectional:
            x = torch.cat((hidden[1,:,:], hidden[3,:,:]), dim=1)

        # x has shape [batch_size, seq_len, directions * hidden_size]
        # hidden has shape [num_layers * directions, batch_size, hidden_size]
        return x, hidden

class ContextEmbedBlock(nn.Module):
    """
    Embeds latent vectors from image into context vectors using attention.
    """
    def __init__(self, latent_dim, ff_dim=64):
        super(ContextEmbedder, self).__init__()
        self.latent_dim = latent_dim
        self.attention = nn.MultiheadAttention(latent_dim, num_heads=2)
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, latent_dim)
        )
        self.ln1 = nn.LayerNorm(latent_dim)
        self.ln2 = nn.LayerNorm(latent_dim)
        
    def forward(self, x):
        attn = self.attention(x)
        x = x + attn
        x = self.ln1(x)
        mlp = self.mlp(x)
        x = x + mlp
        x = self.ln2(x)
        return x

class ContextEmbed(nn.Module):
    """
    Transformer net to encode segments contextually. Expects that a 0 latent as been added at the end.
    """
    def __init__(self, latent_dim, num_layers=2):
        super(ContextEmbed, self).__init__()
        self.blocks = [ContextEmbedBlock(latent_dim) for _ in range(num_layers)]

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return x

#TODO test
def CosineSimilarityWeighting(x1, x2):
    """
    Calculates the cosine similarity between two batches of vectors. 
    """
    cos = nn.functional.cosine_similarity(x1, x2, dim=1)
    return nn.functional.softmax(cos, dim=0)


def embed_svgs(svgs, SVGEncoder, ContextEmbedder):
    """
    Embeds a list of svgs into a context vectorss

    Args:
        svgs: A list of svgs to embed
    Returns:
        context_vectors: a list of context vectors for each svg
    """

    # Input size is 13 because we have segments of dim 8 + colors of dim 3 + transforms of dim 2
    INPUT_SIZE = 13
    EMBED_SIZE = 64
    HIDDEN_SIZE = 128
    OUTPUT_SIZE = 128
    BIDIRECTIONAL = False
    NUM_LAYERS = 2

    svg_encoder = SVGEncoder(input_size=INPUT_SIZE, embed_size=EMBED_SIZE, hidden_size=HIDDEN_SIZE, output_size=OUTPUT_SIZE, bidirectional=BIDIRECTIONAL, num_layers=NUM_LAYERS)

    context_embedder = ContextEmbedder(latent_dim=OUTPUT_SIZE)

    assert len(svgs) == 3

    context_vectors = []

    for svg_info in svgs:
        segments, colors, transforms = svg_info

        segments = torch.Tensor(segments)
        colors = torch.Tensor(colors).unsqueeze(1).expand(-1, max_len, -1)
        transforms = torch.Tensor(transforms).unsqueeze(1).expand(-1, max_len, -1)

        # Concatenate segments, colors, and transforms into a single tensor
        input_tensor = torch.cat((segments, colors, transforms), dim=2)

        latent = svg_encoder(input_tensor)
        context_vector = context_embedder(latent)
        context_vectors.append(context_vector)
    
    return context_vectors

        





        