import torch
import torch.nn as nn
import numpy as np
import os
import sys
sys.path.append('../')
from data.AnimeVectorizedDataset import AnimationVectorizedDataset, get_loader

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
        super(ContextEmbedBlock, self).__init__()
        self.latent_dim = latent_dim
        self.attention = nn.MultiheadAttention(latent_dim, num_heads=2, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, latent_dim)
        )
        self.ln1 = nn.LayerNorm(latent_dim)
        self.ln2 = nn.LayerNorm(latent_dim)
        
    def forward(self, x):
        attn, _ = self.attention(x, x, x)
        x = x + attn
        x = self.ln1(x)
        mlp = self.mlp(x)
        x = x + mlp
        x = self.ln2(x)
        return x

class ContextEmbedder(nn.Module):
    """
    Transformer net to encode segments contextually. Expects that a 0 latent as been added at the end.
    """
    def __init__(self, latent_dim, num_layers=2):
        super(ContextEmbedder, self).__init__()
        self.blocks = [ContextEmbedBlock(latent_dim).cuda() for _ in range(num_layers)]

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
        svgs: A batch of svgs to embed
        SVGEncoder: The SVGEncoder to use
        ContextEmbedder: The ContextEmbedder to use
    
    Returns:
        context_vectors: a list of context vectors for each svg
    """

    context_vectors = []


    # Loops through each triplet in the batch
    for svg_triplet_idx in range(svgs.shape[0]):
        svg_triplet = svgs[svg_triplet_idx]

        svg_frame_1 = svg_triplet[0]
        svg_frame_3 = svg_triplet[len(svg_triplet) - 1]

        # print("BEFORE SVG ENCODER")
        # print(svg_frame_1.shape)
        # print(svg_frame_3.shape)

        # Encodes the svg triplet
        frame_1_embeddings, hidden1 = SVGEncoder(svg_frame_1)
        frame_3_embeddings, hidden3 = SVGEncoder(svg_frame_3)

        frame_1_embeddings = hidden1[1, :, :].unsqueeze(0)
        frame_3_embeddings = hidden3[1, :, :].unsqueeze(0)

        # print("BEFORE CONTEXT EMBEDDING")
        # print(frame_1_embeddings.shape)
        # print(frame_3_embeddings.shape)


        # Calculates the context vector
        frame_1_context = ContextEmbedder(frame_1_embeddings)
        frame_3_context = ContextEmbedder(frame_3_embeddings)

        # print("AFTER CONTEXT EMBEDDING")
        # print(frame_1_context.shape)
        # print(frame_3_context.shape)

        context_vectors.append(frame_1_context)
        context_vectors.append(frame_3_context)
    
    # Returns [batch_size, num_segments, latent_dim]
    context_vectors = torch.stack(context_vectors).squeeze()
    return context_vectors


# INPUT_SIZE = 13
# EMBED_SIZE = 64
# HIDDEN_SIZE = 32
# OUTPUT_SIZE = 32
# BIDIRECTIONAL = False
# NUM_LAYERS = 2

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# svg_encoder = SVGEncoder(input_size=INPUT_SIZE, embed_size=EMBED_SIZE, hidden_size=HIDDEN_SIZE, output_size=OUTPUT_SIZE, bidirectional=BIDIRECTIONAL, num_layers=NUM_LAYERS).to(device)

# context_embedder = ContextEmbedder(latent_dim=OUTPUT_SIZE).to(device)

# data_root = "/media/sahil/DL_5TB/MachineLearning/anime_playlist_downloads/fight_scenes_extracted_frames/extracted_frames"

# csv = "/media/sahil/DL_5TB/MachineLearning/anime_playlist_downloads/fight_scenes_extracted_frames/metadata/all_scenes.csv"

# vectorized_dir = "/media/sahil/DL_5TB/MachineLearning/anime_playlist_downloads/fight_scenes_extracted_frames/extracted_frames_vectorized"

# anime_trips = AnimationVectorizedDataset(csv, data_root, vectorized_dir)

# loader = get_loader("train", csv, data_root, vectorized_dir, 4, False, 0)

# for i, (images, svg_info, time_delta) in enumerate(loader):
#     svg_info = svg_info.to(device)
#     v = embed_svgs(svg_info, svg_encoder, context_embedder)
#     print(v.shape)
#     break

    




        
