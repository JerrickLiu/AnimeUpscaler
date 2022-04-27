#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python generate.py \
    --exp_name CAIN_fin \
    --dataset custom \
    --data_root /media/sahil/DL_5TB/MachineLearning/anime_playlist_downloads/fight_scenes_extracted_frames/extracted_frames/avatar_copy \
    --orig_video_path /media/sahil/DL_5TB/MachineLearning/anime_playlist_downloads/fight_scenes/fight_scenes/avatar_original.webm \
    --save_video_path /home/jerrick/AnimeUpscaler/videos/CAIN_avatar_OP_upscaled.mp4 \
    --img_fmt png \
    --batch_size 32 \
    --test_batch_size 16 \
    --model cain \
    --depth 3 \
    --loss 1*L1 \
    --resume \
    --mode test