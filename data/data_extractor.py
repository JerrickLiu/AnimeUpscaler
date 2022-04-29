from datetime import timedelta
import cv2
import numpy as np
import os
import shutil
from scenedetect import VideoManager 
from scenedetect import SceneManager
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

# For content-aware scene detection:
from scenedetect.detectors import ContentDetector

import scenedetect
from scenedetect import VideoManager 
from scenedetect import SceneManager
from scenedetect import StatsManager
from scenedetect.detectors import ContentDetector
import random
import sys
import splitfolders

sys.path.append('../')

from utils.data_utils import *

CLIP_LENGTH_SECONDS = 180
DATA_PATH = "/media/sahil/DL_5TB/MachineLearning/anime_playlist_downloads/fight_scenes"

CACHE_PATH = "/home/jerrick/ramdisk/fight_cache/" # cache extracted before writing to slow disk later.

# File length for lexical sorting
FILE_LENGTH = 12

def detect_scenes(video_path, save_path, filename):
    # type: (str) -> List[Tuple[FrameTimecode, FrameTimecode]]
    video_manager = VideoManager([video_path])
    stats_manager = StatsManager()
    # Construct our SceneManager and pass it our StatsManager.
    scene_manager = SceneManager(stats_manager)

    # Add ContentDetector algorithm (each detector's constructor
    # takes detector options, e.g. threshold).
    scene_manager.add_detector(ContentDetector())

    # We save our stats file to {FILENAME}.stats.csv.
    stats_file_path = os.path.join(save_path, '%s.stats.csv' % filename)
    scene_csv_path = os.path.join(save_path, '%s.scenes.csv' % filename)

    scene_list = []

    base_timecode = video_manager.get_base_timecode()

    try:
        # If stats file exists, load it.
        if os.path.exists(stats_file_path):
            # Read stats from CSV file opened in read mode:
            with open(stats_file_path, 'r') as stats_file:
                stats_manager.load_from_csv(stats_file)

        # Set downscale factor to improve processing speed.
        video_manager.set_downscale_factor()

        # Start video_manager.
        video_manager.start()

        # Perform scene detection on video_manager.
        scene_manager.detect_scenes(frame_source=video_manager)

        # Obtain list of detected scenes.
        scene_list = scene_manager.get_scene_list()
        # Each scene is a tuple of (start, end) FrameTimecodes.

        # We only write to the stats file if a save is required:
        if stats_manager.is_save_required():
            base_timecode = video_manager.get_base_timecode()
            with open(stats_file_path, 'w') as stats_file:
                stats_manager.save_to_csv(stats_file, base_timecode)

        with open(scene_csv_path, 'w') as scene_file:
                scenedetect.scene_manager.write_scene_list(scene_file, scene_list)

    finally:
        video_manager.release()

    return scene_list

def ignore_files(dir, files):
    return [f for f in files if os.path.isfile(os.path.join(dir, f))]

def extract(video_dir, output_directory, CACHE_DIR=None):
    if CACHE_DIR is None:
        print("WARNING: CACHE_DIR is not set, expect slow performance")
    if not os.path.exists(video_dir):
        print("The directory does not exist")
        sys.exit(1)

    if not os.path.exists(output_directory):
        shutil.copytree(video_dir, output_directory, ignore=ignore_files)

    for root, dirs, files in os.walk(video_dir):
        for file in files:
            if file.endswith(".webm") or file.endswith(".mp4"):
                full_path = os.path.join(root, file)

                filename, _ = os.path.splitext(file)

                # Sanitized filename
                filename = "".join(x if x.isalnum() else "_" for x in filename)
                filename = filename.replace(" ","_")

                frame_folder = filename + "-extracted_frames"
                frame_folder = os.path.join(output_directory, 'extracted_frames', frame_folder)

                csv_folder = filename + "-csv"
                csv_folder = os.path.join(output_directory, 'metadata', csv_folder)

                # Check if the folder exists, if not, create it
                if not os.path.exists(frame_folder):
                    # Check if cache folder exists, else set to frame_folder
                    if CACHE_DIR is None:
                        working_dir = frame_folder
                    else:
                        working_dir = CACHE_DIR
                        if not os.path.exists(working_dir):
                            os.makedirs(working_dir)

                    # make directory with all parent diectories if they do not exist
                    os.makedirs(frame_folder)
                    os.makedirs(csv_folder)

                    clip_path = os.path.join(working_dir, filename + "-clipped" + os.path.splitext(file)[1])
                    

                    video = cv2.VideoCapture(full_path)

                    # get the FPS of the video and total frames
                    fps = video.get(cv2.CAP_PROP_FPS)
                    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

                    video.release()

                    try:
                        duration = int(frame_count / fps)
                    except ZeroDivisionError:
                        print("FPS is 0, skipping")
                        continue

                    # Get 0.25 and 0.75 of the video duration
                    start_seconds = int(duration * 0.25)
                    end_seconds = int(duration * 0.75)

                    start = random.randint(start_seconds, end_seconds)

                    clip_length = min(CLIP_LENGTH_SECONDS, duration - start)

                    # Clip a random 3 minute clip from the middle of the video
                    ffmpeg_extract_subclip(full_path, start, start + clip_length, targetname=clip_path)

                    detect_scenes(clip_path, csv_folder, filename)
                    
                    # read the video file    
                    cap = cv2.VideoCapture(clip_path)
                    # fps = cap.get(cv2.CAP_PROP_FPS)
                    saving_frames_per_second = fps

                    # start the loop
                    count = 0 
                    while True:
                        is_read, frame = cap.read()
                        if not is_read:
                            # break out of the loop if there are no frames to read
                            break

                        new_file = f"{count}.png"

                        # Prepend 0s so that the file name is of length FILE_LENGTH for lexical ordering
                        new_file = "0" * (FILE_LENGTH - len(new_file)) + new_file

                        output_file = working_dir + new_file

                        #resize frame so that the smallest side is height px. Preserves aspect ratio
                        height = 240
                        width = int(frame.shape[1] * height / frame.shape[0])
                        frame = cv2.resize(frame, (width, height))

                        cv2.imwrite(output_file, frame)

                        # increment the frame count
                        count += 1

                    cap.release()
                    # Delete the clipped video file from working dir
                    os.remove(clip_path)                

                    if working_dir != frame_folder:
                        # If the cache folder is set, then we need to move the files to the frame folder
                        for file in os.listdir(working_dir):
                            if file.endswith(".jpg") or file.endswith(".png"):
                                shutil.move(os.path.join(working_dir, file), os.path.join(frame_folder, file))
                        # Delete the cache folder
                        shutil.rmtree(working_dir)
                        if not os.path.exists(working_dir):
                            os.makedirs(working_dir)
                else:
                    print("The folder already exists!")
                    continue


if __name__ == "__main__":
    output_directory = DATA_PATH + "_extracted_frames"

    # Extract the frames
    extract(DATA_PATH, output_directory, CACHE_PATH)
    
    # Combine all the metadata into one csv file. Found in data_utils.py
    combine_csv(output_directory + "/metadata/")

    # Split the data into train and test. Found in data_utils.py
    train_test_split(output_directory + "/extracted_frames/")