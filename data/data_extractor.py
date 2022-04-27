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

sys.path.append('../')

from utils.combine_csv import combine_csv

CLIP_LENGTH_SECONDS = 180
DATA_PATH = "/media/sahil/DL_5TB/MachineLearning/anime_playlist_downloads/fight_scenes"

CACHE_PATH = "/home/jerrick/ramdisk/fight_cache" # cache extracted before writing to slow disk later.

# File length for lexical sorting
FILE_LENTH = 15

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

        # print('List of scenes obtained:')
        # for i, scene in enumerate(scene_list):
        #     print(
        #         'Scene %2d: Start %s / Frame %d, End %s / Frame %d' % (
        #         i+1,
        #         scene[0].get_timecode(), scene[0].get_frames(),
        #         scene[1].get_timecode(), scene[1].get_frames(),))

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

def format_timedelta(td):
    """Utility function to format timedelta objects in a cool way (e.g 00:00:20.05) 
    omitting microseconds and retaining milliseconds"""
    result = str(td)
    try:
        result, ms = result.split(".")
    except ValueError:
        return result + ".00".replace(":", "-")
    ms = int(ms)
    ms = round(ms / 1e4)
    return f"{result}.{ms:02}".replace(":", "-")

def get_ms_timedelta(td):
    """Utility function to convert timedelta objects to milliseconds"""
    return td.total_seconds() * 1000

def get_saving_frames_durations(cap, saving_fps):
    """A function that returns the list of durations where to save the frames"""
    s = []
    # get the clip duration by dividing number of frames by the number of frames per second
    clip_duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    # use np.arange() to make floating-point steps
    for i in np.arange(0, clip_duration, 1 / saving_fps):
        s.append(i)
    return s

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

                # Cehck if the folder exists, if not, create it
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
                    
                    # Clip a random 3 minute clip from the middle of the video
                    video = cv2.VideoCapture(full_path)

                    # get the FPS of the video and total frames
                    fps = video.get(cv2.CAP_PROP_FPS)
                    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

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

                    # ffmpeg_extract_subclip("full.mp4", start_seconds, end_seconds, targetname="cut.mp4")
                    ffmpeg_extract_subclip(full_path, start, start + clip_length, targetname=clip_path)

                    detect_scenes(clip_path, csv_folder, filename)
                    
                    # read the video file    
                    cap = cv2.VideoCapture(clip_path)
                    # fps = cap.get(cv2.CAP_PROP_FPS)
                    saving_frames_per_second = fps
                    # if the SAVING_FRAMES_PER_SECOND is above video FPS, then set it to FPS (as maximum)
                    # saving_frames_per_second = min(fps, SAVING_FRAMES_PER_SECOND)
                    # get the list of duration spots to save
                    saving_frames_durations = get_saving_frames_durations(cap, saving_frames_per_second)

                    # start the loop
                    count = 0 
                    while True:
                        is_read, frame = cap.read()
                        if not is_read:
                            # break out of the loop if there are no frames to read
                            break
                        # get the duration by dividing the frame count by the FPS
                        frame_duration = count / fps
                        try:
                            # get the earliest duration to save
                            closest_duration = saving_frames_durations[0]
                        except IndexError:
                            # the list is empty, all duration frames were saved
                            break

                        if frame_duration >= closest_duration:
                            # if closest duration is less than or equals the frame duration, then save the frame

                            new_file = f"/{count}.png"
                            output_file = working_dir + new_file

                            # Prepend 0s so that the file name is of length FILE_LENGTH for lexical ordering
                            new_file = "0" * (FILE_LENGTH - len(new_file)) + new_file
                            
                            #resize frame so that the smallest side is height px. Preserves aspect ratio
                            height = 240
                            width = int(frame.shape[1] * height / frame.shape[0])
                            frame = cv2.resize(frame, (width, height))

                            cv2.imwrite(output_file, frame) 

                            # drop the duration spot from the list, since this duration spot is already saved
                            try:
                                saving_frames_durations.pop(0)
                            except IndexError:
                                pass
                        # increment the frame count
                        count += 1

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
    
    # Combine all the metadata into one csv file
    combine_csv(output_directory + "/metadata")