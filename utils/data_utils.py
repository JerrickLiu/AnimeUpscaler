import pandas as pd
import glob
import os
import shutil
import random
import sys

def combine_csv(csv_path):
    """
    Combines all the metadata of all the csv files in the csv_path into one csv file
    
    FINAL COMBINED CSV ROWS
    # Scene Number, Start Frame, Start Timecode, Start Time (seconds), End Frame, End Timecoce, End Time (seconds), Length (frames), Length (timecode), Length (seconds), file_name
    """

    # Get all csv files in the metadata folder and subfolders
    csv_files = glob.glob(os.path.join(csv_path, "**/*scenes.csv"), recursive=True)

    # joining files with concat and read_csv
    dfs = []
    for f in csv_files:
        df = pd.read_csv(f, header=1)
        # Get number of rows
        n = df.shape[0]
        # Create a new column with the file name
        file_name = f.split("/")[-1]
        # Remove the scenes.csv from the file name
        file_name = file_name.replace(".scenes.csv", "")

        df['file_name'] = file_name
        dfs.append(df)


    df = pd.concat(dfs, ignore_index=True)
    df.to_csv(csv_path + 'all_scenes.csv', index=False)

def train_test_split(video_dir):
    """
    Split the data into train and test
    """

    if not os.path.exists(video_dir):
        print("The directory does not exist")
        sys.exit(1)

    output_directory = video_dir + "_test"

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    else:
        shutil.rmtree(output_directory)
        os.makedirs(output_directory)
    
    # Move 20% of the files to the test directory
    for directory in os.listdir(video_dir):
        dir_path = os.path.join(video_dir, directory)
        if random.random() < 0.2:
            shutil.move(dir_path, os.path.join(output_directory, directory))


def make_video(original_video_path, frame_folder_path, video_name):
    """
    Make a video from the frames in the frame_folder_path
    """

    images = [img for img in sorted(os.listdir(frame_folder_path)) if img.endswith(".png") or img.endswith(".jpg")]

    frame = cv2.imread(os.path.join(frame_folder_path, images[0]))
    height, width, layers = frame.shape

    videoCapture = cv2.VideoCapture(original_video_path)
    fps = int(videoCapture.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

    video = cv2.VideoWriter(video_name, fourcc, fps, (width,height))

    for image in sorted(images):
        video.write(cv2.imread(os.path.join(frame_folder_path, image)))

    video.release()


def rename_files(directory):
    """
    Rename all the files in the directory to have the same name as the file in the directory with 0s prepended to the file name
    """

    file_length = 15
    for root, dirs, files in os.walk(directory):
            for file in files:
                new_file = file.replace("frame_", "")
                new_file = new_file.replace(".jpg", ".png")
                # Prepend 0s so that the file name is of length 8
                new_file = "0" * (file_length - len(new_file)) + new_file
                full_path = os.path.join(root, file)
                new_full_path = os.path.join(root, new_file)
                os.rename(full_path, new_full_path)
