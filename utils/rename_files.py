import os

DATA_PATH = "/media/sahil/DL_5TB/MachineLearning/anime_playlist_downloads/fight_scenes_extracted_frames/extracted_frames/avatar"

file_length = 15

for root, dirs, files in os.walk(DATA_PATH):
        for file in files:
            new_file = file.replace("frame_", "")
            new_file = new_file.replace(".jpg", ".png")
            # Prepend 0s so that the file name is of length 8
            new_file = "0" * (file_length - len(new_file)) + new_file
            full_path = os.path.join(root, file)
            new_full_path = os.path.join(root, new_file)
            os.rename(full_path, new_full_path)

