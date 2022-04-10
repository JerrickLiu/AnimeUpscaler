import pandas as pd
import glob
import os

files = "/home/jerrick/disk_mount/anime_data/small_extracted_frames/metadata/"

# Get all csv files in the metadata folder and subfolders
csv_files = glob.glob(os.path.join(files, "**/*scenes.csv"), recursive=True)


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
# df.to_csv('/home/jerrick/disk_mount/anime_data/small_extracted_frames/metadata/all_scenes.csv', index=False)