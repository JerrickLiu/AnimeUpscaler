import os
import sys
import shutil

def ignore_files(dir, files):
    return [f for f in files if os.path.isfile(os.path.join(dir, f))]

def vtrace(directory, path_to_vtracer):
    if not os.path.exists(directory):
        print("The directory does not exist")
        sys.exit(1)

    if not os.path.exists(path_to_vtracer):
        print("The path to vtracer does not exist")
        sys.exit(1)

    output_directory = directory + "_vectorized"
    if not os.path.exists(output_directory):
        shutil.copytree(directory, output_directory, ignore=ignore_files)

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".png") or file.endswith(".jpg"):
                output_file = output_directory + root.replace(directory, "") + "/" + file.replace(".png", ".svg")
                os.system(path_to_vtracer + " --input " + os.path.join(root, file) + " --output " + output_file)

if __name__ == "__main__":
    vtrace(sys.argv[1], sys.argv[2])
