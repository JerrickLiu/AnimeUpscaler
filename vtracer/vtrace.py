import os
import sys

# Write a function that goes through a directory of images and executes ./vtrace --input <image> --output <image> on each image.

def vtrace(directory, save_dir, path_to_vtracer):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".png") or file.endswith(".jpg"):
                output = os.path.splitext(file)[0] + ".svg"
                os.system(path_to_vtracer + " --input " + os.path.join(root, file) + " --output " + os.path.join(save_dir, output))

if __name__ == "__main__":
    vtrace(sys.argv[1], sys.argv[2], sys.argv[3])
