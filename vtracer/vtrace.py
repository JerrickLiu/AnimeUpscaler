import os
import sys

def vtrace(directory, path_to_vtracer):

    # Check if the directory exists
    if not os.path.exists(directory):
        print("The directory does not exist")
        sys.exit(1)

    # Check if the path to vtracer exists
    if not os.path.exists(path_to_vtracer):
        print("The path to vtracer does not exist")
        sys.exit(1)

    # Create a directory for the output
    output_directory = directory + "_vectorized"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for root, dirs, files in os.walk(directory):
        # Make all dirs in the output directory
        for d in dirs:
            if not os.path.exists(output_directory + "/" + d):
                os.makedirs(output_directory + "/" + d)

        for file in files:
            if file.endswith(".png") or file.endswith(".jpg"):
                output_file = output_directory + root.replace(directory, "") + "/" + file.replace(".png", ".svg")
                os.system(path_to_vtracer + " --input " + os.path.join(root, file) + " --output " + output_file)
if __name__ == "__main__":
    vtrace(sys.argv[1], sys.argv[2])
