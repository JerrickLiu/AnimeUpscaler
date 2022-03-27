import os
import sys
import shutil
from tqdm import tqdm
from threading import Thread, Lock

mutex = Lock()
fq = []
len_fq = 0

def ignore_files(dir, files):
    return [f for f in files if os.path.isfile(os.path.join(dir, f))]

def threaded_vtrace(path_to_vtracer):
    while True:
        mutex.acquire()
        if len(fq) == 0:
            mutex.release()
            return
        if len_fq - len(fq) % 500 == 0:
            print(str(len_fq - len(fq)) + " files done")
        file, output_file = fq.pop()
        mutex.release()

        os.system(path_to_vtracer + " --input " + file + " --output " + output_file + " >> vtrace_out.txt")

def vtrace(directory, path_to_vtracer):
    global fq
    global len_fq

    if not os.path.exists(directory):
        print("The directory does not exist")
        sys.exit(1)

    if not os.path.exists(path_to_vtracer):
        print("The path to vtracer does not exist")
        sys.exit(1)

    # output_directory = directory + "_vectorized"
    output_directory = "./out"
    if not os.path.exists(output_directory):
        shutil.copytree(directory, output_directory, ignore=ignore_files)

    for root, dirs, files in tqdm(os.walk(directory)):
        for file in files:
            if file.endswith(".png") or file.endswith(".jpg"):
                output_file = output_directory + root.replace(directory, "") + "/" + file.replace(".png", ".svg")
                fq.append((os.path.join(root, file), output_file))
                # os.system(path_to_vtracer + " --input " + os.path.join(root, file) + " --output " + output_file)

    len_fq = len(fq)
    threads = []
    for i in range(32):
        t = Thread(target=threaded_vtrace, args=(path_to_vtracer,))
        t.start()
        threads.append(t)

if __name__ == "__main__":
    vtrace(sys.argv[1], sys.argv[2])
