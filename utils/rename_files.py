import os

DATA_PATH = os.environ["disk"] + "/test_fight_scenes/"

for root, dirs, files in os.walk(DATA_PATH):
        for file in files:
            full_path = os.path.join(root, file)
            r = full_path.replace(" ","")
            if(r != file):
                os.rename(full_path,r)
