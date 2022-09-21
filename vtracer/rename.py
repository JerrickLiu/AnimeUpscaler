import os
from PIL import Image

OUT = "/mnt/WDRed/MachineLearning/data/anime/datasets/"

size = 424, 240
for root, dirs, files in os.walk(OUT):
    for file in files:
        path = os.path.join(root, file)
        # Resize the image to SIZE
        if path.endswith(".png") or path.endswith(".jpg"):
            im = Image.open(path)
            im.thumbnail(size, Image.ANTIALIAS)
            im.save(path)
        