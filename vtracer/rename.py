import os
from PIL import Image

OUT = "/media/sahil/DL_5TB/MachineLearning/anime-ds_cropped"

size = 240, 424
for root, dirs, files in os.walk(OUT):
    for file in files:
        path = os.path.join(root, file)
        # Resize the image to 240 x 424
        if path.endswith(".png") or path.endswith(".jpg"):
            im = Image.open(path)
            im.thumbnail(size, Image.ANTIALIAS)
            im.save(path)
        