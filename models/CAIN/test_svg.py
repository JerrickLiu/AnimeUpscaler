from PIL import Image
import os
import matplotlib.pyplot as plt
import cairosvg
from io import BytesIO

# loads and render an svg

FILE_NAME = "test2.svg"

# image = Image.open(FILE_NAME)
out = BytesIO()
img = cairosvg.svg2png(url=FILE_NAME, write_to=out)
image = Image.open(out)
plt.imshow(image)
plt.show()

