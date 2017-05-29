import pyocr
from PIL import Image

LANG = 'chi_sim'

tool = pyocr.get_available_tools()[0]
print(tool.image_to_string(Image.open('car_scene.png'), lang=LANG))
