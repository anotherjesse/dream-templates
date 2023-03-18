from annotator.util import resize_image, HWC3
from diffusers.utils import load_image
import numpy as np
import cv2
from PIL import Image



def canny(fn, low_threshold=100, high_threshold=200):
    image = load_image(fn)
    image = np.array(image)
    # get canny image
    image = cv2.Canny(image, 100, 200)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    return Image.fromarray(image)
