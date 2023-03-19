import cv2
import numpy as np
from PIL import Image

# this is a copy/paste from controlnet_aux:
# see: https://github.com/patrickvonplaten/controlnet_aux/issues/5

class CannyDetector:
    def __call__(self, img, low_threshold, high_threshold):
        
        input_type = "np"
        if isinstance(img, Image.Image):
            img = np.array(img)
            input_type = "pil"
        
        img = cv2.Canny(img, low_threshold, high_threshold)
        
        if input_type == "pil":
            img = Image.fromarray(img)
            
        return img
