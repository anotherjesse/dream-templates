import cv2
import numpy as np
from PIL import Image

# not able to use: https://github.com/patrickvonplaten/controlnet_aux/issues/5
# it throws an error in the pipeline:
#   image = self.prepare_image(
#   File "/root/.pyenv/versions/3.8.16/lib/python3.8/site-packages/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_controlnet.py", line 515, in prepare_image
#   image = image.transpose(0, 3, 1, 2)
#   ValueError: axes don't match array
#   panic: runtime error: invalid memory address or nil pointer dereference


class CannyDetector:
    def __call__(self, img, low_threshold, high_threshold):
            
        image = np.array(img)
        # get canny image
        image = cv2.Canny(image, low_threshold, high_threshold)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        return Image.fromarray(image)
