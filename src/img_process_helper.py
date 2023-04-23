"""Class to help with image processing.

Decided to abstract away some of the
boilerplate image processing that is shared
between ContentExtractor, StyleExtractor,
and Transferer class.

This allows us to focus on what's actually
important in those classes and not be
distracted by the overhead of the
functions in this class.
"""

import cv2
import numpy as np

class ImgProcessHelper:
    def __init__(self, white_noise_seed=None):
        """Abstracts away some of the image processing boilerplate."""
        # TODO: could eventually allow providing a seed for the white noie
        self.white_noise_seed = white_noise_seed
    
    def _generate_white_noise_img(self):
        """Generates a white noise image based on a seed."""
        uniform_noise = np.zeros((224, 224),dtype=np.uint8)
        cv2.randu(uniform_noise,0,255)
        rgb_uniform_noise = cv2.cvtColor(uniform_noise,cv2.COLOR_GRAY2RGB)
        return rgb_uniform_noise
    
    def _postprocess_img(self, img_tensor):
        """Post-processes the image such that it is viewable for a human in OpenCV."""
        bgr_img = img_tensor.squeeze(0).cpu().detach().numpy().transpose()
        postprocessed_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        return postprocessed_img
