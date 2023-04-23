"""Class that accomplishes the transfer of style onto the content of another image."""
from pathlib import Path

import torch
import torch.nn as nn
import cv2

from content_extractor import ContentExtractor
from style_extractor import StyleExtractor
from vgg_wrapper import VggWrapper

class TransferTotalLoss(nn.Module):
    """Custom total loss function for style-content composition."""
    def __init__(self, weights=None):
        super(TransferTotalLoss, self).__init__()
        self.weights = weights

    def forward(self):
        mse = nn.MSELoss(reduction='mean')
        loss = 0.0

class Transferer:
    def __init__(self,
                 orig_style_img_path,
                 orig_content_img_path):
        """Used for facilitating the transfer of the style of 
        one image onto the content of another."""
        self.vgg = VggWrapper()

        self.contentExtractor = ContentExtractor(orig_content_img_path)
        self.styleExtractor = StyleExtractor(orig_style_img_path, [1, 3, 7, 10, 15, 19])
    
    def generate_styled_content(self,
                                base_img_path=None):
        """Generates an image with the style of one image and the content of another.
        
        Can use a random white noise image as the starting point or use another
        specified image. Gatys found interesting results by using the content
        image as the starting point.
        """
        style_feature_nets = self.styleExtractor.initialize_feature_nets(track_grad=True)
        content_feature_net = self.contentExtractor.initialize_feature_nets()

        if base_img_path is None:
            img_path = 'rand_img.jpg'
            generated_image = self._generate_white_noise_img()
            cv2.imwrite(img_path, generated_image)
            curr_gen_tensor = self.vgg.preprocess_img(img_path)
            curr_gen_tensor.requires_grad = True
        else:
            transformed_gen_img_batch = self.vgg.preprocess_img(base_img_path)
            curr_gen_tensor = transformed_gen_img_batch.clone()
            curr_gen_tensor.requires_grad = True

        

        return None
    
if __name__ == '__main__':
    cuda = torch.cuda.is_available()
    if cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')