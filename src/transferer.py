"""Class that accomplishes the transfer of style onto the content of another image."""
from pathlib import Path

import torch
import torch.nn as nn

from content_extractor import ContentExtractor
from style_extractor import StyleExtractor

class TransferTotalLoss(nn.Module):
    """Custom total loss function for style-content composition."""
    def __init__(self, weights=None):
        super(TransferTotalLoss, self).__init__()
        self.weights = weights

    def forward(self, 
                input_grams, 
                target_grams):
        mse = nn.MSELoss(reduction='mean')
        loss = 0.0

class Transferer:
    def __init__(self,
                 orig_style_img_path,
                 orig_content_img_path):
        """Used for facilitating the transfer of the style of 
        one image onto the content of another."""
        self.contentExtractor = ContentExtractor(orig_content_img_path)
        self.styleExtractor = StyleExtractor(orig_style_img_path, [1, 3, 7, 10, 15, 19])
    
    def generate_styled_content(self,
                                base_img_path=None):
        """Generates an image with the style of one image and the content of another.
        
        Can use a random white noise image as the starting point or use another
        specified image. Gatys found interesting results by using the content
        image as the starting point.
        """
        return None
    
if __name__ == '__main__':
    cuda = torch.cuda.is_available()
    if cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')