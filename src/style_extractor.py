"""Class that extracts the style representation of an image."""

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

from vgg_wrapper import VggWrapper

class StyleExtractor:
    def __init__(self,
                orig_img_path):
        """Class for extracting the style of a given image.
        
        orig_img_path: path to the image that we want to extract the style from
        """
        self.vgg = VggWrapper()

        self.vgg_model = models.vgg16(pretrained=True)
        self.vgg_model.eval() # ensure we don't train the weights of this
        self.preprocess_transform = transforms.Compose([
                                    transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                        std=[0.229, 0.224, 0.225])
                                    ])
        
        self.orig_img_path = orig_img_path

        self.orig_img = self.vgg.preprocess_img(self.orig_img_path, self.preprocess_transform)

    def _preprocess_img(self, img_path, transform):
        """Performs preprocessing for an image, allowing us to hand it to VGG."""
        img = Image.open(img_path)
        transformed_img = transform(img)
        # note that VGG expects an image batch, so we have to add a dimension of the batch size
        # in this case we have just one image, so we add a dimension of size 1 denoting the batch size
        transformed_img_batch = transformed_img.unsqueeze(0)
        transformed_img_batch = transformed_img_batch.to(device='cuda')
        return transformed_img_batch