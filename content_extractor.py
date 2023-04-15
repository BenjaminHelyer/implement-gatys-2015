"""Class that extracts the content representation of an image."""

from torchvision.models import vgg

class ContentExtractor:
    def __init__(self, orig_img_path, feature_layer):
        """Class for extracting the conent of a given image.
        
        orig_img: img path that we want to extract the content of
        feature_layer: layer that has the feature map we are interested in
        """
        self.orig_img = self.preprocess_img(orig_img_path)
        self.feature_layer = feature_layer
        
        self.orig_content = self.get_feature_map(orig_img_path, feature_layer)

    def preprocess_img(self, img_path):
        """Performs preprocessing for an image, allowing us to hand it to VGG."""
        raise NotImplementedError

    def get_feature_map(self, img, layer):
        """Gets the feature map for a given image at a specified layer."""
        raise NotImplementedError
    
