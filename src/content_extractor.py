"""Class that extracts the content representation of an image."""
import json
from pathlib import Path

import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.utils as utils
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

class ContentExtractor:
    def __init__(self, orig_img_path, feature_layer):
        """Class for extracting the conent of a given image.
        
        orig_img: img path that we want to extract the content of
        feature_layer: layer that has the feature map we are interested in
        """
        self.vgg_model = models.vgg16(pretrained=True)
        self.preprocess_transform = transforms.Compose([
                                    transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                        std=[0.229, 0.224, 0.225])
                                    ])

        self.orig_img = self.preprocess_img(orig_img_path, self.preprocess_transform)
        self.feature_layer = feature_layer
        
        self.orig_content = self.get_feature_map(orig_img_path, feature_layer)

    def preprocess_img(self, img_path, transform):
        """Performs preprocessing for an image, allowing us to hand it to VGG."""
        img = Image.open(img_path)
        transformed_img = transform(img)
        # note that VGG expects an image batch, so we have to add a dimension of the batch size
        # in this case we have just one image, so we add a dimension of size 1 denoting the batch size
        transformed_img_batch = transformed_img.unsqueeze(0)
        return transformed_img_batch

    def get_feature_map(self, img, layer):
        """Gets the feature map for a given image at a specified layer."""
        return None
    
    def get_top_k_predictions(self, k=5):
        """Gets the top k predictions of the original image based on VGG output.
        
        Can be used as a sanity test to ensure VGG is processing your image correctly.
        """
        top_preds = []

        predicted_outputs = self.vgg_model(self.orig_img)

        with open('.data/imagenet_class_index.json', 'r') as f:
            class_index = json.load(f)

        _, preds = torch.topk(predicted_outputs, k=k)
        for pred_class in preds[0]:
            predicted_class = class_index[str(pred_class.item())]
            top_preds += [predicted_class]

        return top_preds
    
    def visualize_activations(self):
        """Visualizes the activations from the VGG model."""
        activations = {}
 
        # we perform the forward pass here, just as we would when implementing the model
        # in this case, however, we append the activation values to the dict
        x = self.orig_img
        for module in self.vgg_model.features:
            x = module(x)
            activations[str(module)] = x

        fig, axes = plt.subplots(1, len(activations), figsize=(15, 5))
        for i, layer_name in enumerate(activations):
            activations_tensor = activations[layer_name]
            activations_np = activations_tensor.detach().squeeze(0).cpu().numpy()
            heatmap = np.mean(activations_np, axis=0)
            axes[i].imshow(heatmap, cmap='jet')
            axes[i].set_title(layer_name)
            axes[i].axis('off')

        plt.show()

    
if __name__ == '__main__':
    path_king_crab = Path(__file__).resolve().parent.parent / "tests/test_imgs/alaskan_king_crab.jpg"
    myExtractor = ContentExtractor(path_king_crab, 0)
    myExtractor.visualize_activations()
