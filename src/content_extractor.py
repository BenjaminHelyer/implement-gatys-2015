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
import cv2

class ContentExtractor:
    def __init__(self, 
                 orig_img_path, 
                 feature_layer_num=20):
        """Class for extracting the conent of a given image.
        
        orig_img: img path that we want to extract the content of
        feature_layer: layer that has the feature map we are interested in

        Note that Gatys et al use Conv4_2 for the original content, 
        which corresponds to layer number 19 by the ordering in our implementation.
        """
        self.vgg_model = models.vgg16(pretrained=True)
        self.preprocess_transform = transforms.Compose([
                                    transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                        std=[0.229, 0.224, 0.225])
                                    ])

        self.orig_img = self._preprocess_img(orig_img_path, self.preprocess_transform)
        self.orig_content = self._extract_content(feature_layer_num)

        self.random_image = self._generate_white_noise_img()

    def _preprocess_img(self, img_path, transform):
        """Performs preprocessing for an image, allowing us to hand it to VGG."""
        img = Image.open(img_path)
        transformed_img = transform(img)
        # note that VGG expects an image batch, so we have to add a dimension of the batch size
        # in this case we have just one image, so we add a dimension of size 1 denoting the batch size
        transformed_img_batch = transformed_img.unsqueeze(0)
        return transformed_img_batch
    
    def _generate_white_noise_img(self):
        """Generates a white noise image based on a seed."""
        uniform_noise = np.zeros((224, 224),dtype=np.uint8)
        cv2.randu(uniform_noise,0,255)
        return uniform_noise
    
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
    
    def _get_activations(self):
        """Gets all the activations of the image for each layer of the network."""
        activations = []

        # we perform the forward pass here, just as we would when implementing the model
        # in this case, however, we append the activation values to the dict
        with torch.no_grad():
            x = self.orig_img
            for module in self.vgg_model.features:
                x = module(x)
                activations += [x]

        return activations
    
    def visualize_activations(self, layers_to_plot = [0, 2, 5, 10, 19]):
        """Visualizes the activations from the VGG model."""
        activations = self._get_activations()

        fig, axes = plt.subplots(len(layers_to_plot), 1, figsize=(15, 5))
        for i, layer_num in enumerate(layers_to_plot):
            activations_tensor = activations[layer_num]
            activations_np = activations_tensor.squeeze(0).cpu().numpy()
            heatmap = np.mean(activations_np, axis=0)
            axes[i].imshow(heatmap, cmap='jet')
            axes[i].set_title("Layer: " + str(layer_num))
            axes[i].axis('off')

        plt.show()

    def _extract_content(self, layer_num):
        """Yields a tensor representing the content of the given layer."""
        activations = self._get_activations()

        content = activations[layer_num]

        return content
    
    def visualize_original_content(self):
        """Plots the original content of the image."""
        fig, axes = plt.subplots(1, 1, figsize=(15, 5))
        content_np = self.orig_content.squeeze(0).cpu().numpy()
        content_heatmap = np.mean(content_np, axis=0)
        axes.imshow(content_heatmap, cmap='jet')
        axes.axis('off')
        plt.show()

    def generate_content_image(self):
        """Generates an image that is similar in content to the original image."""
        # generate a random white noise image
            # idea for white noise: go thru pixel by pixel and add uniform random values
        # pass thru new image and get activation
        # perform gradient descent on the new activation vs the old activation
        # update new image somehow
        # continue for certain num of epochs
        return None
    
if __name__ == '__main__':
    path_king_crab = Path(__file__).resolve().parent.parent / "tests/test_imgs/alaskan_king_crab.jpg"
    myExtractor = ContentExtractor(path_king_crab)
    # myExtractor.visualize_activations([0, 1, 2, 3])
    # myExtractor.visualize_activations([26, 27, 28, 29, 30])
    # myExtractor.visualize_original_content()
