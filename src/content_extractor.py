"""Class that extracts the content representation of an image."""
import json
from pathlib import Path 

import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.utils as utils
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2

from vgg_wrapper import VggWrapper

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
        self.vgg = VggWrapper()
        
        self.orig_img_path = orig_img_path
        self.feature_layer_num = feature_layer_num

        self.orig_img = self.vgg.preprocess_img(self.orig_img_path)
        self.orig_content = self._extract_content(self.feature_layer_num)
    
    def _postprocess_img(self, img_tensor):
        """Post-processes the image such that it is viewable for a human in OpenCV."""
        bgr_img = img_tensor.squeeze(0).cpu().detach().numpy().transpose()
        postprocessed_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        return postprocessed_img
    
    def _generate_white_noise_img(self):
        """Generates a white noise image based on a seed."""
        uniform_noise = np.zeros((224, 224),dtype=np.uint8)
        cv2.randu(uniform_noise,0,255)
        rgb_uniform_noise = cv2.cvtColor(uniform_noise,cv2.COLOR_GRAY2RGB)
        return rgb_uniform_noise
    
    def _get_feature_generated_img(self, generated_img):
        """Gets the feature layer for the generated image."""
        activations = []

        # distinct from activations func because we *do* want to track the grad here
        x = generated_img
        for module in self.vgg.model.features:
            x = module(x)
            activations += [x]

        feature_tensor = activations[self.feature_layer_num]
        return feature_tensor

    def _extract_content(self, layer_num):
        """Yields a tensor representing the content of the given layer."""
        activations = self.vgg.get_activations(self.orig_img)

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

    def generate_content_image(self, num_epoch = 10, learn_rate = 0.1, base_img_path=None):
        """Generates an image that is similar in content to the original image."""
        # let's use an actual neural net here which is the subset of the VGG net, since we're only interested in one output
        features = nn.Sequential(*list(self.vgg.model.features.children())[:self.feature_layer_num+1])
        features.eval()

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

        criterion = nn.MSELoss()
        # it seems they technically don't use SGD in the paper, but it should be fine
        optimizer = optim.SGD([curr_gen_tensor], lr=learn_rate)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[500,750], gamma=0.5)

        for _ in range(0, num_epoch):
            optimizer.zero_grad()
            gen_feature_layer = features.forward(curr_gen_tensor)
            loss = criterion(gen_feature_layer, self.orig_content)
            loss.backward()
            optimizer.step() 
            scheduler.step()

        final_img = self._postprocess_img(curr_gen_tensor)
        return final_img
    
if __name__ == '__main__':
    cuda = torch.cuda.is_available()
    if cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    path_king_crab = Path(__file__).resolve().parent.parent / "tests/test_imgs/alaskan_king_crab.jpg"
    path_english_setter = Path(__file__).resolve().parent.parent / "tests/test_imgs/english_setter.jpg"
    path_modified_crab = Path(__file__).resolve().parent.parent / "tests/test_imgs/modified_alaskan_king_crab.jpg"
    
    myExtractor = ContentExtractor(path_king_crab)
    myExtractor.visualize_original_content()
    gen_content = myExtractor.generate_content_image(num_epoch=1000, learn_rate=100, base_img_path=path_modified_crab)
    cv2.imshow('Generated Content Image',gen_content)
    cv2.waitKey()
