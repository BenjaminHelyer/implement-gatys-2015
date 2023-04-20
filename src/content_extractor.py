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
        self.vgg_model.eval() # ensure we don't train the weights of this
        self.preprocess_transform = transforms.Compose([
                                    transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                        std=[0.229, 0.224, 0.225])
                                    ])
        
        self.orig_img_path = orig_img_path
        self.feature_layer_num = feature_layer_num

        self.orig_img = self._preprocess_img(self.orig_img_path, self.preprocess_transform)
        self.orig_content = self._extract_content(self.feature_layer_num)

    def _preprocess_img(self, img_path, transform):
        """Performs preprocessing for an image, allowing us to hand it to VGG."""
        img = Image.open(img_path)
        transformed_img = transform(img)
        # note that VGG expects an image batch, so we have to add a dimension of the batch size
        # in this case we have just one image, so we add a dimension of size 1 denoting the batch size
        transformed_img_batch = transformed_img.unsqueeze(0)
        transformed_img_batch = transformed_img_batch.to(device='cuda')
        return transformed_img_batch
    
    def _generate_white_noise_img(self):
        """Generates a white noise image based on a seed."""
        uniform_noise = np.zeros((224, 224),dtype=np.uint8)
        cv2.randu(uniform_noise,0,255)
        rgb_uniform_noise = cv2.cvtColor(uniform_noise,cv2.COLOR_GRAY2RGB)
        return rgb_uniform_noise
    
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
        # in this case, however, we append the activation values to the list
        # we do not track the grad here as we aren't performing gradient descent
        with torch.no_grad():
            x = self.orig_img
            for module in self.vgg_model.features:
                x = module(x)
                activations += [x]

        return activations
    
    def _get_feature_generated_img(self, generated_img):
        """Gets the feature layer for the generated image."""
        activations = []

        # distinct from activations func because we *do* want to track the grad here
        x = generated_img
        for module in self.vgg_model.features:
            x = module(x)
            activations += [x]

        feature_tensor = activations[self.feature_layer_num]
        return feature_tensor
    
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

    def generate_content_image(self, num_epoch = 10, learn_rate = 0.1, base_img_path=None):
        """Generates an image that is similar in content to the original image."""
        # let's use an actual neural net here which is the subset of the VGG net, since we're only interested in one output
        features = nn.Sequential(*list(self.vgg_model.features.children())[:self.feature_layer_num+1])
        features.eval()

        if base_img_path is None:
            generated_image = self._generate_white_noise_img()

            # need a slightly different transform since we're reading in a numpy array generated via OpenCV
            alt_transform = transforms.Compose([transforms.ToPILImage(),
                                        transforms.Resize((224, 224)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                            std=[0.229, 0.224, 0.225])
                                        ])
            transformed_gen_img = alt_transform(generated_image)
            transformed_gen_img_batch = transformed_gen_img.unsqueeze(0)
            curr_gen_tensor = transformed_gen_img_batch.clone()
            curr_gen_tensor = curr_gen_tensor.to('cuda')
            curr_gen_tensor.requires_grad = True
        else:
            transformed_gen_img_batch = self._preprocess_img(base_img_path, self.preprocess_transform)
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

        bgr_img = curr_gen_tensor.squeeze(0).cpu().detach().numpy().transpose()
        final_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        return final_img
    
if __name__ == '__main__':
    cuda = torch.cuda.is_available()
    if cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    path_king_crab = Path(__file__).resolve().parent.parent / "tests/test_imgs/alaskan_king_crab.jpg"
    path_english_setter = Path(__file__).resolve().parent.parent / "tests/test_imgs/english_setter.jpg"
    path_modified_crab = Path(__file__).resolve().parent.parent / "tests/test_imgs/modified_alaskan_king_crab.jpg"
    myExtractor = ContentExtractor(path_king_crab)
    # myExtractor.visualize_activations([0, 1, 2, 3])
    # myExtractor.visualize_activations([26, 27, 28, 29, 30])
    # myExtractor.visualize_original_content()
    gen_content = myExtractor.generate_content_image(num_epoch=1000, learn_rate=100, base_img_path=path_modified_crab)
    cv2.imshow('Generated Content Image',gen_content)
    cv2.waitKey()
