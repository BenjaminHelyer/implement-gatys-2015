"""Module that serves as a wrapper for the pre-trained VGG model.

The wrapper handles preprocessing the image and performing 
basic image recognition with VGG.
"""
import json
from pathlib import Path

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

class VggWrapper:
    def __init__(self):
        """Class which acts as a wrapper for the VGG model.
        
        Note that one can still access the model directly by using
        the 'model' member of this class.
        """
        self.model = models.vgg16(pretrained=True)
        self.model.eval() # ensure we don't train the weights of this
        
    def preprocess_img(self, 
                       img_path, 
                       transform=transforms.Compose([
                                    transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                        std=[0.229, 0.224, 0.225])
                                    ])):
        """Performs preprocessing for an image, allowing us to hand it to VGG."""
        img = Image.open(img_path)
        transformed_img = transform(img)
        # note that VGG expects an image batch, so we have to add a dimension of the batch size
        # in this case we have just one image, so we add a dimension of size 1 denoting the batch size
        transformed_img_batch = transformed_img.unsqueeze(0)
        transformed_img_batch = transformed_img_batch.to(device='cuda')
        return transformed_img_batch
    
    def get_top_k_predictions(self, img_tensor, k=5):
        """Gets the top k predictions of the original image based on VGG output.
        
        Can be used as a sanity test to ensure VGG is processing your image correctly.

        img_tensor: input image tensor, expected to be preprocessed via preprocess_img function in this wrapper
        """
        top_preds = []

        predicted_outputs = self.model(img_tensor)

        with open('.data/imagenet_class_index.json', 'r') as f:
            class_index = json.load(f)

        _, preds = torch.topk(predicted_outputs, k=k)
        for pred_class in preds[0]:
            predicted_class = class_index[str(pred_class.item())]
            top_preds += [predicted_class]

        return top_preds
    
    def get_activations(self, img_tensor):
        """Gets all the activations of the image for each layer of the network."""
        activations = []

        # we perform the forward pass here, just as we would when implementing the model
        # in this case, however, we append the activation values to the list
        # we do not track the grad here as we aren't performing gradient descent
        with torch.no_grad():
            x = img_tensor
            for module in self.model.features:
                x = module(x)
                activations += [x]

        return activations
    
    def visualize_activations(self, img_tensor, layers_to_plot = [0, 2, 5, 10, 19]):
        """Visualizes the activations from the VGG model."""
        activations = self.get_activations(img_tensor)

        fig, axes = plt.subplots(1, len(layers_to_plot), figsize=(15, 5))
        for i, layer_num in enumerate(layers_to_plot):
            activations_tensor = activations[layer_num]
            activations_np = activations_tensor.squeeze(0).cpu().numpy()
            heatmap = np.mean(activations_np, axis=0)
            axes[i].imshow(heatmap, cmap='jet')
            axes[i].set_title("Layer: " + str(layer_num))
            axes[i].axis('off')

        plt.tight_layout()
        plt.show()
    
if __name__ == '__main__':
    cuda = torch.cuda.is_available()
    if cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    path_king_crab = Path(__file__).resolve().parent.parent / "tests/test_imgs/alaskan_king_crab.jpg"

    myVgg = VggWrapper()
    img_tensor = myVgg.preprocess_img(path_king_crab)
    preds = myVgg.get_top_k_predictions(img_tensor)
    print(preds)
    myVgg.visualize_activations(img_tensor, [0, 1, 2, 3])
    myVgg.visualize_activations(img_tensor, [26, 27, 28, 29, 30])
