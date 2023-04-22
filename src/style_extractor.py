"""Class that extracts the style representation of an image."""
from pathlib import Path
from contextlib import nullcontext

import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
from PIL import Image
import cv2
import numpy as np

from vgg_wrapper import VggWrapper

class StyleTotalLoss(nn.Module):
    """Custom loss function based on the style total loss.
    
    We have a loss function based upon a linear combination
    of several MSE losses. This is because the loss is dependent
    on the loss of a set of Gram matrices, not just one tensor.
    """
    def __init__(self, weights=None):
        super(StyleTotalLoss, self).__init__()
        self.weights = weights

    def forward(self, 
                input_grams, 
                target_grams):
        mse = nn.MSELoss(reduction='mean')
        loss = 0.0

        if len(input_grams) != len(target_grams):
            raise ValueError

        for i in range(len(input_grams)):
            if self.weights is None:
                loss += mse(input_grams[i], target_grams[i])
            else:
                if len(self.weights) != len(input_grams):
                    raise ValueError
                else:
                    loss += self.weights[i]*mse(input_grams[i], target_grams[i])
        return loss

class StyleExtractor:
    def __init__(self,
                orig_img_path,
                feature_layer_nums,
                loss_weights = None):
        """Class for extracting the style of a given image.
        
        orig_img_path: path to the image that we want to extract the style from
        """
        self.vgg = VggWrapper()
        
        self.orig_img_path = orig_img_path
        self.feature_layer_nums = feature_layer_nums

        self.orig_img = self.vgg.preprocess_img(self.orig_img_path)
        self.orig_feature_nets = self.initialize_feature_nets()
        self.orig_grams = self.get_gram_matrices(self.orig_img, self.orig_feature_nets)

        if loss_weights == None:
            self.loss_criterion = StyleTotalLoss()
        else:
            self.loss_criterion = StyleTotalLoss(weights=loss_weights)

    def get_gram_matrices(self, img_tensor, feature_nets, track_grad=False):
        """Gets the Gram matrices for a given image tensor."""
        gram_matrices_list = []
        for net in feature_nets:
                curr_layer = net.forward(img_tensor)
                gram = self.calculate_gram_matrix(curr_layer)
                gram_matrices_list += [gram]
        return gram_matrices_list
    
    def initialize_feature_nets(self, track_grad=False):
        """Initializes the neural networks for all the feature layers.
        
        This avoids the problem of intializing these over and over again,
        which leads to all sorts of issues (not just performance).

        Why we're using these "feature_nets": we define a different neural network
        for each feature layer, since we're interested in doing a forward
        pass on several layers. The neural net depth depends on the feature
        layer we are aiming at, where the last layer in the given neural net
        is the given feature layer.

        Note that the feature layer numbers are set in the
        class constructor, not in this function. This is
        because it is assumed that we want to use the same
        feature layers for the entire life of this class.
        """
        feature_nets = []
        for num in self.feature_layer_nums:
            with torch.no_grad() if track_grad else nullcontext():
                features = nn.Sequential(*list(self.vgg.model.features.children())[:num+1])
                features.eval()
                feature_nets += [features]
        return feature_nets
        
    def calculate_gram_matrix(self, layer):
        """Compute the Gram matrix of a batch of features.
        
        In the case of PyTorch, each layer is called a "feature."
        However, by other convention, each layer is actually
        made up of a set of features. This is what Gatys refers
        to when they use the term "features."
        """
        # get the dimensions for each layer
        batch_size, num_features, height, width = layer.size()
        # we transform the 4D layer into 2D feature matrices
        features = layer.view(batch_size * num_features, height * width)
        # next is the straightforward part of actually computing the Gram matrix
        gram = torch.mm(features, features.t())
        gram.div_(batch_size * num_features * height * width)  # Normalize by size
        return gram

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

    def generate_style_image(self, num_epoch = 10, learn_rate = 0.1, base_img_path=None):
        """Generates an image that is similar in style to the original image."""
        gen_feature_nets = self.initialize_feature_nets(track_grad=True)
        
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

        criterion = self.loss_criterion
        # it seems they technically don't use SGD in the paper, but it should be fine
        optimizer = optim.SGD([curr_gen_tensor], lr=learn_rate)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[400, 450], gamma=0.5)

        for _ in range(0, num_epoch):
            optimizer.zero_grad()
            gen_grams = self.get_gram_matrices(curr_gen_tensor, gen_feature_nets, track_grad=True)
            loss = criterion(gen_grams, self.orig_grams)
            if _ % 50 == 0:
                print(loss)
            loss.backward(retain_graph=True) # TODO: figure out whether we actually should retain graph here
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
    path_van_gogh = Path(__file__).resolve().parent.parent / "tests/test_imgs/van_gogh_1.jpg"
    
    myExtractor = StyleExtractor(path_van_gogh, [1, 3, 7, 10, 15, 19], loss_weights=[10, 50, 100, 100, 50, 10])
    print(myExtractor.orig_grams)
    print(len(myExtractor.orig_grams))
    gen_style = myExtractor.generate_style_image(num_epoch=500, learn_rate=100, base_img_path=path_king_crab)
    cv2.imshow('Generated Style Image',gen_style)
    cv2.waitKey()
