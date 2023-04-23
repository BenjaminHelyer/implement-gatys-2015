"""Class that accomplishes the transfer of style onto the content of another image."""
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import numpy as np

from content_extractor import ContentExtractor
from style_extractor import StyleExtractor
from vgg_wrapper import VggWrapper
from img_process_helper import ImgProcessHelper

class TransferTotalLoss(nn.Module):
    """Custom total loss function for style-content composition."""
    def __init__(self, weight_style_loss, weight_content_loss):
        super(TransferTotalLoss, self).__init__()
        self.weight_style_loss = weight_style_loss
        self.weight_contnet_loss = weight_content_loss

    def forward(self, style_loss, content_loss):
        tot_loss = self.weight_style_loss*style_loss + self.weight_contnet_loss*content_loss
        return tot_loss

class Transferer:
    def __init__(self,
                 orig_style_img_path,
                 orig_content_img_path,
                 weight_style_loss=1,
                 weight_content_loss=1):
        """Used for facilitating the transfer of the style of 
        one image onto the content of another."""
        self.vgg = VggWrapper()
        self.img_helper = ImgProcessHelper()

        self.contentExtractor = ContentExtractor(orig_content_img_path)
        self.styleExtractor = StyleExtractor(orig_style_img_path, [1, 3, 7, 10, 15, 19])

        self.loss_criterion = TransferTotalLoss(weight_style_loss, weight_content_loss)
    
    def generate_styled_content(self, num_epoch = 10, learn_rate = 0.1, base_img_path=None):
        """Generates an image with the style of one image and the content of another.
        
        Can use a random white noise image as the starting point or use another
        specified image. Gatys found interesting results by using the content
        image as the starting point.
        """
        style_feature_nets = self.styleExtractor.initialize_feature_nets(track_grad=True)
        content_feature_net = self.contentExtractor.initialize_feature_nets()

        if base_img_path is None:
            img_path = 'rand_img.jpg'
            generated_image = self.img_helper._generate_white_noise_img()
            cv2.imwrite(img_path, generated_image)
            curr_gen_tensor = self.vgg.preprocess_img(img_path)
            curr_gen_tensor.requires_grad = True
        else:
            transformed_gen_img_batch = self.vgg.preprocess_img(base_img_path)
            curr_gen_tensor = transformed_gen_img_batch.clone()
            curr_gen_tensor.requires_grad = True

        content_criterion = nn.MSELoss()
        total_loss_criterion = self.loss_criterion
        # it seems they technically don't use SGD in the paper, but it should be fine
        optimizer = optim.SGD([curr_gen_tensor], lr=learn_rate)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[500, 1000, 1500, 2000], gamma=0.5)

        for _ in range(0, num_epoch):
            optimizer.zero_grad()
            # calculate style loss
            gen_grams = self.styleExtractor.get_gram_matrices(curr_gen_tensor, style_feature_nets, track_grad=True)
            style_loss = self.styleExtractor.loss_criterion(gen_grams, self.styleExtractor.orig_grams)
            # calculate content loss
            gen_content_layer = content_feature_net.forward(curr_gen_tensor)
            content_loss = content_criterion(gen_content_layer, self.contentExtractor.orig_content)
            # now we get the total loss and perform backpropagation
            loss = total_loss_criterion(style_loss, content_loss)
            if _ % 50 == 0:
                print(loss)
            loss.backward(retain_graph=True) # TODO: figure out whether we actually should retain graph here
            optimizer.step() 
            scheduler.step()

        final_img = self.img_helper._postprocess_img(curr_gen_tensor)
        return final_img
    
if __name__ == '__main__':
    cuda = torch.cuda.is_available()
    if cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    
    path_king_crab = Path(__file__).resolve().parent.parent / "tests/test_imgs/alaskan_king_crab.jpg"
    path_english_setter = Path(__file__).resolve().parent.parent / "tests/test_imgs/english_setter.jpg"
    path_modified_crab = Path(__file__).resolve().parent.parent / "tests/test_imgs/modified_alaskan_king_crab.jpg"
    path_van_gogh = Path(__file__).resolve().parent.parent / "tests/test_imgs/van_gogh_1.jpg"

    myTransferer = Transferer(path_van_gogh, path_king_crab, weight_content_loss=0.01)
    generated_img = myTransferer.generate_styled_content(num_epoch=2500, learn_rate=250)
    cv2.imshow('Generated Image with Style Transfer',generated_img)
    cv2.waitKey()
