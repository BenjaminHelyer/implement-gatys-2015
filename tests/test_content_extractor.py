"""Tests for the content extractor class."""
from pathlib import Path

import pytest
import torch
import cv2
import numpy as np

from src.content_extractor import ContentExtractor

test_file_path = Path(__file__).resolve().parent / "test_imgs"

@pytest.mark.parametrize("img_path,expected_pred", 
                         [(test_file_path / 'alaskan_king_crab.jpg','king_crab'),
                          (test_file_path / 'english_setter.jpg', 'English_setter')])
def test_vgg_predictions(img_path,expected_pred):
    """Tests the top predictions from the VGG model.
    
    Acts as a sanity test to ensure the model which
    ContentExtractor uses is yielding plausible predictions.
    """
    cuda = torch.cuda.is_available()
    if cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        
    uutContentExtractor = ContentExtractor(img_path)
    actual_preds = uutContentExtractor.get_top_k_predictions()
    print(actual_preds) # turn on viewing these preds with -s in pytest
    for pred in actual_preds:
        if pred[1] == expected_pred:
            assert True
            return 
    assert False

@pytest.mark.parametrize("base_img_path", 
                         [(test_file_path / 'alaskan_king_crab.jpg'),
                          (test_file_path / 'english_setter.jpg')])
def test_content_extraction_same_img(base_img_path):
    """Tests the content extraction function, giving it the same image as input.
    
    Since the original image and content extraction base image are the same,
    the loss between the two feature maps should also be the same. Thus,
    the resultant generated image should be the same as the original image.
    """
    cuda = torch.cuda.is_available()
    if cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    uutContentExtractor = ContentExtractor(base_img_path)

    generated_content = uutContentExtractor.generate_content_image(base_img_path=base_img_path)

    # note that we do have to do some post-processing, but this is fair, this is the exact
    # same steps that we did for the generated image
    orig_image_post_transform = uutContentExtractor.orig_img.squeeze(0).cpu().numpy().transpose()
    orig_image_final = cv2.cvtColor(orig_image_post_transform, cv2.COLOR_BGR2RGB)

    np.testing.assert_array_equal(orig_image_final, generated_content)
