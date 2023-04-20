"""Tests for the content extractor class."""
from pathlib import Path
import sys

import pytest
import torch
import numpy as np

sys.path.append("src/")
from content_extractor import ContentExtractor

test_file_path = Path(__file__).resolve().parent / "test_imgs"

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
    orig_image_final = uutContentExtractor._postprocess_img(uutContentExtractor.orig_img)

    np.testing.assert_array_equal(orig_image_final, generated_content)
