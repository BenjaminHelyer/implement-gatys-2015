"""Tests for the content extractor class."""
from pathlib import Path
import sys

import pytest
import torch
import numpy as np

sys.path.append("src/")
from style_extractor import StyleExtractor

test_file_path = Path(__file__).resolve().parent / "test_imgs"

@pytest.mark.parametrize("base_img_path", 
                         [(test_file_path / 'alaskan_king_crab.jpg'),
                          (test_file_path / 'english_setter.jpg')])
def test_style_extraction_same_img(base_img_path):
    """Tests the style extraction function, giving it the same image as input."""
    cuda = torch.cuda.is_available()
    if cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    uutStyleExtractor = StyleExtractor(base_img_path, [1, 3])

    generated_style = uutStyleExtractor.generate_style_image(base_img_path=base_img_path)
    orig_image_final = uutStyleExtractor.img_helper._postprocess_img(uutStyleExtractor.orig_img)

    np.testing.assert_array_equal(orig_image_final, generated_style)
