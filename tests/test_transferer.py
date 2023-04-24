"""Tests for the transferer class."""
from pathlib import Path
import sys

import pytest
import torch
import numpy as np

sys.path.append("src/")
from transferer import Transferer

test_file_path = Path(__file__).resolve().parent / "test_imgs"

@pytest.mark.parametrize("base_img_path", 
                         [(test_file_path / 'alaskan_king_crab.jpg'),
                          (test_file_path / 'english_setter.jpg')])
def test_transferer_same_img(base_img_path):
    """Tests the style-content composition function with the same image.

    Using the same image for the style, content, and base image should
    yield the same image as output.
    """
    cuda = torch.cuda.is_available()
    if cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    uutTransferer = Transferer(base_img_path, base_img_path)

    composed_img = uutTransferer.generate_styled_content(base_img_path=base_img_path)
    orig_image_final = uutTransferer.img_helper._postprocess_img(uutTransferer.styleExtractor.orig_img)

    np.testing.assert_array_equal(orig_image_final, composed_img)
