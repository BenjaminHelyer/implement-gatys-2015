"""Tests for the VggWrapper module."""

from pathlib import Path
import sys

import pytest
import torch

sys.path.append("src/")
from vgg_wrapper import VggWrapper

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
        
    uutVggWrapper = VggWrapper()
    img_tensor = uutVggWrapper.preprocess_img(img_path)
    actual_preds = uutVggWrapper.get_top_k_predictions(img_tensor)
    print(actual_preds) # turn on viewing these preds with -s in pytest
    for pred in actual_preds:
        if pred[1] == expected_pred:
            assert True
            return 
    assert False