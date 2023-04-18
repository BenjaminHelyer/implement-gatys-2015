"""Tests for the content extractor class."""
from pathlib import Path

import pytest

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
    uutContentExtractor = ContentExtractor(img_path)
    actual_preds = uutContentExtractor.get_top_k_predictions()
    print(actual_preds) # turn on viewing these preds with -s in pytest
    for pred in actual_preds:
        if pred[1] == expected_pred:
            assert True
            return 
    assert False
