import pytest
import torch

@pytest.fixture
def dummy_input():
    return torch.randn(2, 3, 128, 128)

@pytest.fixture
def dummy_labels():
    return torch.randint(0, 3, (2,))
