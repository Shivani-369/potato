import torch
from src.model import SimpleCNN

def test_model_forward_pass():
    model = SimpleCNN(num_classes=3)  # example with 3 classes
    dummy_input = torch.randn(2, 3, 128, 128)  # batch_size=2
    output = model(dummy_input)
    assert output.shape == (2, 3), "Output shape mismatch"
