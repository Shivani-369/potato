import torch
from torch.utils.data import DataLoader, TensorDataset
from src.model import SimpleCNN
from src.evaluate import evaluate

def test_evaluate_runs():
    # Dummy dataset
    X = torch.randn(8, 3, 128, 128)
    y = torch.randint(0, 3, (8,))
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=2)

    model = SimpleCNN(num_classes=3)
    criterion = torch.nn.CrossEntropyLoss()

    metrics = evaluate(model, loader, criterion, device="cpu")
    assert "loss" in metrics and "accuracy" in metrics, "Evaluation should return loss and accuracy"
