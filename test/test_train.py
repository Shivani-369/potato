import torch
from torch.utils.data import DataLoader, TensorDataset
from src.model import SimpleCNN
from src.train import train_one_epoch

def test_train_one_epoch_runs():
    # Dummy dataset
    X = torch.randn(10, 3, 128, 128)
    y = torch.randint(0, 3, (10,))
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=2)

    model = SimpleCNN(num_classes=3)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()

    # Just ensure it runs without error
    loss = train_one_epoch(model, loader, criterion, optimizer, device="cpu")
    assert isinstance(loss, float), "Loss should be a float"
