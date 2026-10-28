import torch

def compute_accuracy(outputs, targets):
    """
    Calcola la Top-1 Accuracy del batch.
    """
    _, preds = torch.max(outputs, dim=1)
    return (preds == targets).sum().item() / targets.size(0)
