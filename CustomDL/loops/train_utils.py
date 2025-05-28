import math
import torch

from torch import nn
from torch.utils.data import DataLoader


def train_loop(
    model: nn.Module,
    train_loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: nn.Module,
    track_loss: bool = False,
    use_gpu: bool = False
) -> list[float]:
    """
    Performs backpropogation on `model` using `optimizer`.

    :param nn.Module model: The model on which to perform backpropogation.
    :param nn.utils.data.DataLoader train_loader: A DataLoader dispatching batches
        for each backpropogations.
    :param nn.Module loss_fn: The loss function to based on which to compute gradients.
    :param nn.Module optimizer: The optimization algorithm for gradient descent.
    :param bool track_loss: Whether or not to return loss on each backpropogation.
        This is `False` by default.
    :return: A list of loss values per batch if `track_loss=True` else an empty list.
    :rtype: list[float]
    """
    model.train()
    losses = []

    for batch, (X, y) in enumerate(train_loader, start=1):
        if use_gpu:
            X = X.cuda()
            y = y.cuda()
        pred_value = model(X)
        loss = loss_fn(pred_value, y)

        # Compute the gradient with loss.backward()
        # Then backpropogate with optimizer.step()
        # However, to avoid accumulation of previous backward passes
        # we need to call optimizer.zero_grad() to zero out the gradient
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if track_loss: losses.append(loss.item())
    return losses

@torch.no_grad()
def test_loop(
    model: nn.Module,
    test_loader: DataLoader,
    loss_fn: nn.Module,
    compute_accuracy: bool,
    use_gpu: bool = False
) -> tuple[float, float]:
    """
    Evaluate `model` based on `loss_fn` and return the average score(s).

    :param nn.Module model: The model on which to perform evaluation.
    :param nn.utils.data.DataLoader test_loader: A DataLoader containing test data.
    :param nn.Module loss_fn: The loss function to based on which to compute metrics.
    :param bool compute_accuracy: Whether or not to compute accuracy. This is only
        meaningful in the case the `model` is a classifier.
    :return: The average loss (per batch) and average accuracy (per sample). If
        `compute_accuracy=False` then average accuracy returned is 0.
    :rtype: tuple[float, float]
    """
    model.eval()
    total_loss, total_accuracy = 0, 0
    for X, y in test_loader:
        if use_gpu:
            X = X.cuda()
            y= y.cuda()
        pred = model(X)
        total_loss += loss_fn(pred, y)
        if compute_accuracy:
            labels = (pred.argmax(dim=1) == y)
            total_accuracy += labels.type(torch.int).sum().item()
    return total_loss / len(test_loader), total_accuracy / len(test_loader.dataset)