from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from .metrics import rating_metrics


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    criterion = nn.MSELoss()
    weighted_loss = 0.0
    sample_count = 0

    for features, ratings in loader:
        users = features[:, 0].to(device)
        movies = features[:, 1].to(device)
        ratings = ratings.to(device)

        optimizer.zero_grad(set_to_none=True)
        predictions = model(users, movies)
        loss = criterion(predictions, ratings)
        loss.backward()
        optimizer.step()

        batch_size = ratings.numel()
        weighted_loss += loss.item() * batch_size
        sample_count += batch_size

    return weighted_loss / max(sample_count, 1)


def evaluate_ratings(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []

    with torch.no_grad():
        for features, ratings in loader:
            users = features[:, 0].to(device)
            movies = features[:, 1].to(device)
            predictions.append(model(users, movies).cpu())
            targets.append(ratings.cpu())

    if not predictions:
        raise ValueError("evaluation loader is empty")
    return rating_metrics(torch.cat(predictions), torch.cat(targets))


def save_checkpoint(model: nn.Module, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
