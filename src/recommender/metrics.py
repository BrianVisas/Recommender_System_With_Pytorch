from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable

import pandas as pd
import torch
from torch import nn


def rating_metrics(predictions: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    """Compute RMSE and MAE for explicit-rating predictions."""
    if predictions.numel() == 0:
        raise ValueError("predictions must not be empty")
    errors = predictions.float() - targets.float()
    return {
        "rmse": float(torch.sqrt(torch.mean(errors**2)).item()),
        "mae": float(torch.mean(torch.abs(errors)).item()),
    }


def ndcg_at_k(recommended: Iterable[int], relevant: set[int], k: int) -> float:
    """Binary-relevance NDCG@K."""
    ranked = list(recommended)[:k]
    dcg = sum((1.0 / math.log2(rank + 2)) for rank, item in enumerate(ranked) if item in relevant)
    ideal_hits = min(len(relevant), k)
    if ideal_hits == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(rank + 2) for rank in range(ideal_hits))
    return dcg / idcg


def build_seen_items(train: pd.DataFrame) -> dict[int, set[int]]:
    seen: dict[int, set[int]] = defaultdict(set)
    for row in train[["user_idx", "movie_idx"]].itertuples(index=False):
        seen[int(row.user_idx)].add(int(row.movie_idx))
    return dict(seen)


def evaluate_ranking(
    model: nn.Module,
    train: pd.DataFrame,
    test: pd.DataFrame,
    n_movies: int,
    device: torch.device,
    k: int = 10,
    relevance_threshold: float = 4.0,
    score_batch_size: int = 4096,
) -> dict[str, float]:
    """Evaluate Top-K ranking independently for each eligible user.

    Candidates are movies not present in the user's training history. Precision@K,
    Recall@K and NDCG@K are macro-averaged across users with at least one relevant
    held-out item.
    """
    if k <= 0:
        raise ValueError("k must be positive")

    seen_items = build_seen_items(train)
    all_movies = set(range(n_movies))
    user_results: list[tuple[float, float, float]] = []

    model.eval()
    with torch.no_grad():
        for user_idx, user_test in test.groupby("user_idx"):
            relevant = set(
                int(movie_idx)
                for movie_idx in user_test.loc[
                    user_test["rating"] >= relevance_threshold,
                    "movie_idx",
                ].tolist()
            )
            if not relevant:
                continue

            candidates = sorted(all_movies - seen_items.get(int(user_idx), set()))
            if not candidates:
                continue

            scores: list[torch.Tensor] = []
            for start in range(0, len(candidates), score_batch_size):
                batch_movies = candidates[start : start + score_batch_size]
                movie_tensor = torch.tensor(batch_movies, dtype=torch.long, device=device)
                user_tensor = torch.full_like(movie_tensor, int(user_idx))
                scores.append(model(user_tensor, movie_tensor).detach().cpu())

            candidate_scores = torch.cat(scores)
            top_n = min(k, len(candidates))
            top_positions = torch.topk(candidate_scores, k=top_n).indices.tolist()
            recommended = [candidates[position] for position in top_positions]
            hits = len(set(recommended) & relevant)

            precision = hits / top_n
            recall = hits / len(relevant)
            ndcg = ndcg_at_k(recommended, relevant, top_n)
            user_results.append((precision, recall, ndcg))

    if not user_results:
        return {
            f"precision@{k}": 0.0,
            f"recall@{k}": 0.0,
            f"ndcg@{k}": 0.0,
            "users": 0.0,
        }

    count = len(user_results)
    return {
        f"precision@{k}": sum(result[0] for result in user_results) / count,
        f"recall@{k}": sum(result[1] for result in user_results) / count,
        f"ndcg@{k}": sum(result[2] for result in user_results) / count,
        "users": float(count),
    }
