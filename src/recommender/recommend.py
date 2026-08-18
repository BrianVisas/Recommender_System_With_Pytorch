from __future__ import annotations

import pandas as pd
import torch
from torch import nn


def recommend_top_n(
    model: nn.Module,
    user_idx: int,
    n_movies: int,
    seen_movie_indices: set[int],
    device: torch.device,
    top_n: int = 10,
) -> list[tuple[int, float]]:
    """Recommend unseen movies for one encoded user."""
    candidates = [movie for movie in range(n_movies) if movie not in seen_movie_indices]
    if not candidates:
        return []

    model.eval()
    with torch.no_grad():
        movie_tensor = torch.tensor(candidates, dtype=torch.long, device=device)
        user_tensor = torch.full_like(movie_tensor, user_idx)
        scores = model(user_tensor, movie_tensor)
        count = min(top_n, len(candidates))
        positions = torch.topk(scores, k=count).indices.tolist()

    return [(candidates[position], float(scores[position].item())) for position in positions]


def attach_movie_metadata(
    recommendations: list[tuple[int, float]],
    movie_encoder,
    movies: pd.DataFrame,
) -> pd.DataFrame:
    if not recommendations:
        return pd.DataFrame(columns=["movie_id", "title", "genres", "score"])

    encoded_ids = [movie for movie, _ in recommendations]
    original_ids = movie_encoder.inverse_transform(encoded_ids)
    score_map = {
        int(movie_id): score
        for movie_id, (_, score) in zip(original_ids, recommendations, strict=True)
    }
    result = movies[movies["movie_id"].isin(original_ids)].copy()
    result["score"] = result["movie_id"].map(score_map)
    return result.sort_values("score", ascending=False).reset_index(drop=True)
