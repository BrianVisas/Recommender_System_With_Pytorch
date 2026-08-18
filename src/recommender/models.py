from __future__ import annotations

import torch
from torch import nn


class MLPRecommender(nn.Module):
    """Embedding-based neural collaborative filtering model for explicit ratings."""

    def __init__(self, n_users: int, n_movies: int, embedding_dim: int = 32) -> None:
        super().__init__()
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.movie_embedding = nn.Embedding(n_movies, embedding_dim)
        self.network = nn.Sequential(
            nn.Linear(embedding_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, user_ids: torch.Tensor, movie_ids: torch.Tensor) -> torch.Tensor:
        user_emb = self.user_embedding(user_ids)
        movie_emb = self.movie_embedding(movie_ids)
        features = torch.cat((user_emb, movie_emb), dim=-1)
        return self.network(features).squeeze(-1)
