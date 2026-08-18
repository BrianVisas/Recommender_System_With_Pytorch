from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class PreparedData:
    train: pd.DataFrame
    test: pd.DataFrame
    movies: pd.DataFrame
    user_encoder: LabelEncoder
    movie_encoder: LabelEncoder

    @property
    def n_users(self) -> int:
        return len(self.user_encoder.classes_)

    @property
    def n_movies(self) -> int:
        return len(self.movie_encoder.classes_)


def load_movielens_1m(
    ratings_path: str | Path,
    movies_path: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load MovieLens 1M ratings and movie metadata."""
    ratings = pd.read_csv(
        ratings_path,
        sep="::",
        engine="python",
        names=["user_id", "movie_id", "rating", "timestamp"],
        encoding="latin-1",
    )
    movies = pd.read_csv(
        movies_path,
        sep="::",
        engine="python",
        names=["movie_id", "title", "genres"],
        encoding="latin-1",
    )
    return ratings, movies


def _per_user_holdout(
    ratings: pd.DataFrame,
    test_fraction: float = 0.2,
    seed: int = 42,
    min_interactions_for_test: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a deterministic per-user holdout while retaining training history."""
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")

    test_indices: list[int] = []
    for _, user_rows in ratings.groupby("user_idx", sort=False):
        if len(user_rows) < min_interactions_for_test:
            continue
        n_test = max(1, int(round(len(user_rows) * test_fraction)))
        n_test = min(n_test, len(user_rows) - 1)
        test_indices.extend(user_rows.sample(n=n_test, random_state=seed).index.tolist())

    test_mask = ratings.index.isin(test_indices)
    train = ratings.loc[~test_mask].reset_index(drop=True)
    test = ratings.loc[test_mask].reset_index(drop=True)
    return train, test


def prepare_data(
    ratings_path: str | Path,
    movies_path: str | Path,
    test_fraction: float = 0.2,
    seed: int = 42,
) -> PreparedData:
    ratings, movies = load_movielens_1m(ratings_path, movies_path)

    user_encoder = LabelEncoder().fit(ratings["user_id"])
    movie_encoder = LabelEncoder().fit(ratings["movie_id"])

    ratings = ratings.copy()
    ratings["user_idx"] = user_encoder.transform(ratings["user_id"])
    ratings["movie_idx"] = movie_encoder.transform(ratings["movie_id"])

    train, test = _per_user_holdout(ratings, test_fraction=test_fraction, seed=seed)
    return PreparedData(train, test, movies, user_encoder, movie_encoder)


def make_loader(
    frame: pd.DataFrame,
    batch_size: int = 1024,
    shuffle: bool = False,
) -> DataLoader:
    features = torch.tensor(frame[["user_idx", "movie_idx"]].to_numpy(), dtype=torch.long)
    ratings = torch.tensor(frame["rating"].to_numpy(), dtype=torch.float32)
    return DataLoader(TensorDataset(features, ratings), batch_size=batch_size, shuffle=shuffle)
