import math

import pandas as pd
import torch
from torch import nn

from recommender.metrics import evaluate_ranking, ndcg_at_k, rating_metrics


class LookupModel(nn.Module):
    def forward(self, user_ids: torch.Tensor, movie_ids: torch.Tensor) -> torch.Tensor:
        return movie_ids.float()


def test_rating_metrics() -> None:
    result = rating_metrics(torch.tensor([3.0, 5.0]), torch.tensor([4.0, 5.0]))
    assert math.isclose(result["rmse"], math.sqrt(0.5), rel_tol=1e-6)
    assert math.isclose(result["mae"], 0.5, rel_tol=1e-6)


def test_ndcg_is_one_for_ideal_ranking() -> None:
    assert math.isclose(ndcg_at_k([4, 2, 1], {4, 2}, 2), 1.0)


def test_ranking_is_computed_per_user_and_excludes_seen_items() -> None:
    train = pd.DataFrame(
        {"user_idx": [0, 1], "movie_idx": [3, 2], "rating": [5.0, 5.0]}
    )
    test = pd.DataFrame(
        {"user_idx": [0, 1], "movie_idx": [2, 3], "rating": [5.0, 5.0]}
    )

    result = evaluate_ranking(
        LookupModel(),
        train,
        test,
        n_movies=4,
        device=torch.device("cpu"),
        k=1,
    )

    assert result["users"] == 2.0
    assert result["precision@1"] == 1.0
    assert result["recall@1"] == 1.0
