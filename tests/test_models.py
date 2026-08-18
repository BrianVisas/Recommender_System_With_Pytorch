import torch

from recommender.models import MLPRecommender


def test_mlp_forward_shape() -> None:
    model = MLPRecommender(n_users=4, n_movies=6, embedding_dim=8)
    users = torch.tensor([0, 1, 2])
    movies = torch.tensor([2, 3, 4])
    output = model(users, movies)
    assert output.shape == (3,)
