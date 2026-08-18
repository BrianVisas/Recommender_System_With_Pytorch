import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPRecommender(nn.Module):
    def __init__(self, n_users, n_movies, embedding_dim=32):
        super(MLPRecommender, self).__init__()
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.movie_embedding = nn.Embedding(n_movies, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim * 2, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, user_ids, movie_ids):
        user_emb = self.user_embedding(user_ids)
        movie_emb = self.movie_embedding(movie_ids)
        x = torch.cat([user_emb, movie_emb], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class CNNRecommender(nn.Module):
    def __init__(self, n_users, n_movies, embedding_dim=32):
        super(CNNRecommender, self).__init__()
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.movie_embedding = nn.Embedding(n_movies, embedding_dim)
        self.conv1 = nn.Conv1d(embedding_dim * 2, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, user_ids, movie_ids):
        user_emb = self.user_embedding(user_ids).unsqueeze(2)
        movie_emb = self.movie_embedding(movie_ids).unsqueeze(2)
        x = torch.cat([user_emb, movie_emb], dim=1)
        x = F.relu(self.conv1(x)).squeeze(2)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

