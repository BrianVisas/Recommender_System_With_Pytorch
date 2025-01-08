import pandas as pd
import pickle
import torch
from models import MLPRecommender  # Import your MLP model

# File paths
ratings_path = "/Users/brianvisas/Downloads/ml-1M/ratings.dat"
movies_path = "/Users/brianvisas/Downloads/ml-1M/movies.dat"
model_path = "models/mlp_model_epoch_10.pth"

# Load encoders
with open("user_encoder.pkl", "rb") as f:
    user_encoder = pickle.load(f)
with open("movie_encoder.pkl", "rb") as f:
    movie_encoder = pickle.load(f)

# Load movies with appropriate encoding
movies = pd.read_csv(
    movies_path,
    delimiter="::",
    engine="python",
    names=["movieId", "title", "genres"],
    encoding="latin-1",  # Fixes the UnicodeDecodeError
)

# Load the trained model
device = torch.device("cpu")
n_users = len(user_encoder.classes_)
n_movies = len(movie_encoder.classes_)
model = MLPRecommender(n_users, n_movies)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Make recommendations for a specific user
user_id = 5  # Example user ID
original_user_id = user_encoder.inverse_transform([user_id])[0]  # Convert back to original ID
all_movie_ids = list(range(n_movies))  # Create a list of all movie IDs

# Prepare tensors for prediction
user_tensor = torch.tensor([user_id] * n_movies, dtype=torch.long).to(device)
movie_tensor = torch.tensor(all_movie_ids, dtype=torch.long).to(device)

# Get predictions
with torch.no_grad():
    predictions = model(user_tensor, movie_tensor).squeeze()

# Get top-10 recommendations
top_k = 10
top_k_indices = torch.topk(predictions, top_k).indices.cpu().numpy()
recommended_movie_ids = movie_encoder.inverse_transform(top_k_indices)
recommended_movies = movies[movies["movieId"].isin(recommended_movie_ids)]

# Display recommendations
print(f"Top-{top_k} recommendations for user {original_user_id}:")
for _, row in recommended_movies.iterrows():
    print(f"{row['title']} ({row['genres']})")
