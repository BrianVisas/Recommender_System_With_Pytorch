import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import pickle


def preprocess_data(ratings_path, movies_path, save_encoders=True, batch_size=256):
  
    # Load datasets with appropriate encoding
    print("Loading datasets...")
    ratings = pd.read_csv(ratings_path, delimiter="::", engine="python",
                          names=["userId", "movieId", "rating", "timestamp"], encoding="latin-1")
    movies = pd.read_csv(movies_path, delimiter="::", engine="python",
                         names=["movieId", "title", "genres"], encoding="latin-1")

    # Normalize ratings to range [0, 1]
    print("Normalizing ratings...")
    scaler = MinMaxScaler(feature_range=(0, 1))
    ratings["rating"] = scaler.fit_transform(ratings[["rating"]])

    # Encode userId and movieId
    print("Encoding user and movie IDs...")
    user_encoder = LabelEncoder()
    movie_encoder = LabelEncoder()

    ratings["userId"] = user_encoder.fit_transform(ratings["userId"])
    ratings["movieId"] = movie_encoder.fit_transform(ratings["movieId"])

    n_users = len(user_encoder.classes_)
    n_movies = len(movie_encoder.classes_)
    print(f"Number of users: {n_users}, Number of movies: {n_movies}")

    # Save encoders for consistent recommendation
    if save_encoders:
        print("Saving encoders...")
        with open("user_encoder.pkl", "wb") as f:
            pickle.dump(user_encoder, f)
        with open("movie_encoder.pkl", "wb") as f:
            pickle.dump(movie_encoder, f)

    # Split data into train and test sets
    print("Splitting data into training and testing sets...")
    X = ratings[["userId", "movieId"]]
    y = ratings["rating"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create DataLoaders
    print("Creating DataLoaders...")
    train_loader = create_data_loader(X_train, y_train, batch_size=batch_size, shuffle=True)
    test_loader = create_data_loader(X_test, y_test, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, n_users, n_movies


def create_data_loader(X, y, batch_size=256, shuffle=True):
    """
    Create a PyTorch DataLoader from the input data.

    Args:
        X (pd.DataFrame): Features (userId, movieId).
        y (pd.Series): Target labels (ratings).
        batch_size (int): Batch size for DataLoader.
        shuffle (bool): Whether to shuffle the data.

    Returns:
        DataLoader: A PyTorch DataLoader object.
    """
    # Convert data to PyTorch tensors
    print("Converting data to PyTorch tensors...")
    X_tensor = torch.tensor(X.values, dtype=torch.long)
    y_tensor = torch.tensor(y.values, dtype=torch.float32)

    # Create a TensorDataset
    dataset = TensorDataset(X_tensor, y_tensor)

    # Create and return a DataLoader
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


if __name__ == "__main__":
    # File paths
    ratings_path = "/Users/brianvisas/Downloads/ml-1M/ratings.dat"
    movies_path = "/Users/brianvisas/Downloads/ml-1M/movies.dat"

    # Preprocess data
    print("Preprocessing data...")
    train_loader, test_loader, n_users, n_movies = preprocess_data(ratings_path, movies_path)

    # Verify DataLoaders
    print("Verifying DataLoaders...")
    for batch in train_loader:
        user_movie_ids, ratings = batch
        print(f"User-Movie IDs shape: {user_movie_ids.shape}")
        print(f"Ratings shape: {ratings.shape}")
        break  # Display only the first batch

    print("Preprocessing completed successfully.")
