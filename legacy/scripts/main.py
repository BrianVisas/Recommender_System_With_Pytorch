import torch
import torch.nn as nn
from torch.optim import Adam
import matplotlib.pyplot as plt
from preprocess import preprocess_data
from models import MLPRecommender, CNNRecommender
from train import train_model
from validate import evaluate_model_with_metrics
import pickle
import os


def plot_metrics(epochs, metrics, labels, title, ylabel, filename, breakpoints=None):
    
    plt.figure(figsize=(10, 6))
    
    for i, metric in enumerate(metrics):
        plt.plot(epochs, metric, marker='o', label=labels[i])
        
        # Highlight breakpoints if provided
        if breakpoints:
            for bp in breakpoints:
                if bp < len(metric):
                    plt.scatter(
                        epochs[bp],
                        metric[bp],
                        color='red',
                        marker='*',
                        s=150,  # Size of the star marker
                        label=f"Breakpoint: {labels[i]} Epoch {epochs[bp]}"
                    )
    
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.show()


def main():
    # Paths to dataset
    ratings_path = "/Users/brianvisas/Downloads/ml-1M/ratings.dat"
    movies_path = "/Users/brianvisas/Downloads/ml-1M/movies.dat"
    
    # Preprocess data
    print("Preprocessing data...")
    train_loader, test_loader, n_users, n_movies = preprocess_data(ratings_path, movies_path)
    
    # Define device
    device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # Training configurations
    n_epochs = 200
    embedding_dim = 32
    learning_rate = 0.001
    k_values = [10, 20, 50]
    
    # Create directories to save models and results
    if not os.path.exists("models"):
        os.makedirs("models")
    if not os.path.exists("results"):
        os.makedirs("results")
    
    # Initialize models
    mlp_model = MLPRecommender(n_users, n_movies, embedding_dim).to(device)
    cnn_model = CNNRecommender(n_users, n_movies, embedding_dim).to(device)
    
    # Define loss function and optimizers
    criterion = nn.MSELoss()
    mlp_optimizer = Adam(mlp_model.parameters(), lr=learning_rate)
    cnn_optimizer = Adam(cnn_model.parameters(), lr=learning_rate)
    
    # Metrics storage
    metrics = {
        "mlp": {"loss": [], "rmse": [], "mae": [], "precision": {k: [] for k in k_values}, "recall": {k: [] for k in k_values}},
        "cnn": {"loss": [], "rmse": [], "mae": [], "precision": {k: [] for k in k_values}, "recall": {k: [] for k in k_values}}
    }
    
    # Training loop for MLP model
    print("\nTraining MLP Model...")
    for epoch in range(1, n_epochs + 1):
        print(f"\nEpoch {epoch}/{n_epochs}")
        train_loss = train_model(mlp_model, train_loader, criterion, mlp_optimizer, device)
        metrics["mlp"]["loss"].append(train_loss)
        
        # Evaluate and process metrics
        metrics_result = evaluate_model_with_metrics(mlp_model, test_loader, device, k_values=k_values)
        metrics["mlp"]["rmse"].append(metrics_result["RMSE"])
        metrics["mlp"]["mae"].append(metrics_result["MAE"])
        
        for k in k_values:
            metrics["mlp"]["precision"][k].append(metrics_result[f"Precision@{k}"])
            metrics["mlp"]["recall"][k].append(metrics_result[f"Recall@{k}"])
        
        # Save model
        torch.save(mlp_model.state_dict(), f"models/mlp_model_epoch_{epoch}.pth")
        print(f"MLP Model saved at models/mlp_model_epoch_{epoch}.pth")
    
    # Training loop for CNN model
    print("\nTraining CNN Model...")
    for epoch in range(1, n_epochs + 1):
        print(f"\nEpoch {epoch}/{n_epochs}")
        train_loss = train_model(cnn_model, train_loader, criterion, cnn_optimizer, device)
        metrics["cnn"]["loss"].append(train_loss)
        
        # Evaluate and process metrics
        metrics_result = evaluate_model_with_metrics(cnn_model, test_loader, device, k_values=k_values)
        metrics["cnn"]["rmse"].append(metrics_result["RMSE"])
        metrics["cnn"]["mae"].append(metrics_result["MAE"])
        
        for k in k_values:
            metrics["cnn"]["precision"][k].append(metrics_result[f"Precision@{k}"])
            metrics["cnn"]["recall"][k].append(metrics_result[f"Recall@{k}"])
        
        # Save model
        torch.save(cnn_model.state_dict(), f"models/cnn_model_epoch_{epoch}.pth")
        print(f"CNN Model saved at models/cnn_model_epoch_{epoch}.pth")
    
    # Save training metrics
    with open("results/training_metrics.pkl", "wb") as f:
        pickle.dump(metrics, f)
    print("\nTraining metrics saved at results/training_metrics.pkl")
    
    # Plot metrics
    epochs = list(range(1, n_epochs + 1))
    
    # Loss Plot
    plot_metrics(epochs, [metrics["mlp"]["loss"], metrics["cnn"]["loss"]],
                 labels=["MLP Loss", "CNN Loss"], title="Training Loss per Epoch",
                 ylabel="Loss", filename="results/training_loss.png")
    
    # RMSE Plot
    plot_metrics(epochs, [metrics["mlp"]["rmse"], metrics["cnn"]["rmse"]],
                 labels=["MLP RMSE", "CNN RMSE"], title="Validation RMSE per Epoch",
                 ylabel="RMSE", filename="results/validation_rmse.png")
    
    # MAE Plot
    plot_metrics(epochs, [metrics["mlp"]["mae"], metrics["cnn"]["mae"]],
                 labels=["MLP MAE", "CNN MAE"], title="Validation MAE per Epoch",
                 ylabel="MAE", filename="results/validation_mae.png")
    
    # Precision@k and Recall@k Plots
    for k in k_values:
        plot_metrics(epochs, [metrics["mlp"]["precision"][k], metrics["cnn"]["precision"][k]],
                     labels=[f"MLP Precision@{k}", f"CNN Precision@{k}"],
                     title=f"Precision@{k} Comparison: MLP vs CNN",
                     ylabel=f"Precision@{k}", filename=f"results/precision_at_{k}.png")
        
        plot_metrics(epochs, [metrics["mlp"]["recall"][k], metrics["cnn"]["recall"][k]],
                     labels=[f"MLP Recall@{k}", f"CNN Recall@{k}"],
                     title=f"Recall@{k} Comparison: MLP vs CNN",
                     ylabel=f"Recall@{k}", filename=f"results/recall_at_{k}.png")


if __name__ == "__main__":
    main()
