import torch
from train import compute_precision_recall_at_k


def evaluate_model(model, test_loader, device):
    """
    Evaluate the model on the test set.
    
    Args:
        model: PyTorch model.
        test_loader: DataLoader for testing.
        device: Computation device.
    
    Returns:
        Tuple of (RMSE, MAE).
    """
    model.eval()
    mse_loss = 0
    mae_loss = 0
    with torch.no_grad():
        for batch in test_loader:
            user_movie_ids, ratings = batch
            user_ids = user_movie_ids[:, 0].to(device)
            movie_ids = user_movie_ids[:, 1].to(device)
            ratings = ratings.to(device)

            predictions = model(user_ids, movie_ids).squeeze()
            mse_loss += ((predictions - ratings) ** 2).sum().item()
            mae_loss += torch.abs(predictions - ratings).sum().item()

    rmse = (mse_loss / len(test_loader.dataset)) ** 0.5
    mae = mae_loss / len(test_loader.dataset)
    print(f"Validation RMSE: {rmse:.4f}, MAE: {mae:.4f}")
    return rmse, mae

def evaluate_model_with_metrics(model, test_loader, device, k_values=[10, 20, 50]):
    """
    Evaluate the model on the test set with additional metrics.

    Args:
        model: PyTorch model.
        test_loader: DataLoader for testing.
        device: Computation device.
        k_values: List of k values for Precision@k and Recall@k.

    Returns:
        Dictionary of metrics (RMSE, MAE, Precision@k, Recall@k for each k).
    """
    rmse, mae = evaluate_model(model, test_loader, device)
    metrics = {"RMSE": rmse, "MAE": mae}
    
    for k in k_values:
        precision_at_k, recall_at_k = compute_precision_recall_at_k(model, test_loader, k=k, threshold=0.8, device=device)
        metrics[f"Precision@{k}"] = precision_at_k
        metrics[f"Recall@{k}"] = recall_at_k
    
    return metrics


