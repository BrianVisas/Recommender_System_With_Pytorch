import torch

def train_model(model, train_loader, criterion, optimizer, device):
   
    model.train()
    total_loss = 0
    for batch in train_loader:
        user_movie_ids, ratings = batch
        user_ids = user_movie_ids[:, 0].to(device)
        movie_ids = user_movie_ids[:, 1].to(device)
        ratings = ratings.to(device)

        optimizer.zero_grad()
        predictions = model(user_ids, movie_ids).squeeze()
        loss = criterion(predictions, ratings)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)
    print(f"Training Loss: {average_loss:.4f}")
    return average_loss

def compute_precision_recall_at_k(model, test_loader, k=10, threshold=4.0, device=torch.device("cpu")):
   
    precisions = []
    recalls = []
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            user_movie_ids, ratings = batch
            user_ids = user_movie_ids[:, 0].to(device)
            movie_ids = user_movie_ids[:, 1].to(device)
            ratings = ratings.to(device)

            predictions = model(user_ids, movie_ids).squeeze()
            _, top_k_indices = torch.topk(predictions, k)
            top_k_movie_ids = movie_ids[top_k_indices]
            
            relevant_items = (ratings >= threshold).nonzero(as_tuple=True)[0]
            top_k_relevant = (ratings[top_k_indices] >= threshold).nonzero(as_tuple=True)[0]

            precision = len(set(top_k_relevant.tolist()).intersection(relevant_items.tolist())) / k
            recall = len(set(top_k_relevant.tolist()).intersection(relevant_items.tolist())) / len(relevant_items) if len(relevant_items) > 0 else 0.0

            precisions.append(precision)
            recalls.append(recall)

    precision_at_k = sum(precisions) / len(precisions)
    recall_at_k = sum(recalls) / len(recalls)

    return precision_at_k, recall_at_k

