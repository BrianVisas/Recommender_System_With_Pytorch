from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.optim import Adam

from .data import make_loader, prepare_data
from .metrics import build_seen_items, evaluate_ranking
from .models import MLPRecommender
from .recommend import attach_movie_metadata, recommend_top_n
from .train import evaluate_ratings, save_checkpoint, train_epoch


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_command(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    data = prepare_data(args.ratings, args.movies, test_fraction=args.test_fraction, seed=args.seed)
    train_loader = make_loader(data.train, batch_size=args.batch_size, shuffle=True)
    test_loader = make_loader(data.test, batch_size=args.batch_size, shuffle=False)

    model = MLPRecommender(data.n_users, data.n_movies, args.embedding_dim).to(device)
    optimizer = Adam(model.parameters(), lr=args.learning_rate)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    best_rmse = float("inf")
    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        metrics = evaluate_ratings(model, test_loader, device)
        row = {"epoch": float(epoch), "train_mse": train_loss, **metrics}
        history.append(row)
        print(json.dumps(row))
        if metrics["rmse"] < best_rmse:
            best_rmse = metrics["rmse"]
            save_checkpoint(model, output / "best_model.pt")

    save_checkpoint(model, output / "last_model.pt")
    ranking = evaluate_ranking(
        model,
        data.train,
        data.test,
        data.n_movies,
        device,
        k=args.k,
        relevance_threshold=args.relevance_threshold,
    )
    summary = {"device": str(device), "best_rmse": best_rmse, **ranking, "history": history}
    (output / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(ranking))


def recommend_command(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    data = prepare_data(args.ratings, args.movies, test_fraction=args.test_fraction, seed=args.seed)
    model = MLPRecommender(data.n_users, data.n_movies, args.embedding_dim).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))

    if args.user_id not in set(data.user_encoder.classes_.tolist()):
        raise SystemExit(f"Unknown MovieLens user_id: {args.user_id}")

    user_idx = int(data.user_encoder.transform([args.user_id])[0])
    seen = build_seen_items(data.train).get(user_idx, set())
    recommendations = recommend_top_n(model, user_idx, data.n_movies, seen, device, args.top_n)
    table = attach_movie_metadata(recommendations, data.movie_encoder, data.movies)
    print(table[["title", "genres", "score"]].to_string(index=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and query a neural MovieLens recommender.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--ratings", required=True, help="Path to MovieLens ratings.dat")
    common.add_argument("--movies", required=True, help="Path to MovieLens movies.dat")
    common.add_argument("--embedding-dim", type=int, default=32)
    common.add_argument("--test-fraction", type=float, default=0.2)
    common.add_argument("--seed", type=int, default=42)
    common.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")

    train_parser = subparsers.add_parser("train", parents=[common])
    train_parser.add_argument("--epochs", type=int, default=20)
    train_parser.add_argument("--batch-size", type=int, default=1024)
    train_parser.add_argument("--learning-rate", type=float, default=1e-3)
    train_parser.add_argument("--k", type=int, default=10)
    train_parser.add_argument("--relevance-threshold", type=float, default=4.0)
    train_parser.add_argument("--output", default="artifacts")
    train_parser.set_defaults(func=train_command)

    recommend_parser = subparsers.add_parser("recommend", parents=[common])
    recommend_parser.add_argument("--model", required=True)
    recommend_parser.add_argument("--user-id", type=int, required=True)
    recommend_parser.add_argument("--top-n", type=int, default=10)
    recommend_parser.set_defaults(func=recommend_command)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
