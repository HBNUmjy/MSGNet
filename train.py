import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import MDDD, collate_fn
from early_detection import calculate_delay
from models import MSGNet
from utils import FocalLoss, move_to_device


CONFIG_PATH = Path(__file__).with_name("config.yaml")


def parse_args():
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        defaults = yaml.safe_load(handle)

    parser = argparse.ArgumentParser(description="Train and evaluate MSGNet on MUD3.")
    parser.add_argument("--data_dir", type=Path)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--device", type=str)
    parser.add_argument("--save_dir", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--latency_name", type=str)
    parser.set_defaults(**defaults)
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_epoch(model, loader, loss_fn, device, optimizer=None, threshold=0.5):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_samples = 0
    all_labels = []
    all_predictions = []

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for inputs, labels in tqdm(loader, leave=False, unit="batch"):
            inputs = move_to_device(inputs, device)
            labels = labels.to(device=device, dtype=torch.float32).unsqueeze(1)
            logits = model(inputs)
            loss = loss_fn(logits, labels)

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            predictions = (torch.sigmoid(logits) >= threshold).long().view(-1)
            all_predictions.extend(predictions.cpu().tolist())
            all_labels.extend(labels.long().view(-1).cpu().tolist())

    return {
        "loss": total_loss / max(total_samples, 1),
        "accuracy": accuracy_score(all_labels, all_predictions),
        "precision": precision_score(all_labels, all_predictions, zero_division=0),
        "recall": recall_score(all_labels, all_predictions, zero_division=0),
        "f1": f1_score(all_labels, all_predictions, zero_division=0),
    }


def select_threshold(model, loader, device):
    probabilities = []
    labels = []
    model.eval()
    with torch.no_grad():
        for inputs, batch_labels in loader:
            inputs = move_to_device(inputs, device)
            probabilities.extend(torch.sigmoid(model(inputs)).view(-1).cpu().tolist())
            labels.extend(batch_labels.tolist())

    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in np.arange(0.01, 1.0, 0.01):
        predictions = [int(probability >= threshold) for probability in probabilities]
        score = f1_score(labels, predictions, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)
    return best_threshold


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)

    train_loader = DataLoader(
        MDDD(args.data_dir, "train"),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        MDDD(args.data_dir, "val"),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        MDDD(args.data_dir, "test"),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    model = MSGNet().to(device)
    loss_fn = FocalLoss(alpha=0.75, gamma=2.0, reduction="mean")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    output_dir = args.save_dir / "MSGNet"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best_model_MSGNet.pt"
    best_validation_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, loss_fn, device, optimizer)
        validation_metrics = run_epoch(model, val_loader, loss_fn, device)
        print(f"epoch {epoch:03d} train={train_metrics} val={validation_metrics}")
        if validation_metrics["loss"] < best_validation_loss:
            best_validation_loss = validation_metrics["loss"]
            torch.save(model.state_dict(), checkpoint_path)

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    threshold = select_threshold(model, val_loader, device)
    test_metrics = run_epoch(model, test_loader, loss_fn, device, threshold=threshold)
    results = {"threshold": threshold, "test": test_metrics}
    with (output_dir / "result.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print(results)
    calculate_delay(
        model,
        test_loader,
        device,
        threshold=threshold,
        save_name=args.latency_name,
    )


if __name__ == "__main__":
    main()

