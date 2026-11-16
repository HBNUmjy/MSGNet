import csv
import math
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm

from utils import f_latency, move_to_device


FALSE_POSITIVE_COST = 0.4545


def erde_cost(target, trajectory, offset):
    positive_steps = np.flatnonzero(trajectory)
    prediction = int(positive_steps.size > 0)
    if prediction == 0 and target == 0:
        return 0.0
    if prediction == 0 and target == 1:
        return 1.0
    if prediction == 1 and target == 0:
        return FALSE_POSITIVE_COST
    delay = int(positive_steps[0])
    return 1.0 - 1.0 / (1.0 + math.exp(delay - offset))


def build_prefix_predictions(model, inputs, device, threshold):
    features, lengths, names = inputs
    if not torch.is_tensor(lengths):
        lengths = torch.tensor(lengths, device=device, dtype=torch.long)
    else:
        lengths = lengths.to(device=device, dtype=torch.long)
    max_length = int(lengths.max().item())
    predictions = torch.zeros(
        features.size(0), max_length, dtype=torch.long, device=device
    )

    for prefix_length in range(1, max_length + 1):
        prefix_features = features[:, :prefix_length]
        prefix_lengths = lengths.clamp(max=prefix_length)
        logits = model((prefix_features, prefix_lengths, names)).view(-1)
        active_users = lengths >= prefix_length
        predictions[active_users, prefix_length - 1] = (
            torch.sigmoid(logits[active_users]) >= threshold
        ).long()
    return predictions, lengths


def calculate_delay(model, datasetloader, device, threshold=0.5, save_name="MSGNet"):
    model.eval()
    all_labels = []
    all_predictions = []
    delays = []
    erde_5 = []
    erde_10 = []
    rows = []

    with torch.no_grad():
        for inputs, labels in tqdm(datasetloader, desc="Early detection", unit="batch"):
            inputs = move_to_device(inputs, device)
            labels = labels.to(device).view(-1)
            prefix_predictions, lengths = build_prefix_predictions(
                model, inputs, device, threshold
            )
            names = inputs[2]

            for index in range(labels.size(0)):
                valid_length = int(lengths[index].item())
                trajectory = prefix_predictions[index, :valid_length].cpu().numpy()
                target = int(labels[index].item())
                positive_steps = np.flatnonzero(trajectory)
                delay = int(positive_steps[0]) if positive_steps.size else valid_length
                final_prediction = int(positive_steps.size > 0)

                all_labels.append(target)
                all_predictions.append(final_prediction)
                delays.append(delay if target == 1 else 0)
                erde_5.append(erde_cost(target, trajectory, 5))
                erde_10.append(erde_cost(target, trajectory, 10))
                rows.append(
                    {
                        "name": names[index],
                        "delay": delay if positive_steps.size else "None",
                        "trace": trajectory.tolist(),
                    }
                )

    output_path = Path(f"latency_{save_name}.csv")
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("name", "delay", "trace"))
        writer.writeheader()
        writer.writerows(rows)

    metrics = {
        "f1_early": f1_score(all_labels, all_predictions, zero_division=0),
        "f1_latency": f_latency(
            np.asarray(all_predictions), np.asarray(all_labels), np.asarray(delays)
        ),
        "accuracy": accuracy_score(all_labels, all_predictions),
        "precision": precision_score(all_labels, all_predictions, zero_division=0),
        "recall": recall_score(all_labels, all_predictions, zero_division=0),
        "erde_5": float(np.mean(erde_5)),
        "erde_10": float(np.mean(erde_10)),
    }
    print(metrics)
    return metrics
